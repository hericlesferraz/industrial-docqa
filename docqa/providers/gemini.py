"""The Gemini gateway: the only module that imports `google.genai`.

One client lives for the whole process (a client created inline gets collected and its calls fail with "client has
been closed"). The SDK never retries on its own: every retry decision belongs to the callers, which know the deadline
and the attempt count. Every call records a `gen_ai.call` span with the OpenTelemetry GenAI attributes.
"""

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from docqa.providers.llm import GenConfig, GenResult, ProviderError, Turn
from docqa.telemetry import Span, Telemetry

PROVIDER_NAME = "gcp.gemini"
MESSAGE_MAX = 300
MODEL_CHECK_TIMEOUT_S = 10.0
EMBED_CALL_TIMEOUT_S = 30.0
STOP = "STOP"
RETRY_IN = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)
RETRY_DELAY = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)s\s*$")
PER_DAY = re.compile(r"PerDay|per[ _-]day", re.IGNORECASE)
SPEND_CAP = re.compile(r"spend(?:ing)?[ _-]?cap", re.IGNORECASE)
KEY_PROBLEM = re.compile(r"API_KEY_INVALID|API key not valid|API key expired|API_KEY_EXPIRED", re.IGNORECASE)
TIMEOUT_CLASSES = {"TimeoutException", "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout", "TimeoutError"}
TRANSPORT_CLASSES = {
    "TransportError",
    "NetworkError",
    "ConnectError",
    "ReadError",
    "WriteError",
    "RemoteProtocolError",
    "ProtocolError",
    "ConnectionError",
}


def scrub(text: str, api_key: str) -> str:
    """Remove the key from a provider message and cut it for storage.

    Args:
        text: Message from the provider or the transport.
        api_key: The key to remove.

    Returns:
        The message without the key, at most `MESSAGE_MAX` characters.
    """
    if api_key:
        text = text.replace(api_key, "[key]")
    return " ".join(text.split())[:MESSAGE_MAX]


def _retry_after(exc: genai_errors.APIError, blob: str) -> float | None:
    """Find the wait the provider asked for in a 429.

    Args:
        exc: The provider error.
        blob: The error body as JSON text.

    Returns:
        Seconds to wait, from `RetryInfo.retryDelay`, the `Retry-After` header or "retry in Ns", or None.
    """
    details = exc.details.get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
    for item in details if isinstance(details, list) else []:
        if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
            match = RETRY_DELAY.match(str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}).get("retry-after") if response is not None else None
    if header and header.strip().replace(".", "", 1).isdigit():
        return float(header)
    match = RETRY_IN.search(blob)
    return float(match.group(1)) if match else None


def classify_api_error(exc: genai_errors.APIError, *, api_key: str, model: str | None) -> ProviderError:
    """Classify an HTTP error answered by the provider.

    | Status | Kind |
    |---|---|
    | 429 naming a spend cap | `billing` (account-wide) |
    | 429 naming a per-day quota | `quota_exhausted` |
    | other 429 | `rate_limited`, with the wait the provider asked for |
    | 402 | `billing` (account-wide) |
    | 400 naming the key, 401 | `misconfigured` (account-wide) |
    | 403, 404, 400 `FAILED_PRECONDITION` | `misconfigured` (the model) |
    | other 400 | `invalid_request`: this request is malformed (for example an image too large); no pause |
    | 5xx | `transient` |

    Args:
        exc: The provider error.
        api_key: The key, scrubbed from the message.
        model: Model of the call.

    Returns:
        The classified error.
    """
    code = int(exc.code or 0)
    blob = json.dumps(exc.details, default=str) if exc.details else str(exc)
    message = scrub(f"{code} {exc.status or ''}: {exc.message or blob}", api_key)
    status = str(exc.status or "")

    def error(kind: str, *, account_wide: bool = False, retry_after_s: float | None = None) -> ProviderError:
        """Build the classified error.

        Args:
            kind: Failure kind.
            account_wide: The failure concerns the whole account.
            retry_after_s: Suggested wait.

        Returns:
            The error.
        """
        return ProviderError(
            kind, message, status=code, retry_after_s=retry_after_s, model=model, account_wide=account_wide
        )

    if code == 429:
        if SPEND_CAP.search(blob):
            return error("billing", account_wide=True)
        if PER_DAY.search(blob):
            return error("quota_exhausted", retry_after_s=_retry_after(exc, blob))
        return error("rate_limited", retry_after_s=_retry_after(exc, blob))
    if code == 402:
        return error("billing", account_wide=True)
    if code == 401 or (code == 400 and KEY_PROBLEM.search(blob)):
        return error("misconfigured", account_wide=True)
    if code in (403, 404) or (code == 400 and status == "FAILED_PRECONDITION"):
        return error("misconfigured")
    if code == 400:
        return error("invalid_request")
    if code >= 500:
        return error("transient")
    return error("misconfigured")


def classify_error(exc: BaseException, *, api_key: str, model: str | None) -> ProviderError | None:
    """Classify any exception raised by a provider call.

    Args:
        exc: The exception.
        api_key: The key, scrubbed from the message.
        model: Model of the call.

    Returns:
        The classified error, or None when the exception is not a provider or transport failure (a bug of ours).
    """
    if isinstance(exc, genai_errors.APIError):
        return classify_api_error(exc, api_key=api_key, model=model)
    names = {cls.__name__ for cls in type(exc).__mro__}
    message = scrub(f"{type(exc).__name__}: {exc}", api_key)
    if names & TIMEOUT_CLASSES:
        return ProviderError("timeout", message, model=model)
    if names & TRANSPORT_CLASSES:
        return ProviderError("transient", message, model=model)
    return None


def _name(value: object) -> str | None:
    """Return the plain name of an SDK enum value.

    Args:
        value: An enum member, a string or None.

    Returns:
        `STOP` for `FinishReason.STOP`, or None.
    """
    if value is None:
        return None
    raw = str(getattr(value, "value", value))
    return raw.rsplit(".", 1)[-1] or None


def to_result(response: types.GenerateContentResponse, latency_ms: float) -> GenResult:
    """Read the first candidate's text, finish reason and usage.

    A response without candidates (the prompt itself was blocked) reports the block reason as its finish reason.

    Args:
        response: The SDK response.
        latency_ms: Wall time of the call.

    Returns:
        The result. Thought parts are left out of the text.
    """
    candidate = response.candidates[0] if response.candidates else None
    if candidate is not None:
        finish = _name(candidate.finish_reason) or "FINISH_REASON_UNSPECIFIED"
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []
        text = "".join(part.text for part in parts if part.text and not part.thought) or None
    else:
        feedback = response.prompt_feedback
        finish = _name(feedback.block_reason if feedback else None) or "NO_CANDIDATE"
        text = None
    usage = response.usage_metadata
    return GenResult(
        text=text,
        finish_reason=finish,
        model_version=response.model_version,
        input_tokens=usage.prompt_token_count if usage else None,
        output_tokens=usage.candidates_token_count if usage else None,
        thinking_tokens=usage.thoughts_token_count if usage else None,
        latency_ms=latency_ms,
    )


def record_result(span: Span, result: GenResult) -> None:
    """Set the response attributes of a `gen_ai.call` span.

    `gen_ai.usage.output_tokens` counts the thinking tokens too, because they are billed as output; the share of
    thinking is in `docqa.thinking_tokens`. A finish reason other than `STOP` marks the span as failed with
    `finish_<reason>`.

    Args:
        span: The open span.
        result: The call's result.
    """
    if result.model_version:
        span.set("gen_ai.response.model", result.model_version)
    if result.input_tokens is not None:
        span.set("gen_ai.usage.input_tokens", result.input_tokens)
    span.set("gen_ai.usage.output_tokens", (result.output_tokens or 0) + (result.thinking_tokens or 0))
    span.set("docqa.thinking_tokens", result.thinking_tokens or 0)
    span.set("gen_ai.response.finish_reasons", [result.finish_reason])
    if result.finish_reason != STOP:
        span.error(f"finish_{result.finish_reason.lower()}")


def record_error(span: Span, error: ProviderError) -> None:
    """Mark a `gen_ai.call` span as failed with the provider error's kind.

    Args:
        span: The open span.
        error: The classified error.
    """
    span.set("error.type", error.kind)
    if error.status is not None:
        span.set("docqa.provider_status", error.status)
    span.error(error.kind)


@contextmanager
def call_span(telemetry: Telemetry, *, operation: str, model: str, stage: str, attempt: int = 1) -> Iterator[Span]:
    """Open the `gen_ai.call` span of one model call.

    Args:
        telemetry: Where the span goes.
        operation: `generate_content`, `embeddings` or `get_model`.
        model: Requested model.
        stage: `docqa.stage` value (`ocr`, `answer`, `model_check`, ...).
        attempt: Attempt number.

    Yields:
        The open span.
    """
    attributes: dict[str, str | int] = {
        "gen_ai.provider.name": PROVIDER_NAME,
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "docqa.stage": stage,
        "docqa.attempt": attempt,
    }
    with telemetry.span("gen_ai.call", **attributes) as span:
        yield span


def _part(part: str | bytes) -> types.Part:
    """Convert one request part.

    Args:
        part: Text, or PNG bytes.

    Returns:
        The SDK part.
    """
    if isinstance(part, bytes):
        return types.Part.from_bytes(data=part, mime_type="image/png")
    return types.Part.from_text(text=part)


def to_contents(turns: list[Turn]) -> list[types.Content]:
    """Convert request turns to SDK contents.

    Args:
        turns: Our turns.

    Returns:
        One SDK content per turn, parts in order.
    """
    return [types.Content(role=turn.role, parts=[_part(part) for part in turn.parts]) for turn in turns]


def to_config(config: GenConfig, timeout_s: float) -> types.GenerateContentConfig:
    """Convert generation settings to the SDK's.

    Args:
        config: Our settings.
        timeout_s: HTTP timeout of the call.

    Returns:
        The SDK settings, with the timeout in milliseconds.
    """
    return types.GenerateContentConfig(
        system_instruction=config.system_instruction,
        response_mime_type="application/json" if config.response_json_schema is not None else None,
        response_json_schema=config.response_json_schema,
        thinking_config=types.ThinkingConfig(thinking_level=config.thinking_level) if config.thinking_level else None,
        media_resolution=config.media_resolution,
        max_output_tokens=config.max_output_tokens,
        temperature=config.temperature,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )


class GeminiGateway:
    """The gateway to the Gemini API: one client per process, classified errors, a span per call."""

    def __init__(self, api_key: str, telemetry: Telemetry) -> None:
        """Create the process's client.

        Args:
            api_key: The Gemini API key; never logged.
            telemetry: Where the call spans go.
        """
        self._api_key = api_key
        self._telemetry = telemetry
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=None))

    def _classified(self, exc: Exception, model: str) -> ProviderError | None:
        """Classify an exception of a call.

        Args:
            exc: The exception.
            model: Model of the call.

        Returns:
            The classified error, or None for an exception that is not the provider's.
        """
        return classify_error(exc, api_key=self._api_key, model=model)

    def generate(
        self, *, model: str, contents: list[Turn], config: GenConfig, timeout_s: float, stage: str, attempt: int = 1
    ) -> GenResult:
        """Call a model once and record a `gen_ai.call` span.

        Args:
            model: Model name.
            contents: Request turns.
            config: Generation settings.
            timeout_s: HTTP timeout of this call.
            stage: `docqa.stage` value.
            attempt: Attempt number for the span.

        Returns:
            The first candidate's text, finish reason and usage.

        Raises:
            ProviderError: The call failed; see `classify_api_error`.
        """
        request = to_config(config, timeout_s)
        sdk_contents = to_contents(contents)
        with call_span(
            self._telemetry, operation="generate_content", model=model, stage=stage, attempt=attempt
        ) as span:
            started = time.perf_counter()
            try:
                response = self._client.models.generate_content(model=model, contents=sdk_contents, config=request)
            except Exception as exc:
                error = self._classified(exc, model)
                if error is None:
                    raise
                record_error(span, error)
                raise error from None
            result = to_result(response, (time.perf_counter() - started) * 1000)
            record_result(span, result)
        return result

    def embed(self, *, model: str, texts: list[str], task: str, dim: int, stage: str) -> np.ndarray:
        """Embed texts in one call, one `types.Content` per text, and record a `gen_ai.call` span.

        A plain list of strings would be embedded as one aggregated vector, so each text goes in its own content.

        Args:
            model: Embedding model name.
            texts: Texts to embed.
            task: `RETRIEVAL_DOCUMENT` or `RETRIEVAL_QUERY`.
            dim: Output dimensionality.
            stage: `docqa.stage` value (`embed_chunks`, `embed_query`).

        Returns:
            Array of shape (len(texts), dim), float32, not normalized (reduced Gemini vectors aren't).

        Raises:
            ProviderError: The call failed, or the provider returned a different number of vectors.
        """
        config = types.EmbedContentConfig(
            task_type=task,
            output_dimensionality=dim,
            http_options=types.HttpOptions(timeout=int(EMBED_CALL_TIMEOUT_S * 1000)),
        )
        contents = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts]
        with call_span(self._telemetry, operation="embeddings", model=model, stage=stage) as span:
            span.set("docqa.embed.count", len(texts))
            span.set("docqa.tokens_est", sum(max(1, len(text) // 4) for text in texts))
            try:
                response = self._client.models.embed_content(model=model, contents=contents, config=config)
            except Exception as exc:
                error = self._classified(exc, model)
                if error is None:
                    raise
                record_error(span, error)
                raise error from None
            embeddings = response.embeddings or []
            if len(embeddings) != len(texts):
                error = ProviderError(
                    "transient", f"{len(embeddings)} vectors returned for {len(texts)} texts", model=model
                )
                record_error(span, error)
                raise error
            tokens = [e.statistics.token_count for e in embeddings if e.statistics and e.statistics.token_count]
            if len(tokens) == len(embeddings):
                span.set("gen_ai.usage.input_tokens", int(sum(tokens)))
        return np.array([e.values for e in embeddings], dtype=np.float32)

    def check_model(self, model: str) -> None:
        """Fetch a model's metadata, without generation or cost.

        Args:
            model: Model name.

        Raises:
            ProviderError: The key or the model was rejected, or the provider is unreachable.
        """
        config = types.GetModelConfig(http_options=types.HttpOptions(timeout=int(MODEL_CHECK_TIMEOUT_S * 1000)))
        with call_span(self._telemetry, operation="get_model", model=model, stage="model_check") as span:
            try:
                self._client.models.get(model=model, config=config)
            except Exception as exc:
                error = self._classified(exc, model)
                if error is None:
                    raise
                record_error(span, error)
                raise error from None
