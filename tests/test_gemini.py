"""The Gemini gateway without network: error classification, response reading, request settings and call spans."""

from typing import Any

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from docqa.providers.gemini import GeminiGateway, classify_error, to_config, to_contents, to_result
from docqa.providers.llm import GenConfig, ProviderError, Turn
from docqa.telemetry import MemoryTelemetry, bind

KEY = "AIzaSy-test-key-0123456789"


def _api_error(code: int, message: str, status: str, details: list[dict[str, Any]] | None = None) -> Exception:
    """Build an SDK error as the provider would answer it.

    Args:
        code: HTTP status.
        message: Provider message.
        status: gRPC status name.
        details: `error.details` entries.

    Returns:
        A `ClientError` or `ServerError`.
    """
    body = {"error": {"code": code, "message": message, "status": status, "details": details or []}}
    kind = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return kind(code, body)


RETRY_INFO = {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "38s"}
PER_MINUTE = {
    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel"}],
}
PER_DAY = {
    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
}
KEY_INVALID = {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}


@pytest.mark.parametrize(
    ("error", "kind", "account_wide", "retry_after"),
    [
        (_api_error(429, "Quota exceeded.", "RESOURCE_EXHAUSTED", [PER_MINUTE, RETRY_INFO]), "rate_limited", False, 38),
        (_api_error(429, "Quota exceeded. Please retry in 12.5s.", "RESOURCE_EXHAUSTED"), "rate_limited", False, 12.5),
        (_api_error(429, "Quota exceeded.", "RESOURCE_EXHAUSTED", [PER_DAY, RETRY_INFO]), "quota_exhausted", False, 38),
        (
            _api_error(429, "Your project reached its monthly spending cap.", "RESOURCE_EXHAUSTED"),
            "billing",
            True,
            None,
        ),
        (_api_error(402, "Payment required.", "FAILED_PRECONDITION"), "billing", True, None),
        (_api_error(400, "API key not valid.", "INVALID_ARGUMENT", [KEY_INVALID]), "misconfigured", True, None),
        (_api_error(401, "Unauthenticated.", "UNAUTHENTICATED"), "misconfigured", True, None),
        (_api_error(403, "Permission denied.", "PERMISSION_DENIED"), "misconfigured", False, None),
        (_api_error(404, "models/nope is not found.", "NOT_FOUND"), "misconfigured", False, None),
        (_api_error(400, "User location is not supported.", "FAILED_PRECONDITION"), "misconfigured", False, None),
        (
            _api_error(400, "Request payload size exceeds the limit.", "INVALID_ARGUMENT"),
            "invalid_request",
            False,
            None,
        ),
        (_api_error(500, "Internal error.", "INTERNAL"), "transient", False, None),
        (_api_error(503, "This model is currently experiencing high demand.", "UNAVAILABLE"), "transient", False, None),
        (httpx.ConnectError("connection refused"), "transient", False, None),
        (httpx.RemoteProtocolError("server disconnected"), "transient", False, None),
        (httpx.ReadTimeout("timed out"), "timeout", False, None),
    ],
    ids=[
        "429-minute-retryinfo",
        "429-retry-in-message",
        "429-day",
        "429-spend-cap",
        "402",
        "400-key",
        "401",
        "403",
        "404",
        "400-precondition",
        "400-invalid-argument",
        "500",
        "503",
        "connect",
        "protocol",
        "read-timeout",
    ],
)
def test_errors_are_classified(error: Exception, kind: str, account_wide: bool, retry_after: float | None) -> None:
    """Every provider and transport failure gets the kind that decides retry, reschedule, pause or failure.

    Args:
        error: The raised exception.
        kind: Expected kind.
        account_wide: Whether the failure concerns the whole account.
        retry_after: Expected suggested wait.
    """
    classified = classify_error(error, api_key=KEY, model="gemini-x")
    assert classified is not None
    assert (classified.kind, classified.account_wide, classified.retry_after_s) == (kind, account_wide, retry_after)
    assert classified.model == "gemini-x"


def test_unknown_exceptions_are_not_classified() -> None:
    """An exception that is not the provider's is a bug of ours and is left alone."""
    assert classify_error(ValueError("boom"), api_key=KEY, model="m") is None


def test_messages_are_scrubbed_and_cut() -> None:
    """The key never survives in a message, and messages are cut to 300 characters."""
    error = _api_error(400, f"API key not valid: {KEY} " + "x" * 500, "INVALID_ARGUMENT")
    classified = classify_error(error, api_key=KEY, model="m")
    assert classified is not None
    assert KEY not in classified.message
    assert "[key]" in classified.message
    assert len(classified.message) <= 300


def _response(**fields: object) -> types.GenerateContentResponse:
    """Build an SDK response.

    Args:
        **fields: Response fields.

    Returns:
        The response.
    """
    return types.GenerateContentResponse(**fields)


def test_result_reads_text_finish_and_usage() -> None:
    """Thought parts are left out of the text; the finish reason is a plain name; usage is copied."""
    response = _response(
        candidates=[
            types.Candidate(
                finish_reason="STOP",
                content=types.Content(
                    role="model", parts=[types.Part(text="plan", thought=True), types.Part(text="ok")]
                ),
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1303, candidates_token_count=420, thoughts_token_count=12
        ),
        model_version="gemini-3.5-flash-lite-001",
    )
    result = to_result(response, 12.5)
    assert (result.text, result.finish_reason, result.model_version) == ("ok", "STOP", "gemini-3.5-flash-lite-001")
    assert (result.input_tokens, result.output_tokens, result.thinking_tokens) == (1303, 420, 12)


def test_blocked_prompt_reports_the_block_reason() -> None:
    """A response without candidates reports why the prompt was blocked, and no text."""
    response = _response(prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="PROHIBITED_CONTENT"))
    result = to_result(response, 1.0)
    assert (result.text, result.finish_reason) == (None, "PROHIBITED_CONTENT")


def test_request_settings_are_translated() -> None:
    """Thinking, media resolution, output budget, temperature and the timeout reach the SDK; images become PNG parts."""
    config = to_config(
        GenConfig(
            thinking_level="low", media_resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=8192, temperature=0.0
        ),
        300,
    )
    assert config.thinking_config is not None and config.thinking_config.thinking_level is not None
    assert str(config.media_resolution).endswith("MEDIA_RESOLUTION_HIGH")
    assert config.max_output_tokens == 8192
    assert config.temperature == 0.0
    assert config.http_options is not None and config.http_options.timeout == 300_000
    assert config.response_mime_type is None
    [content] = to_contents([Turn(role="user", parts=[b"\x89PNG", "prompt"])])
    assert content.parts is not None
    image, text = content.parts
    assert image.inline_data is not None and image.inline_data.mime_type == "image/png"
    assert text.text == "prompt"


class _Models:
    """Stands in for `client.models`.

    Attributes:
        outcome: Response to return, or exception to raise.
    """

    def __init__(self, outcome: types.GenerateContentResponse | Exception) -> None:
        """Keep the outcome.

        Args:
            outcome: Response or exception.
        """
        self.outcome = outcome

    def generate_content(self, **kwargs: object) -> types.GenerateContentResponse:
        """Return or raise the outcome.

        Args:
            **kwargs: The SDK call's arguments.

        Returns:
            The response.

        Raises:
            Exception: The scripted exception.
        """
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class _Client:
    """Stands in for `genai.Client`.

    Attributes:
        models: The fake models service.
    """

    def __init__(self, models: _Models) -> None:
        """Keep the models service.

        Args:
            models: The fake models service.
        """
        self.models = models


def _gateway(outcome: types.GenerateContentResponse | Exception) -> tuple[GeminiGateway, MemoryTelemetry]:
    """Build a gateway whose client returns or raises a given outcome.

    Args:
        outcome: Response or exception.

    Returns:
        The gateway and its telemetry.
    """
    telemetry = MemoryTelemetry()
    gateway = GeminiGateway(KEY, telemetry)
    gateway._client = _Client(_Models(outcome))
    return gateway, telemetry


def _generate(gateway: GeminiGateway) -> None:
    """Make one OCR-like call.

    Args:
        gateway: The gateway.
    """
    gateway.generate(
        model="gemini-3.5-flash-lite",
        contents=[Turn(role="user", parts=[b"png", "prompt"])],
        config=GenConfig(thinking_level="low"),
        timeout_s=300,
        stage="ocr",
        attempt=2,
    )


def test_a_call_records_a_gen_ai_span() -> None:
    """The span carries the GenAI attributes, output tokens with thinking, and an error for a non-STOP finish."""
    response = _response(
        candidates=[types.Candidate(finish_reason="RECITATION")],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1303, candidates_token_count=0, thoughts_token_count=7
        ),
        model_version="v1",
    )
    gateway, telemetry = _gateway(response)
    with bind(job_id=7):
        _generate(gateway)
        telemetry.flush()
    [span] = telemetry.named("gen_ai.call")
    assert span.attributes["gen_ai.provider.name"] == "gcp.gemini"
    assert span.attributes["gen_ai.request.model"] == "gemini-3.5-flash-lite"
    assert span.attributes["gen_ai.response.model"] == "v1"
    assert span.attributes["docqa.stage"] == "ocr"
    assert span.attributes["docqa.attempt"] == 2
    assert span.attributes["gen_ai.usage.input_tokens"] == 1303
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    assert span.attributes["docqa.thinking_tokens"] == 7
    assert span.attributes["gen_ai.response.finish_reasons"] == ["RECITATION"]
    assert (span.status, span.error_code, span.ids["job_id"]) == ("error", "finish_recitation", 7)


def test_a_failed_call_raises_the_classified_error_and_records_it() -> None:
    """A provider error comes out as a `ProviderError`, and its span records the kind, never the key."""
    gateway, telemetry = _gateway(_api_error(503, f"high demand {KEY}", "UNAVAILABLE"))
    with bind(job_id=1):
        with pytest.raises(ProviderError) as raised:
            _generate(gateway)
        telemetry.flush()
    assert (raised.value.kind, raised.value.status, raised.value.model) == ("transient", 503, "gemini-3.5-flash-lite")
    [span] = telemetry.named("gen_ai.call")
    assert (span.status, span.error_code, span.attributes["error.type"]) == ("error", "transient", "transient")
    assert KEY not in repr(span.attributes)
