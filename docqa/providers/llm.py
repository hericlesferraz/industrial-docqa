"""Types of the model provider gateway, shared by OCR, embeddings and answers."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class Turn:
    """One message of a model request.

    Attributes:
        role: `user` or `model`.
        parts: Text parts and PNG images (bytes).
    """

    role: Literal["user", "model"]
    parts: list[str | bytes]


@dataclass(frozen=True)
class GenConfig:
    """Generation settings, translated to `google.genai.types` by the gateway.

    Attributes:
        system_instruction: System prompt.
        response_json_schema: JSON schema of the output; also sets `response_mime_type="application/json"`.
        thinking_level: For example `low`.
        media_resolution: For example `MEDIA_RESOLUTION_HIGH`.
        max_output_tokens: Output budget.
        temperature: Sampling temperature; None keeps the model's default (the eval judge sets 0).
    """

    system_instruction: str | None = None
    response_json_schema: dict[str, object] | None = None
    thinking_level: str | None = None
    media_resolution: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class GenResult:
    """Outcome of a call that returned a candidate.

    Attributes:
        text: Output text, or None.
        finish_reason: `STOP`, `RECITATION`, `SAFETY`, `MAX_TOKENS`, ...
        model_version: Model version reported by the provider.
        input_tokens: Prompt tokens (images included).
        output_tokens: Candidate tokens.
        thinking_tokens: Thinking tokens, when reported.
        latency_ms: Wall time of the call.
    """

    text: str | None
    finish_reason: str
    model_version: str | None
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None
    latency_ms: float


class ProviderError(Exception):
    """A provider failure, classified.

    Attributes:
        kind: `transient`, `timeout`, `rate_limited`, `quota_exhausted`, `billing`, `misconfigured`, or
            `invalid_request` (this request is malformed, for example an image too large; the model stays usable).
        status: HTTP status from the provider, if any.
        retry_after_s: Suggested wait, if the provider gave one.
        message: Provider message cut to 300 characters, scrubbed of the key.
        model: Model of the failed call, so a pause or a quota names the right one.
        account_wide: True when the failure concerns the whole account (billing, an invalid key), not one model.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        status: int | None = None,
        retry_after_s: float | None = None,
        model: str | None = None,
        account_wide: bool = False,
    ) -> None:
        """Store the classification.

        Args:
            kind: Failure kind (see Attributes).
            message: Sanitized provider message.
            status: HTTP status from the provider.
            retry_after_s: Suggested wait in seconds.
            model: Model of the failed call.
            account_wide: The failure concerns the whole account.
        """
        super().__init__(kind, message)
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after_s = retry_after_s
        self.model = model
        self.account_wide = account_wide


class LLM(Protocol):
    """The only door to the model provider."""

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
            ProviderError: The call failed.
        """
        ...

    def embed(self, *, model: str, texts: list[str], task: str, dim: int, stage: str) -> np.ndarray:
        """Embed texts, one `types.Content` per text.

        Args:
            model: Embedding model name.
            texts: Texts to embed.
            task: `RETRIEVAL_DOCUMENT` or `RETRIEVAL_QUERY`.
            dim: Output dimensionality.
            stage: `docqa.stage` value.

        Returns:
            Array of shape (len(texts), dim), not yet normalized.

        Raises:
            ProviderError: The call failed.
        """
        ...

    def check_model(self, model: str) -> None:
        """Fetch a model's metadata, without generation or cost.

        Args:
            model: Model name.

        Raises:
            ProviderError: The key or the model was rejected, or the provider is unreachable.
        """
