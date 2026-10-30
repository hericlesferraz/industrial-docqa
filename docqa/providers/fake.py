"""A scripted stand-in for the provider gateway, used by every test that would otherwise call a paid API."""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from docqa.providers.gemini import call_span, record_error, record_result
from docqa.providers.llm import GenConfig, GenResult, ProviderError, Turn
from docqa.telemetry import Telemetry

Outcome = GenResult | ProviderError
Script = list[Outcome] | Callable[["FakeCall"], Outcome]


def result(
    text: str | None = "# Page\n\nText.",
    finish: str = "STOP",
    *,
    input_tokens: int = 1300,
    output_tokens: int = 400,
    thinking_tokens: int | None = None,
    model_version: str | None = None,
    latency_ms: float = 5.0,
) -> GenResult:
    """Build a model result for a script.

    Args:
        text: Output text.
        finish: Finish reason.
        input_tokens: Prompt tokens.
        output_tokens: Candidate tokens.
        thinking_tokens: Thinking tokens.
        model_version: Version reported by the provider.
        latency_ms: Wall time.

    Returns:
        The result.
    """
    return GenResult(
        text=text,
        finish_reason=finish,
        model_version=model_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        latency_ms=latency_ms,
    )


@dataclass(frozen=True)
class FakeCall:
    """One call received by the fake.

    Attributes:
        model: Requested model.
        stage: `docqa.stage` of the call.
        attempt: Attempt number given by the caller.
        config: Generation settings.
        contents: Request turns.
        timeout_s: HTTP timeout asked for.
    """

    model: str
    stage: str
    attempt: int
    config: GenConfig
    contents: list[Turn]
    timeout_s: float


@dataclass
class FakeLLM:
    """Returns scripted results or raises scripted errors, in order, and records every call.

    Attributes:
        script: Outcomes in call order, or a function of the call that returns its outcome.
        telemetry: When given, every call records a `gen_ai.call` span like the real gateway.
        model_errors: Errors `check_model` raises, per model.
        calls: Every call received, in order.
        checked: Models passed to `check_model`, in order.
    """

    script: Script = field(default_factory=list)
    telemetry: Telemetry | None = None
    model_errors: dict[str, ProviderError] = field(default_factory=dict)
    calls: list[FakeCall] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def _next(self, call: FakeCall) -> Outcome:
        """Pick the outcome of a call.

        Args:
            call: The call.

        Returns:
            The scripted outcome.

        Raises:
            AssertionError: The script has no outcome left.
        """
        if callable(self.script):
            return self.script(call)
        assert len(self.calls) <= len(self.script), f"FakeLLM script exhausted at call {len(self.calls)}"
        return self.script[len(self.calls) - 1]

    def _outcome(self, call: FakeCall) -> GenResult:
        """Record a call and resolve its outcome.

        Args:
            call: The call.

        Returns:
            The scripted result.

        Raises:
            ProviderError: The scripted error, stamped with the call's model.
        """
        self.calls.append(call)
        outcome = self._next(call)
        if isinstance(outcome, ProviderError):
            if outcome.model is None:
                outcome.model = call.model
            raise outcome
        return outcome

    def generate(
        self, *, model: str, contents: list[Turn], config: GenConfig, timeout_s: float, stage: str, attempt: int = 1
    ) -> GenResult:
        """Return the next scripted outcome.

        Args:
            model: Model name.
            contents: Request turns.
            config: Generation settings.
            timeout_s: HTTP timeout of this call.
            stage: `docqa.stage` value.
            attempt: Attempt number.

        Returns:
            The scripted result.

        Raises:
            ProviderError: The scripted error.
        """
        call = FakeCall(
            model=model, stage=stage, attempt=attempt, config=config, contents=contents, timeout_s=timeout_s
        )
        if self.telemetry is None:
            return self._outcome(call)
        with call_span(self.telemetry, operation="generate_content", model=model, stage=stage, attempt=attempt) as span:
            try:
                outcome = self._outcome(call)
            except ProviderError as error:
                record_error(span, error)
                raise
            record_result(span, outcome)
        return outcome

    def embed(self, *, model: str, texts: list[str], task: str, dim: int, stage: str) -> np.ndarray:
        """Return deterministic unit vectors.

        Args:
            model: Embedding model name.
            texts: Texts to embed.
            task: Task type.
            dim: Output dimensionality.
            stage: `docqa.stage` value.

        Returns:
            Array of shape (len(texts), dim).
        """
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vectors[row, hash(text) % dim] = 1.0
        return vectors

    def check_model(self, model: str) -> None:
        """Record the check and raise the model's scripted error, if any.

        Args:
            model: Model name.

        Raises:
            ProviderError: The scripted error of this model.
        """
        self.checked.append(model)
        if model in self.model_errors:
            raise self.model_errors[model]
