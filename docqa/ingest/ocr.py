"""Page transcription: the prompt, the OCR version, the canonical cleanup and the call chain (DECISIONS #6).

The chain, per job attempt: the default model; a `RECITATION` refusal is sent again to the same model at most
`OCR_RECITATION_RETRIES` times (the filter is not deterministic); a page that still fails (refused, cut by the
token limit, empty) gets one call on the fallback model. The last attempt goes straight to the fallback. Only a
`STOP` finish with text is accepted. Every call takes a pacing slot first and checks that its model is not paused.
"""

import re
import time
from dataclasses import dataclass, field

from psycopg_pool import ConnectionPool

from docqa.config import Settings
from docqa.providers import pacing
from docqa.providers.llm import LLM, GenConfig, GenResult, Turn
from docqa.providers.pacing import Pause

OCR_PROMPT_LABEL = "ocr-g2-nolatex"
OCR_DPI = 150
OCR_MEDIA_RESOLUTION = "MEDIA_RESOLUTION_HIGH"
OCR_THINKING = "low"
OCR_RECITATION_RETRIES = 2
OCR_MAX_OUTPUT_TOKENS = 8192
OCR_CALL_TIMEOUT_S = 300
OCR_STAGE = "ocr"

OCR_PROMPT = """Transcribe this page of an equipment manual exactly as printed.

Rules:
- Output GitHub-flavored Markdown only. No code fences, no comments, no preamble.
- Copy the text verbatim, in its original language. Do not translate, fix spelling, reformat numbers
  or units, or add anything that is not printed on the page.
- Do not use LaTeX or any math markup. Write symbols, units and formulas with plain characters.
- Follow the reading order. In multi-column layouts, finish one column before starting the next.
- Render every table as a Markdown table: one printed row per table row, every value in its own column.
  Leave empty cells empty. Keep footnote markers (such as * or 1) and the footnotes themselves.
- Use Markdown headings for headings and Markdown lists for lists.
- Transcribe legible text inside figures (labels, captions). Do not describe images.
- If a character or number is not legible, write [illegible] instead of guessing.
- Keep page numbers, running headers and footers, each on its own line."""

OCR_CONFIG = GenConfig(
    thinking_level=OCR_THINKING, media_resolution=OCR_MEDIA_RESOLUTION, max_output_tokens=OCR_MAX_OUTPUT_TOKENS
)

STOP = "STOP"
RECITATION = "RECITATION"
SAFETY_FINISHES = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)


def ocr_version(settings: Settings) -> str:
    """Name everything that shapes a transcription; it is part of the cache key.

    Args:
        settings: Process settings, for the two models.

    Returns:
        For example `ocr-g2-nolatex:150dpi:high:gemini-3.5-flash-lite>gemini-3.8-flash`.
    """
    return f"{OCR_PROMPT_LABEL}:{OCR_DPI}dpi:high:{settings.gemini_ocr_model}>{settings.gemini_ocr_fallback_model}"


def canonical_ocr(text: str) -> str:
    r"""Bring a transcription to the canonical page format, changing nothing a reader would see.

    Removes a code fence wrapping the whole output, unescapes `\*` (footnote markers), strips trailing spaces of
    every line and blank lines around the text. Spelling and numbers are never touched.

    Args:
        text: Raw model output.

    Returns:
        The canonical text.
    """
    fenced = FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    text = text.replace("\\*", "*")
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def code_for(result: GenResult) -> str:
    """Map an unaccepted result to the page's error code.

    Args:
        result: The last call's result.

    Returns:
        `ocr_recitation`, `ocr_safety`, `ocr_max_tokens`, `ocr_empty` or `ocr_finish_other`.
    """
    if result.finish_reason == STOP:
        return "ocr_empty"
    if result.finish_reason == RECITATION:
        return "ocr_recitation"
    if result.finish_reason in SAFETY_FINISHES:
        return "ocr_safety"
    if result.finish_reason == "MAX_TOKENS":
        return "ocr_max_tokens"
    return "ocr_finish_other"


class ModelPaused(Exception):
    """The model the chain needs is paused; the job waits for the pause to end without spending an attempt.

    Attributes:
        pause: The active pause.
    """

    def __init__(self, pause: Pause) -> None:
        """Keep the pause.

        Args:
            pause: The active pause.
        """
        super().__init__(pause.model, pause.reason)
        self.pause = pause


@dataclass
class OcrOutcome:
    """How the call chain ended.

    Attributes:
        text: Canonical transcription when accepted, else None.
        model: Model that produced the text, or of the last call.
        code: Page error code when not accepted, else None.
        recitations: `RECITATION` refusals seen.
        fallback_used: The fallback model was called.
        finish_reasons: `(model, finish reason)` of every call, in order.
    """

    text: str | None = None
    model: str | None = None
    code: str | None = None
    recitations: int = 0
    fallback_used: bool = False
    finish_reasons: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Tell whether the page was transcribed.

        Returns:
            True when there is accepted text.
        """
        return self.text is not None

    def quality(self) -> dict[str, object]:
        """Return the evidence stored in the page's `quality.ocr`.

        Returns:
            Model, cache flag, recitations, fallback flag and the finish reasons in call order.
        """
        return {
            "model": self.model,
            "from_cache": False,
            "recitations": self.recitations,
            "fallback_used": self.fallback_used,
            "finish_reasons": [finish for _, finish in self.finish_reasons],
        }

    def message(self) -> str:
        """Describe the calls for a failed page, without page text.

        Returns:
            For example `gemini-3.5-flash-lite: RECITATION x3; gemini-3.8-flash: MAX_TOKENS`.
        """
        groups: list[tuple[str, str, int]] = []
        for model, finish in self.finish_reasons:
            if groups and groups[-1][:2] == (model, finish):
                groups[-1] = (model, finish, groups[-1][2] + 1)
            else:
                groups.append((model, finish, 1))
        parts: dict[str, list[str]] = {}
        for model, finish, count in groups:
            parts.setdefault(model, []).append(finish if count == 1 else f"{finish} x{count}")
        return "; ".join(f"{model}: {', '.join(finishes)}" for model, finishes in parts.items())


def _accepted(result: GenResult) -> str | None:
    """Return the canonical text of an accepted result.

    Args:
        result: A call's result.

    Returns:
        The canonical text when the finish is `STOP` and the text is not empty, else None.
    """
    if result.finish_reason != STOP or not result.text:
        return None
    text = canonical_ocr(result.text)
    return text if text.strip() else None


@dataclass
class _Chain:
    """One run of the call chain.

    Attributes:
        png: Page image.
        llm: Provider gateway.
        settings: Process settings.
        pool: Database pool, for pauses and slots.
        attempt: Job attempt, recorded on each call's span.
        outcome: What happened so far.
    """

    png: bytes
    llm: LLM
    settings: Settings
    pool: ConnectionPool
    attempt: int
    outcome: OcrOutcome = field(default_factory=OcrOutcome)

    def call(self, model: str) -> GenResult:
        """Make one paced call, unless the model is paused.

        Args:
            model: Model to call.

        Returns:
            The call's result.

        Raises:
            ModelPaused: The model, or the account, is paused.
            ProviderError: The call failed.
        """
        with self.pool.connection() as conn:
            paused = pacing.active_pause(conn, model)
            if paused is not None:
                raise ModelPaused(paused)
            slot = pacing.acquire_slot(conn, model, self.settings.ocr_min_interval_s)
        time.sleep(pacing.seconds_until(slot))
        result = self.llm.generate(
            model=model,
            contents=[Turn(role="user", parts=[self.png, OCR_PROMPT])],
            config=OCR_CONFIG,
            timeout_s=OCR_CALL_TIMEOUT_S,
            stage=OCR_STAGE,
            attempt=self.attempt,
        )
        self.outcome.model = model
        self.outcome.finish_reasons.append((model, result.finish_reason))
        if result.finish_reason == RECITATION:
            self.outcome.recitations += 1
        return result

    def accept(self, result: GenResult) -> bool:
        """Record the text of an accepted result.

        Args:
            result: The last call's result.

        Returns:
            True when the result was accepted.
        """
        text = _accepted(result)
        if text is None:
            return False
        self.outcome.text = text
        return True


def transcribe(
    png: bytes, *, llm: LLM, settings: Settings, pool: ConnectionPool, attempt: int, max_attempts: int
) -> OcrOutcome:
    """Transcribe one page image through the call chain.

    Args:
        png: Page image, PNG at `OCR_DPI`.
        llm: Provider gateway.
        settings: Process settings (models, pacing interval).
        pool: Database pool, for pauses and pacing slots.
        attempt: The job's attempt; the last one goes straight to the fallback.
        max_attempts: The job's attempt limit.

    Returns:
        The transcription, or the failure code with the finish reasons.

    Raises:
        ModelPaused: A model the chain needs is paused.
        ProviderError: A call failed (transport, quota, key...); the caller decides.
    """
    chain = _Chain(png=png, llm=llm, settings=settings, pool=pool, attempt=attempt)
    if attempt < max_attempts:
        default = settings.gemini_ocr_model
        result = chain.call(default)
        retries = 0
        while result.finish_reason == RECITATION and retries < OCR_RECITATION_RETRIES:
            retries += 1
            result = chain.call(default)
        if chain.accept(result):
            return chain.outcome
    chain.outcome.fallback_used = True
    result = chain.call(settings.gemini_ocr_fallback_model)
    if not chain.accept(result):
        chain.outcome.code = code_for(result)
    return chain.outcome
