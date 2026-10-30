"""The OCR prompt, version, canonical cleanup, page-label candidates, pacing, and the call chain with a fake provider."""

import itertools
import threading
from datetime import UTC, datetime, timedelta

import pytest
from psycopg_pool import ConnectionPool

from docqa.config import Settings
from docqa.db import tx
from docqa.ingest.ocr import (
    OCR_CONFIG,
    OCR_PROMPT,
    ModelPaused,
    OcrOutcome,
    canonical_ocr,
    code_for,
    ocr_version,
    transcribe,
)
from docqa.providers import pacing
from docqa.providers.fake import FakeLLM, result
from docqa.providers.llm import ProviderError

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.8-flash"


def test_prompt_is_the_g2_prompt_without_latex() -> None:
    """The prompt keeps the rules of DECISIONS #6 and forbids LaTeX; the request settings are the measured ones."""
    assert OCR_PROMPT.startswith("Transcribe this page of an equipment manual exactly as printed.")
    assert "Do not use LaTeX" in OCR_PROMPT
    assert "write [illegible] instead of guessing" in OCR_PROMPT
    assert (OCR_CONFIG.thinking_level, OCR_CONFIG.media_resolution) == ("low", "MEDIA_RESOLUTION_HIGH")
    assert OCR_CONFIG.max_output_tokens == 8192


def test_ocr_version_names_prompt_render_and_models(settings: Settings) -> None:
    """The version changes with the prompt label and either model, so the cache never serves another setup.

    Args:
        settings: Test settings.
    """
    assert ocr_version(settings) == f"ocr-g2-nolatex:150dpi:high:{LITE}>{FLASH}"


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("```markdown\n# Title\n\n| a | b |\n```", "# Title\n\n| a | b |"),
        ("```\ntext\n```\n", "text"),
        ("| 3600 | \\*1800 |   \n\\* note  \n", "| 3600 | *1800 |\n* note"),
        ("\n\nLubrificação 9,500 h\n\n", "Lubrificação 9,500 h"),
        ("a ``` b", "a ``` b"),
    ],
    ids=["fence-with-language", "fence", "escaped-stars", "blank-lines", "inner-backticks"],
)
def test_canonical_ocr(raw: str, canonical: str) -> None:
    r"""Fences, `\*` escapes, trailing spaces and surrounding blank lines go; nothing else changes.

    Args:
        raw: Model output.
        canonical: Expected canonical text.
    """
    assert canonical_ocr(raw) == canonical


@pytest.mark.parametrize(
    ("finish", "text", "code"),
    [
        ("RECITATION", None, "ocr_recitation"),
        ("SAFETY", None, "ocr_safety"),
        ("PROHIBITED_CONTENT", None, "ocr_safety"),
        ("MAX_TOKENS", "half a page", "ocr_max_tokens"),
        ("STOP", "  ", "ocr_empty"),
        ("OTHER", None, "ocr_finish_other"),
    ],
)
def test_failure_codes(finish: str, text: str | None, code: str) -> None:
    """Each unaccepted finish has its own page error code.

    Args:
        finish: Finish reason.
        text: Output text.
        code: Expected code.
    """
    assert code_for(result(text, finish)) == code


def test_pacing_slots_are_spaced_across_threads(db: ConnectionPool) -> None:
    """Two threads asking for the same model's slot at once get slots at least the interval apart.

    Args:
        db: Pool on an empty database.
    """
    slots: list[datetime] = []
    lock = threading.Lock()

    def take() -> None:
        """Reserve one slot."""
        with db.connection() as conn:
            slot = pacing.acquire_slot(conn, LITE, 2.0)
        with lock:
            slots.append(slot)

    threads = [threading.Thread(target=take) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    slots.sort()
    gaps = [(b - a).total_seconds() for a, b in itertools.pairwise(slots)]
    assert all(gap >= 2.0 - 1e-6 for gap in gaps), gaps
    with db.connection() as conn:
        other = pacing.acquire_slot(conn, FLASH, 2.0)
    assert pacing.seconds_until(other) < 0.5


def test_pauses_are_per_model_and_account(db: ConnectionPool) -> None:
    """A model pause blocks only that model; an account pause blocks every model.

    Args:
        db: Pool on an empty database.
    """
    until = datetime.now(UTC) + timedelta(minutes=5)
    with tx(db) as conn:
        pacing.pause(conn, FLASH, "quota_exhausted", "daily quota", until)
    with db.connection() as conn:
        assert pacing.active_pause(conn, LITE) is None
        paused = pacing.active_pause(conn, FLASH)
        assert paused is not None and (paused.model, paused.reason) == (FLASH, "quota_exhausted")
        assert pacing.first_pause(conn, [LITE, FLASH]) == paused
    with tx(db) as conn:
        written = pacing.pause_for(conn, ProviderError("billing", "402", model=LITE))
    assert written.model == "*"
    with db.connection() as conn:
        lite = pacing.active_pause(conn, LITE)
        assert lite is not None and (lite.model, lite.reason) == ("*", "billing")
        assert pacing.first_pause(conn, [LITE, FLASH]) == lite


def _transcribe(fake: FakeLLM, settings: Settings, pool: ConnectionPool, attempt: int = 1) -> OcrOutcome:
    """Run the chain with a fake provider.

    Args:
        fake: Fake provider.
        settings: Test settings.
        pool: Database pool.
        attempt: Job attempt.

    Returns:
        The outcome.
    """
    return transcribe(b"png", llm=fake, settings=settings, pool=pool, attempt=attempt, max_attempts=4)


def test_stop_is_accepted_on_the_default_model(db: ConnectionPool, settings: Settings) -> None:
    """One call, `STOP` with text: accepted, canonical, no fallback.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    fake = FakeLLM([result("```\n# Page 3\n\n| a | \\*b |\n```")])
    outcome = _transcribe(fake, settings, db)
    assert outcome.ok and outcome.text == "# Page 3\n\n| a | *b |"
    assert (outcome.model, outcome.recitations, outcome.fallback_used) == (LITE, 0, False)
    [call] = fake.calls
    assert (call.model, call.stage, call.attempt, call.timeout_s) == (LITE, "ocr", 1, 300)
    assert call.contents[0].parts == [b"png", OCR_PROMPT]


def test_recitation_is_retried_on_the_same_model(db: ConnectionPool, settings: Settings) -> None:
    """A `RECITATION` refusal is sent again to the default model before any fallback.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    fake = FakeLLM([result(None, "RECITATION"), result("text")])
    outcome = _transcribe(fake, settings, db)
    assert outcome.ok and (outcome.model, outcome.recitations, outcome.fallback_used) == (LITE, 1, False)
    assert [call.model for call in fake.calls] == [LITE, LITE]
    assert outcome.quality()["finish_reasons"] == ["RECITATION", "STOP"]


def test_three_recitations_go_to_the_fallback(db: ConnectionPool, settings: Settings) -> None:
    """After the default model refuses three times, the fallback transcribes the page.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    fake = FakeLLM([result(None, "RECITATION")] * 3 + [result("text")])
    outcome = _transcribe(fake, settings, db)
    assert outcome.ok and (outcome.model, outcome.recitations, outcome.fallback_used) == (FLASH, 3, True)
    assert [call.model for call in fake.calls] == [LITE, LITE, LITE, FLASH]


def test_a_cut_goes_to_the_fallback_and_its_failure_is_the_code(db: ConnectionPool, settings: Settings) -> None:
    """`MAX_TOKENS` is never accepted; the fallback's own failure decides the page's code.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    fake = FakeLLM([result("half", "MAX_TOKENS"), result(None, "SAFETY")])
    outcome = _transcribe(fake, settings, db)
    assert not outcome.ok and outcome.code == "ocr_safety"
    assert outcome.message() == f"{LITE}: MAX_TOKENS; {FLASH}: SAFETY"


def test_the_last_attempt_goes_straight_to_the_fallback(db: ConnectionPool, settings: Settings) -> None:
    """Out of attempts, the default model is skipped.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    fake = FakeLLM([result("text")])
    outcome = _transcribe(fake, settings, db, attempt=4)
    assert outcome.ok and outcome.fallback_used
    assert [call.model for call in fake.calls] == [FLASH]


def test_a_paused_fallback_stops_the_chain(db: ConnectionPool, settings: Settings) -> None:
    """The chain checks the fallback's pause before calling it.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    with tx(db) as conn:
        pacing.pause(conn, FLASH, "quota_exhausted", "daily", datetime.now(UTC) + timedelta(minutes=5))
    fake = FakeLLM([result(None, "RECITATION")] * 3)
    with pytest.raises(ModelPaused) as raised:
        _transcribe(fake, settings, db)
    assert raised.value.pause.model == FLASH
    assert [call.model for call in fake.calls] == [LITE, LITE, LITE]


def test_provider_errors_propagate(db: ConnectionPool, settings: Settings) -> None:
    """A provider error ends the chain; the handler decides what to do.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
    """
    fake = FakeLLM([ProviderError("transient", "503 UNAVAILABLE", status=503)])
    with pytest.raises(ProviderError) as raised:
        _transcribe(fake, settings, db)
    assert (raised.value.kind, raised.value.model) == ("transient", LITE)
