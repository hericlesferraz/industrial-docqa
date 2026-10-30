"""Per-question scoring of the eval: quotes in the PDF, numbers, ranks, flags and the failure stage."""

from pathlib import Path

import pymupdf
import pytest

from docqa.evaluation.client import Reply
from docqa.evaluation.dataset import Question, Source
from docqa.evaluation.judge import Fact, JudgeOutcome
from docqa.evaluation.metrics import (
    Corpus,
    FileOutcome,
    QuestionResult,
    evaluate,
    failure_stage,
    finalize,
    needs_judge,
    normalize,
    percentile,
    quote_check,
    ungrounded_numbers,
)

DOC = "11111111-1111-1111-1111-111111111111"
PAGE_TEXT = (
    "Relubrication Intervals\nNEMA (IEC) Frame Size Rated Speed (RPM)\n3600 1800 1200 900\n"
    "Upto 210 incl.(132) 5500Hrs. 12000Hrs. 18000Hrs. 22000Hrs.\n"
    "Over 210 to 280 incl.(180) 3600Hrs. 9500Hrs. 15000Hrs. 18000Hrs.\n"
)


def _pdf(path: Path) -> Path:
    """Write a two-page PDF whose second page holds a small lubrication table as text.

    Args:
        path: Output file.

    Returns:
        The path.
    """
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Cover page")
    doc.new_page().insert_text((40, 72), PAGE_TEXT, fontsize=8)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    """A corpus with the generated PDF ingested and page 2 indexed.

    Args:
        tmp_path: Pytest temp folder.

    Returns:
        The corpus.
    """
    detail = {"pages": [{"page": 1, "state": "indexed"}, {"page": 2, "state": "indexed"}]}
    item = FileOutcome("LB.pdf", _pdf(tmp_path / "LB.pdf"), 202, "accepted", document_id=DOC, detail=detail)
    return Corpus([item])


def _question(status: str = "answered", sources: tuple[Source, ...] = (Source("LB.pdf", (2,)),)) -> Question:
    """Build a question.

    Args:
        status: Expected status.
        sources: Expected sources.

    Returns:
        The question.
    """
    return Question(
        set_name="public",
        id="q-1",
        question="Relubrication interval for frame 256 at 1800 RPM?",
        expected_status=status,
        expected_answer="9,500 hours.",
        sources=sources if status != "not_found" else (),
        lang="en",
    )


def _citation(quote: str, *, page: int = 2, transcribed: bool = False) -> dict[str, object]:
    """Build an answer citation.

    Args:
        quote: The quote.
        page: Cited page.
        transcribed: Transcribed page flag.

    Returns:
        The citation body.
    """
    return {"document_id": DOC, "filename": "LB.pdf", "page": page, "quote": quote, "transcribed": transcribed}


def _record(ranks: list[tuple[int, int, bool]]) -> dict[str, object]:
    """Build an answer record with candidates and one call.

    Args:
        ranks: `(rank, page, sent_to_llm)` per candidate.

    Returns:
        The record.
    """
    return {
        "versions": {"model": "gemini-3.8-flash"},
        "totals": {"latency_ms": 2500.0},
        "calls": [
            {"purpose": "answer", "attempt": 1, "model": "gemini-3.8-flash", "input_tokens": 3000, "output_tokens": 100}
        ],
        "candidates": [
            {"rank": rank, "document_id": DOC, "page": page, "sent_to_llm": sent} for rank, page, sent in ranks
        ],
        "coverage": {"incomplete": []},
    }


def _answered(answer: str, quotes: list[str], *, coverage: dict[str, object] | None = None) -> Reply:
    """Build a 200 `answered` reply.

    Args:
        answer: Answer text.
        quotes: One citation per quote, on page 2.
        coverage: Coverage block.

    Returns:
        The reply.
    """
    body = {
        "answer_id": "a-1",
        "status": "answered",
        "answer": answer,
        "citations": [_citation(q) for q in quotes],
        "coverage": coverage or {"complete": True, "incomplete": []},
    }
    return Reply(200, body, 2600.0)


ROW = "Over 210 to 280 incl.(180) | 3600Hrs. | 9500Hrs. | 15000Hrs. | 18000Hrs."
HEADER = "NEMA (IEC) Frame Size | Rated Speed | (RPM)"


def test_quotes_are_found_cell_by_cell_ignoring_markup_and_spaces() -> None:
    """Rows match cell by cell; whitespace and Markdown markers don't matter; a changed digit does."""
    text = normalize(PAGE_TEXT)
    assert quote_check(f"{HEADER}\n{ROW}", text) == (2, 2)
    assert quote_check("**Upto 210  incl.(132)** | 5500Hrs.", text) == (1, 1)
    assert quote_check("* Relubrication   Intervals", text) == (1, 1)
    assert quote_check("Over 210 to 280 incl.(180) | 9600Hrs.", text) == (0, 1)
    assert normalize("ar ambiente)\x07 com") == normalize("ar ambiente) com")
    assert normalize("Lift Eye <!-- End of picture text -->") == "LiftEye"


def test_evaluate_scores_a_correct_answer(corpus: Corpus) -> None:
    """Status, source, quote in the PDF, grounded numbers, rank and record numbers are read from the API only.

    Args:
        corpus: The ingested PDF.
    """
    reply = _answered("9,500 hours.", [f"{HEADER}\n{ROW}"])
    result = evaluate(_question(), reply, _record([(1, 2, True), (2, 1, True)]), corpus)
    assert (result.status_ok, result.source_ok, result.ingestion_ok) == (True, True, True)
    assert (result.rank, result.in_prompt) == (1, True)
    assert result.ungrounded == []
    [cite] = result.citations
    assert (cite.lines, cite.lines_found, cite.expected_page) == (2, 2, True)
    assert (result.model, result.latency_ms, result.model_calls) == ("gemini-3.8-flash", 2500.0, 1)
    assert result.tokens == {"gemini-3.8-flash": [3000, 100]}
    assert needs_judge(result)
    raw = result.to_json()
    assert raw["response"] == reply.body and raw["record"] is not None and "body" not in raw
    judge = JudgeOutcome("correct", False, [Fact("9,500 h", True, "stated", "9,500 hours")], "judge")
    result.judge = judge
    finalize(result, judged=True)
    assert (result.correct, result.unsafe, result.over_cautious, result.stage) == (True, False, False, None)


def test_transcribed_citations_are_not_checked_against_the_text_layer(corpus: Corpus) -> None:
    """A quote from a transcribed page is counted apart, never as a miss.

    Args:
        corpus: The ingested PDF.
    """
    body = _answered("9,500 hours.", []).body
    body["citations"] = [_citation("Over 210 | 9500 Hrs.", transcribed=True)]
    result = evaluate(_question(), Reply(200, body, 1.0), _record([(1, 2, True)]), corpus)
    [cite] = result.citations
    assert (cite.transcribed, cite.lines_found) == (True, None)


def test_numbers_are_regrounded_from_the_answer_alone() -> None:
    """A number in no quote of the answer (nor in the question) is reported; option quotes ground their option."""
    body = {"status": "answered", "answer": "15,000 hours", "citations": [_citation(ROW)]}
    assert ungrounded_numbers(body, "frame 256?") == []
    body["answer"] = "9,600 hours"
    assert ungrounded_numbers(body, "frame 256?") == ["9,600"]
    ambiguous = {
        "status": "ambiguous",
        "answer": "It depends on the frame.",
        "citations": [_citation(ROW)],
        "options": [{"condition": "Frame 210 to 280", "answer": "9,500 h", "citations": [_citation(ROW)]}],
    }
    assert ungrounded_numbers(ambiguous, "q") == []
    ambiguous["options"][0]["answer"] = "12,000 h"
    assert ungrounded_numbers(ambiguous, "q") == ["12,000"]


@pytest.mark.parametrize(
    ("changes", "stage"),
    [
        ({"ingestion_ok": False, "rank": None, "in_prompt": False, "http_status": 503}, "ingestion"),
        ({"rank": None, "in_prompt": False, "http_status": 503}, "retrieval"),
        ({"rank": 12, "in_prompt": False, "http_status": 503}, "ranking"),
        ({"http_status": 503, "error_code": "llm_unavailable"}, "provider"),
        ({"http_status": 502, "error_code": "llm_invalid_request"}, "provider"),
        ({"http_status": 500, "error_code": "internal_error"}, "api"),
        ({"reason": "unverified", "status": "not_found"}, "verification"),
        ({}, "generation"),
    ],
)
def test_the_failure_stage_is_the_first_in_pipeline_order(changes: dict[str, object], stage: str) -> None:
    """Ingestion, retrieval, ranking, provider, api, verification, generation: the first failing one wins.

    Args:
        changes: Fields of a failed result.
        stage: Expected stage.
    """
    result = _result(status="ambiguous")
    for key, value in changes.items():
        setattr(result, key, value)
    assert failure_stage(result) == stage


def _result(**changes: object) -> QuestionResult:
    """Build a result of an `answered` question whose expected page reached the prompt.

    Args:
        **changes: Fields to replace.

    Returns:
        The result.
    """
    result = QuestionResult(
        set_name="public",
        id="q-1",
        repeat=1,
        lang="en",
        question="q",
        expected_status="answered",
        expected_answer="9,500 hours.",
        expected_sources=["LB.pdf:2"],
        http_status=200,
        error_code=None,
        status="answered",
        reason=None,
        answer="9,500 hours.",
        options=0,
        answer_id="a",
        model="m",
        latency_ms=1.0,
        wall_ms=1.0,
        model_calls=1,
        verifier_retry=False,
        provider_retry=False,
        tokens={},
        coverage_pending=False,
        status_ok=True,
        source_ok=True,
        citations=[],
        ungrounded=[],
        ingestion_ok=True,
        rank=1,
        in_prompt=True,
    )
    for key, value in changes.items():
        setattr(result, key, value)
    result.status_ok = result.status == result.expected_status
    return result


def _judged(verdict: str, contradicted: bool = False) -> JudgeOutcome:
    """Build a judge outcome.

    Args:
        verdict: The verdict.
        contradicted: A fact is contradicted.

    Returns:
        The outcome.
    """
    return JudgeOutcome(verdict, contradicted, [], "judge")


@pytest.mark.parametrize(
    ("expected", "got", "judge", "flags"),
    [
        ("answered", "answered", _judged("correct"), (True, False, False)),
        ("answered", "answered", _judged("partial"), (False, False, False)),
        ("answered", "answered", _judged("wrong", contradicted=True), (False, True, False)),
        ("answered", "ambiguous", None, (False, False, True)),
        ("answered", "not_found", None, (False, False, True)),
        ("ambiguous", "answered", None, (False, True, False)),
        ("ambiguous", "not_found", None, (False, False, True)),
        ("not_found", "not_found", None, (True, False, False)),
        ("not_found", "answered", None, (False, True, False)),
        ("not_found", "ambiguous", None, (False, True, False)),
    ],
)
def test_headline_flags(expected: str, got: str, judge: JudgeOutcome | None, flags: tuple[bool, bool, bool]) -> None:
    """Correct, unsafe and over-cautious for each pair of expected and returned status.

    Args:
        expected: Expected status.
        got: Returned status.
        judge: The judge's outcome, when judged.
        flags: Expected `(correct, unsafe, over_cautious)`.
    """
    result = _result(expected_status=expected, status=got, judge=judge)
    finalize(result, judged=True)
    assert (result.correct, result.unsafe, result.over_cautious) == flags


def test_source_judge_failure_and_no_judge_mode() -> None:
    """Without a citation on an expected page an answer is not correct; a failed judge leaves it unknown.

    Without the judge, status and source decide.
    """
    no_source = finalize(_result(source_ok=False, judge=_judged("correct")), judged=True)
    assert (no_source.correct, no_source.stage) == (False, "generation")
    failed = finalize(_result(judge=JudgeOutcome(None, False, [], "judge", error="provider: transient")), judged=True)
    assert (failed.correct, failed.stage) == (None, None)
    plain = finalize(_result(), judged=False)
    assert plain.correct is True


def test_error_replies_take_coverage_from_the_record(corpus: Corpus) -> None:
    """A 503 has no answer body: its id, coverage and rank come from the record.

    Args:
        corpus: The ingested PDF.
    """
    reply = Reply(503, {"error": {"code": "llm_unavailable", "request_id": "a-9"}}, 30.0)
    record = _record([(3, 2, True)])
    record["coverage"] = {"incomplete": [{"status": "processing"}]}
    result = finalize(evaluate(_question(), reply, record, corpus), judged=True)
    assert (result.answer_id, result.error_code, result.coverage_pending) == ("a-9", "llm_unavailable", True)
    assert (result.correct, result.stage) == (False, "provider")


def test_failed_documents_do_not_count_as_pending_and_percentiles() -> None:
    """Only queued or processing documents mark coverage as pending; percentiles interpolate."""
    reply = _answered("x", [], coverage={"incomplete": [{"status": "failed"}]})
    corpus = Corpus([])
    assert evaluate(_question("not_found"), reply, None, corpus).coverage_pending is False
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.5) is None
