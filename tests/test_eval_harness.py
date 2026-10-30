"""The whole eval run against a scripted API: ingestion wait, questions, records, judge, results and report."""

import json
from pathlib import Path

import httpx
import pymupdf
import pytest

from docqa.evaluation.__main__ import main
from docqa.evaluation.client import EvalClient
from docqa.evaluation.dataset import load_sets
from docqa.evaluation.harness import RunConfig, run
from docqa.evaluation.judge import Judge, JudgeCache
from docqa.evaluation.report import render
from docqa.providers.fake import FakeLLM, result

DOC = "22222222-2222-2222-2222-222222222222"
QUOTE = "Over 210 to 280 incl.(180) | 9500Hrs."


class FakeApi:
    """Answers the eval's calls like the real API would, from a fixed script.

    Attributes:
        batch_reads: How many times the batch was read.
        asked: Questions received.
    """

    def __init__(self) -> None:
        """Start with nothing asked."""
        self.batch_reads = 0
        self.asked: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Route one request.

        Args:
            request: The request.

        Returns:
            The scripted response.
        """
        path, method = request.url.path, request.method
        if path == "/healthz":
            return httpx.Response(200, json={"status": "ok", "checks": {}, "reasons": []})
        if path == "/v1/admin/tenants":
            return httpx.Response(201, json={"tenant_id": "t-1", "name": "eval", "api_key": "dq_test"})
        assert request.headers["Authorization"] == "Bearer dq_test"
        if path == "/v1/batches" and method == "POST":
            return httpx.Response(201, json={"batch_id": "b-1", "created_at": "2026-10-30T00:00:00Z"})
        if path == "/v1/documents" and method == "POST":
            if b'filename="notes.txt"' in request.content:
                return httpx.Response(422, json={"error": {"code": "not_a_pdf", "request_id": "r"}})
            return httpx.Response(202, json={"document_id": DOC, "filename": "LB.pdf", "status": "queued"})
        if path == "/v1/batches/b-1":
            self.batch_reads += 1
            status = "processing" if self.batch_reads == 1 else "ready"
            item = {"filename": "LB.pdf", "document": {"document_id": DOC, "status": status, "ocr_paused": None}}
            return httpx.Response(200, json={"complete": status == "ready", "items": [item], "counts": {"items": 2}})
        if path == f"/v1/documents/{DOC}":
            return httpx.Response(200, json=_detail())
        if path == "/v1/questions":
            question = json.loads(request.content)["question"]
            self.asked.append(question)
            return _answer(question, len(self.asked))
        if path.startswith("/v1/answers/"):
            return httpx.Response(200, json=_record(path.rsplit("/", 1)[1]))
        return httpx.Response(404, json={"error": {"code": "not_found"}})


def _detail() -> dict[str, object]:
    """Build the document detail of the ingested PDF.

    Returns:
        The detail.
    """
    return {
        "document_id": DOC,
        "filename": "LB.pdf",
        "status": "ready",
        "error": None,
        "page_count": 2,
        "pages_indexed": 2,
        "pages_by_source": {"layer": 2},
        "ocr": {
            "pages_flagged": 1,
            "pages_transcribed": 1,
            "pages_from_cache": 1,
            "by_model": {"gemini-3.5-flash-lite": {"calls": 0, "input_tokens": 0, "output_tokens": 0}},
        },
        "pages": [{"page": 1, "state": "indexed"}, {"page": 2, "state": "indexed"}],
    }


def _answer(question: str, n: int) -> httpx.Response:
    """Answer a question: right for the interval, `not_found` for the IP rating, a 503 for anything else.

    Args:
        question: The question.
        n: Request number, used as the answer id.

    Returns:
        The response.
    """
    coverage = {"complete": True, "incomplete": []}
    citation = {"document_id": DOC, "filename": "LB.pdf", "page": 2, "quote": QUOTE, "transcribed": False}
    if "interval" in question:
        body = {"answer_id": f"a-{n}", "status": "answered", "answer": "9,500 hours.", "citations": [citation]}
        return httpx.Response(200, json={**body, "coverage": coverage})
    if "IP" in question:
        body = {"answer_id": f"a-{n}", "status": "not_found", "answer": "Not stated.", "citations": []}
        return httpx.Response(200, json={**body, "coverage": coverage})
    error = {"code": "llm_unavailable", "request_id": f"a-{n}", "retryable": True, "message": "down"}
    return httpx.Response(503, json={"error": error})


def _record(answer_id: str) -> dict[str, object]:
    """Build the record of an answer.

    Args:
        answer_id: The answer id.

    Returns:
        The record.
    """
    call = {"purpose": "answer", "attempt": 1, "model": "gemini-3.8-flash", "input_tokens": 3000, "output_tokens": 100}
    return {
        "answer_id": answer_id,
        "versions": {"model": "gemini-3.8-flash"},
        "totals": {"latency_ms": 2400.0},
        "calls": [call],
        "candidates": [{"rank": 1, "document_id": DOC, "page": 2, "sent_to_llm": True}],
        "coverage": {"incomplete": []},
    }


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    """A PDF folder (one PDF with the quoted row, one text file) and a question file.

    Args:
        tmp_path: Pytest temp folder.

    Returns:
        The folder and the question file.
    """
    folder = tmp_path / "pdfs"
    folder.mkdir()
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Cover")
    doc.new_page().insert_text((40, 72), "Over 210 to 280 incl.(180) 3600Hrs. 9500Hrs.", fontsize=8)
    doc.save(folder / "LB.pdf")
    doc.close()
    (folder / "notes.txt").write_text("not a pdf")
    rows = [
        {
            "id": "q-1",
            "question": "Relubrication interval for frame 256 at 1800 RPM?",
            "expected_status": "answered",
            "expected_answer": "9,500 hours.",
            "sources": [{"file": "LB.pdf", "pages": [2]}],
            "lang": "en",
        },
        {
            "id": "q-2",
            "question": "What is the IP rating?",
            "expected_status": "not_found",
            "expected_answer": "Not stated.",
            "sources": [],
            "lang": "en",
        },
        {
            "id": "q-3",
            "question": "How much grease?",
            "expected_status": "answered",
            "expected_answer": "0.61 oz.",
            "sources": [{"file": "LB.pdf", "pages": [2]}],
            "lang": "pt",
        },
    ]
    questions = tmp_path / "public.jsonl"
    questions.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return folder, questions


def test_a_full_run_against_a_scripted_api(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    """Upload, wait, ask, read records, judge, score, write results and render the report.

    Args:
        tmp_path: Pytest temp folder.
        inputs: The PDF folder and the question file.
    """
    folder, questions_file = inputs
    api = FakeApi()
    client = EvalClient("http://api", "admin", transport=httpx.MockTransport(api))
    facts = json.dumps({"facts": [{"fact": "9,500 hours", "essential": True, "verdict": "stated", "note": "9,500"}]})
    llm = FakeLLM([result(facts, input_tokens=400, output_tokens=50)])
    judge = Judge(llm, "gemini-3.5-flash-lite", JudgeCache(tmp_path / "cache"))
    config = RunConfig(
        pdf_dir=folder,
        question_files=[questions_file],
        out_dir=tmp_path / "out",
        api_url="http://api",
        judge_model="gemini-3.5-flash-lite",
        poll_s=0.0,
    )
    lines: list[str] = []
    questions, _ = load_sets([questions_file], folder)
    outcome = run(config, questions, client, judge, log=lines.append)
    assert outcome.ingest.complete and api.batch_reads == 2
    assert [(f.file, f.outcome, f.error_code) for f in outcome.ingest.files] == [
        ("LB.pdf", "accepted", None),
        ("notes.txt", "rejected", "not_a_pdf"),
    ]
    first, second, third = outcome.results
    assert (first.correct, first.judge.verdict if first.judge else None, first.citations[0].lines_found) == (
        True,
        "correct",
        1,
    )
    assert (second.correct, second.judge) == (True, None)
    assert (third.correct, third.stage, third.error_code) == (False, "provider", "llm_unavailable")
    assert len(llm.calls) == 1
    rows = [json.loads(line) for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["run", "file", "file", "question", "question", "question"]
    report = render(outcome)
    assert "| public | 3 | 2 (67%) | 0 (0%) | 0 (0%) | 1 |" in report
    assert "| provider | 1 | public/q-3 |" in report
    assert "**provisional**" in report
    assert "| Judge | `gemini-3.5-flash-lite` | 1 | 400 | 50 |" in report
    assert "1/1 lines" in report
    assert any("ingestion complete" in line for line in lines)


def test_without_the_judge_and_with_an_ingestion_timeout(tmp_path: Path, inputs: tuple[Path, Path]) -> None:
    """`--no-judge` makes no judge call and says so; a timeout still asks the questions and is reported.

    Args:
        tmp_path: Pytest temp folder.
        inputs: The PDF folder and the question file.
    """
    folder, questions_file = inputs
    client = EvalClient("http://api", "admin", transport=httpx.MockTransport(FakeApi()))
    config = RunConfig(
        pdf_dir=folder,
        question_files=[questions_file],
        out_dir=tmp_path / "out",
        api_url="http://api",
        ingest_timeout_s=0.0,
        poll_s=0.0,
    )
    ticks = iter(range(100))
    questions, _ = load_sets([questions_file], folder)
    outcome = run(config, questions, client, None, log=lambda _: None, clock=lambda: next(ticks))
    assert outcome.ingest.timed_out and not outcome.ingest.complete
    assert outcome.results[0].correct is True and outcome.results[0].judge is None
    report = render(outcome)
    assert "off (`--no-judge`)" in report and "**timed out**" in report


def test_the_command_line_exit_codes(
    tmp_path: Path, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """0 for a valid input, 2 for invalid input, 1 when the API can't be reached.

    Args:
        tmp_path: Pytest temp folder.
        inputs: The PDF folder and the question file.
        monkeypatch: Sets the environment.
        capsys: Captures the output.
    """
    folder, questions_file = inputs
    monkeypatch.setenv("GEMINI_API_KEY", "unused")
    monkeypatch.setenv("ADMIN_API_KEY", "admin")
    monkeypatch.setenv("EVAL_API_URL", "http://127.0.0.1:9")
    assert main(["validate", "--pdfs", str(folder), "--questions", str(questions_file)]) == 0
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n')
    assert main(["validate", "--pdfs", str(folder), "--questions", str(bad)]) == 2
    args = ["run", "--pdfs", str(folder), "--questions", str(questions_file), "--out", str(tmp_path / "o")]
    assert main([*args, "--no-judge"]) == 1
    assert main([*args, "--repeat", "0"]) == 2
    assert main([*args, "--questions", str(bad), "--no-judge"]) == 1
    assert main(["run", "--pdfs", str(folder), "--questions", str(bad), "--out", str(tmp_path / "o")]) == 2
    assert main([*args[:2], str(tmp_path / "nowhere"), *args[3:]]) == 2
    err = capsys.readouterr().err
    assert "harness error" in err and "bad.jsonl:1: skipped" in err and "PDF folder does not exist" in err


def test_a_hidden_set_with_bad_lines_and_a_missing_file_still_gets_a_report(
    tmp_path: Path, inputs: tuple[Path, Path]
) -> None:
    """Bad lines are skipped and listed; a question whose source file is missing is asked and fails at ingestion.

    Args:
        tmp_path: Pytest temp folder.
        inputs: The PDF folder and the question file.
    """
    folder, questions_file = inputs
    rows = questions_file.read_text().splitlines()
    missing = {
        "id": "q-4",
        "question": "Which IP class for frame 256?",
        "expected_status": "answered",
        "expected_answer": "9,500 hours.",
        "sources": [{"file": "gone.pdf", "pages": [1]}],
    }
    broken = [*rows, "{broken", json.dumps({**missing, "id": "q-5", "expected_status": "needs_clarification"})]
    hidden = tmp_path / "hidden.jsonl"
    hidden.write_text("\n".join([*broken, json.dumps(missing), json.dumps({**missing, "question": "IP rating?"})]))
    questions, problems = load_sets([hidden], folder)
    assert [p.line for p in problems if p.skipped] == [4, 5, 7]
    client = EvalClient("http://api", "admin", transport=httpx.MockTransport(FakeApi()))
    config = RunConfig(
        pdf_dir=folder, question_files=[hidden], out_dir=tmp_path / "out", api_url="http://api", poll_s=0.0
    )
    outcome = run(config, questions, client, None, log=lambda _: None, problems=problems)
    last = outcome.results[-1]
    assert (last.id, last.sources_missing, last.source_ok, last.stage) == ("q-4", ["gone.pdf"], None, "ingestion")
    report = render(outcome)
    assert "## Input problems" in report and "3 line(s) skipped" in report
    assert "| hidden.jsonl | 4 | skipped | not valid JSON" in report
    assert "unknown `expected_status` 'needs_clarification'" in report and "duplicate id 'q-4'" in report
    assert "| q-4 ✗ | - | answered | not_found | - | file missing from folder | - | ingestion |" in report
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    assert kinds.count("input_problem") == len(problems)


def test_an_unreadable_file_is_listed_not_fatal(
    tmp_path: Path, inputs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file the harness can't open is an upload error in the report; the run goes on.

    Args:
        tmp_path: Pytest temp folder.
        inputs: The PDF folder and the question file.
        monkeypatch: Replaces the upload.
    """
    folder, questions_file = inputs
    client = EvalClient("http://api", "admin", transport=httpx.MockTransport(FakeApi()))
    real = client.upload

    def upload(path: Path, batch_id: str) -> object:
        """Fail on the text file as an unreadable file would.

        Args:
            path: The file.
            batch_id: The batch.

        Returns:
            The real reply for other files.

        Raises:
            PermissionError: For `notes.txt`.
        """
        if path.name == "notes.txt":
            raise PermissionError(13, "Permission denied")
        return real(path, batch_id)

    monkeypatch.setattr(client, "upload", upload)
    config = RunConfig(
        pdf_dir=folder, question_files=[questions_file], out_dir=tmp_path / "out", api_url="http://api", poll_s=0.0
    )
    questions, _ = load_sets([questions_file], folder)
    outcome = run(config, questions, client, None, log=lambda _: None)
    assert [(f.file, f.outcome, f.error_code) for f in outcome.ingest.files] == [
        ("LB.pdf", "accepted", None),
        ("notes.txt", "upload_error", "unreadable_file (Permission denied)"),
    ]
