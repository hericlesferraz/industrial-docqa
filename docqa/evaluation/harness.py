"""The run: a fresh tenant, the folder uploaded as one batch, the questions asked, everything scored and written."""

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from docqa.evaluation.client import EvalClient, Reply
from docqa.evaluation.dataset import InputProblem, Question, pdf_folder_files
from docqa.evaluation.judge import Judge, JudgeInput, disagreement, numeric_check, response_text
from docqa.evaluation.metrics import (
    FINAL_STATUSES,
    Corpus,
    FileOutcome,
    QuestionResult,
    evaluate,
    finalize,
    needs_judge,
)

POLL_S = 2.0


@dataclass(frozen=True)
class RunConfig:
    """Options of one run.

    Attributes:
        pdf_dir: Folder whose files are uploaded.
        question_files: Question sets.
        out_dir: Where `results.jsonl` and `report.md` go.
        api_url: API base URL.
        repeat: Times each question is asked.
        concurrency: Questions in flight at once.
        judge_model: Judge model, or None with `--no-judge`.
        ingest_timeout_s: How long to wait for the batch; questions are asked anyway after it.
        poll_s: Seconds between two reads of the batch.
    """

    pdf_dir: Path
    question_files: list[Path]
    out_dir: Path
    api_url: str
    repeat: int = 1
    concurrency: int = 1
    judge_model: str | None = None
    ingest_timeout_s: float = 3600.0
    poll_s: float = POLL_S


@dataclass
class IngestResult:
    """The batch as the harness saw it.

    Attributes:
        tenant_id: Tenant created for the run.
        batch_id: The batch.
        files: Outcome of each file.
        complete: The batch reached a final state.
        timed_out: The wait passed `ingest_timeout_s`.
        paused: The wait stopped because every unfinished document was held by a provider pause.
        elapsed_s: Seconds from the first upload to the end of the wait.
        counts: The batch's counts at the end.
    """

    tenant_id: str
    batch_id: str
    files: list[FileOutcome]
    complete: bool
    timed_out: bool
    paused: bool
    elapsed_s: float
    counts: dict[str, object] = field(default_factory=dict)


@dataclass
class RunResult:
    """Everything the report needs.

    Attributes:
        config: The run's options.
        started_at: Start time (UTC).
        finished_at: End time (UTC).
        health: `GET /healthz` at the start.
        ingest: The batch.
        questions: The question sets as loaded.
        results: One result per question and repeat.
        input_problems: Lines of the question files skipped, or asked with part of them ignored.
    """

    config: RunConfig
    started_at: datetime
    finished_at: datetime
    health: dict[str, object]
    ingest: IngestResult
    questions: list[Question]
    results: list[QuestionResult]
    input_problems: list[InputProblem] = field(default_factory=list)


def _upload_outcome(path: Path, reply: Reply) -> FileOutcome:
    """Turn an upload reply into a file outcome.

    Args:
        path: The file.
        reply: The upload reply.

    Returns:
        The outcome.
    """
    if reply.status in (200, 202):
        outcome = "duplicate" if reply.body.get("duplicate") else "accepted"
        return FileOutcome(path.name, path, reply.status, outcome, document_id=str(reply.body.get("document_id")))
    if reply.status in (413, 422):
        return FileOutcome(path.name, path, reply.status, "rejected", error_code=reply.error_code)
    return FileOutcome(path.name, path, reply.status, "upload_error", error_code=reply.error_code or "http_error")


def ingest(
    client: EvalClient, config: RunConfig, tenant_id: str, *, log: Callable[[str], None], clock: Callable[[], float]
) -> IngestResult:
    """Upload every file of the folder as one batch and wait for a final state.

    Args:
        client: API client, already holding the tenant key.
        config: Run options.
        tenant_id: The run's tenant.
        log: Progress printer.
        clock: Monotonic clock, replaceable in tests.

    Returns:
        The batch as seen at the end of the wait, with each document's detail.
    """
    batch_id = client.create_batch()
    started = clock()
    files = []
    for path in pdf_folder_files(config.pdf_dir):
        try:
            outcome = _upload_outcome(path, client.upload(path, batch_id))
        except OSError as exc:
            outcome = FileOutcome(path.name, path, 0, "upload_error", error_code=f"unreadable_file ({exc.strerror})")
        log(f"upload {path.name}: {outcome.outcome} {outcome.error_code or ''}".rstrip())
        files.append(outcome)
    final_at: dict[str, float] = {}
    timed_out = paused = complete = False
    view: dict[str, object] = {}
    last_note = ""
    while True:
        view = client.batch(batch_id)
        elapsed = clock() - started
        documents = [i.get("document") for i in view.get("items") or [] if isinstance(i, dict)]
        documents = [d for d in documents if isinstance(d, dict)]
        for doc in documents:
            if doc.get("status") in FINAL_STATUSES:
                final_at.setdefault(str(doc.get("document_id")), elapsed)
        complete = bool(view.get("complete"))
        unfinished = [d for d in documents if d.get("status") not in FINAL_STATUSES]
        note = f"waiting for {len(unfinished)} document(s), {elapsed:.0f}s"
        if note.split(",")[0] != last_note.split(",")[0]:
            log(note)
            last_note = note
        if complete:
            break
        if elapsed > config.ingest_timeout_s:
            timed_out = True
            break
        if unfinished and all(d.get("ocr_paused") for d in unfinished):
            paused = True
            break
        time.sleep(config.poll_s)
    elapsed = clock() - started
    for item in files:
        if item.document_id:
            item.detail = client.document(item.document_id)
            item.final_after_s = final_at.get(item.document_id)
            error = item.detail.get("error") if item.detail else None
            if isinstance(error, dict) and error.get("code"):
                item.error_code = str(error["code"])
    log(f"ingestion {'complete' if complete else 'incomplete'} after {elapsed:.0f}s")
    counts = view.get("counts") if isinstance(view.get("counts"), dict) else {}
    return IngestResult(tenant_id, batch_id, files, complete, timed_out, paused, elapsed, counts)


def ask_all(
    client: EvalClient,
    questions: list[Question],
    corpus: Corpus,
    config: RunConfig,
    *,
    log: Callable[[str], None],
) -> list[QuestionResult]:
    """Ask every question `repeat` times and score what needs no judge.

    Args:
        client: API client.
        questions: The questions.
        corpus: The ingested files.
        config: Run options.
        log: Progress printer.

    Returns:
        Results in question order, repeats together.
    """
    jobs = [(q, r) for q in questions for r in range(1, config.repeat + 1)]
    done = [0]
    lock = threading.Lock()

    def one(job: tuple[Question, int]) -> QuestionResult:
        """Ask one question and read its record.

        Args:
            job: The question and the repeat number.

        Returns:
            The scored result.
        """
        question, repeat = job
        reply = client.ask(question.question)
        error = reply.body.get("error") if isinstance(reply.body.get("error"), dict) else {}
        answer_id = reply.body.get("answer_id") if reply.status == 200 else error.get("request_id")
        record = client.answer_record(str(answer_id)) if answer_id else None
        result = evaluate(question, reply, record, corpus, repeat=repeat)
        with lock:
            done[0] += 1
            got = result.status or result.error_code or f"http_{result.http_status}"
            log(
                f"[{done[0]}/{len(jobs)}] {question.set_name} {question.id}: {question.expected_status} -> {got} "
                f"({result.wall_ms / 1000:.1f}s)"
            )
        return result

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        return list(pool.map(one, jobs))


def judge_all(results: list[QuestionResult], judge: Judge | None, *, log: Callable[[str], None]) -> None:
    """Judge the content of every answer whose status matches, and run the numeric cross-check.

    Args:
        results: Results to update in place.
        judge: The judge, or None with `--no-judge`.
        log: Progress printer.
    """
    todo = [r for r in results if needs_judge(r)]
    for n, result in enumerate(todo, start=1):
        body = result.body
        text = response_text(body)
        result.numeric = numeric_check(result.question, result.expected_answer, text)
        if judge is None:
            continue
        item = JudgeInput(
            question=result.question,
            expected_status=result.expected_status,
            expected_answer=result.expected_answer,
            response_status=str(result.status),
            response_text=text,
        )
        result.judge = judge.judge(item)
        result.judge_review = disagreement(result.judge, result.numeric)
        verdict = result.judge.verdict or f"failed ({result.judge.error})"
        log(
            f"judge [{n}/{len(todo)}] {result.set_name} {result.id}: {verdict}{' (cached)' if result.judge.cached else ''}"
        )
    for result in results:
        finalize(result, judged=judge is not None)


def write_results(path: Path, run: RunResult) -> None:
    """Write the raw results: `run`, then `input_problem`, `file` and `question` lines, one per item.

    Args:
        path: Output file.
        run: The run.
    """
    with path.open("w", encoding="utf-8") as handle:
        meta = {
            "kind": "run",
            "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat(),
            "api_url": run.config.api_url,
            "repeat": run.config.repeat,
            "concurrency": run.config.concurrency,
            "judge_model": run.config.judge_model,
            "question_files": [p.name for p in run.config.question_files],
            "health": run.health,
            "tenant_id": run.ingest.tenant_id,
            "batch_id": run.ingest.batch_id,
            "complete": run.ingest.complete,
            "timed_out": run.ingest.timed_out,
            "paused": run.ingest.paused,
            "ingest_s": round(run.ingest.elapsed_s, 1),
            "counts": run.ingest.counts,
        }
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for problem in run.input_problems:
            handle.write(json.dumps({"kind": "input_problem", **problem.to_json()}, ensure_ascii=False) + "\n")
        for item in run.ingest.files:
            row = {
                "kind": "file",
                "file": item.file,
                "upload_status": item.upload_status,
                "outcome": item.outcome,
                "error_code": item.error_code,
                "document_id": item.document_id,
                "final_after_s": item.final_after_s,
                "detail": item.detail,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for result in run.results:
            handle.write(json.dumps({"kind": "question", **result.to_json()}, ensure_ascii=False) + "\n")


def run(
    config: RunConfig,
    questions: list[Question],
    client: EvalClient,
    judge: Judge | None,
    *,
    log: Callable[[str], None],
    problems: list[InputProblem] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Run the whole eval.

    Args:
        config: Run options.
        questions: Questions of every set, as loaded.
        client: API client holding the admin key.
        judge: The judge, or None.
        log: Progress printer.
        problems: Lines of the question files skipped or partly ignored, for the report.
        clock: Monotonic clock.

    Returns:
        The run, also written to `config.out_dir` as `results.jsonl`.

    Raises:
        HarnessError: The API is unreachable or refused a call the run depends on.
    """
    started_at = datetime.now(UTC)
    health = client.health()
    log(f"API {config.api_url}: HTTP {health.status} {health.body.get('status', '')}")
    tenant_name = f"eval-{started_at:%Y%m%dT%H%M%SZ}"
    tenant_id = client.create_tenant(tenant_name)
    log(f"tenant {tenant_name} ({tenant_id})")
    ingested = ingest(client, config, tenant_id, log=log, clock=clock)
    corpus = Corpus(ingested.files)
    results = ask_all(client, questions, corpus, config, log=log)
    judge_all(results, judge, log=log)
    outcome = RunResult(
        config=config,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        health=health.body,
        ingest=ingested,
        questions=questions,
        results=results,
        input_problems=list(problems or []),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    write_results(config.out_dir / "results.jsonl", outcome)
    return outcome
