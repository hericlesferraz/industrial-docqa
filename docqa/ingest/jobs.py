"""The ingestion jobs: `extract` creates a document's page rows, `ocr_page` transcribes one flagged page."""

import json
import logging
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from psycopg import Connection

from docqa import queue
from docqa.db import tx
from docqa.handler import JobContext, JobSignal, PermanentFailure, Reschedule, Retry
from docqa.ingest.detect import BLANK, NEEDS_OCR, TEXT_OK
from docqa.ingest.labels import confirm_labels, edge_line_candidates
from docqa.ingest.ocr import OCR_DPI, ModelPaused, OcrOutcome, ocr_version, transcribe
from docqa.ingest.sandbox import EXIT_MEMORY, SandboxResult, run_sandboxed
from docqa.providers import pacing
from docqa.providers.llm import ProviderError
from docqa.queue import Job
from docqa.repo import documents, ocr_cache, pages
from docqa.repo.documents import DocumentRow
from docqa.repo.pages import PageInit
from docqa.telemetry import Span

EXTRACT_VERSION = "x1:pymupdf-2.0:det-g2:lbl-g3"
EXTRACT_STALL_S = 300
CHILD_MODULE = "docqa.ingest.extract_child"
RENDER_MODULE = "docqa.ingest.render_child"
RENDER_TIMEOUT_S = 120
RATE_LIMIT_WAIT_S = 30.0
PAGE_ERROR_CODES = {
    "render_failed",
    "ocr_recitation",
    "ocr_safety",
    "ocr_max_tokens",
    "ocr_empty",
    "ocr_finish_other",
    "ocr_invalid_request",
    "ocr_unavailable",
    "attempts_exhausted",
    "lease_expired",
}
TRANSIENT_KINDS = ("transient", "timeout")
MESSAGE_MAX = 300
PAGE_STATES = {TEXT_OK: "text_ready", NEEDS_OCR: "needs_ocr", BLANK: "blank"}

logger = logging.getLogger("docqa.ingest")


@dataclass
class ChildOutput:
    """What the extraction child wrote.

    Attributes:
        page_count: Pages announced in the header, or None.
        error: `(code, message)` of an error line, or None.
        pages: Page records in output order.
    """

    page_count: int | None = None
    error: tuple[str, str] | None = None
    pages: list[dict[str, object]] = field(default_factory=list)


def read_child_output(path: Path) -> ChildOutput:
    """Parse the JSON lines of the extraction child.

    Args:
        path: The child's output file.

    Returns:
        The header, the error line if any, and the page records.
    """
    output = ChildOutput()
    if not path.exists():
        return output
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if "error" in record:
                output.error = (str(record["error"]), str(record.get("message", "")))
            elif "page_count" in record:
                output.page_count = int(record["page_count"])
            else:
                output.pages.append(record)
    return output


def _last_line(text: str) -> str:
    """Return the last non-empty line of a text, cut for storage.

    Args:
        text: Usually the child's stderr tail.

    Returns:
        The line, at most `MESSAGE_MAX` characters.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:MESSAGE_MAX] if lines else ""


def check_child(result: SandboxResult, output: ChildOutput, job: Job, *, timeout_s: int, memory_mb: int) -> None:
    """Turn a failed extraction into the job signal that records it.

    A child killed by a signal the parent didn't send (for example the kernel's out-of-memory killer acting on
    another process's load) is retried within the job's attempts; every other failure is permanent, because the same
    file fails the same way.

    Args:
        result: How the child ended.
        output: What it wrote.
        job: The extract job, for its attempt count.
        timeout_s: The total time limit that was applied.
        memory_mb: The memory limit that was applied.

    Raises:
        PermanentFailure: The document can't be extracted; the code goes on the document.
        Retry: The child was killed by a signal it didn't get from us, and attempts remain.
    """
    if result.stalled:
        raise PermanentFailure("extract_timeout", f"no page finished for {EXTRACT_STALL_S} s")
    if result.timed_out:
        raise PermanentFailure("extract_timeout", f"extraction took longer than {timeout_s} s")
    if result.memory_exceeded or result.returncode == EXIT_MEMORY:
        raise PermanentFailure("extract_memory", f"extraction used more than {memory_mb} MiB")
    signal_number = result.crash_signal
    if signal_number is not None:
        message = f"the extraction process was killed by signal {signal_number}"
        if job.attempts < job.max_attempts:
            raise Retry("extract_crashed", message)
        raise PermanentFailure("extract_crashed", message)
    if result.returncode != 0:
        detail = _last_line(result.stderr_tail)
        raise PermanentFailure("extract_crashed", f"the extraction process failed: {detail}"[:MESSAGE_MAX])
    if output.error is not None:
        code, message = output.error
        raise PermanentFailure(code, message[:MESSAGE_MAX])
    if output.page_count is None or len(output.pages) != output.page_count:
        raise PermanentFailure("extract_crashed", "the extraction process ended without every page")


def build_pages(records: list[dict[str, object]]) -> list[PageInit]:
    """Turn the child's page records into page rows, with labels confirmed across the whole document.

    Args:
        records: Page records in page order.

    Returns:
        One row per page.
    """
    candidates = [[str(token) for token in record["label_candidates"]] for record in records]
    labels = confirm_labels(candidates)
    rows = []
    for record, label, layer_candidates in zip(records, labels, candidates, strict=True):
        state = PAGE_STATES[str(record["state"])]
        quality = dict(record["quality"])
        quality["label_candidates"] = {"layer": layer_candidates}
        rows.append(
            PageInit(
                page_no=int(record["page_no"]),
                state=state,
                text_source="layer" if state == "text_ready" else None,
                layer_text=str(record["layer_text"]),
                quality=quality,
                page_label=label,
                text_rev=1 if state == "text_ready" else 0,
                extract_version=EXTRACT_VERSION,
            )
        )
    return rows


def _reason_counts(rows: list[PageInit]) -> list[str]:
    """Count the pages sent to OCR by reason, for the span.

    Args:
        rows: Page rows.

    Returns:
        `"<reason>=<count>"` strings, sorted.
    """
    counts = Counter(
        str(row.quality["detector"].get("reason"))
        for row in rows
        if row.state == "needs_ocr" and isinstance(row.quality.get("detector"), dict)
    )
    return [f"{reason}={count}" for reason, count in sorted(counts.items())]


class ExtractHandler:
    """Runs `extract` jobs: one per document."""

    def run(self, ctx: JobContext) -> None:
        """Extract a document and create its pages, the OCR jobs of flagged pages and the `index` job.

        Args:
            ctx: The claimed job and the process dependencies.

        Raises:
            PermanentFailure: The PDF can't be extracted (encrypted, corrupt, too many pages, time or memory limit).
            Retry: The child was killed by a signal it didn't get from us.
            LeaseLost: The lease moved to another owner.
        """
        job = ctx.job
        assert job.document_id is not None
        with ctx.pool.connection() as conn:
            doc = documents.get_document(conn, tenant_id=job.tenant_id, document_id=job.document_id)
        if doc is None or doc.page_count is not None or doc.status == "failed":
            with tx(ctx.pool) as conn:
                queue.assert_lease(conn, job)
                if doc is not None and doc.page_count is not None:
                    queue.enqueue(conn, tenant_id=job.tenant_id, kind="index", document_id=job.document_id)
                queue.complete(conn, job)
            return
        settings = ctx.settings
        with ctx.telemetry.span("extract.document") as span:
            with tempfile.TemporaryDirectory(prefix="docqa-extract-") as scratch, ctx.blobs.open(doc.blob_key) as pdf:
                out = Path(scratch) / "pages.jsonl"
                result = run_sandboxed(
                    CHILD_MODULE,
                    [str(pdf), str(out), str(settings.max_pdf_pages)],
                    timeout_s=settings.extract_timeout_s,
                    memory_mb=settings.extract_memory_mb,
                    progress_path=out,
                    stall_s=EXTRACT_STALL_S,
                )
                span.set("docqa.child_ms", round(result.wall_s * 1000, 1))
                span.set("docqa.child_peak_rss_mb", round(result.peak_rss_mb, 1))
                span.set("docqa.child_exit", result.returncode)
                output = read_child_output(out) if result.returncode == 0 else ChildOutput()
            check_child(result, output, job, timeout_s=settings.extract_timeout_s, memory_mb=settings.extract_memory_mb)
            rows = build_pages(output.pages)
            states = Counter(row.state for row in rows)
            span.set("docqa.pages", len(rows))
            span.set("docqa.pages.text", states["text_ready"])
            span.set("docqa.pages.needs_ocr", states["needs_ocr"])
            span.set("docqa.pages.blank", states["blank"])
            span.set("docqa.reasons", _reason_counts(rows))
            with tx(ctx.pool) as conn:
                queue.assert_lease(conn, job)
                pages.insert_pages(conn, tenant_id=job.tenant_id, document_id=doc.id, pages=rows)
                for row in rows:
                    if row.state == "needs_ocr":
                        queue.enqueue(
                            conn, tenant_id=job.tenant_id, kind="ocr_page", document_id=doc.id, page_no=row.page_no
                        )
                documents.set_page_count(conn, tenant_id=job.tenant_id, document_id=doc.id, page_count=len(rows))
                status = documents.recompute_document(conn, tenant_id=job.tenant_id, document_id=doc.id)
                queue.enqueue(conn, tenant_id=job.tenant_id, kind="index", document_id=doc.id)
                queue.complete(conn, job)
            span.set("docqa.document.status", status)

    def on_failure(self, conn: Connection, job: Job, code: str, message: str) -> None:
        """Mark the document failed with the extraction's error code.

        Args:
            conn: Connection inside the `fail` transaction.
            job: The failing job.
            code: Error code.
            message: Detail without customer text.
        """
        if job.document_id is None:
            return
        documents.set_document_failed(
            conn, tenant_id=job.tenant_id, document_id=job.document_id, code=code, message=message
        )


def page_error_code(code: str) -> str:
    """Map the code a failed `ocr_page` job ends with to the page's error code.

    Args:
        code: The job's last error code.

    Returns:
        The code itself when it is a page code; `ocr_unavailable` after transient provider errors;
        `attempts_exhausted` otherwise.
    """
    if code in PAGE_ERROR_CODES:
        return code
    if code in TRANSIENT_KINDS:
        return "ocr_unavailable"
    return "attempts_exhausted"


class OcrPageHandler:
    """Runs `ocr_page` jobs: one per page the detector flagged."""

    def run(self, ctx: JobContext) -> None:
        """Transcribe the page, from the cross-tenant cache or through the model chain, and queue the `index`.

        Args:
            ctx: The claimed job and the process dependencies.

        Raises:
            PermanentFailure: The page can't be transcribed (render failure, refusal, cut, empty, invalid request,
                or provider errors on the last attempt).
            Retry: A transient provider or render failure, with attempts left.
            Reschedule: A per-minute rate limit, or a paused model; no attempt is spent.
            LeaseLost: The lease moved to another owner.
        """
        job = ctx.job
        assert job.document_id is not None and job.page_no is not None and ctx.llm is not None
        with ctx.pool.connection() as conn:
            doc = documents.get_document(conn, tenant_id=job.tenant_id, document_id=job.document_id)
            page = pages.get_page_for_ocr(
                conn, tenant_id=job.tenant_id, document_id=job.document_id, page_no=job.page_no
            )
        if doc is None or page is None or page.state != "needs_ocr":
            with tx(ctx.pool) as conn:
                queue.assert_lease(conn, job)
                queue.complete(conn, job)
            return
        version = ocr_version(ctx.settings)
        with ctx.telemetry.span("ocr.page") as span:
            with ctx.pool.connection() as conn:
                cached = ocr_cache.get(conn, sha256=doc.sha256, page_no=job.page_no, ocr_version=version)
            span.set("docqa.ocr.from_cache", cached is not None)
            if cached is not None:
                span.set("docqa.ocr.model_final", cached.model)
                self._write(ctx, cached.text, version, {"model": cached.model, "from_cache": True})
                return
            png = self._render(ctx, doc)
            try:
                outcome = transcribe(
                    png,
                    llm=ctx.llm,
                    settings=ctx.settings,
                    pool=ctx.pool,
                    attempt=job.attempts,
                    max_attempts=job.max_attempts,
                )
            except ModelPaused as paused:
                raise Reschedule(paused.pause.reason, paused.pause.until) from None
            except ProviderError as error:
                raise self._signal(ctx, error) from None
            self._record(span, outcome)
            if not outcome.ok:
                raise PermanentFailure(str(outcome.code), outcome.message())
            assert outcome.text is not None and outcome.model is not None
            with tx(ctx.pool) as conn:
                ocr_cache.put(
                    conn,
                    sha256=doc.sha256,
                    page_no=job.page_no,
                    ocr_version=version,
                    text=outcome.text,
                    model=outcome.model,
                )
            self._write(ctx, outcome.text, version, outcome.quality())

    @staticmethod
    def _record(span: Span, outcome: OcrOutcome) -> None:
        """Set the outcome attributes of the `ocr.page` span.

        Args:
            span: The open span.
            outcome: How the chain ended.
        """
        span.set("docqa.ocr.recitations", outcome.recitations)
        span.set("docqa.ocr.fallback_used", outcome.fallback_used)
        span.set("docqa.ocr.calls", len(outcome.finish_reasons))
        if outcome.model:
            span.set("docqa.ocr.model_final", outcome.model)

    @staticmethod
    def _render(ctx: JobContext, doc: DocumentRow) -> bytes:
        """Draw the job's page as a PNG in the sandboxed render child.

        Args:
            ctx: The claimed job and the process dependencies.
            doc: The page's document.

        Returns:
            The PNG bytes.

        Raises:
            Retry: The child was killed by a signal it didn't get from us, and attempts remain.
            PermanentFailure: `render_failed`.
        """
        job = ctx.job
        with ctx.telemetry.span("ocr.render", **{"docqa.render.dpi": OCR_DPI}) as span:
            with tempfile.TemporaryDirectory(prefix="docqa-render-") as scratch, ctx.blobs.open(doc.blob_key) as pdf:
                out = Path(scratch) / "page.png"
                result = run_sandboxed(
                    RENDER_MODULE,
                    [str(pdf), str(job.page_no), str(OCR_DPI), str(out)],
                    timeout_s=RENDER_TIMEOUT_S,
                    memory_mb=ctx.settings.extract_memory_mb,
                )
                if result.returncode == 0 and out.exists():
                    png = out.read_bytes()
                    span.set("docqa.render.png_bytes", len(png))
                    return png
            if result.crash_signal is not None and job.attempts < job.max_attempts:
                raise Retry("render_failed", f"the render process was killed by signal {result.crash_signal}")
            if result.timed_out or result.memory_exceeded:
                limit = "time" if result.timed_out else "memory"
                raise PermanentFailure("render_failed", f"rendering the page passed the {limit} limit")
            raise PermanentFailure("render_failed", f"the page could not be rendered: {_last_line(result.stderr_tail)}")

    @staticmethod
    def _signal(ctx: JobContext, error: ProviderError) -> JobSignal:
        """Turn a provider error into the queue signal that handles it.

        Args:
            ctx: The claimed job and the process dependencies.
            error: The classified error.

        Returns:
            `Retry` for transient errors with attempts left (`PermanentFailure('ocr_unavailable')` on the last one);
            `Reschedule` after a rate limit (holding the model's next slot) or a pause (written here);
            `PermanentFailure('ocr_invalid_request')` for a malformed request.
        """
        job = ctx.job
        now = datetime.now(UTC)
        if error.kind in TRANSIENT_KINDS:
            if job.attempts < job.max_attempts:
                return Retry(error.kind, error.message)
            return PermanentFailure("ocr_unavailable", error.message)
        if error.kind == "rate_limited":
            wait = error.retry_after_s or RATE_LIMIT_WAIT_S
            if error.model:
                with ctx.pool.connection() as conn:
                    pacing.hold(conn, error.model, wait)
            return Reschedule("rate_limited", now + timedelta(seconds=wait))
        if error.kind in pacing.PAUSE_KINDS:
            with tx(ctx.pool) as conn:
                paused = pacing.pause_for(conn, error, now=now)
            ctx.telemetry.event(
                "provider.paused",
                **{
                    "gen_ai.request.model": paused.model,
                    "docqa.pause.reason": paused.reason,
                    "docqa.pause.until": paused.until.isoformat(),
                },
            )
            logger.warning(
                "provider paused",
                extra={"fields": {"model": paused.model, "reason": paused.reason, "until": paused.until.isoformat()}},
            )
            return Reschedule(paused.reason, paused.until)
        if error.kind == "invalid_request":
            return PermanentFailure("ocr_invalid_request", error.message)
        return Retry("internal_error", f"unknown provider error kind {error.kind}")

    @staticmethod
    def _write(ctx: JobContext, text: str, version: str, quality_ocr: dict[str, object]) -> None:
        """Store the transcription on the page, queue the `index` and complete, in one fenced transaction.

        Args:
            ctx: The claimed job and the process dependencies.
            text: Canonical transcription.
            version: OCR version.
            quality_ocr: Evidence for `quality.ocr`.

        Raises:
            LeaseLost: The lease moved to another owner.
        """
        job = ctx.job
        assert job.document_id is not None and job.page_no is not None
        with tx(ctx.pool) as conn:
            queue.assert_lease(conn, job)
            pages.write_ocr_text(
                conn,
                tenant_id=job.tenant_id,
                document_id=job.document_id,
                page_no=job.page_no,
                text=text,
                ocr_version=version,
                quality_ocr=quality_ocr,
                label_candidates=edge_line_candidates(text),
            )
            queue.enqueue(conn, tenant_id=job.tenant_id, kind="index", document_id=job.document_id)
            queue.complete(conn, job)

    def on_failure(self, conn: Connection, job: Job, code: str, message: str) -> None:
        """Mark the page failed, queue the `index` so the document reaches a final state, and recount the document.

        Args:
            conn: Connection inside the `fail` transaction.
            job: The failing job.
            code: The job's error code; mapped by `page_error_code`.
            message: Detail without customer text.
        """
        if job.document_id is None or job.page_no is None:
            return
        pages.set_page_failed(
            conn,
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            page_no=job.page_no,
            code=page_error_code(code),
            message=message,
        )
        queue.enqueue(conn, tenant_id=job.tenant_id, kind="index", document_id=job.document_id)
        documents.recompute_document(conn, tenant_id=job.tenant_id, document_id=job.document_id)
