"""The `extract` job: page rows, OCR jobs, labels, document status, and every way a hostile PDF can fail."""

from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest
from psycopg_pool import ConnectionPool

from docqa.ingest import jobs as extract_jobs
from docqa.ingest.sandbox import SandboxResult, run_sandboxed
from docqa.storage import LocalBlobStore
from docqa.telemetry import MemoryTelemetry
from docqa.worker import WorkerDeps, run_one
from tests.helpers import make_runnable, new_tenant
from tests.ingest_helpers import document, drain, jobs_of, page_rows, store_pdf
from tests.pdfs import (
    LB5001,
    MN414,
    case_pdf,
    encrypted_pdf,
    garbage_pdf,
    rasterized_pdf,
    text_pdf,
    truncated_pdf,
)


def test_text_pdf_becomes_ready(worker_deps: WorkerDeps, blobs: LocalBlobStore, telemetry: MemoryTelemetry) -> None:
    """A healthy manual: page rows with layer text, `processing` after extract, `ready` after the index job.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        telemetry: In-memory telemetry.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, case_pdf(MN414), "MN414.pdf")
    assert run_one(worker_deps, "extract")
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.page_count, doc.pages_pending, doc.pages_indexed) == ("processing", 16, 15, 0)
    rows = page_rows(pool, document_id)
    assert [row["state"] for row in rows].count("blank") == 1
    text = [row for row in rows if row["state"] == "text_ready"]
    assert all(row["text_source"] == "layer" and row["text_rev"] == 1 and row["layer_text"] for row in text)
    assert rows[0]["extract_version"] == extract_jobs.EXTRACT_VERSION
    assert "label_candidates" in rows[0]["quality"]
    assert [job["kind"] for job in jobs_of(pool, document_id)] == ["extract", "index"]
    span = telemetry.named("extract.document")[0]
    assert span.attributes["docqa.pages"] == 16
    assert span.attributes["docqa.pages.blank"] == 1
    assert span.attributes["docqa.child_peak_rss_mb"] > 0
    assert run_one(worker_deps, "index")
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.pages_pending, doc.pages_indexed, doc.pages_failed) == ("ready", 0, 15, 0)
    assert {row["index_version"] for row in page_rows(pool, document_id) if row["state"] == "indexed"} == {"null"}


def test_rasterized_page_is_flagged_for_ocr(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A scanned page is `needs_ocr` with reason `no_text_layer`, gets an `ocr_page` job, and keeps the document open.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    drain(worker_deps)
    [row] = page_rows(pool, document_id)
    assert (row["state"], row["text_source"], row["text_rev"]) == ("needs_ocr", None, 0)
    assert row["quality"]["detector"]["reason"] == "no_text_layer"
    ocr_jobs = [job for job in jobs_of(pool, document_id) if job["kind"] == "ocr_page"]
    assert [(job["page_no"], job["state"]) for job in ocr_jobs] == [(1, "queued")]
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.pages_pending) == ("processing", 1)


def test_all_blank_document_fails_with_no_text(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A document whose pages are all blank is `failed` with `no_text`, not `ready`.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, text_pdf(["", ""]), "blank.pdf")
    drain(worker_deps)
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.error_code, doc.page_count) == ("failed", "no_text", 2)


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (encrypted_pdf(), "pdf_encrypted"),
        (garbage_pdf(), "pdf_corrupt"),
        (truncated_pdf(1024), "pdf_corrupt"),
    ],
    ids=["encrypted", "garbage", "truncated-1k"],
)
def test_bad_pdf_fails_with_its_code(worker_deps: WorkerDeps, blobs: LocalBlobStore, data: bytes, code: str) -> None:
    """A file the extractor can't read fails at once with its code; the job is not retried.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        data: The bad file.
        code: Expected document error code.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, data, "bad.pdf")
    assert run_one(worker_deps, "extract")
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.error_code) == ("failed", code)
    assert doc.error_message
    [job] = jobs_of(pool, document_id)
    assert (job["state"], job["attempts"], job["last_error_code"]) == ("failed", 1, code)


def test_truncated_half_reaches_a_final_state(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A file cut in half ends in a final state, and a failure carries a code; PyMuPDF decides which.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, truncated_pdf(0.5), "half.pdf")
    drain(worker_deps)
    doc = document(pool, tenant_id, document_id)
    assert doc.status in ("ready", "partial", "failed")
    assert doc.status != "failed" or doc.error_code


def test_owner_password_only_is_read(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A PDF with only an owner password (permissions) opens without a password, so it is extracted normally.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, encrypted_pdf(user_password=""), "owner.pdf")
    drain(worker_deps)
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.page_count, doc.pages_indexed) == ("ready", 2, 2)


def test_page_limit(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A PDF over `MAX_PDF_PAGES` fails with `pdf_too_many_pages`.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    deps = replace(worker_deps, settings=replace(worker_deps.settings, max_pdf_pages=1))
    tenant_id = new_tenant(deps.pool)
    document_id = store_pdf(deps.pool, blobs, tenant_id, case_pdf(LB5001), "LB5001.pdf")
    assert run_one(deps, "extract")
    assert document(deps.pool, tenant_id, document_id).error_code == "pdf_too_many_pages"


def test_hung_extraction_times_out(
    worker_deps: WorkerDeps, blobs: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that hangs is killed at the time limit: `extract_timeout`, and the queue moves on.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        monkeypatch: Swaps the child for one that hangs.
    """

    def hanging_child(module: str, args: list[str], **limits: float) -> SandboxResult:
        """Run a fake child that writes one line and hangs, with the handler's limits.

        Args:
            module: Ignored.
            args: The handler's arguments; the output path is used.
            **limits: The handler's limits.

        Returns:
            The sandbox result.
        """
        return run_sandboxed("tests.sandbox_children", ["stall", args[1]], **limits)

    monkeypatch.setattr(extract_jobs, "run_sandboxed", hanging_child)
    deps = replace(worker_deps, settings=replace(worker_deps.settings, extract_timeout_s=1))
    tenant_id = new_tenant(deps.pool)
    first = store_pdf(deps.pool, blobs, tenant_id, case_pdf(LB5001), "hangs.pdf")
    assert run_one(deps, "extract")
    doc = document(deps.pool, tenant_id, first)
    assert (doc.status, doc.error_code) == ("failed", "extract_timeout")


def _killed(returncode: int = -9) -> SandboxResult:
    """Build the result of a child killed by a signal the parent didn't send.

    Args:
        returncode: Negated signal number.

    Returns:
        The result.
    """
    return SandboxResult(
        returncode=returncode,
        timed_out=False,
        stalled=False,
        memory_exceeded=False,
        stderr_tail="",
        wall_s=0.1,
        peak_rss_mb=10.0,
    )


def test_child_killed_by_signal_is_retried_then_fails(
    worker_deps: WorkerDeps, blobs: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child killed by the kernel is retried within the attempts; the last attempt fails with `extract_crashed`.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        monkeypatch: Makes every child die from a signal.
    """
    monkeypatch.setattr(extract_jobs, "run_sandboxed", lambda module, args, **limits: _killed())
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, case_pdf(LB5001), "LB5001.pdf")
    for attempt in (1, 2):
        assert run_one(worker_deps, "extract")
        [job] = jobs_of(pool, document_id)
        assert (job["state"], job["attempts"], job["last_error_code"]) == ("queued", attempt, "extract_crashed")
        assert document(pool, tenant_id, document_id).status == "queued"
        make_runnable(pool, job["id"])
    assert run_one(worker_deps, "extract")
    [job] = jobs_of(pool, document_id)
    assert (job["state"], job["attempts"]) == ("failed", 3)
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.error_code) == ("failed", "extract_crashed")


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (replace(_killed(), returncode=-9, memory_exceeded=True), "extract_memory"),
        (replace(_killed(), returncode=3), "extract_memory"),
        (replace(_killed(), returncode=1, stderr_tail="Traceback\nValueError: boom\n"), "extract_crashed"),
        (replace(_killed(), returncode=-9, stalled=True), "extract_timeout"),
    ],
    ids=["memory-limit", "memory-error", "exception", "stalled"],
)
def test_child_failures_are_permanent(
    worker_deps: WorkerDeps,
    blobs: LocalBlobStore,
    monkeypatch: pytest.MonkeyPatch,
    result: SandboxResult,
    code: str,
) -> None:
    """Limits, memory errors and exceptions fail the document at once: the same file would fail the same way.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        monkeypatch: Fakes the child's outcome.
        result: The faked outcome.
        code: Expected error code.
    """
    monkeypatch.setattr(extract_jobs, "run_sandboxed", lambda module, args, **limits: result)
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, case_pdf(LB5001), "LB5001.pdf")
    assert run_one(worker_deps, "extract")
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.error_code) == ("failed", code)
    assert jobs_of(pool, document_id)[0]["attempts"] == 1


def test_extract_runs_twice_without_duplicating_pages(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A second `extract` of an extracted document only re-enqueues `index`; delivery can be at least once.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = store_pdf(pool, blobs, tenant_id, case_pdf(LB5001), "LB5001.pdf")
    drain(worker_deps)
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO jobs (tenant_id, kind, document_id, max_attempts) VALUES (%s, 'extract', %s, 3)",
            (tenant_id, document_id),
        )
    drain(worker_deps)
    assert len(page_rows(pool, document_id)) == 2
    assert document(pool, tenant_id, document_id).status == "ready"


def test_page_labels_are_stored(worker_deps: WorkerDeps, blobs: LocalBlobStore, tmp_path: Path) -> None:
    """Confirmed printed numbers land in `page_label`; unconfirmed ones stay null.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        tmp_path: Temp folder.
    """
    pool: ConnectionPool = worker_deps.pool
    tenant_id = new_tenant(pool)
    doc = pymupdf.open(stream=text_pdf(["", "", ""]), filetype="pdf")
    for number, page in enumerate(doc, start=7):
        page.insert_text((300, page.rect.height - 20), str(number), fontsize=9)
        page.insert_text((72, 200), "Maintenance schedule for the gearbox and the motor bearings.", fontsize=11)
    path = tmp_path / "numbered.pdf"
    doc.save(path)
    document_id = store_pdf(pool, blobs, tenant_id, path.read_bytes(), "numbered.pdf")
    drain(worker_deps)
    assert [row["page_label"] for row in page_rows(pool, document_id)] == ["7", "8", "9"]
