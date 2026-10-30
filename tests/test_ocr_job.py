"""The `ocr_page` job end to end with a fake provider: transcription, cache, failures, retries, pauses."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.ingest import jobs as ingest_jobs
from docqa.ingest.jobs import OcrPageHandler, page_error_code
from docqa.ingest.sandbox import SandboxResult
from docqa.providers.fake import FakeLLM, result
from docqa.providers.llm import ProviderError
from docqa.storage import LocalBlobStore
from docqa.worker import JobContext, WorkerDeps, run_one
from tests.helpers import expire_lease, make_runnable, new_tenant
from tests.ingest_helpers import document, drain, jobs_of, page_rows, store_pdf
from tests.pdfs import mixed_pdf, rasterized_pdf

LITE = "gemini-3.5-flash-lite"
FLASH = "gemini-3.8-flash"
ALL_KINDS = ("extract", "ocr_page", "index")
TEXT = "```markdown\n# Lubrication\n\n| Speed | Hours |\n|---|---|\n| 3600 | 5500Hrs. |\n```"


def _with(deps: WorkerDeps, fake: FakeLLM) -> WorkerDeps:
    """Give the worker a fake provider.

    Args:
        deps: Worker dependencies.
        fake: Fake provider.

    Returns:
        The dependencies with the fake as `llm`.
    """
    return replace(deps, llm=fake)


def _extracted(deps: WorkerDeps, blobs: LocalBlobStore, tenant_id: UUID, data: bytes, name: str) -> UUID:
    """Store a PDF and run its extraction, leaving its `ocr_page` jobs queued.

    Args:
        deps: Worker dependencies.
        blobs: Blob store.
        tenant_id: Owner.
        data: PDF bytes.
        name: Filename.

    Returns:
        The document id.
    """
    document_id = store_pdf(deps.pool, blobs, tenant_id, data, name)
    assert run_one(deps, "extract")
    return document_id


def _ocr_job(pool: ConnectionPool, document_id: UUID) -> dict[str, Any]:
    """Read the document's single `ocr_page` job.

    Args:
        pool: Database pool.
        document_id: Document id.

    Returns:
        The job row.
    """
    [job] = [job for job in jobs_of(pool, document_id) if job["kind"] == "ocr_page"]
    return job


def _cache_rows(pool: ConnectionPool) -> list[dict[str, Any]]:
    """Read the OCR cache.

    Args:
        pool: Database pool.

    Returns:
        Every cache row.
    """
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM ocr_cache ORDER BY sha256, page_no").fetchall()


def test_transcription_makes_the_page_searchable(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """`STOP` writes the canonical text, the cache entry and the evidence; the index takes the document to `ready`.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([result(TEXT)])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, mixed_pdf(), "mixed.pdf")
    assert [row["state"] for row in page_rows(deps.pool, document_id)] == ["text_ready", "needs_ocr"]
    assert run_one(deps, "ocr_page")
    page = page_rows(deps.pool, document_id)[1]
    assert (page["state"], page["text_source"], page["text_rev"]) == ("text_ready", "ocr", 1)
    assert page["ocr_text"] == "# Lubrication\n\n| Speed | Hours |\n|---|---|\n| 3600 | 5500Hrs. |"
    assert page["ocr_version"] == f"ocr-g2-nolatex:150dpi:high:{LITE}>{FLASH}"
    assert page["quality"]["ocr"] == {
        "model": LITE,
        "from_cache": False,
        "recitations": 0,
        "fallback_used": False,
        "finish_reasons": ["STOP"],
    }
    assert page["quality"]["detector"]["reason"] == "no_text_layer"
    [cached] = _cache_rows(deps.pool)
    assert (cached["page_no"], cached["model"], cached["text"]) == (2, LITE, page["ocr_text"])
    assert [(job["kind"], job["state"]) for job in jobs_of(deps.pool, document_id)] == [
        ("extract", "done"),
        ("ocr_page", "done"),
        ("index", "queued"),
    ]
    drain(deps, ALL_KINDS)
    doc = document(deps.pool, tenant_id, document_id)
    assert (doc.status, doc.pages_indexed, doc.pages_pending) == ("ready", 2, 0)
    assert len(fake.calls) == 1


def test_another_tenant_gets_the_cached_text_without_a_call(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """The same bytes in another tenant copy the transcription from the cache: zero provider calls.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    first = _with(worker_deps, FakeLLM([result(TEXT)]))
    data = rasterized_pdf()
    tenant_a = new_tenant(first.pool, "a")
    doc_a = store_pdf(first.pool, blobs, tenant_a, data, "scan.pdf")
    drain(first, ALL_KINDS)
    silent = FakeLLM([])
    second = _with(worker_deps, silent)
    tenant_b = new_tenant(second.pool, "b")
    doc_b = store_pdf(second.pool, blobs, tenant_b, data, "other-name.pdf")
    drain(second, ALL_KINDS)
    assert silent.calls == []
    page_a, page_b = page_rows(first.pool, doc_a)[0], page_rows(second.pool, doc_b)[0]
    assert page_b["ocr_text"] == page_a["ocr_text"]
    assert page_b["quality"]["ocr"] == {"model": LITE, "from_cache": True}
    assert document(second.pool, tenant_b, doc_b).status == "ready"
    assert len(_cache_rows(second.pool)) == 1


def test_fallback_failure_fails_the_page_and_the_document_is_partial(
    worker_deps: WorkerDeps, blobs: LocalBlobStore
) -> None:
    """Cut on both models: the page fails with its code, the counters change in the same transaction, nothing is cached.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([result("half", "MAX_TOKENS"), result("half", "MAX_TOKENS")])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, mixed_pdf(), "mixed.pdf")
    assert run_one(deps, "ocr_page")
    page = page_rows(deps.pool, document_id)[1]
    assert (page["state"], page["error_code"]) == ("failed", "ocr_max_tokens")
    assert page["error_message"] == f"{LITE}: MAX_TOKENS; {FLASH}: MAX_TOKENS"
    doc = document(deps.pool, tenant_id, document_id)
    assert (doc.pages_failed, doc.pages_pending) == (1, 1)
    assert _ocr_job(deps.pool, document_id)["state"] == "failed"
    assert _cache_rows(deps.pool) == []
    drain(deps, ALL_KINDS)
    doc = document(deps.pool, tenant_id, document_id)
    assert (doc.status, doc.pages_indexed, doc.pages_failed) == ("partial", 1, 1)


def test_transient_errors_retry_and_the_last_attempt_uses_the_fallback(
    worker_deps: WorkerDeps, blobs: LocalBlobStore
) -> None:
    """503s spend attempts with backoff; attempt 4 goes straight to the fallback; its 503 fails the page at once.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM(lambda call: ProviderError("transient", "503 UNAVAILABLE: high demand", status=503))
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    for attempt in (1, 2, 3):
        assert run_one(deps, "ocr_page")
        job = _ocr_job(deps.pool, document_id)
        assert (job["state"], job["attempts"], job["last_error_code"]) == ("queued", attempt, "transient")
        make_runnable(deps.pool, job["id"])
    assert run_one(deps, "ocr_page")
    assert [call.model for call in fake.calls] == [LITE, LITE, LITE, FLASH]
    job = _ocr_job(deps.pool, document_id)
    assert (job["state"], job["attempts"], job["last_error_code"]) == ("failed", 4, "ocr_unavailable")
    page = page_rows(deps.pool, document_id)[0]
    assert (page["state"], page["error_code"]) == ("failed", "ocr_unavailable")
    assert _cache_rows(deps.pool) == []
    drain(deps, ALL_KINDS)
    doc = document(deps.pool, tenant_id, document_id)
    assert (doc.status, doc.error_code) == ("failed", "all_pages_failed")


def test_per_minute_rate_limit_reschedules_without_spending_an_attempt(
    worker_deps: WorkerDeps, blobs: LocalBlobStore
) -> None:
    """A per-minute 429 puts the job back after the asked wait and holds the model's next slot.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([ProviderError("rate_limited", "429", status=429, retry_after_s=20)])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    assert run_one(deps, "ocr_page")
    job = _ocr_job(deps.pool, document_id)
    assert (job["state"], job["attempts"], job["last_error_code"]) == ("queued", 0, "rate_limited")
    assert 18 < (job["run_after"] - datetime.now(UTC)).total_seconds() <= 20
    with deps.pool.connection() as conn:
        state = conn.execute(
            "SELECT next_call_at, paused_until FROM provider_state WHERE model = %s", (LITE,)
        ).fetchone()
    assert state is not None and state["paused_until"] is None
    assert (state["next_call_at"] - datetime.now(UTC)).total_seconds() > 18
    assert page_rows(deps.pool, document_id)[0]["state"] == "needs_ocr"


def test_daily_quota_pauses_the_model_without_spending_an_attempt(
    worker_deps: WorkerDeps, blobs: LocalBlobStore
) -> None:
    """A daily quota pauses the model until the pause ends; no attempt is spent and no OCR job is claimed meanwhile.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([ProviderError("quota_exhausted", "429 per day", status=429)])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    assert run_one(deps, "ocr_page")
    job = _ocr_job(deps.pool, document_id)
    assert (job["state"], job["attempts"], job["last_error_code"]) == ("queued", 0, "quota_exhausted")
    with deps.pool.connection() as conn:
        rows = conn.execute(
            "SELECT model, pause_reason, paused_until FROM provider_state WHERE paused_until > now()"
        ).fetchall()
    assert [(row["model"], row["pause_reason"]) for row in rows] == [(LITE, "quota_exhausted")]
    assert abs((job["run_after"] - rows[0]["paused_until"]).total_seconds()) < 1
    make_runnable(deps.pool, job["id"])
    assert not run_one(deps, "ocr_page")
    assert page_rows(deps.pool, document_id)[0]["state"] == "needs_ocr"


def test_billing_pauses_the_whole_account(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A 402 pauses every model: no OCR job is claimed until the pause ends.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([ProviderError("billing", "402", status=402, account_wide=True)])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    assert run_one(deps, "ocr_page")
    with deps.pool.connection() as conn:
        rows = conn.execute("SELECT model, pause_reason FROM provider_state WHERE paused_until > now()").fetchall()
    assert [(row["model"], row["pause_reason"]) for row in rows] == [("*", "billing")]
    make_runnable(deps.pool, _ocr_job(deps.pool, document_id)["id"])
    assert not run_one(deps, "ocr_page")


def test_invalid_request_fails_the_page(worker_deps: WorkerDeps, blobs: LocalBlobStore) -> None:
    """A malformed request (for example an image over the size limit) fails the page without pausing the model.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([ProviderError("invalid_request", "400 INVALID_ARGUMENT: payload too large", status=400)])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    assert run_one(deps, "ocr_page")
    page = page_rows(deps.pool, document_id)[0]
    assert (page["state"], page["error_code"]) == ("failed", "ocr_invalid_request")
    with deps.pool.connection() as conn:
        assert conn.execute("SELECT count(*) AS n FROM provider_state WHERE paused_until IS NOT NULL").fetchone() == {
            "n": 0
        }


def test_render_failure_is_permanent(
    worker_deps: WorkerDeps, blobs: LocalBlobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page that can't be drawn fails with `render_failed`, without any model call.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
        monkeypatch: Makes the render child fail.
    """
    real = ingest_jobs.run_sandboxed

    def fake_run(module: str, args: list[str], **limits: object) -> SandboxResult:
        """Run the extraction for real and fail the render.

        Args:
            module: Child module.
            args: Child arguments.
            **limits: Sandbox limits.

        Returns:
            The child's outcome.
        """
        if module == ingest_jobs.RENDER_MODULE:
            return SandboxResult(1, False, False, False, "Traceback\nValueError: broken page\n", 0.1, 50.0)
        return real(module, args, **limits)

    monkeypatch.setattr(ingest_jobs, "run_sandboxed", fake_run)
    fake = FakeLLM([])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    assert run_one(deps, "ocr_page")
    page = page_rows(deps.pool, document_id)[0]
    assert (page["state"], page["error_code"]) == ("failed", "render_failed")
    assert "ValueError: broken page" in page["error_message"]
    assert fake.calls == []


def test_a_stale_lease_cannot_write_the_page_but_keeps_the_paid_text(
    worker_deps: WorkerDeps, blobs: LocalBlobStore
) -> None:
    """A worker whose lease expired loses the page write; the transcription it paid for stays in the cache.

    Args:
        worker_deps: Worker dependencies.
        blobs: Blob store.
    """
    fake = FakeLLM([result(TEXT)])
    deps = _with(worker_deps, fake)
    tenant_id = new_tenant(deps.pool)
    document_id = _extracted(deps, blobs, tenant_id, rasterized_pdf(), "scan.pdf")
    with deps.pool.connection() as conn:
        old = queue.claim(conn, kind="ocr_page", owner="old", ocr_cap=3, ocr_model=LITE)
    assert old is not None
    expire_lease(deps.pool, old.id)
    with deps.pool.connection() as conn:
        queue.requeue_expired(conn)
        new = queue.claim(conn, kind="ocr_page", owner="new", ocr_cap=3, ocr_model=LITE)
    assert new is not None and new.lease_token != old.lease_token
    context = JobContext(
        job=old,
        pool=deps.pool,
        settings=deps.settings,
        telemetry=deps.telemetry,
        blobs=deps.blobs,
        llm=fake,
        embedder=None,
    )
    with pytest.raises(queue.LeaseLost):
        OcrPageHandler().run(context)
    assert page_rows(deps.pool, document_id)[0]["state"] == "needs_ocr"
    assert len(_cache_rows(deps.pool)) == 1
    OcrPageHandler().run(replace(context, job=new))
    page = page_rows(deps.pool, document_id)[0]
    assert (page["state"], page["text_rev"], page["quality"]["ocr"]["from_cache"]) == ("text_ready", 1, True)
    assert len(fake.calls) == 1


def test_failed_job_codes_map_to_page_codes() -> None:
    """Page codes pass through; transient provider codes become `ocr_unavailable`; anything else is exhausted."""
    assert page_error_code("ocr_recitation") == "ocr_recitation"
    assert page_error_code("lease_expired") == "lease_expired"
    assert page_error_code("transient") == "ocr_unavailable"
    assert page_error_code("timeout") == "ocr_unavailable"
    assert page_error_code("internal_error") == "attempts_exhausted"
