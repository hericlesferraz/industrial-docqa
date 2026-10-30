"""The job queue: claims, leases, fencing, retries, deduplication and the OCR cap."""

import threading
from datetime import UTC, datetime, timedelta

import pytest
from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.db import tx
from docqa.queue import Job, LeaseLost
from tests.helpers import expire_lease, job_row, new_document, new_tenant


def _enqueue(pool: ConnectionPool, kind: str = "noop", **kwargs: object) -> int | None:
    """Enqueue a job in its own transaction.

    Args:
        pool: Database pool.
        kind: Job kind.
        **kwargs: `tenant_id`, `document_id`, `page_no`, `run_after`.

    Returns:
        The job id, or None when deduplicated.
    """
    with tx(pool) as conn:
        return queue.enqueue(conn, kind=kind, **kwargs)


def _claim(pool: ConnectionPool, kind: str = "noop", owner: str = "t", **kwargs: object) -> Job | None:
    """Claim a job on a fresh connection.

    Args:
        pool: Database pool.
        kind: Job kind.
        owner: Lease owner.
        **kwargs: Extra arguments of `queue.claim`.

    Returns:
        The job, or None.
    """
    with pool.connection() as conn:
        return queue.claim(conn, kind=kind, owner=owner, **kwargs)


def test_backoff_doubles_and_caps() -> None:
    """The retry wait doubles from 10 s and stops at 300 s."""
    assert [queue.backoff_s(n) for n in (1, 2, 3, 4, 5, 6, 10)] == [10, 20, 40, 80, 160, 300, 300]


def test_unknown_kind_is_rejected(db: ConnectionPool) -> None:
    """Enqueueing an unknown kind raises.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    with pytest.raises(ValueError, match="unknown job kind"):
        _enqueue(db, "mystery", tenant_id=tenant_id)


def test_concurrent_claims_get_distinct_jobs(db: ConnectionPool) -> None:
    """Many threads claiming at once never get the same job, and every job is claimed once.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    ids = {_enqueue(db, tenant_id=tenant_id) for _ in range(20)}
    claimed: list[int] = []
    lock = threading.Lock()
    start = threading.Barrier(5)

    def claimer() -> None:
        """Claim until the queue is empty."""
        start.wait()
        while (job := _claim(db, owner=threading.current_thread().name)) is not None:
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=claimer) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(claimed) == sorted(ids)
    assert len(set(claimed)) == len(claimed)


def test_claim_sets_lease_and_respects_run_after_and_fifo(db: ConnectionPool) -> None:
    """A claim takes the oldest runnable job, sets the lease and counts the attempt; future jobs wait.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    later = _enqueue(db, tenant_id=tenant_id, run_after=datetime.now(UTC) + timedelta(hours=1))
    first = _enqueue(db, tenant_id=tenant_id)
    second = _enqueue(db, tenant_id=tenant_id)
    job = _claim(db, owner="host:1:main")
    assert job is not None
    assert job.id == first
    assert job.attempts == 1
    assert job.lease_token is not None
    row = job_row(db, first)
    assert row["state"] == "running"
    assert row["lease_owner"] == "host:1:main"
    assert row["lease_expires_at"] > datetime.now(UTC) + timedelta(seconds=50)
    next_job = _claim(db)
    assert next_job is not None
    assert next_job.id == second
    assert _claim(db) is None
    assert job_row(db, later)["state"] == "queued"


def test_expired_lease_is_requeued_and_fencing_rejects_old_owner(db: ConnectionPool) -> None:
    """An expired lease returns the job; the new claim has a new token; the old owner's writes are rejected.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    job_id = _enqueue(db, tenant_id=tenant_id)
    old = _claim(db, owner="old")
    assert old is not None
    with db.connection() as conn:
        assert queue.requeue_expired(conn) == 0
    expire_lease(db, job_id)
    with db.connection() as conn:
        assert queue.requeue_expired(conn) == 1
    row = job_row(db, job_id)
    assert row["state"] == "queued"
    assert row["last_error_code"] == "lease_expired"
    new = _claim(db, owner="new")
    assert new is not None
    assert new.id == job_id
    assert new.attempts == 2
    assert new.lease_token != old.lease_token
    with pytest.raises(LeaseLost), tx(db) as conn:
        queue.assert_lease(conn, old)
    with pytest.raises(LeaseLost), tx(db) as conn:
        queue.complete(conn, old)
    with pytest.raises(LeaseLost), tx(db) as conn:
        queue.retry(conn, old, code="transient", message="x")
    with db.connection() as conn:
        assert queue.heartbeat(conn, old) is False
    with tx(db) as conn:
        queue.assert_lease(conn, new)
        queue.complete(conn, new)
    row = job_row(db, job_id)
    assert row["state"] == "done"
    assert row["finished_at"] is not None
    assert row["lease_token"] is None


def test_heartbeat_extends_the_lease(db: ConnectionPool) -> None:
    """A heartbeat moves the lease end forward.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    job_id = _enqueue(db, tenant_id=tenant_id)
    job = _claim(db, lease_s=5)
    assert job is not None
    before = job_row(db, job_id)["lease_expires_at"]
    with db.connection() as conn:
        assert queue.heartbeat(conn, job, lease_s=600) is True
    assert job_row(db, job_id)["lease_expires_at"] > before + timedelta(seconds=500)


def test_index_and_ocr_jobs_are_deduplicated(db: ConnectionPool) -> None:
    """At most one queued `index` per document and one live `ocr_page` per page.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_id = new_document(db, tenant_id)
    assert _enqueue(db, "index", tenant_id=tenant_id, document_id=document_id) is not None
    assert _enqueue(db, "index", tenant_id=tenant_id, document_id=document_id) is None
    assert _enqueue(db, "ocr_page", tenant_id=tenant_id, document_id=document_id, page_no=3) is not None
    assert _enqueue(db, "ocr_page", tenant_id=tenant_id, document_id=document_id, page_no=3) is None
    assert _enqueue(db, "ocr_page", tenant_id=tenant_id, document_id=document_id, page_no=4) is not None
    with db.connection() as conn:
        counts = conn.execute("SELECT kind, count(*) AS n FROM jobs GROUP BY kind ORDER BY kind").fetchall()
    assert [(row["kind"], row["n"]) for row in counts] == [("index", 1), ("ocr_page", 2)]


def test_one_running_index_per_document(db: ConnectionPool) -> None:
    """While an `index` of a document runs, a second one of the same document can't be claimed; others can.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_a = new_document(db, tenant_id, "a" * 64)
    document_b = new_document(db, tenant_id, "b" * 64)
    _enqueue(db, "index", tenant_id=tenant_id, document_id=document_a)
    running = _claim(db, "index")
    assert running is not None
    second_a = _enqueue(db, "index", tenant_id=tenant_id, document_id=document_a)
    assert second_a is not None
    first_b = _enqueue(db, "index", tenant_id=tenant_id, document_id=document_b)
    other = _claim(db, "index")
    assert other is not None
    assert other.id == first_b
    assert _claim(db, "index") is None
    with tx(db) as conn:
        queue.complete(conn, running)
    after = _claim(db, "index")
    assert after is not None
    assert after.id == second_a


def test_index_retry_is_superseded_by_a_queued_index(db: ConnectionPool) -> None:
    """Retrying an `index` while another `index` of the document is queued closes it as `superseded`.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_id = new_document(db, tenant_id)
    first = _enqueue(db, "index", tenant_id=tenant_id, document_id=document_id)
    job = _claim(db, "index")
    assert job is not None
    queued = _enqueue(db, "index", tenant_id=tenant_id, document_id=document_id)
    with tx(db) as conn:
        assert queue.retry(conn, job, code="internal_error", message="ValueError") is False
    row = job_row(db, first)
    assert row["state"] == "done"
    assert row["last_error_code"] == "superseded"
    assert job_row(db, queued)["state"] == "queued"


def test_expired_index_is_superseded_by_a_queued_index(db: ConnectionPool) -> None:
    """An expired `index` whose document already has a queued `index` is closed instead of requeued.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_id = new_document(db, tenant_id)
    first = _enqueue(db, "index", tenant_id=tenant_id, document_id=document_id)
    assert _claim(db, "index") is not None
    queued = _enqueue(db, "index", tenant_id=tenant_id, document_id=document_id)
    expire_lease(db, first)
    with db.connection() as conn:
        assert queue.requeue_expired(conn) == 1
    assert job_row(db, first)["last_error_code"] == "superseded"
    assert job_row(db, first)["state"] == "done"
    assert job_row(db, queued)["state"] == "queued"


def test_retry_sets_backoff_and_reschedule_gives_the_attempt_back(db: ConnectionPool) -> None:
    """A retry waits `backoff_s(attempts)` and keeps the attempt; a reschedule returns it.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    retried = _enqueue(db, tenant_id=tenant_id)
    rescheduled = _enqueue(db, tenant_id=tenant_id)
    first = _claim(db)
    second = _claim(db)
    assert first is not None
    assert second is not None
    with tx(db) as conn:
        assert queue.retry(conn, first, code="transient", message="503") is True
    row = job_row(db, retried)
    assert row["state"] == "queued"
    assert row["attempts"] == 1
    assert row["last_error_code"] == "transient"
    assert row["last_error"] == "503"
    assert timedelta(seconds=8) < row["run_after"] - datetime.now(UTC) <= timedelta(seconds=10)
    at = datetime.now(UTC) + timedelta(minutes=5)
    with tx(db) as conn:
        assert queue.reschedule(conn, second, run_after=at, code="rate_limited") is True
    row = job_row(db, rescheduled)
    assert row["state"] == "queued"
    assert row["attempts"] == 0
    assert row["last_error_code"] == "rate_limited"
    assert abs(row["run_after"] - at) < timedelta(seconds=1)
    assert _claim(db) is None


def test_fail_runs_the_hook_in_the_same_transaction(db: ConnectionPool) -> None:
    """`fail` marks the job failed and calls `on_failure` with the code, inside the same transaction.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    job_id = _enqueue(db, tenant_id=tenant_id)
    job = _claim(db)
    assert job is not None
    seen: list[tuple[int, str, str, bool]] = []

    def hook(conn: object, failed: Job, code: str, message: str) -> None:
        """Record the call and whether it runs inside a transaction.

        Args:
            conn: Connection of the `fail` transaction.
            failed: The failing job.
            code: Error code.
            message: Detail.
        """
        in_tx = getattr(conn, "info", None) is not None and conn.info.transaction_status != 0
        seen.append((failed.id, code, message, in_tx))

    with tx(db) as conn:
        queue.fail(conn, job, code="pdf_corrupt", message="cannot open", on_failure=hook)
    assert seen == [(job_id, "pdf_corrupt", "cannot open", True)]
    row = job_row(db, job_id)
    assert row["state"] == "failed"
    assert row["last_error_code"] == "pdf_corrupt"


def test_ocr_claims_respect_the_global_cap_and_pauses(db: ConnectionPool) -> None:
    """No more than `ocr_cap` `ocr_page` jobs run at once, and a pause of the model or `'*'` blocks claims.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_id = new_document(db, tenant_id)
    for page in (1, 2, 3):
        _enqueue(db, "ocr_page", tenant_id=tenant_id, document_id=document_id, page_no=page)
    cap = {"ocr_cap": 2, "ocr_model": "flash-lite"}
    first = _claim(db, "ocr_page", **cap)
    assert first is not None
    assert _claim(db, "ocr_page", **cap) is not None
    assert _claim(db, "ocr_page", **cap) is None
    with tx(db) as conn:
        queue.complete(conn, first)
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO provider_state (model, paused_until, pause_reason) VALUES ('*', now() + interval '1 hour',"
            " 'billing')"
        )
    assert _claim(db, "ocr_page", **cap) is None
    with db.connection() as conn:
        conn.execute("UPDATE provider_state SET model = 'other-model'")
    assert _claim(db, "ocr_page", **cap) is not None
