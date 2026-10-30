"""The job queue: a Postgres table with claims, leases, heartbeats and fencing tokens.

A claim commits before the work starts and no transaction stays open during the work. Every write of a job's
result starts with the fencing check, so a worker whose lease expired can't overwrite the new owner's work.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from psycopg import Connection

LEASE_S = 60
HEARTBEAT_S = 15
OCR_CLAIM_LOCK = 7310002
MAX_ATTEMPTS: dict[str, int] = {"extract": 3, "ocr_page": 4, "index": 3, "noop": 3}

OnFailure = Callable[[Connection, "Job", str, str], None]


def backoff_s(attempts: int) -> int:
    """Return the wait before a retried job runs again.

    Args:
        attempts: Attempts spent so far, the failed one included.

    Returns:
        Seconds: 10, 20, 40, ... capped at 300.
    """
    return min(10 * 2 ** max(attempts - 1, 0), 300)


class LeaseLost(Exception):
    """The job's lease moved to another owner, or the job is no longer running; the result must be dropped."""


@dataclass(frozen=True)
class Job:
    """A claimed job.

    Attributes:
        id: Job id.
        tenant_id: Tenant the job works for.
        kind: `extract`, `ocr_page`, `index` or `noop`.
        document_id: Document the job works on, if any.
        page_no: Page the job works on, if any.
        attempts: Claims so far, this one included.
        max_attempts: Attempts allowed for the kind.
        lease_token: Fencing token of this claim.
        last_error_code: Code of the previous failure, if any.
    """

    id: int
    tenant_id: UUID
    kind: str
    document_id: UUID | None
    page_no: int | None
    attempts: int
    max_attempts: int
    lease_token: UUID | None
    last_error_code: str | None


def _job(row: dict[str, Any]) -> Job:
    """Build a `Job` from a `jobs` row.

    Args:
        row: Row as a dict.

    Returns:
        The job.
    """
    return Job(
        id=row["id"],
        tenant_id=row["tenant_id"],
        kind=row["kind"],
        document_id=row["document_id"],
        page_no=row["page_no"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        lease_token=row["lease_token"],
        last_error_code=row["last_error_code"],
    )


def enqueue(
    conn: Connection,
    *,
    tenant_id: UUID,
    kind: str,
    document_id: UUID | None = None,
    page_no: int | None = None,
    run_after: datetime | None = None,
) -> int | None:
    """Add a job, inside the caller's transaction, with the state change that triggers it.

    A second queued `index` of a document, or a second queued or running `ocr_page` of a page, is not added.

    Args:
        conn: Connection, normally inside the caller's transaction.
        tenant_id: Tenant the job works for.
        kind: Job kind; must be a key of `MAX_ATTEMPTS`.
        document_id: Document the job works on.
        page_no: Page the job works on.
        run_after: Earliest start; now when None.

    Returns:
        The new job id, or None when the job was deduplicated.

    Raises:
        ValueError: The kind is unknown.
    """
    if kind not in MAX_ATTEMPTS:
        raise ValueError(f"unknown job kind {kind!r}")
    row = conn.execute(
        "INSERT INTO jobs (tenant_id, kind, document_id, page_no, max_attempts, run_after)"
        " VALUES (%s, %s, %s, %s, %s, coalesce(%s, now()))"
        " ON CONFLICT DO NOTHING RETURNING id",
        (tenant_id, kind, document_id, page_no, MAX_ATTEMPTS[kind], run_after),
    ).fetchone()
    return int(row["id"]) if row else None


CLAIM_SQL = """
UPDATE jobs SET state = 'running', attempts = attempts + 1, lease_token = gen_random_uuid(),
       lease_owner = %(owner)s, lease_expires_at = now() + make_interval(secs => %(lease)s), updated_at = now()
WHERE id = (
  SELECT j.id FROM jobs j
  WHERE j.state = 'queued' AND j.kind = %(kind)s AND j.run_after <= now()
    AND NOT (j.kind = 'index' AND EXISTS (SELECT 1 FROM jobs r WHERE r.kind = 'index' AND r.state = 'running'
                                          AND r.document_id = j.document_id))
  ORDER BY j.run_after, j.id
  FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING *
"""


def _ocr_slot_free(conn: Connection, *, ocr_cap: int, ocr_model: str | None) -> bool:
    """Check the global OCR cap and the provider pauses, under a transaction-level advisory lock.

    Args:
        conn: Connection inside the claim transaction.
        ocr_cap: Maximum running `ocr_page` jobs across workers.
        ocr_model: Default OCR model; a pause on it or on `'*'` blocks the claim.

    Returns:
        True when an `ocr_page` job may be claimed now.
    """
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (OCR_CLAIM_LOCK,))
    row = conn.execute(
        "SELECT (SELECT count(*) FROM jobs WHERE kind = 'ocr_page' AND state = 'running') AS running,"
        " EXISTS (SELECT 1 FROM provider_state WHERE model IN (%s, '*') AND paused_until > now()) AS paused",
        (ocr_model or "*",),
    ).fetchone()
    assert row is not None
    return int(row["running"]) < ocr_cap and not row["paused"]


def claim(
    conn: Connection,
    *,
    kind: str,
    owner: str,
    lease_s: int = LEASE_S,
    ocr_cap: int | None = None,
    ocr_model: str | None = None,
) -> Job | None:
    """Claim the oldest runnable job of a kind and commit the claim before returning.

    The queue serves every tenant, so this function is not filtered by tenant. FIFO by `(run_after, id)`.

    Args:
        conn: Connection in autocommit mode, outside any transaction.
        kind: Job kind to claim.
        owner: `"<hostname>:<pid>:<thread>"` of the claiming thread.
        lease_s: Lease length in seconds.
        ocr_cap: Global cap of running `ocr_page` jobs; only for `ocr_page`.
        ocr_model: Default OCR model, whose pause blocks the claim; only for `ocr_page`.

    Returns:
        The claimed job with its new fencing token, or None when nothing is runnable.
    """
    with conn.transaction():
        if ocr_cap is not None and not _ocr_slot_free(conn, ocr_cap=ocr_cap, ocr_model=ocr_model):
            return None
        row = conn.execute(CLAIM_SQL, {"owner": owner, "lease": lease_s, "kind": kind}).fetchone()
    return _job(row) if row else None


def heartbeat(conn: Connection, job: Job, *, lease_s: int = LEASE_S) -> bool:
    """Extend a held lease.

    Args:
        conn: Any connection.
        job: The held job.
        lease_s: New lease length from now, in seconds.

    Returns:
        False when the lease was lost.
    """
    cur = conn.execute(
        "UPDATE jobs SET lease_expires_at = now() + make_interval(secs => %s), updated_at = now()"
        " WHERE id = %s AND lease_token = %s AND state = 'running'",
        (lease_s, job.id, job.lease_token),
    )
    return cur.rowcount == 1


def assert_lease(conn: Connection, job: Job) -> None:
    """Lock the job row and check that this claim still owns it; first statement of every result transaction.

    Args:
        conn: Connection inside the result transaction.
        job: The held job.

    Raises:
        LeaseLost: Another claim owns the job, or it is no longer running.
    """
    row = conn.execute(
        "SELECT 1 FROM jobs WHERE id = %s AND lease_token = %s AND state = 'running' FOR UPDATE",
        (job.id, job.lease_token),
    ).fetchone()
    if row is None:
        raise LeaseLost(f"job {job.id} lease lost")


def _fenced_update(conn: Connection, job: Job, assignments: str, params: tuple[object, ...]) -> None:
    """Update a running job only if the caller still holds its lease.

    Args:
        conn: Connection.
        job: The held job.
        assignments: SQL `SET` list.
        params: Parameters of the `SET` list.

    Raises:
        LeaseLost: The lease was lost.
    """
    cur = conn.execute(
        f"UPDATE jobs SET {assignments}, updated_at = now() WHERE id = %s AND lease_token = %s AND state = 'running'",
        (*params, job.id, job.lease_token),
    )
    if cur.rowcount != 1:
        raise LeaseLost(f"job {job.id} lease lost")


CLEAR_LEASE = "lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL"


def complete(conn: Connection, job: Job) -> None:
    """Mark a job done.

    Args:
        conn: Connection, normally inside the result transaction.
        job: The held job.

    Raises:
        LeaseLost: The lease was lost.
    """
    _fenced_update(conn, job, f"state = 'done', finished_at = now(), {CLEAR_LEASE}", ())


def _requeue(conn: Connection, job: Job, assignments: str, params: tuple[object, ...], message: str) -> bool:
    """Put a held job back in the queue, or close it as `superseded` when it is an `index` already queued again.

    At most one `index` per document may be queued (`jobs_one_queued_index`); the queued one indexes the latest
    page texts, so the retry is dropped instead of violating the index.

    Args:
        conn: Connection.
        job: The held job.
        assignments: SQL `SET` list that requeues the job.
        params: Parameters of the `SET` list.
        message: Detail stored when the job is superseded.

    Returns:
        True when requeued, False when superseded.

    Raises:
        LeaseLost: The lease was lost.
    """
    if job.kind != "index":
        _fenced_update(conn, job, assignments, params)
        return True
    try:
        with conn.transaction():
            _fenced_update(conn, job, assignments, params)
    except psycopg.errors.UniqueViolation:
        _fenced_update(
            conn,
            job,
            f"state = 'done', finished_at = now(), last_error_code = 'superseded', last_error = %s, {CLEAR_LEASE}",
            (message,),
        )
        return False
    return True


def retry(conn: Connection, job: Job, *, code: str, message: str) -> bool:
    """Put a job back in the queue after a transient failure; the attempt stays spent.

    Args:
        conn: Connection.
        job: The held job.
        code: Error code.
        message: Detail without customer text or keys.

    Returns:
        True when requeued; False when an `index` was closed as `superseded` because another one is queued.

    Raises:
        LeaseLost: The lease was lost.
    """
    return _requeue(
        conn,
        job,
        "state = 'queued', run_after = now() + make_interval(secs => %s), last_error_code = %s,"
        f" last_error = %s, {CLEAR_LEASE}",
        (backoff_s(job.attempts), code, message),
        f"{code}: {message}",
    )


def reschedule(conn: Connection, job: Job, *, run_after: datetime, code: str) -> bool:
    """Put a job back in the queue for later without spending the attempt (rate limit, provider pause).

    Args:
        conn: Connection.
        job: The held job.
        run_after: When the job may be claimed again.
        code: Reason, stored in `last_error_code`.

    Returns:
        True when requeued; False when an `index` was closed as `superseded` because another one is queued.

    Raises:
        LeaseLost: The lease was lost.
    """
    return _requeue(
        conn,
        job,
        f"state = 'queued', attempts = attempts - 1, run_after = %s, last_error_code = %s, {CLEAR_LEASE}",
        (run_after, code),
        code,
    )


def fail(conn: Connection, job: Job, *, code: str, message: str, on_failure: OnFailure | None) -> None:
    """Mark a job failed and record the failure on its subject in the same transaction.

    Args:
        conn: Connection.
        job: The held job.
        code: Error code.
        message: Detail without customer text or keys.
        on_failure: The kind's hook that records the failure on the document or page.

    Raises:
        LeaseLost: The lease was lost.
    """
    with conn.transaction():
        _fenced_update(
            conn,
            job,
            f"state = 'failed', finished_at = now(), last_error_code = %s, last_error = %s, {CLEAR_LEASE}",
            (code, message),
        )
        if on_failure is not None:
            on_failure(conn, job, code, message)


REQUEUE_SQL = """
WITH superseded AS (
  UPDATE jobs j SET state = 'done', last_error_code = 'superseded', finished_at = now(), updated_at = now(),
         lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL
  WHERE j.state = 'running' AND j.lease_expires_at < now() AND j.kind = 'index'
    AND EXISTS (SELECT 1 FROM jobs q WHERE q.kind = 'index' AND q.state = 'queued' AND q.document_id = j.document_id)
  RETURNING j.id
), requeued AS (
  UPDATE jobs SET state = 'queued', lease_token = NULL, lease_owner = NULL, lease_expires_at = NULL,
         last_error_code = coalesce(last_error_code, 'lease_expired'), run_after = now(), updated_at = now()
  WHERE state = 'running' AND lease_expires_at < now() AND id NOT IN (SELECT id FROM superseded)
  RETURNING id
)
SELECT (SELECT count(*) FROM superseded) + (SELECT count(*) FROM requeued) AS n
"""


def requeue_expired(conn: Connection) -> int:
    """Return jobs with expired leases to the queue; across every tenant.

    An expired `index` whose document already has a queued `index` is closed as `superseded` instead.

    Args:
        conn: Any connection.

    Returns:
        Number of jobs requeued or superseded.
    """
    row = conn.execute(REQUEUE_SQL).fetchone()
    return int(row["n"]) if row else 0
