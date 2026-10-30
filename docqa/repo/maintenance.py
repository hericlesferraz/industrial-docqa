"""Periodic cleanup and the index-version reconcile, across tenants; not filtered by tenant by design."""

from datetime import timedelta

from psycopg import Connection

from docqa import queue
from docqa.repo import documents

DONE_JOBS_KEEP = timedelta(days=7)
TMP_GRACE = timedelta(hours=1)
HISTORY_GRACE = timedelta(hours=24)


def delete_done_jobs(conn: Connection, *, keep: timedelta = DONE_JOBS_KEEP) -> int:
    """Delete finished jobs older than the retention; failed jobs stay as evidence.

    Args:
        conn: Any connection.
        keep: Retention of `done` jobs.

    Returns:
        Number of jobs deleted.
    """
    cur = conn.execute(
        "DELETE FROM jobs WHERE state = 'done' AND finished_at < now() - make_interval(secs => %s)",
        (keep.total_seconds(),),
    )
    return cur.rowcount


def cleanup_history(conn: Connection, *, grace: timedelta = HISTORY_GRACE) -> int:
    """Delete archived chunks that no answer references, once they are older than the grace period.

    The grace period is longer than any answer takes, so an answer still being generated, which records its chunks
    only at the end, never loses them (DECISIONS #5).

    Args:
        conn: Any connection.
        grace: Minimum age of the archived rows to delete.

    Returns:
        Number of rows deleted.
    """
    cur = conn.execute(
        "DELETE FROM chunks_history h WHERE h.archived_at < now() - make_interval(secs => %s)"
        " AND NOT EXISTS (SELECT 1 FROM answer_candidates a WHERE a.chunk_id = h.id)",
        (grace.total_seconds(),),
    )
    return cur.rowcount


def reconcile_index_version(conn: Connection, *, current: str) -> int:
    """Send back to indexing every page indexed under another index version, in every tenant.

    The pages become `text_ready`, their documents' counters and status are recomputed, and one `index` job is
    enqueued per document. Search only reads chunks of the current version, so the old chunks stop counting at once
    and move to history when their page is reindexed.

    Args:
        conn: Connection inside a transaction.
        current: The deployment's index version.

    Returns:
        Number of pages sent back.
    """
    rows = conn.execute(
        "UPDATE pages SET state = 'text_ready', updated_at = now()"
        " WHERE state = 'indexed' AND index_version IS DISTINCT FROM %s RETURNING tenant_id, document_id",
        (current,),
    ).fetchall()
    affected = sorted({(row["tenant_id"], row["document_id"]) for row in rows}, key=str)
    for tenant_id, document_id in affected:
        documents.recompute_document(conn, tenant_id=tenant_id, document_id=document_id)
        queue.enqueue(conn, tenant_id=tenant_id, kind="index", document_id=document_id)
    return len(rows)
