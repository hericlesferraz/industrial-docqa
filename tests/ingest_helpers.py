"""Helpers of the ingestion tests: store a PDF as an uploaded document, drain the queue, read rows."""

import hashlib
from typing import Any
from uuid import UUID

from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.db import tx
from docqa.repo import documents
from docqa.storage import LocalBlobStore
from docqa.worker import WorkerDeps, run_one


def store_pdf(pool: ConnectionPool, blobs: LocalBlobStore, tenant_id: UUID, data: bytes, filename: str) -> UUID:
    """Store bytes and create their document and `extract` job, as an upload does.

    Args:
        pool: Database pool.
        blobs: Blob store.
        tenant_id: Owner.
        data: File bytes.
        filename: Client filename.

    Returns:
        The document id.
    """
    tmp = blobs.new_temp()
    tmp.write_bytes(data)
    sha256 = hashlib.sha256(data).hexdigest()
    key = blobs.commit_temp(tmp, tenant_id, sha256)
    with tx(pool) as conn:
        row, created = documents.insert_or_get_document(
            conn, tenant_id=tenant_id, sha256=sha256, filename=filename, blob_key=key
        )
        if created:
            queue.enqueue(conn, tenant_id=tenant_id, kind="extract", document_id=row.id)
    return row.id


def drain(deps: WorkerDeps, kinds: tuple[str, ...] = ("extract", "index"), limit: int = 50) -> int:
    """Run queued jobs of the given kinds until none is runnable now.

    Args:
        deps: Worker dependencies.
        kinds: Job kinds to run, in this order on each round.
        limit: Safety cap on the number of jobs.

    Returns:
        Number of jobs run.
    """
    ran = 0
    progress = True
    while progress and ran < limit:
        progress = False
        for kind in kinds:
            while ran < limit and run_one(deps, kind):
                ran += 1
                progress = True
    return ran


def document(pool: ConnectionPool, tenant_id: UUID, document_id: UUID) -> documents.DocumentRow:
    """Read a document.

    Args:
        pool: Database pool.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        The row.
    """
    with pool.connection() as conn:
        row = documents.get_document(conn, tenant_id=tenant_id, document_id=document_id)
    assert row is not None
    return row


def page_rows(pool: ConnectionPool, document_id: UUID) -> list[dict[str, Any]]:
    """Read a document's page rows in page order.

    Args:
        pool: Database pool.
        document_id: Document id.

    Returns:
        The rows as dicts.
    """
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM pages WHERE document_id = %s ORDER BY page_no", (document_id,)).fetchall()


def jobs_of(pool: ConnectionPool, document_id: UUID) -> list[dict[str, Any]]:
    """Read a document's jobs in creation order.

    Args:
        pool: Database pool.
        document_id: Document id.

    Returns:
        The rows as dicts.
    """
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM jobs WHERE document_id = %s ORDER BY id", (document_id,)).fetchall()
