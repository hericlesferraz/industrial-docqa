"""Upload batches of a tenant: one item per file sent with the batch id, written once and never changed."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg import Connection
from psycopg.rows import class_row

ITEM_OUTCOMES = ("accepted", "duplicate", "rejected")


@dataclass(frozen=True)
class BatchRow:
    """A `batches` row.

    Attributes:
        id: Batch id.
        tenant_id: Owner.
        created_at: Creation time.
    """

    id: UUID
    tenant_id: UUID
    created_at: datetime


@dataclass(frozen=True)
class BatchItemRow:
    """A batch item with the current state of its document.

    Attributes:
        filename: Name given by the client for this upload.
        outcome: `accepted`, `duplicate` or `rejected`.
        error_code: Why the file was rejected; None otherwise.
        created_at: Upload time.
        document_id: The document, None for rejected files.
        status: Document status.
        page_count: Pages in the file; None until extracted.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed.
        pages_failed: Pages that failed.
        document_error_code: Failure code of the document, if any.
        needs_ocr: Pages of the document still waiting for a transcription.
    """

    filename: str
    outcome: str
    error_code: str | None
    created_at: datetime
    document_id: UUID | None
    status: str | None
    page_count: int | None
    pages_indexed: int | None
    pages_pending: int | None
    pages_failed: int | None
    document_error_code: str | None
    needs_ocr: int


@dataclass(frozen=True)
class BatchViewRow:
    """A batch and its items.

    Attributes:
        batch: The batch row.
        items: Its items in upload order.
    """

    batch: BatchRow
    items: list[BatchItemRow]


def create_batch(conn: Connection, *, tenant_id: UUID) -> BatchRow:
    """Create an empty batch.

    Args:
        conn: Connection inside the caller's transaction.
        tenant_id: Owner.

    Returns:
        The new batch.
    """
    with conn.cursor(row_factory=class_row(BatchRow)) as cur:
        row = cur.execute(
            "INSERT INTO batches (tenant_id) VALUES (%s) RETURNING id, tenant_id, created_at", (tenant_id,)
        ).fetchone()
    assert row is not None
    return row


def get_batch(conn: Connection, *, tenant_id: UUID, batch_id: UUID) -> BatchRow | None:
    """Load one batch of a tenant.

    Args:
        conn: Any connection.
        tenant_id: Tenant that must own the batch.
        batch_id: Batch id.

    Returns:
        The batch, or None when it doesn't exist or belongs to another tenant.
    """
    with conn.cursor(row_factory=class_row(BatchRow)) as cur:
        return cur.execute(
            "SELECT id, tenant_id, created_at FROM batches WHERE tenant_id = %s AND id = %s", (tenant_id, batch_id)
        ).fetchone()


def add_item(
    conn: Connection,
    *,
    tenant_id: UUID,
    batch_id: UUID,
    filename: str,
    outcome: str,
    document_id: UUID | None = None,
    error_code: str | None = None,
) -> None:
    """Record the outcome of one file sent with a batch id.

    Args:
        conn: Connection, inside the upload's transaction when a document was created.
        tenant_id: Owner.
        batch_id: Batch the file was sent with.
        filename: Name given by the client.
        outcome: `accepted`, `duplicate` or `rejected`.
        document_id: The document, for accepted and duplicate files.
        error_code: Why the file was rejected.

    Raises:
        ValueError: The outcome is unknown, or the document and error code don't match the outcome.
    """
    if outcome not in ITEM_OUTCOMES:
        raise ValueError(f"unknown batch item outcome {outcome!r}")
    if (outcome == "rejected") != (document_id is None) or (outcome == "rejected") != (error_code is not None):
        raise ValueError("a rejected item has an error code and no document; other items have a document")
    conn.execute(
        "INSERT INTO batch_items (tenant_id, batch_id, filename, outcome, document_id, error_code)"
        " VALUES (%s, %s, %s, %s, %s, %s)",
        (tenant_id, batch_id, filename, outcome, document_id, error_code),
    )


def batch_view(conn: Connection, *, tenant_id: UUID, batch_id: UUID) -> BatchViewRow | None:
    """Load a batch with every item and the current state of each item's document.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        batch_id: Batch id.

    Returns:
        The batch and its items in upload order, or None when the tenant has no such batch.
    """
    batch = get_batch(conn, tenant_id=tenant_id, batch_id=batch_id)
    if batch is None:
        return None
    with conn.cursor(row_factory=class_row(BatchItemRow)) as cur:
        items = cur.execute(
            "SELECT i.filename, i.outcome, i.error_code, i.created_at, i.document_id, d.status, d.page_count,"
            " d.pages_indexed, d.pages_pending, d.pages_failed, d.error_code AS document_error_code,"
            " (SELECT count(*) FROM pages p WHERE p.tenant_id = i.tenant_id AND p.document_id = i.document_id"
            " AND p.state = 'needs_ocr') AS needs_ocr"
            " FROM batch_items i LEFT JOIN documents d ON d.id = i.document_id AND d.tenant_id = i.tenant_id"
            " WHERE i.tenant_id = %s AND i.batch_id = %s ORDER BY i.created_at, i.id",
            (tenant_id, batch_id),
        ).fetchall()
    return BatchViewRow(batch=batch, items=items)
