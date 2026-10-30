"""Documents of a tenant: upload rows, extraction outcome, and the status recomputed from the page states."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from psycopg import Connection
from psycopg.rows import class_row

DOCUMENT_COLUMNS = (
    "id, tenant_id, sha256, filename, blob_key, status, error_code, error_message, page_count,"
    " pages_indexed, pages_pending, pages_failed, created_at, updated_at"
)
FINAL_STATUSES = ("ready", "partial", "failed")
DERIVED_ERRORS = {
    "all_pages_failed": "Every page with content failed.",
    "no_text": "No page has text: every page is blank.",
}


@dataclass(frozen=True)
class DocumentRow:
    """A `documents` row.

    Attributes:
        id: Document id.
        tenant_id: Owner.
        sha256: Digest of the stored bytes.
        filename: Name given by the client; metadata only.
        blob_key: Key in the blob store.
        status: `queued`, `processing`, `ready`, `partial` or `failed`.
        error_code: Failure code of the document, if any.
        error_message: Failure detail, if any.
        page_count: Pages in the file; None until extracted.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed.
        pages_failed: Pages that failed.
        created_at: Upload time.
        updated_at: Last change.
    """

    id: UUID
    tenant_id: UUID
    sha256: str
    filename: str
    blob_key: str
    status: str
    error_code: str | None
    error_message: str | None
    page_count: int | None
    pages_indexed: int
    pages_pending: int
    pages_failed: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PageSummaryRow:
    """What the document view shows of one page.

    Attributes:
        page_no: 1-based position in the file.
        state: Page state.
        text_source: `layer`, `ocr` or None.
        page_label: Printed page number, when confirmed.
        flag_reason: Why the detector sent the page to OCR, if it did.
        error_code: Failure code of the page, if any.
        error_message: Failure detail, if any.
    """

    page_no: int
    state: str
    text_source: str | None
    page_label: str | None
    flag_reason: str | None
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True)
class DocumentDetailRow:
    """A document with its pages, for `GET /v1/documents/{id}`.

    Attributes:
        document: The document row.
        pages: Its pages in page order.
    """

    document: DocumentRow
    pages: list[PageSummaryRow]


def get_document(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> DocumentRow | None:
    """Load one document of a tenant.

    Args:
        conn: Any connection.
        tenant_id: Tenant that must own the document.
        document_id: Document id.

    Returns:
        The document, or None when it doesn't exist or belongs to another tenant.
    """
    with conn.cursor(row_factory=class_row(DocumentRow)) as cur:
        return cur.execute(
            f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE tenant_id = %s AND id = %s",
            (tenant_id, document_id),
        ).fetchone()


def insert_or_get_document(
    conn: Connection, *, tenant_id: UUID, sha256: str, filename: str, blob_key: str
) -> tuple[DocumentRow, bool]:
    """Create a document, or return the tenant's existing document with the same bytes.

    The unique constraint on `(tenant_id, sha256)` settles two concurrent uploads of the same file.

    Args:
        conn: Connection inside the caller's transaction.
        tenant_id: Owner.
        sha256: Digest computed while the file streamed in.
        filename: Name given by the client.
        blob_key: Key of the stored file.

    Returns:
        `(row, created)`: the new row and True, or the existing row and False.
    """
    with conn.cursor(row_factory=class_row(DocumentRow)) as cur:
        created = cur.execute(
            "INSERT INTO documents (tenant_id, sha256, filename, blob_key) VALUES (%s, %s, %s, %s)"
            f" ON CONFLICT (tenant_id, sha256) DO NOTHING RETURNING {DOCUMENT_COLUMNS}",
            (tenant_id, sha256, filename, blob_key),
        ).fetchone()
        if created is not None:
            return created, True
        existing = cur.execute(
            f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE tenant_id = %s AND sha256 = %s",
            (tenant_id, sha256),
        ).fetchone()
    assert existing is not None
    return existing, False


def list_documents(conn: Connection, *, tenant_id: UUID, limit: int, offset: int) -> tuple[list[DocumentRow], int]:
    """List a tenant's documents, newest first.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        limit: Page size.
        offset: Rows to skip.

    Returns:
        The rows of the page and the tenant's total number of documents.
    """
    with conn.cursor(row_factory=class_row(DocumentRow)) as cur:
        rows = cur.execute(
            f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE tenant_id = %s"
            " ORDER BY created_at DESC, id LIMIT %s OFFSET %s",
            (tenant_id, limit, offset),
        ).fetchall()
    total = conn.execute("SELECT count(*) AS n FROM documents WHERE tenant_id = %s", (tenant_id,)).fetchone()
    assert total is not None
    return rows, int(total["n"])


def missing_documents(conn: Connection, *, tenant_id: UUID, document_ids: list[UUID]) -> list[UUID]:
    """Find the ids of a scope that are not documents of the tenant.

    Another tenant's document and an unknown id look the same (DECISIONS #14).

    Args:
        conn: Any connection.
        tenant_id: Tenant asking.
        document_ids: The requested scope.

    Returns:
        The ids the tenant doesn't own, in request order.
    """
    rows = conn.execute(
        "SELECT id FROM documents WHERE tenant_id = %s AND id = ANY(%s)", (tenant_id, document_ids)
    ).fetchall()
    found = {row["id"] for row in rows}
    return [document_id for document_id in document_ids if document_id not in found]


@dataclass(frozen=True)
class ScopeTotals:
    """Counts over the documents of a question's scope, read from the document rows (DECISIONS #16).

    Attributes:
        documents: Documents in scope.
        incomplete: Documents that are not `ready`.
        pending_documents: Documents `queued` or `processing`.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed.
        pages_failed: Pages that failed.
    """

    documents: int
    incomplete: int
    pending_documents: int
    pages_indexed: int
    pages_pending: int
    pages_failed: int


def scope_coverage(
    conn: Connection, *, tenant_id: UUID, document_ids: list[UUID] | None, cap: int
) -> tuple[ScopeTotals, list[DocumentRow]]:
    """Count a scope's documents and pages and list the documents that aren't ready, never reading pages.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_ids: The scope, or None for every document of the tenant.
        cap: Most documents to list.

    Returns:
        The totals and up to `cap` documents that aren't `ready`: `queued` and `processing` first, then oldest first.
    """
    scope = "" if document_ids is None else " AND id = ANY(%(documents)s)"
    params = {"tenant": tenant_id, "documents": document_ids, "cap": cap}
    totals = conn.execute(
        "SELECT count(*) AS documents, count(*) FILTER (WHERE status <> 'ready') AS incomplete,"
        " count(*) FILTER (WHERE status IN ('queued', 'processing')) AS pending_documents,"
        " coalesce(sum(pages_indexed), 0) AS pages_indexed, coalesce(sum(pages_pending), 0) AS pages_pending,"
        " coalesce(sum(pages_failed), 0) AS pages_failed"
        f" FROM documents WHERE tenant_id = %(tenant)s{scope}",
        params,
    ).fetchone()
    assert totals is not None
    with conn.cursor(row_factory=class_row(DocumentRow)) as cur:
        rows = cur.execute(
            f"SELECT {DOCUMENT_COLUMNS} FROM documents WHERE tenant_id = %(tenant)s AND status <> 'ready'{scope}"
            " ORDER BY status NOT IN ('queued', 'processing'), created_at, id LIMIT %(cap)s",
            params,
        ).fetchall()
    return ScopeTotals(**{name: int(value) for name, value in totals.items()}), rows


def set_document_failed(conn: Connection, *, tenant_id: UUID, document_id: UUID, code: str, message: str) -> None:
    """Mark a document failed with the code of the extraction failure.

    Args:
        conn: Connection inside the caller's transaction.
        tenant_id: Owner.
        document_id: Document id.
        code: Error code (`pdf_encrypted`, `pdf_corrupt`, `extract_timeout`, ...).
        message: Detail without customer text.
    """
    conn.execute(
        "UPDATE documents SET status = 'failed', error_code = %s, error_message = %s, updated_at = now()"
        " WHERE tenant_id = %s AND id = %s",
        (code, message, tenant_id, document_id),
    )


def set_page_count(conn: Connection, *, tenant_id: UUID, document_id: UUID, page_count: int) -> None:
    """Record the number of pages found by the extraction.

    Args:
        conn: Connection inside the extraction's result transaction.
        tenant_id: Owner.
        document_id: Document id.
        page_count: Pages in the file.
    """
    conn.execute(
        "UPDATE documents SET page_count = %s, updated_at = now() WHERE tenant_id = %s AND id = %s",
        (page_count, tenant_id, document_id),
    )


def document_status(
    *, page_count: int | None, error_code: str | None, indexed: int, pending: int, failed: int
) -> tuple[str, str | None]:
    """Apply the document status rules, first match wins.

    Args:
        page_count: Pages in the file; None until extracted.
        error_code: The document's current error code.
        indexed: Pages in state `indexed`.
        pending: Pages in state `needs_ocr` or `text_ready`.
        failed: Pages in state `failed`.

    Returns:
        `(status, error_code)`. An extraction error is kept; the codes `all_pages_failed` and `no_text` are derived
        from the pages here.
    """
    if error_code is not None and error_code not in DERIVED_ERRORS:
        return "failed", error_code
    if page_count is None:
        return "queued", None
    if pending > 0:
        return "processing", None
    if indexed == 0:
        return "failed", "all_pages_failed" if failed > 0 else "no_text"
    if failed > 0:
        return "partial", None
    return "ready", None


def recompute_document(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> str:
    """Recount a document's pages and write its coverage counters and status, in the caller's transaction.

    The document row is locked first, so two transactions that change pages of the same document recount in turn,
    each seeing the other's committed page states.

    Args:
        conn: Connection inside the transaction that changed the page states.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        The new status; `failed` without changes when the document doesn't exist.
    """
    doc = conn.execute(
        "SELECT page_count, error_code, error_message FROM documents WHERE tenant_id = %s AND id = %s FOR UPDATE",
        (tenant_id, document_id),
    ).fetchone()
    if doc is None:
        return "failed"
    counts = conn.execute(
        "SELECT count(*) FILTER (WHERE state = 'indexed') AS indexed,"
        " count(*) FILTER (WHERE state IN ('needs_ocr', 'text_ready')) AS pending,"
        " count(*) FILTER (WHERE state = 'failed') AS failed"
        " FROM pages WHERE tenant_id = %s AND document_id = %s",
        (tenant_id, document_id),
    ).fetchone()
    assert counts is not None
    status, code = document_status(
        page_count=doc["page_count"],
        error_code=doc["error_code"],
        indexed=counts["indexed"],
        pending=counts["pending"],
        failed=counts["failed"],
    )
    message = DERIVED_ERRORS.get(code, doc["error_message"]) if code else None
    conn.execute(
        "UPDATE documents SET status = %s, error_code = %s, error_message = %s, pages_indexed = %s,"
        " pages_pending = %s, pages_failed = %s, updated_at = now() WHERE tenant_id = %s AND id = %s",
        (status, code, message, counts["indexed"], counts["pending"], counts["failed"], tenant_id, document_id),
    )
    return status


def document_detail(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> DocumentDetailRow | None:
    """Load a document and the summary of each of its pages.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        The document and its pages in page order, or None when the tenant has no such document.
    """
    document = get_document(conn, tenant_id=tenant_id, document_id=document_id)
    if document is None:
        return None
    with conn.cursor(row_factory=class_row(PageSummaryRow)) as cur:
        pages = cur.execute(
            "SELECT page_no, state, text_source, page_label, quality->'detector'->>'reason' AS flag_reason,"
            " error_code, error_message FROM pages WHERE tenant_id = %s AND document_id = %s ORDER BY page_no",
            (tenant_id, document_id),
        ).fetchall()
    return DocumentDetailRow(document=document, pages=pages)


@dataclass(frozen=True)
class OcrModelUsage:
    """OCR calls of a document to one model, summed from its `gen_ai.call` spans.

    Attributes:
        model: Requested model.
        calls: Calls made, failed ones included.
        input_tokens: Prompt tokens, images included.
        output_tokens: Output tokens, thinking included (billed as output).
    """

    model: str
    calls: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class OcrSummaryRow:
    """What the document view shows about OCR.

    Attributes:
        pages_flagged: Pages the detector sent to OCR.
        pages_transcribed: Pages whose text comes from a transcription.
        pages_from_cache: Transcribed pages copied from the cross-tenant cache.
        recitations: `RECITATION` refusals on the transcribed pages.
        fallback_used: Transcribed pages that needed the fallback model.
        needs_ocr: Pages still waiting for a transcription.
        by_model: Calls and tokens per model.
    """

    pages_flagged: int
    pages_transcribed: int
    pages_from_cache: int
    recitations: int
    fallback_used: int
    needs_ocr: int
    by_model: list[OcrModelUsage]


def ocr_summary(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> OcrSummaryRow:
    """Count a document's OCR pages from `pages.quality` and its OCR calls and tokens from the spans.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        The summary; zeros for a document without flagged pages.
    """
    counts = conn.execute(
        "SELECT count(*) FILTER (WHERE quality->'detector'->>'reason' IS NOT NULL) AS pages_flagged,"
        " count(*) FILTER (WHERE text_source = 'ocr') AS pages_transcribed,"
        " count(*) FILTER (WHERE (quality->'ocr'->>'from_cache')::boolean) AS pages_from_cache,"
        " coalesce(sum((quality->'ocr'->>'recitations')::int), 0) AS recitations,"
        " count(*) FILTER (WHERE (quality->'ocr'->>'fallback_used')::boolean) AS fallback_used,"
        " count(*) FILTER (WHERE state = 'needs_ocr') AS needs_ocr"
        " FROM pages WHERE tenant_id = %s AND document_id = %s",
        (tenant_id, document_id),
    ).fetchone()
    assert counts is not None
    with conn.cursor(row_factory=class_row(OcrModelUsage)) as cur:
        usage = cur.execute(
            "SELECT attributes->>'gen_ai.request.model' AS model, count(*) AS calls,"
            " coalesce(sum((attributes->>'gen_ai.usage.input_tokens')::bigint), 0) AS input_tokens,"
            " coalesce(sum((attributes->>'gen_ai.usage.output_tokens')::bigint), 0) AS output_tokens"
            " FROM spans WHERE tenant_id = %s AND document_id = %s AND name = 'gen_ai.call'"
            " AND attributes->>'docqa.stage' = 'ocr' GROUP BY 1 ORDER BY 1",
            (tenant_id, document_id),
        ).fetchall()
    return OcrSummaryRow(**{key: int(value) for key, value in counts.items()}, by_model=usage)
