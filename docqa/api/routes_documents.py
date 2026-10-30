"""Document and batch routes: streaming upload, batches, the document list and the document view."""

from functools import partial
from pathlib import Path
from typing import Annotated
from uuid import UUID

import anyio
from fastapi import APIRouter, Query, Request, Response
from psycopg import Connection
from starlette.requests import ClientDisconnect

from docqa import queue
from docqa.api.deps import Deps, DepsDep, TenantDep
from docqa.api.models import (
    BatchCounts,
    BatchCreated,
    BatchDocument,
    BatchItem,
    BatchView,
    DocumentDetail,
    DocumentList,
    DocumentSummary,
    ErrorInfo,
    OcrPause,
    OcrUsage,
    OcrView,
    PageView,
    UploadResult,
)
from docqa.config import Settings
from docqa.db import tx
from docqa.errors import BATCH_NOT_FOUND, DOCUMENT_NOT_FOUND, ApiError, ErrorBody
from docqa.ingest.upload import (
    HEADER_LIMIT,
    UploadedFile,
    UploadReader,
    UploadRejected,
    boundary_of,
    invalid,
    too_large,
)
from docqa.providers import pacing
from docqa.repo import batches, documents, pages
from docqa.repo.documents import FINAL_STATUSES, DocumentRow
from docqa.storage import BlobStore

router = APIRouter(prefix="/v1", tags=["documents"])

ITEM_REJECT_CODES = ("file_too_large", "empty_file", "not_a_pdf")
DOCUMENT_STATUSES = ("queued", "processing", "ready", "partial", "failed")
UPLOAD_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {"file": {"type": "string", "format": "binary", "description": "One PDF file."}},
                    "required": ["file"],
                }
            }
        },
    }
}
ERRORS_401 = {401: {"model": ErrorBody}}


def ocr_pause(conn: Connection, settings: Settings) -> OcrPause | None:
    """Return the provider pause that holds transcriptions: the account's, the default OCR model's or the fallback's.

    Args:
        conn: Any connection.
        settings: Process settings, for the two OCR models.

    Returns:
        The pause, or None.
    """
    found = pacing.first_pause(conn, [settings.gemini_ocr_model, settings.gemini_ocr_fallback_model])
    return OcrPause(model=found.model, reason=found.reason, until=found.until) if found else None


def _batch_not_found() -> ApiError:
    """Build the 404 of an unknown batch.

    Returns:
        The error; another tenant's batch gets the same answer.
    """
    return ApiError(404, BATCH_NOT_FOUND, "Unknown batch id.")


@router.post(
    "/batches",
    status_code=201,
    response_model=BatchCreated,
    responses=ERRORS_401,
    summary="Create an upload batch",
)
def create_batch(deps: DepsDep, tenant_id: TenantDep) -> BatchCreated:
    """Create an empty batch; send each file with its id to follow the whole upload in one view.

    Args:
        deps: Process objects.
        tenant_id: The authenticated tenant.

    Returns:
        The batch id and creation time.
    """
    with tx(deps.pool) as conn:
        batch = batches.create_batch(conn, tenant_id=tenant_id)
    return BatchCreated(batch_id=batch.id, created_at=batch.created_at)


@router.get(
    "/batches/{batch_id}",
    response_model=BatchView,
    responses={**ERRORS_401, 404: {"model": ErrorBody}},
    summary="Follow an upload batch",
)
def get_batch(batch_id: UUID, deps: DepsDep, tenant_id: TenantDep) -> BatchView:
    """Return every item of a batch with its document's status, and whether the batch is complete.

    Args:
        batch_id: Batch id.
        deps: Process objects.
        tenant_id: The authenticated tenant.

    Returns:
        The batch view.

    Raises:
        ApiError: 404 `batch_not_found`, also for another tenant's batch.
    """
    with deps.pool.connection() as conn:
        view = batches.batch_view(conn, tenant_id=tenant_id, batch_id=batch_id)
        waiting = view is not None and any(row.needs_ocr for row in view.items)
        paused = ocr_pause(conn, deps.settings) if waiting else None
    if view is None:
        raise _batch_not_found()
    outcomes = dict.fromkeys(("accepted", "duplicate", "rejected"), 0)
    statuses: dict[UUID, str] = {}
    items = []
    for row in view.items:
        outcomes[row.outcome] += 1
        document = None
        if row.document_id is not None and row.status is not None:
            statuses[row.document_id] = row.status
            document = BatchDocument(
                document_id=row.document_id,
                status=row.status,
                page_count=row.page_count,
                pages_indexed=row.pages_indexed or 0,
                pages_pending=row.pages_pending or 0,
                pages_failed=row.pages_failed or 0,
                error_code=row.document_error_code,
                ocr_paused=paused if row.needs_ocr else None,
            )
        items.append(
            BatchItem(
                filename=row.filename,
                outcome=row.outcome,
                error_code=row.error_code,
                created_at=row.created_at,
                document=document,
            )
        )
    by_status = dict.fromkeys(DOCUMENT_STATUSES, 0)
    for status in statuses.values():
        by_status[status] += 1
    complete = all(
        row.outcome == "rejected" or row.status is None or row.status in FINAL_STATUSES for row in view.items
    )
    counts = BatchCounts(items=len(items), documents=by_status, **outcomes)
    return BatchView(
        batch_id=view.batch.id, created_at=view.batch.created_at, complete=complete, counts=counts, items=items
    )


def _content_length(request: Request) -> int | None:
    """Read the request's declared body size.

    Args:
        request: The upload request.

    Returns:
        `Content-Length`, or None when absent or malformed.
    """
    raw = request.headers.get("content-length")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


async def _receive(request: Request, blobs: BlobStore, max_bytes: int) -> tuple[Path, UploadedFile]:
    """Stream the request body into a temp file, checking it as it arrives.

    The size declared by `Content-Length` is checked as soon as the part's headers (and so the filename) are read,
    so an oversized file is refused without reading the rest of it.

    Args:
        request: The upload request.
        blobs: File store that owns the temp folder.
        max_bytes: File size limit.

    Returns:
        The temp file and what was received.

    Raises:
        UploadRejected: The upload breaks a rule; the temp file is gone and `filename` is set when known.
    """
    boundary = boundary_of(request.headers.get("content-type"))
    declared = _content_length(request)
    tmp = await anyio.to_thread.run_sync(blobs.new_temp)
    reader: UploadReader | None = None
    try:
        reader = await anyio.to_thread.run_sync(UploadReader, boundary, tmp, max_bytes)
        received, length_checked = 0, False
        async for chunk in request.stream():
            received += len(chunk)
            await anyio.to_thread.run_sync(reader.feed, chunk)
            if not reader.headers_done:
                if received > HEADER_LIMIT:
                    raise invalid("The part headers are larger than 64 KiB.")
            elif not length_checked:
                length_checked = True
                if declared is not None and declared > max_bytes + HEADER_LIMIT:
                    raise too_large(max_bytes)
        uploaded = await anyio.to_thread.run_sync(reader.finish)
    except (UploadRejected, ClientDisconnect) as exc:
        if reader is not None:
            reader.close()
        await anyio.to_thread.run_sync(partial(tmp.unlink, missing_ok=True))
        rejected = exc if isinstance(exc, UploadRejected) else invalid("The upload was interrupted.")
        rejected.filename = reader.filename if reader is not None else None
        raise rejected from None
    return tmp, uploaded


def _record_rejection(deps: Deps, tenant_id: UUID, batch_id: UUID, filename: str, code: str) -> None:
    """Record a rejected file in its batch.

    Args:
        deps: Process objects.
        tenant_id: The tenant.
        batch_id: The batch the file was sent with.
        filename: Cleaned filename.
        code: Rejection code.
    """
    with tx(deps.pool) as conn:
        batches.add_item(
            conn, tenant_id=tenant_id, batch_id=batch_id, filename=filename, outcome="rejected", error_code=code
        )


def _store_document(
    deps: Deps, tenant_id: UUID, batch_id: UUID | None, uploaded: UploadedFile, key: str
) -> tuple[DocumentRow, bool]:
    """Create the document and its `extract` job, or find the duplicate, and record the batch item; one transaction.

    Args:
        deps: Process objects.
        tenant_id: The tenant.
        batch_id: Batch the file was sent with, if any.
        uploaded: What was received.
        key: Key of the stored file.

    Returns:
        `(document, created)`.
    """
    with tx(deps.pool) as conn:
        row, created = documents.insert_or_get_document(
            conn, tenant_id=tenant_id, sha256=uploaded.sha256, filename=uploaded.filename, blob_key=key
        )
        if created:
            queue.enqueue(conn, tenant_id=tenant_id, kind="extract", document_id=row.id)
        if batch_id is not None:
            batches.add_item(
                conn,
                tenant_id=tenant_id,
                batch_id=batch_id,
                filename=uploaded.filename,
                outcome="accepted" if created else "duplicate",
                document_id=row.id,
            )
    return row, created


def _find_batch(deps: Deps, tenant_id: UUID, batch_id: UUID) -> bool:
    """Tell whether a batch belongs to the tenant.

    Args:
        deps: Process objects.
        tenant_id: The tenant.
        batch_id: Batch id.

    Returns:
        True when the tenant owns the batch.
    """
    with deps.pool.connection() as conn:
        return batches.get_batch(conn, tenant_id=tenant_id, batch_id=batch_id) is not None


@router.post(
    "/documents",
    status_code=202,
    response_model=UploadResult,
    responses={
        200: {"model": UploadResult, "description": "The same bytes were already uploaded; nothing is reprocessed."},
        **ERRORS_401,
        404: {"model": ErrorBody, "description": "`batch_not_found`"},
        413: {"model": ErrorBody, "description": "`file_too_large`"},
        422: {"model": ErrorBody, "description": "`empty_file`, `not_a_pdf`, `invalid_upload`, `invalid_request`"},
    },
    openapi_extra=UPLOAD_BODY,
    summary="Upload one PDF",
)
async def upload_document(
    request: Request,
    response: Response,
    deps: DepsDep,
    tenant_id: TenantDep,
    batch_id: Annotated[UUID | None, Query(description="Batch to record this file in.")] = None,
) -> UploadResult:
    """Store one PDF sent as multipart part `file` and queue its extraction.

    The body is streamed to disk while its sha256 is computed, never buffered and never parsed as a PDF here. A new
    document answers 202; the same bytes uploaded again by the tenant answer 200 with `duplicate: true` and the
    existing document as it is.

    Args:
        request: The upload request.
        response: Lets the route answer 200 for a duplicate.
        deps: Process objects.
        tenant_id: The authenticated tenant.
        batch_id: Batch to record this file in.

    Returns:
        The document, its status and whether it was a duplicate.

    Raises:
        ApiError: 404 `batch_not_found`, 413 `file_too_large`, 422 `empty_file`, `not_a_pdf` or `invalid_upload`.
    """
    max_bytes = deps.settings.max_upload_mb * 2**20
    if batch_id is not None and not await anyio.to_thread.run_sync(_find_batch, deps, tenant_id, batch_id):
        raise _batch_not_found()
    with deps.telemetry.span("upload.store") as span:
        try:
            tmp, uploaded = await _receive(request, deps.blobs, max_bytes)
        except UploadRejected as exc:
            span.set("docqa.upload.outcome", "rejected")
            span.set("docqa.error_code", exc.code)
            if batch_id is not None and exc.filename and exc.code in ITEM_REJECT_CODES:
                await anyio.to_thread.run_sync(_record_rejection, deps, tenant_id, batch_id, exc.filename, exc.code)
            details: dict[str, object] | None = {"max_bytes": max_bytes} if exc.status == 413 else None
            raise ApiError(exc.status, exc.code, exc.message, details=details) from None
        key = await anyio.to_thread.run_sync(deps.blobs.commit_temp, tmp, tenant_id, uploaded.sha256)
        row, created = await anyio.to_thread.run_sync(_store_document, deps, tenant_id, batch_id, uploaded, key)
        span.set("docqa.upload.bytes", uploaded.size)
        span.set("docqa.upload.outcome", "accepted" if created else "duplicate")
        span.set("docqa.upload.duplicate", not created)
    response.status_code = 202 if created else 200
    return UploadResult(document_id=row.id, filename=row.filename, status=row.status, duplicate=not created)


def _summary(row: DocumentRow) -> DocumentSummary:
    """Render a document row for the list.

    Args:
        row: The document.

    Returns:
        Its summary.
    """
    return DocumentSummary(
        document_id=row.id,
        filename=row.filename,
        status=row.status,
        page_count=row.page_count,
        pages_indexed=row.pages_indexed,
        pages_pending=row.pages_pending,
        pages_failed=row.pages_failed,
        error_code=row.error_code,
        created_at=row.created_at,
    )


@router.get("/documents", response_model=DocumentList, responses=ERRORS_401, summary="List documents")
def list_documents(
    deps: DepsDep,
    tenant_id: TenantDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    """List the tenant's documents, newest first.

    Args:
        deps: Process objects.
        tenant_id: The authenticated tenant.
        limit: Page size.
        offset: Documents to skip.

    Returns:
        A page of documents and the total.
    """
    with deps.pool.connection() as conn:
        rows, total = documents.list_documents(conn, tenant_id=tenant_id, limit=limit, offset=offset)
    return DocumentList(documents=[_summary(row) for row in rows], total=total)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentDetail,
    responses={**ERRORS_401, 404: {"model": ErrorBody}},
    summary="Inspect a document and its pages",
)
def get_document(document_id: UUID, deps: DepsDep, tenant_id: TenantDep) -> DocumentDetail:
    """Return a document with its counters, its pages by state and by text source, its OCR, and each page's state.

    Args:
        document_id: Document id.
        deps: Process objects.
        tenant_id: The authenticated tenant.

    Returns:
        The document view.

    Raises:
        ApiError: 404 `document_not_found`, also for another tenant's document.
    """
    with deps.pool.connection() as conn:
        detail = documents.document_detail(conn, tenant_id=tenant_id, document_id=document_id)
        if detail is None:
            raise ApiError(404, DOCUMENT_NOT_FOUND, "Unknown document id.")
        by_state, by_source = pages.page_counts(conn, tenant_id=tenant_id, document_id=document_id)
        ocr = documents.ocr_summary(conn, tenant_id=tenant_id, document_id=document_id)
        paused = ocr_pause(conn, deps.settings) if ocr.needs_ocr else None
    doc = detail.document
    return DocumentDetail(
        document_id=doc.id,
        filename=doc.filename,
        status=doc.status,
        error=ErrorInfo(code=doc.error_code, message=doc.error_message) if doc.error_code else None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        page_count=doc.page_count,
        pages_indexed=doc.pages_indexed,
        pages_pending=doc.pages_pending,
        pages_failed=doc.pages_failed,
        pages_by_state=by_state,
        pages_by_source=by_source,
        ocr=OcrView(
            pages_flagged=ocr.pages_flagged,
            pages_transcribed=ocr.pages_transcribed,
            pages_from_cache=ocr.pages_from_cache,
            recitations=ocr.recitations,
            fallback_used=ocr.fallback_used,
            calls=sum(usage.calls for usage in ocr.by_model),
            input_tokens=sum(usage.input_tokens for usage in ocr.by_model),
            output_tokens=sum(usage.output_tokens for usage in ocr.by_model),
            by_model={
                usage.model: OcrUsage(
                    calls=usage.calls, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens
                )
                for usage in ocr.by_model
            },
            paused=paused,
        ),
        pages=[
            PageView(
                page=page.page_no,
                state=page.state,
                text_source=page.text_source,
                page_label=page.page_label,
                flag_reason=page.flag_reason,
                error=ErrorInfo(code=page.error_code, message=page.error_message) if page.error_code else None,
            )
            for page in detail.pages
        ],
    )
