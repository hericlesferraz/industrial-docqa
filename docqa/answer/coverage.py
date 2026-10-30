"""The coverage block of every answer (DECISIONS #16): how much of the scope the search could read.

`complete` means every document in scope is `ready`. Pending pages (the answer may change later) and failed pages (no
answer will cover them) are counted apart. The block reads document rows only, whose counters change in the same
transaction as the page states, so its cost doesn't grow with the number of pages.

`pending_work` is internal: it adds the coverage note to the prompt only when a document in scope is still `queued` or
`processing`. A failed or partial document never triggers it, because "still being processed" would be false.
"""

from dataclasses import dataclass
from uuid import UUID

from psycopg import Connection

from docqa.repo import documents

COVERAGE_INCOMPLETE_CAP = 20


@dataclass(frozen=True)
class IncompleteDocument:
    """A document of the scope that isn't `ready`.

    Attributes:
        document_id: Document id.
        filename: Document filename.
        status: `queued`, `processing`, `partial` or `failed`.
        pages_pending: Pages still being processed; None before extraction.
        pages_failed: Pages that failed; None before extraction.
        error_code: The document's failure code, if any.
    """

    document_id: UUID
    filename: str
    status: str
    pages_pending: int | None
    pages_failed: int | None
    error_code: str | None


@dataclass(frozen=True)
class Coverage:
    """Coverage of a question's scope.

    Attributes:
        complete: Every document in scope is `ready`.
        documents: Documents in scope.
        documents_incomplete: Documents that aren't `ready`.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed.
        pages_failed: Pages that failed.
        incomplete: Up to `COVERAGE_INCOMPLETE_CAP` documents that aren't ready, pending ones first.
        pending_work: A document in scope is `queued` or `processing`.
    """

    complete: bool
    documents: int
    documents_incomplete: int
    pages_indexed: int
    pages_pending: int
    pages_failed: int
    incomplete: list[IncompleteDocument]
    pending_work: bool

    def to_json(self) -> dict[str, object]:
        """Give the block returned with the answer and stored in its record.

        Returns:
            Every field except `pending_work`, ids as strings.
        """
        return {
            "complete": self.complete,
            "documents": self.documents,
            "documents_incomplete": self.documents_incomplete,
            "pages_indexed": self.pages_indexed,
            "pages_pending": self.pages_pending,
            "pages_failed": self.pages_failed,
            "incomplete": [
                {
                    "document_id": str(item.document_id),
                    "filename": item.filename,
                    "status": item.status,
                    "pages_pending": item.pages_pending,
                    "pages_failed": item.pages_failed,
                    "error_code": item.error_code,
                }
                for item in self.incomplete
            ],
        }


def coverage(conn: Connection, *, tenant_id: UUID, document_ids: list[UUID] | None) -> Coverage:
    """Build the coverage block of a scope.

    Args:
        conn: Any connection.
        tenant_id: Tenant asking.
        document_ids: The scope, or None for every document of the tenant.

    Returns:
        The coverage.
    """
    totals, rows = documents.scope_coverage(
        conn, tenant_id=tenant_id, document_ids=document_ids, cap=COVERAGE_INCOMPLETE_CAP
    )
    incomplete = [
        IncompleteDocument(
            document_id=row.id,
            filename=row.filename,
            status=row.status,
            pages_pending=row.pages_pending if row.page_count is not None else None,
            pages_failed=row.pages_failed if row.page_count is not None else None,
            error_code=row.error_code,
        )
        for row in rows
    ]
    return Coverage(
        complete=totals.incomplete == 0,
        documents=totals.documents,
        documents_incomplete=totals.incomplete,
        pages_indexed=totals.pages_indexed,
        pages_pending=totals.pages_pending,
        pages_failed=totals.pages_failed,
        incomplete=incomplete,
        pending_work=totals.pending_documents > 0,
    )
