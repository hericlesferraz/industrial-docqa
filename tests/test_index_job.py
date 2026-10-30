"""Document status rules, coverage counters, and the `index` job with `NullIndexer`."""

from uuid import UUID

import pytest
from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.db import tx
from docqa.index.jobs import IndexHandler, NoEntries, NullIndexer, PageEntries
from docqa.queue import Job
from docqa.repo import documents, pages
from docqa.repo.documents import document_status
from docqa.repo.pages import PageInit, PageText
from docqa.telemetry import MemoryTelemetry
from docqa.worker import HANDLERS, WorkerDeps, run_one
from tests.helpers import new_document, new_tenant
from tests.ingest_helpers import document, jobs_of, page_rows


@pytest.mark.parametrize(
    ("page_count", "error_code", "counts", "expected"),
    [
        (None, None, (0, 0, 0), ("queued", None)),
        (None, "pdf_encrypted", (0, 0, 0), ("failed", "pdf_encrypted")),
        (3, None, (1, 2, 0), ("processing", None)),
        (3, None, (0, 1, 2), ("processing", None)),
        (3, None, (0, 0, 2), ("failed", "all_pages_failed")),
        (3, None, (0, 0, 0), ("failed", "no_text")),
        (3, None, (2, 0, 1), ("partial", None)),
        (3, None, (3, 0, 0), ("ready", None)),
        (3, "no_text", (3, 0, 0), ("ready", None)),
    ],
)
def test_document_status_rules(
    page_count: int | None, error_code: str | None, counts: tuple[int, int, int], expected: tuple[str, str | None]
) -> None:
    """The status rules, first match wins; extraction errors stay, derived codes follow the pages.

    Args:
        page_count: Pages in the file.
        error_code: The document's current code.
        counts: `(indexed, pending, failed)`.
        expected: `(status, error_code)`.
    """
    indexed, pending, failed = counts
    got = document_status(page_count=page_count, error_code=error_code, indexed=indexed, pending=pending, failed=failed)
    assert got == expected


def _pages(pool: ConnectionPool, tenant_id: UUID, document_id: UUID, states: list[str]) -> None:
    """Create page rows in the given states and set the page count.

    Args:
        pool: Database pool.
        tenant_id: Owner.
        document_id: Document id.
        states: State of each page.
    """
    rows = [
        PageInit(
            page_no=n,
            state=state,
            text_source="layer" if state in ("text_ready", "indexed") else None,
            layer_text=f"text of page {n}" if state != "blank" else "",
            quality={},
            page_label=None,
            text_rev=1 if state in ("text_ready", "indexed") else 0,
            extract_version="test",
        )
        for n, state in enumerate(states, start=1)
    ]
    with tx(pool) as conn:
        pages.insert_pages(conn, tenant_id=tenant_id, document_id=document_id, pages=rows)
        documents.set_page_count(conn, tenant_id=tenant_id, document_id=document_id, page_count=len(rows))
        documents.recompute_document(conn, tenant_id=tenant_id, document_id=document_id)


def test_recompute_writes_counters_and_status(db: ConnectionPool) -> None:
    """Counters count indexed, pending (`needs_ocr` + `text_ready`) and failed pages; blank pages count nowhere.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_id = new_document(db, tenant_id)
    _pages(db, tenant_id, document_id, ["indexed", "text_ready", "needs_ocr", "blank", "failed"])
    doc = document(db, tenant_id, document_id)
    assert (doc.status, doc.pages_indexed, doc.pages_pending, doc.pages_failed) == ("processing", 1, 2, 1)


def _enqueue_index(pool: ConnectionPool, tenant_id: UUID, document_id: UUID) -> None:
    """Enqueue the document's `index` job.

    Args:
        pool: Database pool.
        tenant_id: Owner.
        document_id: Document id.
    """
    with tx(pool) as conn:
        queue.enqueue(conn, tenant_id=tenant_id, kind="index", document_id=document_id)


def test_null_indexer_takes_text_pages_to_ready(worker_deps: WorkerDeps, telemetry: MemoryTelemetry) -> None:
    """Every `text_ready` page becomes `indexed` with version `null`, and the document becomes `ready`.

    Args:
        worker_deps: Worker dependencies.
        telemetry: In-memory telemetry.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = new_document(pool, tenant_id)
    _pages(pool, tenant_id, document_id, ["text_ready", "blank", "text_ready"])
    _enqueue_index(pool, tenant_id, document_id)
    assert run_one(worker_deps, "index")
    assert [row["state"] for row in page_rows(pool, document_id)] == ["indexed", "blank", "indexed"]
    doc = document(pool, tenant_id, document_id)
    assert (doc.status, doc.pages_indexed, doc.pages_pending) == ("ready", 2, 0)
    span = telemetry.named("index.document")[0]
    assert (span.attributes["docqa.pages"], span.attributes["docqa.index_version"]) == (2, "null")


class _Probe:
    """An indexer that records the document's counters before each page, and can change a page's text meanwhile.

    Attributes:
        pool: Database pool.
        tenant_id: Owner.
        document_id: Document id.
        rewrite: Page whose text is replaced while it is being indexed.
        seen: `pages_indexed` of the document before each page.
        index_version: Version recorded on indexed pages.
    """

    index_version = "probe"

    def __init__(self, pool: ConnectionPool, tenant_id: UUID, document_id: UUID, rewrite: int | None) -> None:
        """Store the target.

        Args:
            pool: Database pool.
            tenant_id: Owner.
            document_id: Document id.
            rewrite: Page whose text changes during indexing.
        """
        self.pool = pool
        self.tenant_id = tenant_id
        self.document_id = document_id
        self.rewrite = rewrite
        self.seen: list[int] = []

    def prepare(self, filename: str, page: PageText) -> PageEntries:
        """Record the counter and maybe bump the page's text revision, as a late OCR result would.

        Args:
            filename: Document filename.
            page: The page.

        Returns:
            No entries.
        """
        self.seen.append(document(self.pool, self.tenant_id, self.document_id).pages_indexed)
        if page.page_no == self.rewrite:
            with self.pool.connection() as conn:
                conn.execute(
                    "UPDATE pages SET text_rev = text_rev + 1 WHERE document_id = %s AND page_no = %s",
                    (self.document_id, page.page_no),
                )
        return NoEntries()


def test_counters_move_with_each_page(worker_deps: WorkerDeps) -> None:
    """The coverage counters change in the same transaction as each page, not only at the end of the job.

    Args:
        worker_deps: Worker dependencies.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = new_document(pool, tenant_id)
    _pages(pool, tenant_id, document_id, ["text_ready", "text_ready", "text_ready"])
    probe = _Probe(pool, tenant_id, document_id, rewrite=None)
    _enqueue_index(pool, tenant_id, document_id)
    assert run_one(worker_deps, "index", handlers={**HANDLERS, "index": IndexHandler(probe)})
    assert probe.seen == [0, 1, 2]


def test_page_whose_text_changed_stays_text_ready(worker_deps: WorkerDeps) -> None:
    """A page rewritten while being indexed is not marked indexed; its chunks would hold the old text.

    Args:
        worker_deps: Worker dependencies.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = new_document(pool, tenant_id)
    _pages(pool, tenant_id, document_id, ["text_ready", "text_ready"])
    probe = _Probe(pool, tenant_id, document_id, rewrite=2)
    _enqueue_index(pool, tenant_id, document_id)
    assert run_one(worker_deps, "index", handlers={**HANDLERS, "index": IndexHandler(probe)})
    assert [row["state"] for row in page_rows(pool, document_id)] == ["indexed", "text_ready"]
    assert document(pool, tenant_id, document_id).status == "processing"


def test_index_failure_fails_the_waiting_pages(worker_deps: WorkerDeps) -> None:
    """When indexing gives up, its `text_ready` pages fail with `index_failed` and the document is recomputed.

    Args:
        worker_deps: Worker dependencies.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = new_document(pool, tenant_id)
    _pages(pool, tenant_id, document_id, ["indexed", "text_ready"])
    _enqueue_index(pool, tenant_id, document_id)
    [job_row] = jobs_of(pool, document_id)
    job = Job(
        id=job_row["id"],
        tenant_id=tenant_id,
        kind="index",
        document_id=document_id,
        page_no=None,
        attempts=3,
        max_attempts=3,
        lease_token=None,
        last_error_code="internal_error",
    )
    with tx(pool) as conn:
        IndexHandler(NullIndexer()).on_failure(conn, job, "internal_error", "boom")
    rows = page_rows(pool, document_id)
    assert [(row["state"], row["error_code"]) for row in rows] == [("indexed", None), ("failed", "index_failed")]
    assert document(pool, tenant_id, document_id).status == "partial"


def test_null_indexer_writes_nothing(db: ConnectionPool) -> None:
    """`NullIndexer` builds empty entries under version `null`.

    Args:
        db: Pool on an empty database.
    """
    entries = NullIndexer().prepare("manual.pdf", PageText(page_no=1, text_rev=1, text_source="layer", text="x"))
    assert NullIndexer.index_version == "null"
    with db.connection() as conn:
        assert entries.write(conn, tenant_id=UUID(int=0), document_id=UUID(int=0)) == 0
