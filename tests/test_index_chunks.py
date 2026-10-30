"""The `index` job with the real indexer: chunks, history, optimistic text check, reconcile and cleanup."""

import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import numpy as np
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.db import tx
from docqa.index.chunking import chunk_page
from docqa.index.embed import index_version
from docqa.index.jobs import ChunkIndexer
from docqa.providers.llm import ProviderError
from docqa.repo import chunks as chunk_rows
from docqa.repo import maintenance
from docqa.repo.pages import PageText
from docqa.telemetry import MemoryTelemetry
from docqa.worker import WorkerDeps, run_one, startup
from tests.helpers import new_tenant
from tests.index_helpers import FakeEmbedder, index_all, make_document, with_embedder
from tests.ingest_helpers import document, jobs_of, page_rows

PAGE_1 = "# **Lubrication**\n\nRegrease every 9500 hours at 1800 RPM. Use Polyrex EM grease.\n\n|a|b|\n|---|---|\n|1|2|"
PAGE_2 = "# **Warranty**\n\nThe warranty covers 18 months from shipment."


def _chunks(pool: ConnectionPool, document_id: UUID) -> list[dict[str, Any]]:
    """Read a document's chunks in page and chunk order.

    Args:
        pool: Database pool.
        document_id: Document id.

    Returns:
        The rows, with the vector as text.
    """
    with pool.connection() as conn:
        return conn.execute(
            "SELECT id, page_no, ordinal, text_rev, text_source, index_version, text, units, embedding::text AS vec"
            " FROM chunks WHERE document_id = %s ORDER BY page_no, ordinal",
            (document_id,),
        ).fetchall()


def _history(pool: ConnectionPool, document_id: UUID) -> list[dict[str, Any]]:
    """Read a document's archived chunks.

    Args:
        pool: Database pool.
        document_id: Document id.

    Returns:
        The rows.
    """
    with pool.connection() as conn:
        return conn.execute("SELECT * FROM chunks_history WHERE document_id = %s", (document_id,)).fetchall()


def test_index_writes_chunks_vectors_and_units(worker_deps: WorkerDeps, telemetry: MemoryTelemetry) -> None:
    """Each page gets its chunks with units and a normalized vector; pages and document end indexed and ready.

    Args:
        worker_deps: Worker dependencies.
        telemetry: In-memory telemetry.
    """
    deps = with_embedder(worker_deps)
    tenant_id = new_tenant(deps.pool)
    document_id = make_document(deps.pool, tenant_id, [PAGE_1, PAGE_2, ""])
    assert index_all(deps) == 1
    rows = _chunks(deps.pool, document_id)
    expected = chunk_page("manual.pdf", 1, PAGE_1) + chunk_page("manual.pdf", 2, PAGE_2)
    assert [row["text"] for row in rows] == [chunk.text for chunk in expected]
    version = index_version(FakeEmbedder())
    assert {row["index_version"] for row in rows} == {version}
    assert rows[0]["units"][0] == {
        "pos": 0,
        "kind": "heading",
        "md": "# **Lubrication**",
        "table": None,
        "header": False,
    }
    vector = np.array([float(x) for x in rows[0]["vec"].strip("[]").split(",")])
    assert vector.shape == (1024,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
    states = [(row["state"], row["index_version"]) for row in page_rows(deps.pool, document_id)]
    assert states == [("indexed", version)] * 3
    doc = document(deps.pool, tenant_id, document_id)
    assert (doc.status, doc.pages_indexed, doc.pages_pending) == ("ready", 3, 0)
    span = telemetry.named("index.document")[0]
    assert (span.attributes["docqa.pages"], span.attributes["docqa.chunks"]) == (3, len(expected))
    assert span.attributes["docqa.peak_rss_mb"] > 0
    embed_spans = telemetry.named("embed.chunks")
    assert [s.attributes["docqa.embed.count"] for s in embed_spans] == [1, 1]
    assert embed_spans[0].attributes["docqa.embed.provider"] == "fake"


def test_reindex_moves_old_chunks_to_history(worker_deps: WorkerDeps) -> None:
    """A page with new text gets new chunks; the old ones move to history with the same ids and no vector.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_id = new_tenant(deps.pool)
    document_id = make_document(deps.pool, tenant_id, [PAGE_1])
    index_all(deps)
    old = _chunks(deps.pool, document_id)
    with deps.pool.connection() as conn:
        conn.execute(
            "UPDATE pages SET state = 'text_ready', layer_text = %s, text_rev = 2 WHERE document_id = %s",
            (PAGE_2, document_id),
        )
    with tx(deps.pool) as conn:
        queue.enqueue(conn, tenant_id=tenant_id, kind="index", document_id=document_id)
    index_all(deps)
    new = _chunks(deps.pool, document_id)
    history = _history(deps.pool, document_id)
    assert {row["id"] for row in history} == {row["id"] for row in old}
    assert {row["text"] for row in history} == {row["text"] for row in old}
    assert {row["text_rev"] for row in history} == {1}
    assert [row["text"] for row in new] == [chunk.text for chunk in chunk_page("manual.pdf", 1, PAGE_2)]
    assert {row["id"] for row in new}.isdisjoint({row["id"] for row in old})
    assert {row["text_rev"] for row in new} == {2}


class _RewritingEmbedder(FakeEmbedder):
    """An embedder that changes a page's text revision while it embeds, as a late OCR result would.

    Attributes:
        pool: Database pool.
        document_id: Document id.
        page_no: Page to rewrite.
    """

    def __init__(self, pool: ConnectionPool, document_id: UUID, page_no: int) -> None:
        """Store the target page.

        Args:
            pool: Database pool.
            document_id: Document id.
            page_no: Page to rewrite.
        """
        super().__init__()
        self.pool = pool
        self.document_id = document_id
        self.page_no = page_no

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Bump the target page's revision once, then embed.

        Args:
            texts: Texts.

        Returns:
            Normalized vectors.
        """
        if self.calls == self.page_no - 1:
            with self.pool.connection() as conn:
                conn.execute(
                    "UPDATE pages SET text_rev = text_rev + 1 WHERE document_id = %s AND page_no = %s",
                    (self.document_id, self.page_no),
                )
        return super().embed_documents(texts)


def test_text_changed_mid_run_leaves_no_chunks_and_page_text_ready(worker_deps: WorkerDeps) -> None:
    """A page rewritten while embedding stays `text_ready` and its chunks are rolled back with the mark.

    Args:
        worker_deps: Worker dependencies.
    """
    tenant_id = new_tenant(worker_deps.pool)
    document_id = make_document(worker_deps.pool, tenant_id, [PAGE_1, PAGE_2])
    deps = with_embedder(worker_deps, _RewritingEmbedder(worker_deps.pool, document_id, page_no=2))
    index_all(deps)
    assert [row["state"] for row in page_rows(deps.pool, document_id)] == ["indexed", "text_ready"]
    assert {row["page_no"] for row in _chunks(deps.pool, document_id)} == {1}
    assert document(deps.pool, tenant_id, document_id).status == "processing"


def test_reconcile_reindexes_pages_of_another_version(worker_deps: WorkerDeps) -> None:
    """Pages indexed under another version (`NullIndexer`, or another provider) go back to indexing at start.

    Args:
        worker_deps: Worker dependencies.
    """
    tenant_id = new_tenant(worker_deps.pool)
    document_id = make_document(worker_deps.pool, tenant_id, [PAGE_1, PAGE_2])
    assert run_one(worker_deps, "index")
    assert {row["index_version"] for row in page_rows(worker_deps.pool, document_id)} == {"null"}
    deps = with_embedder(worker_deps, FakeEmbedder("fake-hash@2"))
    assert startup(deps) == 2
    assert [row["state"] for row in page_rows(deps.pool, document_id)] == ["text_ready", "text_ready"]
    doc = document(deps.pool, tenant_id, document_id)
    assert (doc.status, doc.pages_indexed, doc.pages_pending) == ("processing", 0, 2)
    assert [job["state"] for job in jobs_of(deps.pool, document_id) if job["kind"] == "index"] == ["done", "queued"]
    index_all(deps)
    version = index_version(FakeEmbedder("fake-hash@2"))
    assert {row["index_version"] for row in page_rows(deps.pool, document_id)} == {version}
    assert document(deps.pool, tenant_id, document_id).status == "ready"
    assert startup(deps) == 0
    other = with_embedder(worker_deps, FakeEmbedder("fake-hash@3"))
    assert startup(other) == 2
    index_all(other)
    assert len(_history(deps.pool, document_id)) == len(_chunks(deps.pool, document_id))


def test_startup_without_embedder_does_nothing(worker_deps: WorkerDeps) -> None:
    """The ingestion tests' worker has no embedder, so the reconcile has no current version to compare.

    Args:
        worker_deps: Worker dependencies.
    """
    assert startup(worker_deps) == 0


def _archive(pool: ConnectionPool, tenant_id: UUID, document_id: UUID, archived_at: datetime) -> UUID:
    """Insert one archived chunk.

    Args:
        pool: Database pool.
        tenant_id: Owner.
        document_id: Document id.
        archived_at: Archive time.

    Returns:
        The chunk id.
    """
    chunk_id = uuid.uuid4()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO chunks_history (id, tenant_id, document_id, page_no, ordinal, text_rev, text_source,"
            " index_version, text, units, archived_at) VALUES (%s, %s, %s, 1, 0, 1, 'layer', 'v', 't', %s, %s)",
            (chunk_id, tenant_id, document_id, Jsonb([]), archived_at),
        )
    return chunk_id


def test_history_cleanup_keeps_referenced_and_recent_rows(worker_deps: WorkerDeps) -> None:
    """Old unreferenced history goes; history an answer points to, or archived recently, stays.

    Args:
        worker_deps: Worker dependencies.
    """
    pool = worker_deps.pool
    tenant_id = new_tenant(pool)
    document_id = make_document(pool, tenant_id, [PAGE_1])
    old = datetime(2026, 1, 1, tzinfo=UTC)
    gone = _archive(pool, tenant_id, document_id, old)
    cited = _archive(pool, tenant_id, document_id, old)
    recent = _archive(pool, tenant_id, document_id, datetime.now(UTC))
    answer_id = uuid.uuid4()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO answers (id, tenant_id, question, search_query, http_status, model, prompt_version,"
            " assembly_version, index_version, search_version, coverage, response) VALUES (%s, %s, 'q', 'q', 200,"
            " 'm', 'p', 'a', 'v', 's', '{}', '{}')",
            (answer_id, tenant_id),
        )
        conn.execute(
            "INSERT INTO answer_candidates (answer_id, tenant_id, rank, chunk_id, document_id, page_no,"
            " chunk_ordinal, sent_to_llm) VALUES (%s, %s, 1, %s, %s, 1, 0, true)",
            (answer_id, tenant_id, cited, document_id),
        )
        assert maintenance.cleanup_history(conn) == 1
        left = {row["id"] for row in conn.execute("SELECT id FROM chunks_history").fetchall()}
    assert left == {cited, recent}
    assert gone not in left


def test_page_units_dedupes_a_repeated_table_header(worker_deps: WorkerDeps) -> None:
    """A split table repeats its first row in every chunk; the page's units list it once, in position order.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    rows = "\n".join(f"|Frame {n} incl.|{n * 100}Hrs.|{n * 200}Hrs.|" for n in range(1, 50))
    big = f"|**Frame**|**3600**|**1800**|\n|---|---|---|\n{rows}"
    tenant_id = new_tenant(deps.pool)
    document_id = make_document(deps.pool, tenant_id, [big])
    index_all(deps)
    assert len(_chunks(deps.pool, document_id)) > 1
    version = index_version(FakeEmbedder())
    with deps.pool.connection() as conn:
        found = chunk_rows.page_units(conn, tenant_id=tenant_id, pairs=[(document_id, 1)], index_version=version)
        other = chunk_rows.page_units(
            conn, tenant_id=new_tenant(deps.pool), pairs=[(document_id, 1)], index_version=version
        )
    page = found[(document_id, 1)]
    assert [unit.pos for unit in page.units] == list(range(50))
    assert (page.text_rev, page.text_source) == (1, "layer")
    assert other == {}


class _FailingEmbedder(FakeEmbedder):
    """An embedder whose provider fails with a given error.

    Attributes:
        error: Raised by `embed_documents`.
    """

    provider = "gemini"

    def __init__(self, error: ProviderError) -> None:
        """Keep the error.

        Args:
            error: Raised by `embed_documents`.
        """
        super().__init__()
        self.error = error

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Fail.

        Args:
            texts: Texts.

        Raises:
            ProviderError: Always.
        """
        raise self.error


def test_provider_errors_steer_the_index_job(worker_deps: WorkerDeps) -> None:
    """A rate limit reschedules without spending the attempt; a malformed request fails the waiting pages.

    Args:
        worker_deps: Worker dependencies.
    """
    tenant_id = new_tenant(worker_deps.pool)
    document_id = make_document(worker_deps.pool, tenant_id, [PAGE_1])
    limited = ProviderError("rate_limited", "429", retry_after_s=40, model="gemini-embedding-2")
    assert run_one(with_embedder(worker_deps, _FailingEmbedder(limited)), "index")
    [job] = jobs_of(worker_deps.pool, document_id)
    assert (job["state"], job["attempts"], job["last_error_code"]) == ("queued", 0, "rate_limited")
    with worker_deps.pool.connection() as conn:
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job["id"],))
    bad = ProviderError("invalid_request", "400 INVALID_ARGUMENT", model="gemini-embedding-2")
    assert run_one(with_embedder(worker_deps, _FailingEmbedder(bad)), "index")
    [job] = jobs_of(worker_deps.pool, document_id)
    assert (job["state"], job["last_error_code"]) == ("failed", "embed_invalid_request")
    [page] = page_rows(worker_deps.pool, document_id)
    assert (page["state"], page["error_code"]) == ("failed", "index_failed")
    assert document(worker_deps.pool, tenant_id, document_id).status == "failed"


def test_chunk_indexer_on_a_page_without_units(worker_deps: WorkerDeps) -> None:
    """A page whose text has no units is indexed with no chunks and no embedding call.

    Args:
        worker_deps: Worker dependencies.
    """
    embedder = FakeEmbedder()
    indexer = ChunkIndexer(embedder, MemoryTelemetry())
    entries = indexer.prepare("m.pdf", PageText(page_no=1, text_rev=1, text_source="layer", text="  \n "))
    assert embedder.calls == 0
    tenant_id = new_tenant(worker_deps.pool)
    document_id = make_document(worker_deps.pool, tenant_id, [PAGE_1])
    with worker_deps.pool.connection() as conn, conn.transaction():
        assert entries.write(conn, tenant_id=tenant_id, document_id=document_id) == 0
