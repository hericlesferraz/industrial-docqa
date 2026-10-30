"""The `index` job: make a document's ready pages searchable, page by page, and recompute its status.

`ChunkIndexer` cuts each page into chunks of whole units and embeds them (DECISIONS #10); a page's chunks replace its
previous ones, which move to `chunks_history` (DECISIONS #5). `NullIndexer` writes no search entries: the job runs
with it when the worker has no embedder (tests of the ingestion stages), and the worker-start reconcile reindexes the
pages it marked, because their index version differs.
"""

import logging
import resource
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

import numpy as np
from psycopg import Connection

from docqa import queue
from docqa.db import tx
from docqa.handler import JobContext, JobSignal, PermanentFailure, Reschedule, Retry
from docqa.index.chunking import Chunk, chunk_page
from docqa.index.embed import Embedder, index_version
from docqa.ingest.labels import refresh_labels
from docqa.providers import pacing
from docqa.providers.llm import ProviderError
from docqa.queue import Job
from docqa.repo import chunks as chunk_rows
from docqa.repo import documents, pages
from docqa.repo.pages import PageText
from docqa.telemetry import Telemetry

NULL_INDEX_VERSION = "null"
RATE_LIMIT_WAIT_S = 30.0
TRANSIENT_KINDS = ("transient", "timeout")

logger = logging.getLogger("docqa.index")


class PageEntries(Protocol):
    """Search entries built for one page, written inside that page's index transaction."""

    def write(self, conn: Connection, *, tenant_id: UUID, document_id: UUID) -> int:
        """Replace the page's search entries.

        Args:
            conn: Connection inside the page's index transaction, after the fencing check.
            tenant_id: Owner.
            document_id: Document id.

        Returns:
            Number of entries written.
        """
        ...


class Indexer(Protocol):
    """Builds a page's search entries; the heavy work runs outside any transaction.

    Attributes:
        index_version: Version of the chunker and embedder, recorded on indexed pages.
    """

    index_version: str

    def prepare(self, filename: str, page: PageText) -> PageEntries:
        """Build the search entries of one page (chunking and embedding), outside any transaction.

        Args:
            filename: Document filename, for the context line of the chunks.
            page: The page and its canonical text.

        Returns:
            Entries ready to be written.
        """
        ...


@dataclass(frozen=True)
class NoEntries:
    """The empty set of search entries `NullIndexer` builds."""

    def write(self, conn: Connection, *, tenant_id: UUID, document_id: UUID) -> int:
        """Write nothing.

        Args:
            conn: Connection inside the page's index transaction.
            tenant_id: Owner.
            document_id: Document id.

        Returns:
            0.
        """
        return 0


class NullIndexer:
    """An indexer that writes no search entries; pages become `indexed` with index version `null`.

    Attributes:
        index_version: Always `null`.
    """

    index_version = NULL_INDEX_VERSION

    def prepare(self, filename: str, page: PageText) -> PageEntries:
        """Build no entries.

        Args:
            filename: Document filename.
            page: The page.

        Returns:
            An empty set of entries.
        """
        return NoEntries()


@dataclass(frozen=True)
class ChunkEntries:
    """A page's chunks and their vectors, ready to replace the page's current chunks.

    Attributes:
        page: The page they were built from.
        chunks: The page's chunks; none for a page without units.
        vectors: One normalized vector per chunk.
        index_version: Version of the chunker and embedder.
    """

    page: PageText
    chunks: list[Chunk]
    vectors: np.ndarray
    index_version: str

    def write(self, conn: Connection, *, tenant_id: UUID, document_id: UUID) -> int:
        """Archive the page's current chunks and insert these.

        Args:
            conn: Connection inside the page's index transaction, after the fencing check.
            tenant_id: Owner.
            document_id: Document id.

        Returns:
            Number of chunks written.
        """
        chunk_rows.replace_page_chunks(
            conn,
            tenant_id=tenant_id,
            document_id=document_id,
            page_no=self.page.page_no,
            text_rev=self.page.text_rev,
            text_source=self.page.text_source,
            index_version=self.index_version,
            chunks=self.chunks,
            vectors=self.vectors,
        )
        return len(self.chunks)


class ChunkIndexer:
    """The real indexer: chunks of whole units, embedded outside any transaction.

    Attributes:
        embedder: The deployment's embedder.
        telemetry: Where the `embed.chunks` spans go.
        index_version: `win250-head4|<embedder model id>`.
    """

    def __init__(self, embedder: Embedder, telemetry: Telemetry) -> None:
        """Keep the embedder and derive the index version.

        Args:
            embedder: The deployment's embedder.
            telemetry: Where the `embed.chunks` spans go.
        """
        self.embedder = embedder
        self.telemetry = telemetry
        self.index_version = index_version(embedder)

    def prepare(self, filename: str, page: PageText) -> PageEntries:
        """Chunk one page and embed its chunks.

        Args:
            filename: Document filename, for the context line of the chunks.
            page: The page and its canonical text.

        Returns:
            The page's chunks with their vectors.

        Raises:
            ProviderError: The Gemini embedder failed.
        """
        chunks = chunk_page(filename, page.page_no, page.text)
        vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        if chunks:
            attributes = {
                "docqa.embed.provider": self.embedder.provider,
                "docqa.embed.count": len(chunks),
                "docqa.embed.batch_size": int(getattr(self.embedder, "batch_size", len(chunks))),
            }
            with self.telemetry.span("embed.chunks", **attributes):
                vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        return ChunkEntries(page=page, chunks=chunks, vectors=vectors, index_version=self.index_version)


def peak_rss_mb() -> int:
    """Return the peak resident memory of this process so far.

    Returns:
        Peak RSS in MiB (Linux reports `ru_maxrss` in KiB).
    """
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


class _TextChanged(Exception):
    """The page's text changed while it was being indexed; its transaction is rolled back."""


class IndexHandler:
    """Runs `index` jobs: one per document, never more than one queued or running.

    Attributes:
        indexer: A fixed indexer, or None to build one per job from the worker's embedder.
    """

    def __init__(self, indexer: Indexer | None = None) -> None:
        """Keep the indexer.

        Args:
            indexer: A fixed indexer (tests), or None: `ChunkIndexer` over the worker's embedder, or `NullIndexer`
                when the worker has none.
        """
        self.indexer = indexer

    def indexer_for(self, ctx: JobContext) -> Indexer:
        """Choose the indexer of a job.

        Args:
            ctx: The claimed job and the process dependencies.

        Returns:
            The fixed indexer, else `ChunkIndexer` over `ctx.embedder`, else `NullIndexer`.
        """
        if self.indexer is not None:
            return self.indexer
        if ctx.embedder is not None:
            return ChunkIndexer(ctx.embedder, ctx.telemetry)
        return NullIndexer()

    def run(self, ctx: JobContext) -> None:
        """Index every `text_ready` page of the document, one transaction per page, then complete.

        Each page transaction writes the page's entries, marks the page `indexed` if its text is unchanged and
        recomputes the document's counters and status, so a page answers questions as soon as it is indexed and the
        coverage counts stay exact. The last transaction recomputes the page labels of the whole document, because
        transcribed pages bring label candidates in any order.

        Args:
            ctx: The claimed job and the process dependencies.

        Raises:
            LeaseLost: The lease moved to another owner.
            JobSignal: The embedding provider failed (Gemini only); the pages done so far stay indexed.
        """
        job = ctx.job
        assert job.document_id is not None
        tenant_id, document_id = job.tenant_id, job.document_id
        indexer = self.indexer_for(ctx)
        with ctx.telemetry.span("index.document", **{"docqa.index_version": indexer.index_version}) as span:
            with ctx.pool.connection() as conn:
                doc = documents.get_document(conn, tenant_id=tenant_id, document_id=document_id)
                todo = [] if doc is None else pages.pages_to_index(conn, tenant_id=tenant_id, document_id=document_id)
            indexed = entries_written = 0
            try:
                for page in todo:
                    assert doc is not None
                    try:
                        entries = indexer.prepare(doc.filename, page)
                    except ProviderError as error:
                        raise _signal(ctx, error) from None
                    try:
                        with tx(ctx.pool) as conn:
                            queue.assert_lease(conn, job)
                            written = entries.write(conn, tenant_id=tenant_id, document_id=document_id)
                            if not pages.mark_indexed(
                                conn,
                                tenant_id=tenant_id,
                                document_id=document_id,
                                page_no=page.page_no,
                                text_rev=page.text_rev,
                                index_version=indexer.index_version,
                            ):
                                raise _TextChanged
                            documents.recompute_document(conn, tenant_id=tenant_id, document_id=document_id)
                    except _TextChanged:
                        continue
                    indexed += 1
                    entries_written += written
            finally:
                span.set("docqa.pages", indexed)
                span.set("docqa.chunks", entries_written)
                span.set("docqa.peak_rss_mb", peak_rss_mb())
            labels_changed = 0
            with tx(ctx.pool) as conn:
                queue.assert_lease(conn, job)
                if doc is not None:
                    labels_changed = refresh_labels(conn, tenant_id=tenant_id, document_id=document_id)
                    documents.recompute_document(conn, tenant_id=tenant_id, document_id=document_id)
                queue.complete(conn, job)
            span.set("docqa.labels_changed", labels_changed)

    def on_failure(self, conn: Connection, job: Job, code: str, message: str) -> None:
        """Mark failed the pages that indexing gave up on, and recompute the document.

        Args:
            conn: Connection inside the `fail` transaction.
            job: The failing job.
            code: Error code of the job; the pages get `index_failed`.
            message: Detail without customer text.
        """
        if job.document_id is None:
            return
        pages.fail_pages(
            conn,
            tenant_id=job.tenant_id,
            document_id=job.document_id,
            from_state="text_ready",
            code="index_failed",
            message=f"indexing gave up: {code}",
        )
        documents.recompute_document(conn, tenant_id=job.tenant_id, document_id=job.document_id)


def _signal(ctx: JobContext, error: ProviderError) -> JobSignal:
    """Turn an embedding provider error into the queue signal that handles it.

    Args:
        ctx: The claimed job and the process dependencies.
        error: The classified error.

    Returns:
        `Retry` for transient errors (`PermanentFailure('embed_unavailable')` on the last attempt); `Reschedule` after a
        rate limit or a pause (written here, without spending an attempt); `PermanentFailure('embed_invalid_request')`
        for a malformed request.
    """
    job = ctx.job
    now = datetime.now(UTC)
    if error.kind in TRANSIENT_KINDS:
        if job.attempts < job.max_attempts:
            return Retry(error.kind, error.message)
        return PermanentFailure("embed_unavailable", error.message)
    if error.kind == "rate_limited":
        return Reschedule("rate_limited", now + timedelta(seconds=error.retry_after_s or RATE_LIMIT_WAIT_S))
    if error.kind in pacing.PAUSE_KINDS:
        with tx(ctx.pool) as conn:
            paused = pacing.pause_for(conn, error, now=now)
        ctx.telemetry.event(
            "provider.paused",
            **{
                "gen_ai.request.model": paused.model,
                "docqa.pause.reason": paused.reason,
                "docqa.pause.until": paused.until.isoformat(),
            },
        )
        logger.warning(
            "provider paused",
            extra={"fields": {"model": paused.model, "reason": paused.reason, "until": paused.until.isoformat()}},
        )
        return Reschedule(paused.reason, paused.until)
    if error.kind == "invalid_request":
        return PermanentFailure("embed_invalid_request", error.message)
    return Retry("internal_error", f"unknown provider error kind {error.kind}")
