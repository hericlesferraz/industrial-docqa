"""Helpers of the index and search tests: a deterministic embedder and documents indexed from given page texts."""

import hashlib
import re
from dataclasses import replace
from uuid import UUID

import numpy as np
from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.db import tx
from docqa.index.embed import EMBED_DIM, normalize
from docqa.repo import documents, pages
from docqa.repo.pages import PageInit
from docqa.worker import WorkerDeps, run_one


class FakeEmbedder:
    """Bag-of-words vectors from hashed lowercase words: texts that share words are close, no model needed.

    Attributes:
        model_id: Enters the index version.
        dim: 1024.
        provider: `fake`.
        batch_size: Reported in the `embed.chunks` span.
        calls: Number of `embed_documents` calls.
    """

    dim = EMBED_DIM
    provider = "fake"
    batch_size = 8

    def __init__(self, model_id: str = "fake-hash@1") -> None:
        """Name the model.

        Args:
            model_id: Identifier that enters the index version.
        """
        self.model_id = model_id
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        """Hash each word of a text into one of the dimensions.

        Args:
            text: Any text.

        Returns:
            The unnormalized count vector.
        """
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed chunk texts.

        Args:
            texts: Texts.

        Returns:
            Normalized vectors, one row per text.
        """
        self.calls += 1
        return normalize(np.array([self._vector(text) for text in texts]).reshape(len(texts), self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a question.

        Args:
            text: The question.

        Returns:
            Normalized vector.
        """
        return normalize(self._vector(text).reshape(1, self.dim))[0]


def make_document(
    pool: ConnectionPool, tenant_id: UUID, texts: list[str], *, filename: str = "manual.pdf", source: str = "layer"
) -> UUID:
    """Create a document whose pages are `text_ready` with the given texts, and enqueue its `index` job.

    Args:
        pool: Database pool.
        tenant_id: Owner.
        texts: Canonical text of each page, in page order.
        filename: Document filename.
        source: `layer` or `ocr`.

    Returns:
        The document id.
    """
    sha256 = hashlib.sha256(f"{tenant_id}{filename}{texts}".encode()).hexdigest()
    rows = [
        PageInit(
            page_no=n,
            state="text_ready",
            text_source="layer",
            layer_text=text,
            quality={},
            page_label=None,
            text_rev=1,
            extract_version="test",
        )
        for n, text in enumerate(texts, start=1)
    ]
    with tx(pool) as conn:
        row, _ = documents.insert_or_get_document(
            conn, tenant_id=tenant_id, sha256=sha256, filename=filename, blob_key=f"{tenant_id}/{sha256}.pdf"
        )
        pages.insert_pages(conn, tenant_id=tenant_id, document_id=row.id, pages=rows)
        if source == "ocr":
            conn.execute(
                "UPDATE pages SET text_source = 'ocr', ocr_text = layer_text WHERE document_id = %s", (row.id,)
            )
        documents.set_page_count(conn, tenant_id=tenant_id, document_id=row.id, page_count=len(rows))
        documents.recompute_document(conn, tenant_id=tenant_id, document_id=row.id)
        queue.enqueue(conn, tenant_id=tenant_id, kind="index", document_id=row.id)
    return row.id


def with_embedder(deps: WorkerDeps, embedder: object | None = None) -> WorkerDeps:
    """Give worker dependencies an embedder, so the `index` job runs the real indexer.

    Args:
        deps: Worker dependencies.
        embedder: The embedder; a new `FakeEmbedder` when None.

    Returns:
        The same dependencies with the embedder.
    """
    return replace(deps, embedder=embedder or FakeEmbedder())


def index_all(deps: WorkerDeps, limit: int = 20) -> int:
    """Run queued `index` jobs until none is runnable.

    Args:
        deps: Worker dependencies with an embedder.
        limit: Safety cap.

    Returns:
        Number of jobs run.
    """
    ran = 0
    while ran < limit and run_one(deps, "index"):
        ran += 1
    return ran
