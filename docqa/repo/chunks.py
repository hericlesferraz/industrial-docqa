"""Chunks of a tenant's pages: written by the index, archived to history on reindex, read back for their units."""

from dataclasses import dataclass
from uuid import UUID

import numpy as np
from psycopg import Connection
from psycopg.types.json import Jsonb

from docqa.index.chunking import Chunk
from docqa.text.units import Unit

ARCHIVE_PAGE = (
    "WITH moved AS (DELETE FROM chunks WHERE tenant_id = %s AND document_id = %s AND page_no = %s"
    " RETURNING id, tenant_id, document_id, page_no, ordinal, text_rev, text_source, index_version, text, units)"
    " INSERT INTO chunks_history (id, tenant_id, document_id, page_no, ordinal, text_rev, text_source,"
    " index_version, text, units) SELECT * FROM moved ON CONFLICT (id) DO NOTHING"
)
INSERT_CHUNK = (
    "INSERT INTO chunks (tenant_id, document_id, page_no, ordinal, text_rev, text_source, index_version, text, units,"
    " embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)"
)


def vector_literal(vector: np.ndarray) -> str:
    """Write a vector in pgvector's text form, so no type adapter has to be registered on the connections.

    Args:
        vector: One vector.

    Returns:
        `[x1,x2,...]` with enough digits to round-trip a float32.
    """
    return "[" + ",".join(f"{value:.9g}" for value in np.asarray(vector, dtype=np.float32).tolist()) + "]"


def replace_page_chunks(
    conn: Connection,
    *,
    tenant_id: UUID,
    document_id: UUID,
    page_no: int,
    text_rev: int,
    text_source: str,
    index_version: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
) -> None:
    """Archive a page's current chunks to `chunks_history`, same ids and no vector, then insert the new ones.

    Args:
        conn: Connection inside the page's index transaction.
        tenant_id: Owner.
        document_id: Document id.
        page_no: Page.
        text_rev: Page text revision the chunks came from.
        text_source: `layer` or `ocr`.
        index_version: Version of the chunker and embedder.
        chunks: The page's new chunks.
        vectors: One normalized vector per chunk.

    Raises:
        ValueError: The number of vectors differs from the number of chunks.
    """
    if len(vectors) != len(chunks):
        raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
    conn.execute(ARCHIVE_PAGE, (tenant_id, document_id, page_no))
    rows = [
        (
            tenant_id,
            document_id,
            page_no,
            chunk.ordinal,
            text_rev,
            text_source,
            index_version,
            chunk.text,
            Jsonb([unit.to_json() for unit in chunk.units]),
            vector_literal(vector),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(INSERT_CHUNK, rows)


@dataclass(frozen=True)
class PageUnits:
    """The units of a page, gathered from its current chunks.

    Attributes:
        document_id: Document id.
        page_no: Page.
        text_rev: Page text revision the chunks came from.
        text_source: `layer` or `ocr`.
        units: Units in page order, one per position.
    """

    document_id: UUID
    page_no: int
    text_rev: int
    text_source: str
    units: list[Unit]


def page_units(
    conn: Connection, *, tenant_id: UUID, pairs: list[tuple[UUID, int]], index_version: str
) -> dict[tuple[UUID, int], PageUnits]:
    """Read the units of some pages from their current chunks, deduplicated by position.

    A table header repeated in every piece of a split table appears once.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        pairs: `(document_id, page_no)` of the pages.
        index_version: Only chunks of this version count.

    Returns:
        The pages that have chunks, by `(document_id, page_no)`.
    """
    if not pairs:
        return {}
    rows = conn.execute(
        "SELECT c.document_id, c.page_no, c.text_rev, c.text_source, c.units FROM chunks c"
        " JOIN unnest(%s::uuid[], %s::int[]) AS p(document_id, page_no)"
        " ON p.document_id = c.document_id AND p.page_no = c.page_no"
        " WHERE c.tenant_id = %s AND c.index_version = %s ORDER BY c.document_id, c.page_no, c.ordinal",
        ([document for document, _ in pairs], [page for _, page in pairs], tenant_id, index_version),
    ).fetchall()
    found: dict[tuple[UUID, int], dict[int, Unit]] = {}
    meta: dict[tuple[UUID, int], tuple[int, str]] = {}
    for row in rows:
        key = (row["document_id"], row["page_no"])
        meta[key] = (row["text_rev"], row["text_source"])
        by_pos = found.setdefault(key, {})
        for data in row["units"]:
            unit = Unit.from_json(data)
            by_pos.setdefault(unit.pos, unit)
    return {
        key: PageUnits(
            document_id=key[0],
            page_no=key[1],
            text_rev=meta[key][0],
            text_source=meta[key][1],
            units=[by_pos[pos] for pos in sorted(by_pos)],
        )
        for key, by_pos in found.items()
    }
