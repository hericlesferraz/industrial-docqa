"""Hybrid search (DECISIONS #10, #14): vectors by cosine plus the question's rare terms, fused with RRF.

The vector list comes from the HNSW index with iterative scans, so a tenant or document filter still yields full lists.
The keyword list searches only the question's rare lexemes: codes and rare words keep their selectivity in any
language, while common words would add cross-language noise. A lexeme is rare when it occurs in at least one and at
most `rare_max_df` chunks of the tenant, a bound that grows with the tenant (3 chunks, or 0.3% of them when that is
more), and is not a function word of the six likely question languages. Rarity is counted over the whole tenant, never
over the scope. `SEARCH_VERSION` names the rule and goes on every answer record.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from psycopg import Connection, sql

from docqa.repo.chunks import vector_literal
from docqa.retrieval.stopwords import STOPWORDS
from docqa.telemetry import Span, Telemetry
from docqa.text.units import Unit

LIST_N = 50
TOP_PROMPT = 8
TOP_RECORD = 30
RRF_K = 60
RARE_MIN_DF = 3
RARE_SHARE = 0.003
HNSW_EF_SEARCH = 100
SEARCH_VERSION = "rrf60-df0.3pct-stop"


@dataclass(frozen=True)
class Candidate:
    """A chunk returned by the fused search.

    Attributes:
        rank: 1-based fused rank.
        chunk_id: Chunk id.
        document_id: Document of the chunk.
        filename: Document filename.
        page_no: 1-based page.
        ordinal: Chunk position within the page.
        text_rev: Page text revision the chunk came from.
        text_source: `layer` or `ocr`.
        units: Units of the chunk.
        vec_rank: Rank in the vector list, or None.
        lex_rank: Rank in the rare-term list, or None.
    """

    rank: int
    chunk_id: UUID
    document_id: UUID
    filename: str
    page_no: int
    ordinal: int
    text_rev: int
    text_source: str
    units: list[Unit]
    vec_rank: int | None
    lex_rank: int | None


@dataclass(frozen=True)
class _Listed:
    """A chunk in one of the two ranked lists, with the keys that break ties.

    Attributes:
        chunk_id: Chunk id.
        document_id: Document id.
        page_no: Page.
        ordinal: Chunk position within the page.
    """

    chunk_id: UUID
    document_id: UUID
    page_no: int
    ordinal: int


def _scope_filter(document_ids: list[UUID] | None) -> sql.Composable:
    """Build the optional document filter of both searches.

    Args:
        document_ids: The scope, or None for the whole tenant.

    Returns:
        An empty fragment, or `AND document_id = ANY(%(documents)s)`.
    """
    if document_ids is None:
        return sql.SQL("")
    return sql.SQL(" AND document_id = ANY(%(documents)s)")


def vector_list(
    conn: Connection,
    *,
    tenant_id: UUID,
    query_vec: np.ndarray,
    index_version: str,
    document_ids: list[UUID] | None = None,
    limit: int = LIST_N,
) -> list[_Listed]:
    """Rank chunks by cosine distance to the question vector.

    Runs in its own transaction so `SET LOCAL` keeps the HNSW settings to this query: a wider candidate list and
    iterative scans in strict order, which keep scanning until enough rows pass the tenant and scope filters.

    Args:
        conn: Connection outside any transaction.
        tenant_id: Tenant whose chunks are searched.
        query_vec: Normalized question vector.
        index_version: Only chunks of this version are searched.
        document_ids: Optional scope.
        limit: List size.

    Returns:
        Chunks in rank order.
    """
    query = sql.SQL(
        "SELECT id, document_id, page_no, ordinal FROM chunks WHERE tenant_id = %(tenant)s"
        " AND index_version = %(version)s{scope} ORDER BY embedding <=> %(vec)s::vector LIMIT %(limit)s"
    ).format(scope=_scope_filter(document_ids))
    params = {
        "tenant": tenant_id,
        "version": index_version,
        "documents": document_ids,
        "vec": vector_literal(query_vec),
        "limit": limit,
    }
    with conn.transaction():
        conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(HNSW_EF_SEARCH)))
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")
        rows = conn.execute(query, params).fetchall()
    return [_Listed(row["id"], row["document_id"], row["page_no"], row["ordinal"]) for row in rows]


def rare_max_df(chunk_count: int) -> int:
    """Return the highest chunk count of a rare term in a tenant of a given size.

    Args:
        chunk_count: The tenant's chunks of the searched index version.

    Returns:
        `RARE_MIN_DF`, or `RARE_SHARE` of the chunks when that is more.
    """
    return max(RARE_MIN_DF, math.floor(RARE_SHARE * chunk_count))


def tenant_chunk_count(conn: Connection, *, tenant_id: UUID, index_version: str) -> int:
    """Count the tenant's chunks of an index version (an index-only scan of `chunks_tenant_version`).

    Args:
        conn: Any connection.
        tenant_id: The tenant.
        index_version: The searched version.

    Returns:
        The count.
    """
    row = conn.execute(
        "SELECT count(*) AS n FROM chunks WHERE tenant_id = %s AND index_version = %s", (tenant_id, index_version)
    ).fetchone()
    return int(row["n"]) if row else 0


def is_rare(lexeme: str, df: int, max_df: int) -> bool:
    """Tell whether a question lexeme goes to the keyword list.

    Args:
        lexeme: Unaccented, lowercased lexeme.
        df: Chunks of the tenant that contain it (capped above `max_df`).
        max_df: The tenant's bound, from `rare_max_df`.

    Returns:
        True when it occurs in 1 to `max_df` chunks and is not a function word.
    """
    return 0 < df <= max_df and lexeme not in STOPWORDS


def lexeme_counts(
    conn: Connection, *, tenant_id: UUID, index_version: str, query: str, cap: int = RARE_MIN_DF + 1
) -> list[tuple[str, int]]:
    """Count, for each lexeme of the question, the tenant's chunks that contain it, up to `cap`.

    Each count is a `LIMIT cap` lookup on the full-text index, so a common word costs little more than a rare one.

    Args:
        conn: Any connection.
        tenant_id: Rarity is counted over the whole tenant, never the scope.
        index_version: Only chunks of this version count.
        query: Question text.
        cap: Highest count reported; one more than the rare bound is enough.

    Returns:
        `(lexeme, count)` for every lexeme of the question, unaccented and lowercased, in lexeme order.
    """
    rows = conn.execute(
        "SELECT q.lexeme, d.df FROM unnest(tsvector_to_array(to_tsvector('simple', docqa_unaccent(%(query)s))))"
        " AS q(lexeme), LATERAL (SELECT count(*) AS df FROM (SELECT 1 FROM chunks WHERE tenant_id = %(tenant)s"
        " AND index_version = %(version)s AND tsv @@ to_tsquery('simple', quote_literal(q.lexeme))"
        " LIMIT %(cap)s) hits) d ORDER BY q.lexeme",
        {"query": query, "tenant": tenant_id, "version": index_version, "cap": cap},
    ).fetchall()
    return [(row["lexeme"], int(row["df"])) for row in rows]


def rare_terms(
    conn: Connection, *, tenant_id: UUID, index_version: str, query: str, max_df: int | None = None
) -> list[str]:
    """Return the query lexemes that are rare in the tenant (`is_rare`).

    Args:
        conn: Any connection.
        tenant_id: Rarity is counted over the whole tenant, never the scope.
        index_version: Only chunks of this version count.
        query: Question text.
        max_df: Highest chunk count for a rare term; by default `rare_max_df` of the tenant's size.

    Returns:
        The rare lexemes, unaccented and lowercased by `to_tsvector('simple', ...)`.
    """
    if max_df is None:
        max_df = rare_max_df(tenant_chunk_count(conn, tenant_id=tenant_id, index_version=index_version))
    counts = lexeme_counts(conn, tenant_id=tenant_id, index_version=index_version, query=query, cap=max_df + 1)
    return [lexeme for lexeme, df in counts if is_rare(lexeme, df, max_df)]


def lexical_list(
    conn: Connection,
    *,
    tenant_id: UUID,
    terms: list[str],
    index_version: str,
    document_ids: list[UUID] | None = None,
    limit: int = LIST_N,
) -> list[_Listed]:
    """Rank chunks that contain any of the rare terms with `ts_rank` (length-normalized).

    Args:
        conn: Any connection.
        tenant_id: Tenant whose chunks are searched.
        terms: Rare lexemes, OR-ed.
        index_version: Only chunks of this version are searched.
        document_ids: Optional scope.
        limit: List size.

    Returns:
        Chunks in rank order; empty without terms.
    """
    if not terms:
        return []
    query = sql.SQL(
        "SELECT id, document_id, page_no, ordinal FROM chunks,"
        " to_tsquery('simple', (SELECT string_agg(quote_literal(term), ' | ') FROM unnest(%(terms)s::text[]) term))"
        " AS q WHERE tenant_id = %(tenant)s AND index_version = %(version)s{scope} AND tsv @@ q"
        " ORDER BY ts_rank(tsv, q, 1) DESC, document_id, page_no, ordinal LIMIT %(limit)s"
    ).format(scope=_scope_filter(document_ids))
    rows = conn.execute(
        query,
        {"terms": terms, "tenant": tenant_id, "version": index_version, "documents": document_ids, "limit": limit},
    ).fetchall()
    return [_Listed(row["id"], row["document_id"], row["page_no"], row["ordinal"]) for row in rows]


def rrf(vec: list[_Listed], lex: list[_Listed], *, k: int = RRF_K) -> list[tuple[_Listed, int | None, int | None]]:
    """Fuse the two lists with reciprocal rank fusion: each list gives a chunk `1 / (k + rank)`.

    Ties break on the better of the two ranks, then on document, page and chunk position, so the order is
    deterministic.

    Args:
        vec: Vector list in rank order.
        lex: Rare-term list in rank order.
        k: RRF constant.

    Returns:
        `(chunk, vec_rank, lex_rank)` in fused order; a rank is None when the chunk is not in that list.
    """
    entries: dict[UUID, _Listed] = {}
    ranks: dict[UUID, list[int | None]] = {}
    for column, ranked in enumerate((vec, lex)):
        for rank, item in enumerate(ranked, start=1):
            entries.setdefault(item.chunk_id, item)
            ranks.setdefault(item.chunk_id, [None, None])[column] = rank

    def key(chunk_id: UUID) -> tuple[float, int, str, int, int]:
        """Sort key of a fused chunk.

        Args:
            chunk_id: The chunk.

        Returns:
            Negative score, best rank, document, page and chunk position.
        """
        own = [rank for rank in ranks[chunk_id] if rank is not None]
        item = entries[chunk_id]
        score = sum(1.0 / (k + rank) for rank in own)
        return (-score, min(own), str(item.document_id), item.page_no, item.ordinal)

    order = sorted(entries, key=key)
    return [(entries[chunk_id], ranks[chunk_id][0], ranks[chunk_id][1]) for chunk_id in order]


def _details(conn: Connection, *, tenant_id: UUID, chunk_ids: list[UUID]) -> dict[UUID, dict[str, object]]:
    """Load what a candidate carries: filename, revision, source and units.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        chunk_ids: Chunks to load.

    Returns:
        Rows by chunk id.
    """
    if not chunk_ids:
        return {}
    rows = conn.execute(
        "SELECT c.id, c.document_id, d.filename, c.page_no, c.ordinal, c.text_rev, c.text_source, c.units"
        " FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE c.tenant_id = %s AND d.tenant_id = %s AND c.id = ANY(%s)",
        (tenant_id, tenant_id, chunk_ids),
    ).fetchall()
    return {row["id"]: row for row in rows}


@contextmanager
def _maybe_span(telemetry: Telemetry | None, name: str) -> Iterator[Span | None]:
    """Open a span when telemetry is given.

    Args:
        telemetry: Where spans go, or None.
        name: Span name.

    Yields:
        The span, or None.
    """
    if telemetry is None:
        yield None
        return
    with telemetry.span(name) as span:
        yield span


def search(
    conn: Connection,
    *,
    tenant_id: UUID,
    query: str,
    query_vec: np.ndarray,
    index_version: str,
    document_ids: list[UUID] | None = None,
    top: int = TOP_RECORD,
    telemetry: Telemetry | None = None,
) -> list[Candidate]:
    """Run the vector and rare-term searches and fuse them with RRF.

    Args:
        conn: Connection outside any transaction; the vector query sets `hnsw.*` with `SET LOCAL` in its own one.
        tenant_id: Tenant whose chunks are searched.
        query: Question text, for the rare terms.
        query_vec: Question vector.
        index_version: Only chunks of this version are searched.
        document_ids: Optional scope; both searches get it, rarity is still counted over the tenant.
        top: Number of fused candidates to return.
        telemetry: Records `search.vector`, `search.lexical` (with the tenant's rare bound) and `search.fuse` when
            given.

    Returns:
        Candidates ordered by fused rank.
    """
    with _maybe_span(telemetry, "search.vector") as span:
        vec = vector_list(
            conn, tenant_id=tenant_id, query_vec=query_vec, index_version=index_version, document_ids=document_ids
        )
        if span is not None:
            span.set("docqa.results", len(vec))
    with _maybe_span(telemetry, "search.lexical") as span:
        max_df = rare_max_df(tenant_chunk_count(conn, tenant_id=tenant_id, index_version=index_version))
        counts = lexeme_counts(conn, tenant_id=tenant_id, index_version=index_version, query=query, cap=max_df + 1)
        terms = [lexeme for lexeme, df in counts if is_rare(lexeme, df, max_df)]
        lex = lexical_list(
            conn, tenant_id=tenant_id, terms=terms, index_version=index_version, document_ids=document_ids
        )
        if span is not None:
            span.set("docqa.lexemes", len(counts))
            span.set("docqa.rare_max_df", max_df)
            span.set("docqa.rare_terms", len(terms))
            span.set("docqa.results", len(lex))
    with _maybe_span(telemetry, "search.fuse") as span:
        fused = rrf(vec, lex)[:top]
        rows = _details(conn, tenant_id=tenant_id, chunk_ids=[item.chunk_id for item, _, _ in fused])
        candidates = []
        for item, vec_rank, lex_rank in fused:
            row = rows.get(item.chunk_id)
            if row is None:
                continue
            candidates.append(
                Candidate(
                    rank=len(candidates) + 1,
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    filename=str(row["filename"]),
                    page_no=item.page_no,
                    ordinal=item.ordinal,
                    text_rev=int(str(row["text_rev"])),
                    text_source=str(row["text_source"]),
                    units=[Unit.from_json(data) for data in row["units"]],
                    vec_rank=vec_rank,
                    lex_rank=lex_rank,
                )
            )
        if span is not None:
            span.set("docqa.candidates", len(candidates))
    return candidates
