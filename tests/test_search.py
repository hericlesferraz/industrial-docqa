"""Hybrid search: tenant isolation, scope, tenant-wide rarity, RRF order and the index-version filter."""

from uuid import UUID, uuid4

import pytest

from docqa.index.embed import index_version
from docqa.retrieval.search import (
    RARE_MIN_DF,
    Candidate,
    _Listed,
    is_rare,
    lexeme_counts,
    rare_max_df,
    rare_terms,
    rrf,
    search,
)
from docqa.retrieval.stopwords import STOPWORDS
from docqa.telemetry import MemoryTelemetry
from docqa.worker import WorkerDeps
from tests.helpers import new_tenant
from tests.index_helpers import FakeEmbedder, index_all, make_document, with_embedder

VERSION = index_version(FakeEmbedder())
GEAR = [
    "# **Gear unit**\n\nChange the oil of the gear unit every 10000 hours. Use synthetic oil ISO VG 220.",
    "# **Warranty**\n\nThe gear unit warranty covers 12 months. Plug HN10-WD is the breather.",
]
MOTOR = [
    "# **Motor**\n\nRegrease the motor bearings every 9500 hours. The W3Seal protects the bearing.",
    "# **Warranty**\n\nThe motor warranty covers 18 months from shipment.",
    "# **Seal**\n\nCommon words here: motor bearings oil.",
]


def _corpus(deps: WorkerDeps) -> tuple[UUID, UUID, UUID]:
    """Index a gear-unit manual and a motor manual in one tenant.

    Args:
        deps: Worker dependencies with an embedder.

    Returns:
        The tenant, the gear manual and the motor manual.
    """
    tenant_id = new_tenant(deps.pool)
    gear = make_document(deps.pool, tenant_id, GEAR, filename="gear.pdf")
    motor = make_document(deps.pool, tenant_id, MOTOR, filename="motor.pdf")
    index_all(deps)
    return tenant_id, gear, motor


def _search(deps: WorkerDeps, tenant_id: UUID, query: str, **kwargs: object) -> list[Candidate]:
    """Run a search with the fake embedder's question vector.

    Args:
        deps: Worker dependencies.
        tenant_id: Tenant.
        query: Question.
        **kwargs: More arguments of `search`.

    Returns:
        The candidates.
    """
    embedder = FakeEmbedder()
    with deps.pool.connection() as conn:
        return search(
            conn,
            tenant_id=tenant_id,
            query=query,
            query_vec=embedder.embed_query(query),
            index_version=str(kwargs.pop("index_version", VERSION)),
            **kwargs,
        )


def test_search_returns_ranked_candidates_with_units(worker_deps: WorkerDeps) -> None:
    """Candidates carry rank, page, filename, revision, source, units and both list ranks.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_id, gear, _ = _corpus(deps)
    found = _search(deps, tenant_id, "How often do I change the gear unit oil?")
    assert [c.rank for c in found] == list(range(1, len(found) + 1))
    top = found[0]
    assert (top.document_id, top.page_no, top.filename) == (gear, 1, "gear.pdf")
    assert (top.text_rev, top.text_source, top.ordinal) == (1, "layer", 0)
    assert top.units[0].md == "# **Gear unit**"
    assert top.vec_rank == 1


def test_needle_code_comes_first_through_rare_terms(worker_deps: WorkerDeps) -> None:
    """A code that occurs in one chunk is a rare term; the keyword list puts its chunk first.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_id, gear, motor = _corpus(deps)
    with deps.pool.connection() as conn:
        assert rare_terms(conn, tenant_id=tenant_id, index_version=VERSION, query="W3Seal") == ["w3seal"]
    top = _search(deps, tenant_id, "W3Seal")[0]
    assert (top.document_id, top.page_no, top.lex_rank) == (motor, 1, 1)
    plug = _search(deps, tenant_id, "o que é o bujão HN10-WD?")[0]
    assert (plug.document_id, plug.page_no) == (gear, 2)


def test_rarity_is_counted_over_the_whole_tenant(worker_deps: WorkerDeps) -> None:
    """A word in more than 3 chunks of the tenant is not rare even if the scope holds only two of them.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_id, gear, _ = _corpus(deps)
    with deps.pool.connection() as conn:
        counts = dict(lexeme_counts(conn, tenant_id=tenant_id, index_version=VERSION, query="warranty the oil"))
        assert counts == {"warranty": 2, "the": RARE_MIN_DF + 1, "oil": 2}
        assert rare_terms(conn, tenant_id=tenant_id, index_version=VERSION, query="the") == []
    scoped = _search(deps, tenant_id, "the", document_ids=[gear])
    assert scoped
    assert all(c.lex_rank is None for c in scoped)


def test_scope_keeps_only_the_given_documents(worker_deps: WorkerDeps) -> None:
    """Both lists get the document filter: a scoped question never sees another manual's chunks.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_id, gear, motor = _corpus(deps)
    unscoped = _search(deps, tenant_id, "warranty months")
    assert {c.document_id for c in unscoped} == {gear, motor}
    scoped = _search(deps, tenant_id, "warranty months W3Seal", document_ids=[gear])
    assert {c.document_id for c in scoped} == {gear}
    assert _search(deps, tenant_id, "warranty", document_ids=[uuid4()]) == []


def test_other_tenants_and_versions_are_invisible(worker_deps: WorkerDeps) -> None:
    """A tenant only finds its own chunks, and only those of the current index version.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_a, _, _ = _corpus(deps)
    tenant_b = new_tenant(deps.pool)
    other = make_document(deps.pool, tenant_b, ["# **Secret**\n\nTenant B part code ZX9000."], filename="b.pdf")
    index_all(deps)
    assert _search(deps, tenant_a, "ZX9000 secret")[0].document_id != other
    assert all(c.document_id != other for c in _search(deps, tenant_a, "ZX9000 secret"))
    assert _search(deps, tenant_b, "ZX9000")[0].document_id == other
    assert _search(deps, tenant_a, "warranty", index_version="win250-g4|another@1") == []


def _listed(n: int) -> _Listed:
    """Build a list entry whose tie-break keys follow its number.

    Args:
        n: Entry number.

    Returns:
        The entry.
    """
    return _Listed(chunk_id=UUID(int=n), document_id=UUID(int=0), page_no=n, ordinal=0)


def test_rrf_scores_and_breaks_ties_deterministically() -> None:
    """Scores add over both lists; equal scores break on the best rank, then document, page and position."""
    a, b, c, d = (_listed(n) for n in range(1, 5))
    fused = rrf([a, b, c], [c, d])
    assert [item.chunk_id for item, _, _ in fused] == [c.chunk_id, a.chunk_id, b.chunk_id, d.chunk_id]
    assert [(v, x) for _, v, x in fused] == [(3, 1), (1, None), (2, None), (None, 2)]
    assert rrf([b], [a]) == [(a, None, 1), (b, 1, None)]
    assert rrf([], []) == []


def test_search_records_its_spans(worker_deps: WorkerDeps) -> None:
    """The vector, lexical and fusion steps each record a span with their sizes, never the question.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = with_embedder(worker_deps)
    tenant_id, _, _ = _corpus(deps)
    telemetry = MemoryTelemetry()
    question = "W3Seal warranty"
    with deps.pool.connection() as conn:
        found = search(
            conn,
            tenant_id=tenant_id,
            query=question,
            query_vec=FakeEmbedder().embed_query(question),
            index_version=VERSION,
            top=3,
            telemetry=telemetry,
        )
    telemetry.flush()
    assert len(found) == 3
    lexical = telemetry.named("search.lexical")[0].attributes
    assert (lexical["docqa.lexemes"], lexical["docqa.rare_terms"], lexical["docqa.rare_max_df"]) == (2, 2, 3)
    assert telemetry.named("search.vector")[0].attributes["docqa.results"] == 5
    assert telemetry.named("search.fuse")[0].attributes["docqa.candidates"] == 3
    assert all(question not in str(span.attributes) for span in telemetry.spans)


@pytest.mark.parametrize("query", ["", "   ", "?!"])
def test_query_without_lexemes_still_searches_by_vector(worker_deps: WorkerDeps, query: str) -> None:
    """A question with no words has no rare terms; the vector list alone ranks the chunks.

    Args:
        worker_deps: Worker dependencies.
        query: A question without lexemes.
    """
    deps = with_embedder(worker_deps)
    tenant_id, _, _ = _corpus(deps)
    found = _search(deps, tenant_id, query)
    assert all(c.lex_rank is None for c in found)


def test_the_rare_bound_grows_with_the_tenant() -> None:
    """3 chunks in a small tenant, 0.3% of the chunks in a large one, so a code repeated across manuals stays rare."""
    assert [rare_max_df(n) for n in (0, 535, 1333, 1334, 3095, 100_000)] == [3, 3, 3, 4, 9, 300]
    assert is_rare("m16", 9, rare_max_df(3095)) and not is_rare("m16", 9, rare_max_df(535))
    assert not is_rare("w3seal", 0, 3)


def test_function_words_are_never_rare(worker_deps: WorkerDeps) -> None:
    """A function word that occurs once in the tenant is still left out: `em` in an English corpus is not a term.

    Args:
        worker_deps: Worker dependencies.
    """
    assert {"em", "the", "der", "fur", "qual", "cuales", "quelle", "perche", "whose"} <= STOPWORDS
    assert not {"m16", "oil", "graxa", "w3seal", "ip55"} & STOPWORDS
    deps = with_embedder(worker_deps)
    tenant_id = new_tenant(deps.pool)
    make_document(deps.pool, tenant_id, ["# **Glossary**\n\nEM electromagnetic. W3Seal shaft seal."], filename="g.pdf")
    index_all(deps)
    with deps.pool.connection() as conn:
        counts = dict(lexeme_counts(conn, tenant_id=tenant_id, index_version=VERSION, query="em W3Seal"))
        assert counts == {"em": 1, "w3seal": 1}
        assert rare_terms(conn, tenant_id=tenant_id, index_version=VERSION, query="Qual é o selo em W3Seal?") == [
            "w3seal"
        ]
