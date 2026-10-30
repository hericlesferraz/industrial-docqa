"""Answer records (DECISIONS #11): what a question saw, what the model returned and what the client got.

One row per question that passed validation, 5xx outcomes included, with the top 30 candidates of the fused search.
The record and the request's stage spans are written in one transaction before the HTTP response is sent, so a read
right after the answer is complete.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from uuid import UUID

from psycopg import Connection
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from docqa.retrieval.search import TOP_PROMPT, Candidate

INSERT_ANSWER = (
    "INSERT INTO answers (id, tenant_id, question, search_query, document_ids, http_status, status, reason, error_code,"
    " model, prompt_version, assembly_version, index_version, search_version, coverage, labels, calls, response)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
INSERT_CANDIDATE = (
    "INSERT INTO answer_candidates (answer_id, tenant_id, rank, chunk_id, document_id, page_no, chunk_ordinal,"
    " vec_rank, lex_rank, sent_to_llm) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


@dataclass(frozen=True)
class AnswerRecordIn:
    """What the pipeline records about one question.

    Attributes:
        answer_id: The request id.
        question: The question as typed.
        search_query: What the search received; equal to `question` in the MVP.
        document_ids: The scope, or None.
        http_status: Status of the HTTP response.
        status: `answered`, `ambiguous` or `not_found`; None on an HTTP error.
        reason: `unverified` for the decline, else None.
        error_code: Error code when the HTTP status is 5xx.
        model: Answer model requested.
        prompt_version: System prompt version.
        assembly_version: Passage assembly version.
        index_version: Index version searched.
        search_version: Search rule used (`SEARCH_VERSION`).
        coverage: The coverage block.
        labels: Request label to what it points to.
        calls: One entry per model call, in order.
        response: The body returned to the client.
    """

    answer_id: UUID
    question: str
    search_query: str
    document_ids: list[UUID] | None
    http_status: int
    status: str | None
    reason: str | None
    error_code: str | None
    model: str
    prompt_version: str
    assembly_version: str
    index_version: str
    search_version: str
    coverage: dict[str, object]
    labels: dict[str, object]
    calls: list[dict[str, object]]
    response: dict[str, object]


def record_answer(conn: Connection, *, tenant_id: UUID, record: AnswerRecordIn, candidates: list[Candidate]) -> None:
    """Write an answer row and its candidates, in the caller's transaction.

    Args:
        conn: Connection inside the record transaction.
        tenant_id: Owner.
        record: The answer record.
        candidates: The fused candidates, top 30; the first `TOP_PROMPT` went to the model.
    """
    conn.execute(
        INSERT_ANSWER,
        (
            record.answer_id,
            tenant_id,
            record.question,
            record.search_query,
            record.document_ids,
            record.http_status,
            record.status,
            record.reason,
            record.error_code,
            record.model,
            record.prompt_version,
            record.assembly_version,
            record.index_version,
            record.search_version,
            Jsonb(record.coverage),
            Jsonb(record.labels),
            Jsonb(record.calls),
            Jsonb(record.response),
        ),
    )
    rows = [
        (
            record.answer_id,
            tenant_id,
            candidate.rank,
            candidate.chunk_id,
            candidate.document_id,
            candidate.page_no,
            candidate.ordinal,
            candidate.vec_rank,
            candidate.lex_rank,
            candidate.rank <= TOP_PROMPT,
        )
        for candidate in candidates
    ]
    if rows:
        with conn.cursor() as cur:
            cur.executemany(INSERT_CANDIDATE, rows)


@dataclass(frozen=True)
class CandidateRow:
    """A recorded candidate, with its document's filename.

    Attributes:
        rank: 1-based fused rank.
        chunk_id: Chunk id, in `chunks` or `chunks_history`.
        document_id: Document id.
        filename: Document filename, or None if the document was deleted.
        page_no: 1-based page.
        chunk_ordinal: Chunk position within the page.
        vec_rank: Rank in the vector list, or None.
        lex_rank: Rank in the rare-term list, or None.
        sent_to_llm: Whether its units went to the model.
    """

    rank: int
    chunk_id: UUID
    document_id: UUID
    filename: str | None
    page_no: int
    chunk_ordinal: int
    vec_rank: int | None
    lex_rank: int | None
    sent_to_llm: bool


@dataclass(frozen=True)
class SpanRow:
    """A span of the question's request.

    Attributes:
        name: Span name.
        started_at: Start time.
        duration_ms: Duration.
        status: `ok` or `error`.
        error_code: Error code, if any.
        attributes: Span attributes.
    """

    name: str
    started_at: datetime
    duration_ms: float
    status: str
    error_code: str | None
    attributes: dict[str, object]


@dataclass(frozen=True)
class AnswerRecordRow:
    """An answer record as stored, for `GET /v1/answers/{id}`.

    Attributes:
        id: Answer id.
        created_at: When it was recorded.
        question: The question as typed.
        search_query: What the search received.
        document_ids: The scope, or None.
        http_status: HTTP status returned.
        status: Answer status, or None on an HTTP error.
        reason: `unverified` or None.
        error_code: Error code, or None.
        model: Answer model.
        prompt_version: System prompt version.
        assembly_version: Passage assembly version.
        index_version: Index version searched.
        search_version: Search rule used.
        coverage: Coverage block.
        labels: Label map.
        calls: Model calls.
        response: Body returned.
        candidates: Recorded candidates in rank order.
        spans: The request's spans, oldest first.
    """

    id: UUID
    created_at: datetime
    question: str
    search_query: str
    document_ids: list[UUID] | None
    http_status: int
    status: str | None
    reason: str | None
    error_code: str | None
    model: str
    prompt_version: str
    assembly_version: str
    index_version: str
    search_version: str
    coverage: dict[str, object]
    labels: dict[str, object]
    calls: list[dict[str, object]]
    response: dict[str, object]
    candidates: list[CandidateRow] = field(default_factory=list)
    spans: list[SpanRow] = field(default_factory=list)


def get_answer_record(conn: Connection, *, tenant_id: UUID, answer_id: UUID) -> AnswerRecordRow | None:
    """Load an answer record with its candidates and its request's spans.

    Args:
        conn: Any connection.
        tenant_id: Tenant that must own the answer.
        answer_id: Answer id.

    Returns:
        The record, or None when it doesn't exist or belongs to another tenant.
    """
    with conn.cursor(row_factory=class_row(AnswerRecordRow)) as cur:
        row = cur.execute(
            "SELECT id, created_at, question, search_query, document_ids, http_status, status, reason, error_code,"
            " model, prompt_version, assembly_version, index_version, search_version, coverage, labels, calls,"
            " response FROM answers WHERE tenant_id = %s AND id = %s",
            (tenant_id, answer_id),
        ).fetchone()
    if row is None:
        return None
    with conn.cursor(row_factory=class_row(CandidateRow)) as cur:
        candidates = cur.execute(
            "SELECT c.rank, c.chunk_id, c.document_id, d.filename, c.page_no, c.chunk_ordinal, c.vec_rank,"
            " c.lex_rank, c.sent_to_llm FROM answer_candidates c"
            " LEFT JOIN documents d ON d.id = c.document_id AND d.tenant_id = c.tenant_id"
            " WHERE c.tenant_id = %s AND c.answer_id = %s ORDER BY c.rank",
            (tenant_id, answer_id),
        ).fetchall()
    with conn.cursor(row_factory=class_row(SpanRow)) as cur:
        spans = cur.execute(
            "SELECT name, started_at, duration_ms, status, error_code, attributes FROM spans"
            " WHERE request_id = %s AND tenant_id = %s ORDER BY started_at, span_id",
            (answer_id, tenant_id),
        ).fetchall()
    return replace(row, candidates=candidates, spans=spans)
