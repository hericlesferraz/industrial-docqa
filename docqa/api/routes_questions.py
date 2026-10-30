"""Question routes: ask a question, and read the record of an answer."""

from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from docqa.answer.service import answer_question, public_body
from docqa.api.deps import DepsDep, TenantDep
from docqa.api.models import (
    AnswerCall,
    AnswerOutcome,
    AnswerRecord,
    AnswerResponse,
    AnswerStage,
    AnswerTotals,
    AnswerVersions,
    CandidateView,
    CoverageView,
    LabelView,
    QuestionRequest,
)
from docqa.errors import ANSWER_NOT_FOUND, ApiError, ErrorBody, request_id_of
from docqa.repo import answers
from docqa.repo.answers import AnswerRecordRow

router = APIRouter(prefix="/v1", tags=["questions"])

QUESTION_ERRORS = {
    401: {"model": ErrorBody, "description": "Missing or invalid API key."},
    404: {"model": ErrorBody, "description": "`document_not_found`: an id of `document_ids` is unknown."},
    422: {"model": ErrorBody, "description": "`invalid_request`: blank question, empty `document_ids`, ..."},
    502: {"model": ErrorBody, "description": "`llm_invalid_output`, `llm_refused` or `llm_invalid_request`."},
    503: {
        "model": ErrorBody,
        "description": "`llm_unavailable`, `llm_rate_limited` (with `Retry-After`), `llm_billing` or"
        " `llm_misconfigured`; `details.passages` holds the most relevant passages, which are not an answer.",
    },
    504: {"model": ErrorBody, "description": "`llm_timeout`."},
}


@router.post(
    "/questions",
    response_model=AnswerResponse,
    responses=QUESTION_ERRORS,
    summary="Ask a question about the tenant's documents",
)
def ask_question(body: QuestionRequest, request: Request, deps: DepsDep, tenant_id: TenantDep) -> JSONResponse:
    """Answer a question with verifiable citations, or say that the documents don't answer it.

    Args:
        body: The question and an optional scope.
        request: The current request, for its id.
        deps: Process objects.
        tenant_id: Tenant of the API key.

    Returns:
        The answer; its `answer_id` is the request id and opens the record in `GET /v1/answers/{answer_id}`.
    """
    scope = list(dict.fromkeys(body.document_ids)) if body.document_ids is not None else None
    response = answer_question(
        deps, tenant_id=tenant_id, request_id=UUID(request_id_of(request)), question=body.question, document_ids=scope
    )
    return JSONResponse(public_body(response))


def _totals(row: AnswerRecordRow) -> AnswerTotals:
    """Sum the answer calls and read the pipeline's own duration.

    Args:
        row: The stored record.

    Returns:
        Latency from the `answer.question` span (None if its spans are missing) and the tokens of the answer calls.
    """
    root = next((span for span in row.spans if span.name == "answer.question"), None)

    def total(key: str) -> int:
        """Sum one token count over the calls.

        Args:
            key: Field of the call entries.

        Returns:
            The sum, missing values counting 0.
        """
        return sum(int(str(call.get(key) or 0)) for call in row.calls)

    return AnswerTotals(
        latency_ms=round(root.duration_ms, 1) if root else None,
        model_calls=len(row.calls),
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        thinking_tokens=total("thinking_tokens"),
    )


def _labels(row: AnswerRecordRow) -> dict[str, LabelView]:
    """Render the stored label map.

    Args:
        row: The stored record.

    Returns:
        Each label and its unit.
    """
    views = {}
    for label, data in row.labels.items():
        info = dict(data) if isinstance(data, dict) else {}
        views[label] = LabelView(
            chunk_id=info["chunk_id"],
            document_id=info["document_id"],
            page=info["page_no"],
            pos=info["pos"],
            kind=info["kind"],
            text=info["md"],
        )
    return views


@router.get(
    "/answers/{answer_id}",
    response_model=AnswerRecord,
    responses={401: {"model": ErrorBody}, 404: {"model": ErrorBody, "description": "`answer_not_found`."}},
    summary="Read the record of an answer",
)
def get_answer(answer_id: UUID, deps: DepsDep, tenant_id: TenantDep) -> AnswerRecord:
    """Show everything needed to explain one answer: candidates, labels, raw model outputs, verdicts and timings.

    Args:
        answer_id: The `answer_id` of a response, or the `request_id` of an error.
        deps: Process objects.
        tenant_id: Tenant of the API key.

    Returns:
        The record.

    Raises:
        ApiError: 404 `answer_not_found`, also for another tenant's answer.
    """
    with deps.pool.connection() as conn:
        row = answers.get_answer_record(conn, tenant_id=tenant_id, answer_id=answer_id)
    if row is None:
        raise ApiError(404, ANSWER_NOT_FOUND, "Unknown answer id.")
    return AnswerRecord(
        answer_id=row.id,
        created_at=row.created_at,
        question=row.question,
        search_query=row.search_query,
        document_ids=row.document_ids,
        outcome=AnswerOutcome(
            http_status=row.http_status, status=row.status, reason=row.reason, error_code=row.error_code
        ),
        versions=AnswerVersions(
            model=row.model,
            prompt=row.prompt_version,
            assembly=row.assembly_version,
            index=row.index_version,
            search=row.search_version,
        ),
        coverage=CoverageView.model_validate(row.coverage),
        candidates=[
            CandidateView(
                rank=candidate.rank,
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                filename=candidate.filename,
                page=candidate.page_no,
                chunk_ordinal=candidate.chunk_ordinal,
                vec_rank=candidate.vec_rank,
                lex_rank=candidate.lex_rank,
                sent_to_llm=candidate.sent_to_llm,
            )
            for candidate in row.candidates
        ],
        labels=_labels(row),
        calls=[AnswerCall.model_validate(call) for call in row.calls],
        stages=[
            AnswerStage(
                name=span.name,
                started_at=span.started_at,
                duration_ms=round(span.duration_ms, 1),
                status=span.status,
                error_code=span.error_code,
                attributes=span.attributes,
            )
            for span in row.spans
        ],
        totals=_totals(row),
        response=row.response,
    )
