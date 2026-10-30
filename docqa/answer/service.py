"""The question pipeline: scope, coverage, search, the answer call, the verifier with one retry, and the record.

Steps: check the scope (404 before anything is recorded), read coverage (DECISIONS #16), embed the question
and run the hybrid search (top 30 recorded, top 8 to the model), label the units (DECISIONS #7), call the model with
the schema of DECISIONS #12, validate the output on our side, verify it with the three expansions; on a failed check,
call once more with the verifier's message, then decline as `not_found` + `unverified` with the passages (DECISIONS
#17).

Provider failures follow DECISIONS #15: one retry with a short backoff on transient errors, per-minute rate limits and
timeouts, when it fits the deadline; never on the daily quota, billing, a rejected key or model, or a malformed request.
The per-call timeout is half the deadline, so a timed-out call still leaves room for its retry. A 503 carries the most
relevant passages, marked as not an answer.

Every question that passes validation is recorded, 5xx outcomes included, with its stage spans and before the response
is sent, so `GET /v1/answers/{id}` right after is complete. Latency comes from the pipeline's own `answer.question`
span, never from the HTTP span.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from uuid import UUID

from docqa.answer.coverage import Coverage, coverage
from docqa.answer.prompt import (
    ASSEMBLY_VERSION,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    LabelInfo,
    build_passages,
    user_content,
)
from docqa.answer.schema import Ambiguous, Answered, InvalidOutput, ModelOutput, parse_output, response_schema
from docqa.answer.verify import Evidence, VerifyResult, retry_message, verify
from docqa.api.deps import Deps
from docqa.api.models import AnswerOption, AnswerResponse, Citation, CoverageView, Passage
from docqa.db import tx
from docqa.errors import (
    DOCUMENT_NOT_FOUND,
    LLM_BILLING,
    LLM_INVALID_OUTPUT,
    LLM_INVALID_REQUEST,
    LLM_MISCONFIGURED,
    LLM_RATE_LIMITED,
    LLM_REFUSED,
    LLM_TIMEOUT,
    LLM_UNAVAILABLE,
    SERVICE_UNAVAILABLE,
    ApiError,
    error_body,
)
from docqa.index.embed import index_version
from docqa.providers.llm import GenConfig, GenResult, ProviderError, Turn
from docqa.repo import answers, chunks, documents, pages
from docqa.repo.chunks import PageUnits
from docqa.retrieval.search import SEARCH_VERSION, TOP_PROMPT, TOP_RECORD, Candidate, search
from docqa.telemetry import Telemetry
from docqa.text.units import quote_form

ANSWER_DEADLINE_S = 60.0
PROVIDER_RETRY_BACKOFF_S = 1.0
ANSWER_CALL_TIMEOUT_S = (ANSWER_DEADLINE_S - PROVIDER_RETRY_BACKOFF_S) / 2
MIN_CALL_S = 2.0
ANSWER_MAX_OUTPUT_TOKENS = 4096
ANSWER_THINKING = "low"
PASSAGES_IN_ERRORS = 3
RATE_LIMIT_RETRY_AFTER_S = 30
QUOTA_RETRY_AFTER_S = 3600
UNVERIFIED_ANSWER = "The answer could not be verified against the cited passages."
STOP = "STOP"
REFUSALS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}
RETRIED_KINDS = {"transient", "rate_limited", "timeout"}
PASSAGES_NOTE = " details.passages holds the most relevant passages; they are search results, not an answer."

logger = logging.getLogger("docqa.answer")


@dataclass
class _Run:
    """State of one question while the pipeline runs; what the record is built from.

    Attributes:
        request_id: The request id, which is the answer id.
        tenant_id: Tenant asking.
        question: The question as typed.
        document_ids: The scope, or None.
        deadline: Monotonic time by which the answer must be ready.
        model: Answer model.
        index_version: Index version searched.
        coverage: Coverage of the scope, once read.
        candidates: Fused search results, top 30.
        labels: Request labels.
        calls: One entry per model call.
    """

    request_id: UUID
    tenant_id: UUID
    question: str
    document_ids: list[UUID] | None
    deadline: float
    model: str
    index_version: str
    coverage: Coverage | None = None
    candidates: list[Candidate] = field(default_factory=list)
    labels: dict[str, LabelInfo] = field(default_factory=dict)
    calls: list[dict[str, object]] = field(default_factory=list)

    def remaining(self) -> float:
        """Return the time left before the deadline.

        Returns:
            Seconds, negative once the deadline has passed.
        """
        return self.deadline - time.monotonic()


@dataclass(frozen=True)
class _Outcome:
    """How the pipeline ended: an answer body or an error.

    Attributes:
        response: The answer, when there is one.
        error: The error to raise, when there is one.
    """

    response: AnswerResponse | None = None
    error: ApiError | None = None


@dataclass(frozen=True)
class _Generated:
    """A model output that passed our validation.

    Attributes:
        output: The parsed output.
        raw: The output text, for the retry's model turn.
    """

    output: ModelOutput
    raw: str


def public_body(response: AnswerResponse) -> dict[str, object]:
    """Render an answer as returned to the client and stored in the record.

    `options`, `reason` and `passages` appear only when they apply; every other field is always present.

    Args:
        response: The answer.

    Returns:
        The JSON body.
    """
    return response.model_dump(mode="json", exclude_unset=True)


def _call_entry(
    run: _Run,
    *,
    purpose: str,
    attempt: int,
    latency_ms: float,
    result: GenResult | None = None,
    error_code: str | None = None,
) -> dict[str, object]:
    """Append the record entry of one model call.

    Args:
        run: The question's state.
        purpose: `answer` or `verifier_retry`.
        attempt: 1, or 2 after a provider failure.
        latency_ms: Wall time of the call.
        result: The call's result, when it returned one.
        error_code: Failure of the call, if any.

    Returns:
        The entry, already appended to `run.calls`; the caller fills `parse_error` and `verifier`.
    """
    entry: dict[str, object] = {
        "n": len(run.calls) + 1,
        "purpose": purpose,
        "attempt": attempt,
        "model": run.model,
        "finish_reason": result.finish_reason if result else None,
        "raw_output": result.text if result else None,
        "parse_error": None,
        "verifier": None,
        "error_code": error_code,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": result.input_tokens if result else None,
        "output_tokens": (result.output_tokens or 0) + (result.thinking_tokens or 0) if result else None,
        "thinking_tokens": (result.thinking_tokens or 0) if result else None,
    }
    run.calls.append(entry)
    return entry


def _passages(deps: Deps, run: _Run) -> list[Passage]:
    """Build the passages shown with a 503 or a decline: the best candidates, which are not an answer.

    Args:
        deps: Process objects.
        run: The question's state.

    Returns:
        Up to `PASSAGES_IN_ERRORS` passages in rank order.
    """
    best = run.candidates[:PASSAGES_IN_ERRORS]
    if not best:
        return []
    with deps.pool.connection() as conn:
        labels = pages.page_labels(
            conn, tenant_id=run.tenant_id, pairs=[(candidate.document_id, candidate.page_no) for candidate in best]
        )
    return [
        Passage(
            document_id=candidate.document_id,
            filename=candidate.filename,
            page=candidate.page_no,
            page_label=labels.get((candidate.document_id, candidate.page_no)),
            text="\n".join(quote_form(unit) for unit in candidate.units),
            transcribed=candidate.text_source == "ocr",
        )
        for candidate in best
    ]


def _provider_error(deps: Deps, run: _Run, error: ProviderError, *, with_passages: bool = True) -> ApiError:
    """Map a provider failure on the answer path to its HTTP error (DECISIONS #15).

    Args:
        deps: Process objects.
        run: The question's state.
        error: The classified provider failure, after any retry.
        with_passages: Whether the search ran, so a 503 can carry its passages.

    Returns:
        The error: 503 with passages for an unavailable, rate-limited, unpaid or misconfigured provider; 504 for a
        timeout; 502 `llm_invalid_request` for a request the provider rejected as malformed.
    """
    if error.kind == "timeout":
        return ApiError(504, LLM_TIMEOUT, "The model provider did not answer in time.", retryable=True)
    if error.kind == "invalid_request":
        return ApiError(502, LLM_INVALID_REQUEST, "The model provider rejected the request as invalid.")
    headers: dict[str, str] | None = None
    if error.kind == "rate_limited":
        wait = error.retry_after_s or RATE_LIMIT_RETRY_AFTER_S
        status, code, retryable, message = 503, LLM_RATE_LIMITED, True, "The model provider is rate limiting us."
        headers = {"Retry-After": str(math.ceil(wait))}
    elif error.kind == "quota_exhausted":
        status, code, retryable, message = 503, LLM_RATE_LIMITED, True, "The model provider's daily quota is used up."
        headers = {"Retry-After": str(QUOTA_RETRY_AFTER_S)}
    elif error.kind == "billing":
        status, code, retryable, message = 503, LLM_BILLING, False, "The model provider account has no credit."
    elif error.kind == "misconfigured":
        status, code, retryable, message = (
            503,
            LLM_MISCONFIGURED,
            False,
            "The model provider rejected the key or model.",
        )
    else:
        status, code, retryable, message = 503, LLM_UNAVAILABLE, True, "The model provider is unavailable."
    details: dict[str, object] | None = None
    if with_passages:
        details = {"passages": [passage.model_dump(mode="json") for passage in _passages(deps, run)]}
        message += PASSAGES_NOTE
    return ApiError(status, code, message, retryable=retryable, details=details, headers=headers)


def _retry_wait(error: ProviderError, attempt: int, run: _Run) -> float | None:
    """Decide whether a failed call is retried, and after how long.

    Args:
        error: The classified failure.
        attempt: The attempt that failed.
        run: The question's state, for the deadline.

    Returns:
        Seconds to wait before the retry, or None when there is no retry.
    """
    if attempt > 1 or error.kind not in RETRIED_KINDS:
        return None
    wait = PROVIDER_RETRY_BACKOFF_S
    if error.kind == "rate_limited" and error.retry_after_s:
        wait = error.retry_after_s
    return wait if run.remaining() - wait >= MIN_CALL_S else None


def _generate(
    deps: Deps, run: _Run, contents: list[Turn], config: GenConfig, *, purpose: str, stage: str
) -> GenResult | ApiError:
    """Call the answer model, retrying once on a transient failure that fits the deadline.

    Args:
        deps: Process objects.
        run: The question's state.
        contents: Request turns.
        config: Generation settings.
        purpose: `answer` or `verifier_retry`, for the record.
        stage: `answer` or `answer_retry`, for the span.

    Returns:
        The model's result, or the HTTP error to return.
    """
    if deps.llm is None:
        return _provider_error(deps, run, ProviderError("misconfigured", "no provider gateway", model=run.model))
    for attempt in (1, 2):
        remaining = run.remaining()
        if remaining < MIN_CALL_S:
            return ApiError(
                504, LLM_TIMEOUT, "The question's deadline passed before the model answered.", retryable=True
            )
        started = time.perf_counter()
        try:
            result = deps.llm.generate(
                model=run.model,
                contents=contents,
                config=config,
                timeout_s=min(ANSWER_CALL_TIMEOUT_S, remaining),
                stage=stage,
                attempt=attempt,
            )
        except ProviderError as error:
            latency_ms = (time.perf_counter() - started) * 1000
            _call_entry(run, purpose=purpose, attempt=attempt, latency_ms=latency_ms, error_code=error.kind)
            logger.warning(
                "answer model call failed",
                extra={"fields": {"model": run.model, "kind": error.kind, "status": error.status, "attempt": attempt}},
            )
            wait = _retry_wait(error, attempt, run)
            if wait is None:
                return _provider_error(deps, run, error)
            time.sleep(wait)
            continue
        _call_entry(
            run, purpose=purpose, attempt=attempt, latency_ms=(time.perf_counter() - started) * 1000, result=result
        )
        return result
    raise AssertionError("unreachable: the second attempt always returns")


def _validated(run: _Run, result: GenResult) -> _Generated | ApiError:
    """Check a model result's finish reason and validate its output on our side (DECISIONS #12).

    Args:
        run: The question's state; the last call entry gets the failure.
        result: The model's result.

    Returns:
        The parsed output, or 502 `llm_refused` for a refusal and `llm_invalid_output` for anything else.
    """
    entry = run.calls[-1]
    if result.finish_reason != STOP:
        entry["error_code"] = f"finish_{result.finish_reason.lower()}"
        if result.finish_reason in REFUSALS:
            return ApiError(502, LLM_REFUSED, "The model provider refused to answer.")
        return ApiError(502, LLM_INVALID_OUTPUT, "The model stopped before a complete answer.", retryable=True)
    try:
        output = parse_output(result.text, set(run.labels))
    except InvalidOutput as exc:
        entry["error_code"] = "invalid_output"
        entry["parse_error"] = str(exc)
        return ApiError(502, LLM_INVALID_OUTPUT, "The model's output did not match the answer schema.", retryable=True)
    return _Generated(output=output, raw=result.text or "")


def _check(
    deps: Deps, run: _Run, output: ModelOutput, page_units: dict[tuple[UUID, int], PageUnits], round_: int
) -> VerifyResult:
    """Verify an output and note the verdict on the last call entry and in a span.

    Args:
        deps: Process objects.
        run: The question's state.
        output: The parsed output.
        page_units: Units of the labels' pages.
        round_: 1 for the first output, 2 for the retry's.

    Returns:
        The verdict.
    """
    with deps.telemetry.span("answer.verify", **{"docqa.verify.round": round_}) as span:
        result = verify(output, run.labels, page_units, run.question)
        span.set("docqa.verify.passed", result.passed)
        span.set("docqa.verify.problems", len(result.problems))
        span.set("docqa.verify.added", len(result.added))
    run.calls[-1]["verifier"] = result.to_json()
    return result


def _citations(deps: Deps, run: _Run, groups: list[list[Evidence]]) -> list[list[Citation]]:
    """Turn evidence into citations: one per page, units in page order, in the order the model first cited each page.

    Args:
        deps: Process objects.
        run: The question's state.
        groups: Evidence per group (the answer, or each option, then the merged options).

    Returns:
        The citations of each group.
    """
    pairs = {(item.document_id, item.page_no) for group in groups for item in group}
    with deps.pool.connection() as conn:
        labels = pages.page_labels(conn, tenant_id=run.tenant_id, pairs=sorted(pairs))
    out = []
    for group in groups:
        by_page: dict[tuple[UUID, int], dict[int, Evidence]] = {}
        for item in group:
            by_page.setdefault((item.document_id, item.page_no), {}).setdefault(item.unit.pos, item)
        out.append(
            [
                Citation(
                    document_id=key[0],
                    filename=next(iter(units.values())).filename,
                    page=key[1],
                    page_label=labels.get(key),
                    quote="\n".join(quote_form(units[pos].unit) for pos in sorted(units)),
                    transcribed=next(iter(units.values())).text_source == "ocr",
                )
                for key, units in by_page.items()
            ]
        )
    return out


def _coverage_view(run: _Run) -> CoverageView:
    """Render the coverage block.

    Args:
        run: The question's state, with coverage read.

    Returns:
        The block.
    """
    assert run.coverage is not None
    return CoverageView.model_validate(run.coverage.to_json())


def _answer_response(deps: Deps, run: _Run, output: ModelOutput, verdict: VerifyResult) -> AnswerResponse:
    """Build the response of a verified output.

    Args:
        deps: Process objects.
        run: The question's state.
        output: The verified output.
        verdict: The verifier's result, with the evidence of each group.

    Returns:
        The answer: `answered` with its citations, `ambiguous` with options and merged citations, or `not_found`.
    """
    if isinstance(output, Answered):
        (citations,) = _citations(deps, run, [verdict.groups[0].evidence])
        return AnswerResponse(
            answer_id=run.request_id,
            status="answered",
            answer=output.answer,
            citations=citations,
            coverage=_coverage_view(run),
        )
    if isinstance(output, Ambiguous):
        groups = [group.evidence for group in verdict.groups]
        *per_option, merged = _citations(deps, run, [*groups, [item for group in groups for item in group]])
        options = [
            AnswerOption(condition=option.condition, answer=option.answer, citations=citations)
            for option, citations in zip(output.options, per_option, strict=True)
        ]
        return AnswerResponse(
            answer_id=run.request_id,
            status="ambiguous",
            answer=output.depends_on,
            citations=merged,
            options=options,
            coverage=_coverage_view(run),
        )
    return AnswerResponse(
        answer_id=run.request_id,
        status="not_found",
        answer=output.answer,
        citations=[],
        coverage=_coverage_view(run),
    )


def _decline(deps: Deps, run: _Run) -> AnswerResponse:
    """Build the decline after a second failed verification: `not_found`, `unverified`, with the passages.

    The unverified text is never returned; it stays in the record.

    Args:
        deps: Process objects.
        run: The question's state.

    Returns:
        The decline.
    """
    return AnswerResponse(
        answer_id=run.request_id,
        status="not_found",
        answer=UNVERIFIED_ANSWER,
        citations=[],
        reason="unverified",
        passages=_passages(deps, run),
        coverage=_coverage_view(run),
    )


def _retrieve(deps: Deps, run: _Run) -> ApiError | None:
    """Read coverage, embed the question and run the search.

    Args:
        deps: Process objects.
        run: The question's state; gets the coverage and the candidates.

    Returns:
        An error when the embedder is missing or its provider failed, else None.
    """
    telemetry = deps.telemetry
    with deps.pool.connection() as conn, telemetry.span("answer.coverage") as span:
        run.coverage = coverage(conn, tenant_id=run.tenant_id, document_ids=run.document_ids)
        span.set("docqa.coverage.complete", run.coverage.complete)
        span.set("docqa.documents", run.coverage.documents)
        span.set("docqa.pending_work", run.coverage.pending_work)
    if deps.embedder is None:
        return ApiError(503, SERVICE_UNAVAILABLE, "The embedder is not loaded.", retryable=True)
    with telemetry.span("answer.embed_query"):
        try:
            vector = deps.embedder.embed_query(run.question)
        except ProviderError as error:
            return _provider_error(deps, run, error, with_passages=False)
    with deps.pool.connection() as conn, telemetry.span("answer.search") as span:
        run.candidates = search(
            conn,
            tenant_id=run.tenant_id,
            query=run.question,
            query_vec=vector,
            index_version=run.index_version,
            document_ids=run.document_ids,
            top=TOP_RECORD,
            telemetry=telemetry,
        )
        span.set("docqa.candidates", len(run.candidates))
    return None


def _pipeline(deps: Deps, run: _Run) -> _Outcome:
    """Run the question from coverage to a verified answer, a decline or an error.

    Args:
        deps: Process objects.
        run: The question's state.

    Returns:
        The outcome.
    """
    failure = _retrieve(deps, run)
    if failure is not None:
        return _Outcome(error=failure)
    assert run.coverage is not None
    with deps.telemetry.span("answer.prompt") as span:
        passages, run.labels = build_passages(run.candidates[:TOP_PROMPT])
        question_turn = user_content(run.question, passages, pending_work=run.coverage.pending_work)
        config = GenConfig(
            system_instruction=SYSTEM_PROMPT,
            response_json_schema=response_schema(list(run.labels)),
            thinking_level=ANSWER_THINKING,
            max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS,
        )
        span.set("docqa.labels", len(run.labels))
        span.set("docqa.prompt_tokens_est", (len(SYSTEM_PROMPT) + len(question_turn)) // 4)
    contents = [Turn(role="user", parts=[question_turn])]
    result = _generate(deps, run, contents, config, purpose="answer", stage="answer")
    if isinstance(result, ApiError):
        return _Outcome(error=result)
    first = _validated(run, result)
    if isinstance(first, ApiError):
        return _Outcome(error=first)
    with deps.pool.connection() as conn:
        page_units = chunks.page_units(
            conn,
            tenant_id=run.tenant_id,
            pairs=sorted({(info.document_id, info.page_no) for info in run.labels.values()}),
            index_version=run.index_version,
        )
    verdict = _check(deps, run, first.output, page_units, 1)
    if verdict.passed:
        return _Outcome(response=_answer_response(deps, run, first.output, verdict))
    retry = [
        *contents,
        Turn(role="model", parts=[first.raw]),
        Turn(role="user", parts=[retry_message(verdict.problems)]),
    ]
    if run.remaining() < MIN_CALL_S:
        return _Outcome(response=_decline(deps, run))
    result = _generate(deps, run, retry, config, purpose="verifier_retry", stage="answer_retry")
    if isinstance(result, ApiError):
        return _Outcome(error=result)
    second = _validated(run, result)
    if isinstance(second, ApiError):
        return _Outcome(response=_decline(deps, run))
    verdict = _check(deps, run, second.output, page_units, 2)
    if verdict.passed:
        return _Outcome(response=_answer_response(deps, run, second.output, verdict))
    return _Outcome(response=_decline(deps, run))


def _record(deps: Deps, run: _Run, outcome: _Outcome, body: dict[str, object], http_status: int) -> None:
    """Write the answer record and the request's stage spans in one transaction.

    Args:
        deps: Process objects.
        run: The question's state.
        outcome: How the pipeline ended.
        body: The body returned to the client.
        http_status: The HTTP status returned.
    """
    response = outcome.response
    record = answers.AnswerRecordIn(
        answer_id=run.request_id,
        question=run.question,
        search_query=run.question,
        document_ids=run.document_ids,
        http_status=http_status,
        status=response.status if response else None,
        reason=response.reason if response else None,
        error_code=outcome.error.code if outcome.error else None,
        model=run.model,
        prompt_version=PROMPT_VERSION,
        assembly_version=ASSEMBLY_VERSION,
        index_version=run.index_version,
        search_version=SEARCH_VERSION,
        coverage=run.coverage.to_json() if run.coverage else {},
        labels={label: info.to_json() for label, info in run.labels.items()},
        calls=run.calls,
        response=body,
    )
    with deps.telemetry.span("answer.record"), tx(deps.pool) as conn:
        answers.record_answer(conn, tenant_id=run.tenant_id, record=record, candidates=run.candidates)
        deps.telemetry.flush(conn)


def _check_scope(deps: Deps, tenant_id: UUID, document_ids: list[UUID] | None) -> None:
    """Check that every id of the scope is a document of the tenant (DECISIONS #14).

    Args:
        deps: Process objects.
        tenant_id: Tenant asking.
        document_ids: The scope, or None.

    Raises:
        ApiError: 404 `document_not_found` with the missing ids, the same for another tenant's id and an unknown one.
    """
    if document_ids is None:
        return
    with deps.pool.connection() as conn, deps.telemetry.span("answer.scope") as span:
        missing = documents.missing_documents(conn, tenant_id=tenant_id, document_ids=document_ids)
        span.set("docqa.documents", len(document_ids))
        if missing:
            span.error(DOCUMENT_NOT_FOUND)
            raise ApiError(
                404,
                DOCUMENT_NOT_FOUND,
                "Unknown document ids.",
                details={"document_ids": [str(document_id) for document_id in missing]},
            )


def answer_question(
    deps: Deps, *, tenant_id: UUID, request_id: UUID, question: str, document_ids: list[UUID] | None
) -> AnswerResponse:
    """Answer a question with verified citations, record it, and return the answer.

    Args:
        deps: Process objects.
        tenant_id: Tenant asking.
        request_id: The request id, which becomes the answer id.
        question: The question as typed.
        document_ids: Optional scope, without duplicates.

    Returns:
        The answer.

    Raises:
        ApiError: 404 for an unknown document in the scope (not recorded); after recording, 502, 503 or 504 for a
            provider failure (DECISIONS #15).
    """
    _check_scope(deps, tenant_id, document_ids)
    run = _Run(
        request_id=request_id,
        tenant_id=tenant_id,
        question=question,
        document_ids=document_ids,
        deadline=time.monotonic() + ANSWER_DEADLINE_S,
        model=deps.settings.gemini_answer_model,
        index_version=index_version(deps.embedder) if deps.embedder is not None else "none",
    )
    telemetry: Telemetry = deps.telemetry
    with telemetry.span("answer.question", **{"gen_ai.request.model": run.model}) as span:
        outcome = _pipeline(deps, run)
        if outcome.response is not None:
            span.set("docqa.answer.status", outcome.response.status)
            span.set("http.response.status_code", 200)
            if outcome.response.reason:
                span.set("docqa.answer.reason", outcome.response.reason)
        elif outcome.error is not None:
            span.set("http.response.status_code", outcome.error.status)
            span.error(outcome.error.code)
        span.set("docqa.model_calls", len(run.calls))
    if outcome.error is not None:
        error = outcome.error
        body = error_body(error.code, error.message, str(request_id), retryable=error.retryable, details=error.details)
        _record(deps, run, outcome, body, error.status)
        raise error
    assert outcome.response is not None
    _record(deps, run, outcome, public_body(outcome.response), 200)
    return outcome.response
