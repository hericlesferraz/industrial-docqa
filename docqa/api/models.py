"""Request and response models of the HTTP API."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints

from docqa.errors import ErrorBody

__all__ = [
    "AnswerCall",
    "AnswerOption",
    "AnswerOutcome",
    "AnswerRecord",
    "AnswerResponse",
    "AnswerStage",
    "AnswerTotals",
    "AnswerVersions",
    "BatchCounts",
    "BatchCreated",
    "BatchDocument",
    "BatchItem",
    "BatchView",
    "CandidateView",
    "Citation",
    "CoverageView",
    "DocumentDetail",
    "DocumentList",
    "DocumentSummary",
    "ErrorBody",
    "ErrorInfo",
    "HealthView",
    "IncompleteDocument",
    "LabelView",
    "OcrPause",
    "OcrUsage",
    "OcrView",
    "PageView",
    "Passage",
    "QuestionRequest",
    "TenantCreate",
    "TenantCreated",
    "UploadResult",
    "VerifierView",
]

DocumentStatus = Literal["queued", "processing", "ready", "partial", "failed"]
PageState = Literal["needs_ocr", "text_ready", "indexed", "blank", "failed"]
TextSource = Literal["layer", "ocr"]
ItemOutcome = Literal["accepted", "duplicate", "rejected"]
AnswerStatus = Literal["answered", "ambiguous", "not_found"]
QUESTION_MAX_CHARS = 2000
DOCUMENT_IDS_MAX = 100


class HealthView(BaseModel):
    """Body of `GET /healthz`.

    Attributes:
        status: `ok`, or `degraded` when a dependency the answer path needs failed its start-up check.
        checks: Status of each dependency: `database` now; `models` and `embedder` when they exist.
        reasons: Why the status is `degraded`.
    """

    status: str
    checks: dict[str, str | dict[str, str]]
    reasons: list[str]


class TenantCreate(BaseModel):
    """Body of `POST /v1/admin/tenants`.

    Attributes:
        name: Display name of the tenant.
    """

    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class TenantCreated(BaseModel):
    """A new tenant and its API key, shown only once.

    Attributes:
        tenant_id: Id of the tenant.
        name: Display name.
        api_key: Bearer key of the tenant; only its hash is stored.
    """

    tenant_id: UUID
    name: str
    api_key: str


class BatchCreated(BaseModel):
    """A new, empty batch.

    Attributes:
        batch_id: Id to send with each file, as `POST /v1/documents?batch_id=`.
        created_at: Creation time.
    """

    batch_id: UUID
    created_at: datetime


class UploadResult(BaseModel):
    """Outcome of one upload: a new document (202) or the tenant's existing one with the same bytes (200).

    Attributes:
        document_id: The document.
        filename: The document's filename; for a duplicate, the name of the first upload.
        status: Document status now.
        duplicate: True when the same bytes were already uploaded by this tenant.
    """

    document_id: UUID
    filename: str
    status: DocumentStatus
    duplicate: bool


class OcrPause(BaseModel):
    """A provider pause that holds a document's transcriptions.

    Attributes:
        model: The paused model, or `*` when the whole account is paused.
        reason: `quota_exhausted` (daily quota), `billing` (prepaid balance or spend cap) or `misconfigured`
            (key or model rejected).
        until: When the pause ends; the first call after it probes the provider again.
    """

    model: str
    reason: str
    until: datetime


class BatchDocument(BaseModel):
    """Current state of the document a batch item points to.

    Attributes:
        document_id: The document.
        status: Document status.
        page_count: Pages in the file; null until extracted.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed.
        pages_failed: Pages that failed.
        error_code: Failure code of the document, if any.
        ocr_paused: The provider pause holding this document's transcriptions, when it has pages waiting for one.
    """

    document_id: UUID
    status: DocumentStatus
    page_count: int | None
    pages_indexed: int
    pages_pending: int
    pages_failed: int
    error_code: str | None
    ocr_paused: OcrPause | None


class BatchItem(BaseModel):
    """One file sent with the batch.

    Attributes:
        filename: Name of the file as sent.
        outcome: `accepted` (new document), `duplicate` (existing document) or `rejected`.
        error_code: Why the file was rejected: `file_too_large`, `empty_file` or `not_a_pdf`.
        created_at: Upload time.
        document: The document, null for rejected files.
    """

    filename: str
    outcome: ItemOutcome
    error_code: str | None
    created_at: datetime
    document: BatchDocument | None


class BatchCounts(BaseModel):
    """Counts of a batch.

    Attributes:
        items: Files sent with the batch.
        accepted: New documents.
        duplicate: Files that matched an existing document.
        rejected: Files rejected at upload.
        documents: Distinct documents of the batch by status.
    """

    items: int
    accepted: int
    duplicate: int
    rejected: int
    documents: dict[str, int]


class BatchView(BaseModel):
    """A batch with every item and its document's status.

    Attributes:
        batch_id: Batch id.
        created_at: Creation time.
        complete: True when every item is rejected or its document is `ready`, `partial` or `failed`.
        counts: Counts by outcome and by document status.
        items: Items in upload order.
    """

    batch_id: UUID
    created_at: datetime
    complete: bool
    counts: BatchCounts
    items: list[BatchItem]


class DocumentSummary(BaseModel):
    """A document in the list.

    Attributes:
        document_id: The document.
        filename: Name given at upload.
        status: Document status.
        page_count: Pages in the file; null until extracted.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed.
        pages_failed: Pages that failed.
        error_code: Failure code of the document, if any.
        created_at: Upload time.
    """

    document_id: UUID
    filename: str
    status: DocumentStatus
    page_count: int | None
    pages_indexed: int
    pages_pending: int
    pages_failed: int
    error_code: str | None
    created_at: datetime


class DocumentList(BaseModel):
    """A page of the tenant's documents, newest first.

    Attributes:
        documents: The documents of this page.
        total: The tenant's total number of documents.
    """

    documents: list[DocumentSummary]
    total: int


class ErrorInfo(BaseModel):
    """A stored failure of a document or a page.

    Attributes:
        code: Error code.
        message: Detail.
    """

    code: str
    message: str | None


class PageView(BaseModel):
    """One page of a document.

    Attributes:
        page: 1-based position in the file.
        state: `needs_ocr`, `text_ready`, `indexed`, `blank` or `failed`.
        text_source: `layer` (the PDF's text) or `ocr` (transcribed from the page image); null without text.
        page_label: Number printed on the page, when confirmed by its neighbours.
        flag_reason: Why the page was sent to OCR: `garbled:<reasons>`, `no_text_layer` or `layer_error`.
        error: The page's failure, if any.
    """

    page: int
    state: PageState
    text_source: TextSource | None
    page_label: str | None
    flag_reason: str | None
    error: ErrorInfo | None


class OcrUsage(BaseModel):
    """OCR calls to one model.

    Attributes:
        calls: Calls made, failed ones included.
        input_tokens: Prompt tokens, images included.
        output_tokens: Output tokens, thinking included (billed as output).
    """

    calls: int
    input_tokens: int
    output_tokens: int


class OcrView(BaseModel):
    """What happened to the pages sent to OCR.

    Attributes:
        pages_flagged: Pages the detector sent to OCR.
        pages_transcribed: Pages whose text is a transcription.
        pages_from_cache: Transcribed pages copied from an earlier transcription of the same file (no model call).
        recitations: `RECITATION` refusals before the transcribed pages passed.
        fallback_used: Transcribed pages that needed the fallback model.
        calls: Model calls made for this document's pages.
        input_tokens: Prompt tokens of those calls.
        output_tokens: Output tokens of those calls, thinking included.
        by_model: Calls and tokens per model, for cost.
        paused: The provider pause holding pages that still wait for OCR, if any.
    """

    pages_flagged: int
    pages_transcribed: int
    pages_from_cache: int
    recitations: int
    fallback_used: int
    calls: int
    input_tokens: int
    output_tokens: int
    by_model: dict[str, OcrUsage]
    paused: OcrPause | None


class DocumentDetail(BaseModel):
    """A document with its pages, for clients and on-call.

    Attributes:
        document_id: The document.
        filename: Name given at upload.
        status: Document status.
        error: The document's failure, if any.
        created_at: Upload time.
        updated_at: Last change.
        page_count: Pages in the file; null until extracted.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed (`needs_ocr` and `text_ready`).
        pages_failed: Pages that failed.
        pages_by_state: Pages per state.
        pages_by_source: Pages with text per source.
        ocr: Transcription counts, model calls and tokens, and the provider pause holding the document.
        pages: Every page in order.
    """

    document_id: UUID
    filename: str
    status: DocumentStatus
    error: ErrorInfo | None
    created_at: datetime
    updated_at: datetime
    page_count: int | None
    pages_indexed: int
    pages_pending: int
    pages_failed: int
    pages_by_state: dict[str, int]
    pages_by_source: dict[str, int]
    ocr: OcrView
    pages: list[PageView]


class QuestionRequest(BaseModel):
    """Body of `POST /v1/questions`.

    Attributes:
        question: The question, in any language; not blank.
        document_ids: Optional scope: only these documents of the tenant are searched (DECISIONS #14).
    """

    question: Annotated[str, StringConstraints(min_length=1, max_length=QUESTION_MAX_CHARS, pattern=r"\S")]
    document_ids: Annotated[list[UUID], Field(min_length=1, max_length=DOCUMENT_IDS_MAX)] | None = None


class Citation(BaseModel):
    """Where an answer comes from: one page and its verbatim text (DECISIONS #7, #8).

    Attributes:
        document_id: The document.
        filename: Name given at upload.
        page: 1-based position of the page in the file.
        page_label: Page number printed on the page, when confirmed; null otherwise.
        quote: The cited units of the page, in page order, one per line, verbatim in the page's language: cited rows,
            the header rows and footnotes attached to them, and attached sentences.
        transcribed: True when the page text was transcribed from its image, not read from the text layer.
    """

    document_id: UUID
    filename: str
    page: int
    page_label: str | None
    quote: str
    transcribed: bool


class AnswerOption(BaseModel):
    """One case of an ambiguous answer.

    Attributes:
        condition: The case.
        answer: The answer for that case.
        citations: Its evidence, one citation per page.
    """

    condition: str
    answer: str
    citations: list[Citation]


class Passage(BaseModel):
    """A search result shown when there is no verified answer. It is not an answer.

    Attributes:
        document_id: The document.
        filename: Name given at upload.
        page: 1-based page.
        page_label: Printed page number, or null.
        text: The retrieved chunk's units, one per line.
        transcribed: True for a transcribed page.
    """

    document_id: UUID
    filename: str
    page: int
    page_label: str | None
    text: str
    transcribed: bool


class IncompleteDocument(BaseModel):
    """A document of the scope that isn't ready.

    Attributes:
        document_id: The document.
        filename: Name given at upload.
        status: `queued`, `processing`, `partial` or `failed`.
        pages_pending: Pages still being processed; null before extraction.
        pages_failed: Pages that failed; null before extraction.
        error_code: The document's failure code, if any.
    """

    document_id: UUID
    filename: str
    status: DocumentStatus
    pages_pending: int | None
    pages_failed: int | None
    error_code: str | None


class CoverageView(BaseModel):
    """How much of the scope the search could read (DECISIONS #16).

    Attributes:
        complete: Every document in scope is `ready`. When false, a `not_found` may change once pending pages finish.
        documents: Documents in scope.
        documents_incomplete: Documents that aren't `ready`.
        pages_indexed: Pages searchable now.
        pages_pending: Pages still being processed: asking again later may help.
        pages_failed: Pages that failed: no answer will cover them.
        incomplete: Up to 20 documents that aren't ready, those still processing first.
    """

    complete: bool
    documents: int
    documents_incomplete: int
    pages_indexed: int
    pages_pending: int
    pages_failed: int
    incomplete: list[IncompleteDocument]


class AnswerResponse(BaseModel):
    """Body of a 200 from `POST /v1/questions`.

    Attributes:
        answer_id: Id of the answer record, `GET /v1/answers/{answer_id}`; also the `X-Request-ID`.
        status: `answered`, `ambiguous` (the answer depends on something the question doesn't say) or `not_found`.
        answer: The answer in the question's language; for `ambiguous`, what the answer depends on.
        citations: One per page; for `ambiguous`, the options' citations merged per page; empty for `not_found`.
        options: Only for `ambiguous`: each case with its condition, answer and citations.
        reason: Only when an answer was produced but failed verification twice: `unverified`. The status is then
            `not_found` and `answer` a fixed English sentence.
        passages: Only with `reason`: the most relevant passages, which are not an answer.
        coverage: Coverage of the scope.
    """

    answer_id: UUID
    status: AnswerStatus
    answer: str
    citations: list[Citation]
    options: list[AnswerOption] | None = None
    reason: Literal["unverified"] | None = None
    passages: list[Passage] | None = None
    coverage: CoverageView


class AnswerOutcome(BaseModel):
    """What the client got.

    Attributes:
        http_status: HTTP status of the response.
        status: Answer status; null on an HTTP error.
        reason: `unverified` for the decline, else null.
        error_code: Error code of a 5xx, else null.
    """

    http_status: int
    status: AnswerStatus | None
    reason: str | None
    error_code: str | None


class AnswerVersions(BaseModel):
    """Versions of everything that shaped the answer.

    Attributes:
        model: Answer model requested.
        prompt: System prompt version.
        assembly: Passage assembly version.
        index: Index version searched (chunker and embedding model).
        search: Search rule (fusion and rare-term bound), to compare records across a change of the rule.
    """

    model: str
    prompt: str
    assembly: str
    index: str
    search: str


class CandidateView(BaseModel):
    """A chunk of the fused search, top 30.

    Attributes:
        rank: 1-based fused rank.
        chunk_id: Chunk id.
        document_id: Document id.
        filename: Document filename; null if the document was deleted.
        page: 1-based page.
        chunk_ordinal: Chunk position within the page.
        vec_rank: Rank in the vector list, or null.
        lex_rank: Rank in the rare-term list, or null.
        sent_to_llm: Whether its units went to the model (top 8).
    """

    rank: int
    chunk_id: UUID
    document_id: UUID
    filename: str | None
    page: int
    chunk_ordinal: int
    vec_rank: int | None
    lex_rank: int | None
    sent_to_llm: bool


class LabelView(BaseModel):
    """What a request label pointed to.

    Attributes:
        chunk_id: Chunk the unit was retrieved in.
        document_id: Document id.
        page: 1-based page.
        pos: Unit position in the page.
        kind: `heading`, `text` or `row`.
        text: The unit's stored Markdown.
    """

    chunk_id: UUID
    document_id: UUID
    page: int
    pos: int
    kind: str
    text: str


class VerifierView(BaseModel):
    """The verifier's verdict on one model output.

    Attributes:
        passed: Every check passed.
        problems: What failed; the retry message lists them.
        added_labels: Uncited sentences attached because they hold a number of the answer.
    """

    passed: bool
    problems: list[str]
    added_labels: list[str]


class AnswerCall(BaseModel):
    """One call to the answer model.

    Attributes:
        n: 1-based call number.
        purpose: `answer`, or `verifier_retry` after a failed verification.
        attempt: 1, or 2 for the retry after a provider failure.
        model: Model requested.
        finish_reason: Finish reason, null when the call failed.
        raw_output: The model's text, as returned.
        parse_error: Why the output was rejected by our validation, if it was.
        verifier: The verifier's verdict, when the output was valid.
        error_code: Provider failure kind (`transient`, `timeout`, ...), `finish_<reason>` or `invalid_output`.
        latency_ms: Wall time of the call.
        input_tokens: Prompt tokens.
        output_tokens: Output tokens, thinking included (billed as output).
        thinking_tokens: The thinking share of `output_tokens`.
    """

    n: int
    purpose: str
    attempt: int
    model: str
    finish_reason: str | None
    raw_output: str | None
    parse_error: str | None
    verifier: VerifierView | None
    error_code: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    thinking_tokens: int | None


class AnswerStage(BaseModel):
    """A span of the question's request, written by the pipeline.

    Attributes:
        name: Span name, such as `answer.search` or `gen_ai.call`.
        started_at: Start time.
        duration_ms: Duration.
        status: `ok` or `error`.
        error_code: Error code, if any.
        attributes: Span attributes; never customer text.
    """

    name: str
    started_at: datetime
    duration_ms: float
    status: str
    error_code: str | None
    attributes: dict[str, object]


class AnswerTotals(BaseModel):
    """Totals of one question, measured by the pipeline itself.

    Attributes:
        latency_ms: Wall time of the pipeline, from the scope check to the response (the `answer.question` span).
        model_calls: Calls to the answer model.
        input_tokens: Prompt tokens of the answer calls.
        output_tokens: Output tokens of the answer calls, thinking included.
        thinking_tokens: The thinking share of `output_tokens`.
    """

    latency_ms: float | None
    model_calls: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int


class AnswerRecord(BaseModel):
    """Body of `GET /v1/answers/{answer_id}`: everything needed to explain one answer (UC3).

    Attributes:
        answer_id: The answer.
        created_at: When it was recorded.
        question: The question as typed.
        search_query: What the search received; equal to `question` today.
        document_ids: The scope, or null for the whole tenant.
        outcome: What the client got.
        versions: Model, prompt, assembly and index versions.
        coverage: Coverage of the scope when the question was asked.
        candidates: The fused search's top 30.
        labels: Each request label and the unit it pointed to.
        calls: Every call to the answer model, with its raw output and verdict.
        stages: The request's spans, oldest first.
        totals: Latency and tokens.
        response: The body that was returned, answer or error.
    """

    answer_id: UUID
    created_at: datetime
    question: str
    search_query: str
    document_ids: list[UUID] | None
    outcome: AnswerOutcome
    versions: AnswerVersions
    coverage: CoverageView
    candidates: list[CandidateView]
    labels: dict[str, LabelView]
    calls: list[AnswerCall]
    stages: list[AnswerStage]
    totals: AnswerTotals
    response: dict[str, object]
