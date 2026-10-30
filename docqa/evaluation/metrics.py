"""Per-question scoring, in pipeline order, from the API's answer and its record only.

Nothing here reads the database: the answer body, `GET /v1/answers/{id}` (candidates, calls, totals), `GET
/v1/documents/{id}` (page states) and the PDF files themselves are the inputs. Metrics come from the record's own
measurements (calls, tokens, pipeline latency), never from the request's spans, which may land after the response.
"""

import re
import statistics
import threading
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf

from docqa.evaluation.client import Reply
from docqa.evaluation.dataset import Question
from docqa.evaluation.judge import JudgeOutcome, NumericCheck
from docqa.text.numbers import ungrounded

TOP_PROMPT = 8
FINAL_STATUSES = ("ready", "partial", "failed")
PENDING_STATUSES = ("queued", "processing")
STAGES = ("ingestion", "retrieval", "ranking", "provider", "api", "verification", "generation")
MARKUP = re.compile(r"[\s*#|_`]+")
HTML = re.compile(r"<!--.*?-->|</?(?:br|sup|sub|u|b|i)\s*/?>", re.IGNORECASE | re.DOTALL)
DASHES = re.compile(r"[-‐‑−­]")


@dataclass
class FileOutcome:
    """What happened to one file of the folder.

    Attributes:
        file: File name.
        path: Local path, read for the quote check.
        upload_status: HTTP status of the upload.
        outcome: `accepted`, `duplicate`, `rejected` or `upload_error`.
        error_code: Rejection code, or the document's failure code.
        document_id: The document, when accepted or duplicate.
        detail: `GET /v1/documents/{id}` after ingestion, or None.
        final_after_s: Seconds from the first upload until the harness saw the document in a final state.
    """

    file: str
    path: Path
    upload_status: int
    outcome: str
    error_code: str | None = None
    document_id: str | None = None
    detail: dict[str, object] | None = None
    final_after_s: float | None = None

    def page_states(self) -> dict[int, str]:
        """Map each page to its state.

        Returns:
            Page number to state, from the document detail.
        """
        pages = self.detail.get("pages") if self.detail else None
        return {int(p["page"]): str(p["state"]) for p in pages or [] if isinstance(p, dict)}


def normalize(text: str) -> str:
    """Normalize text for the quote check: NFKC, no whitespace, no Markdown or HTML markup, no control characters.

    Control and format characters (a stray BEL character in a font's text, a soft hyphen) are invisible on the page, so they
    are dropped on both sides.

    Args:
        text: A quote or a page's text layer.

    Returns:
        The normalized text.
    """
    text = HTML.sub("", unicodedata.normalize("NFKC", text))
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return MARKUP.sub("", text)


class Corpus:
    """The ingested files, by name and by document, with their page texts read on demand by PyMuPDF."""

    def __init__(self, files: list[FileOutcome]) -> None:
        """Index the files.

        Args:
            files: Outcome of every file of the folder.
        """
        self.files = files
        self.by_name = {f.file: f for f in files}
        self.by_document: dict[str, FileOutcome] = {}
        for item in files:
            if item.document_id and item.document_id not in self.by_document:
                self.by_document[item.document_id] = item
        self._texts: dict[tuple[str, int], str | None] = {}
        self._lock = threading.Lock()

    def expected(self, question: Question) -> set[tuple[str, int]]:
        """Return the `(document_id, page)` pairs that count as the question's source.

        Args:
            question: The question.

        Returns:
            The pairs whose file reached a document.
        """
        found = set()
        for file, page in question.expected_pages():
            item = self.by_name.get(file)
            if item is not None and item.document_id:
                found.add((item.document_id, page))
        return found

    def any_expected_indexed(self, question: Question) -> bool:
        """Tell whether at least one expected page is searchable.

        Args:
            question: The question.

        Returns:
            True when some expected page is `indexed`.
        """
        for file, page in question.expected_pages():
            item = self.by_name.get(file)
            if item is not None and item.page_states().get(page) == "indexed":
                return True
        return False

    def page_text(self, document_id: str, page: int) -> str | None:
        """Read a page's text layer from the uploaded file.

        Args:
            document_id: The document.
            page: 1-based page.

        Returns:
            The normalized text layer, or None when the file can't be read.
        """
        key = (document_id, page)
        with self._lock:
            return self._read(key)

    def _read(self, key: tuple[str, int]) -> str | None:
        """Read and cache one page's normalized text layer; the caller holds the lock.

        Args:
            key: `(document_id, page)`.

        Returns:
            The normalized text, or None.
        """
        document_id, page = key
        if key not in self._texts:
            item = self.by_document.get(document_id)
            text = None
            if item is not None:
                try:
                    with pymupdf.open(item.path) as doc:
                        if 1 <= page <= doc.page_count:
                            text = normalize(doc[page - 1].get_text())
                except (RuntimeError, ValueError):
                    text = None
            self._texts[key] = text
        return self._texts[key]


def segment_found(segment: str, page_text: str) -> bool:
    """Tell whether a quote segment is in the page text, ignoring markup and whitespace (and dashes as a fallback).

    Args:
        segment: One cell or one sentence of a quote.
        page_text: The normalized page text.

    Returns:
        True when found, or when the segment is empty after normalization.
    """
    needle = normalize(segment)
    if not needle or needle in page_text:
        return True
    return DASHES.sub("", needle) in DASHES.sub("", page_text)


def quote_check(quote: str, page_text: str) -> tuple[int, int]:
    """Check a citation's quote, line by line: a line is found when each of its cells is in the page text.

    Table rows are checked cell by cell because the text layer may read a table in another order than the rows.

    Args:
        quote: The citation's quote.
        page_text: The normalized page text.

    Returns:
        `(lines found, lines)`.
    """
    lines = [line for line in quote.split("\n") if line.strip()]
    found = sum(all(segment_found(cell, page_text) for cell in line.split(" | ")) for line in lines)
    return found, len(lines)


@dataclass
class CitationCheck:
    """A citation of the answer and whether its quote is in the PDF.

    Attributes:
        file: Filename of the cited document.
        page: Cited page.
        transcribed: The page text was transcribed; its quote can't be found in the text layer.
        lines: Lines of the quote.
        lines_found: Lines found in the page's text layer (null for transcribed pages or unreadable files).
        expected_page: The citation is one of the question's expected pages.
    """

    file: str
    page: int
    transcribed: bool
    lines: int
    lines_found: int | None
    expected_page: bool


@dataclass
class QuestionResult:
    """Everything measured for one asked question.

    Attributes:
        set_name: Question set.
        id: Question id.
        repeat: 1-based repeat number.
        lang: Language, when given.
        question: The question.
        expected_status: Expected status.
        expected_answer: Reference answer.
        expected_sources: Expected `file:page` pairs.
        http_status: HTTP status of the answer.
        error_code: Error code of a non-200.
        status: Status returned, or None on an error.
        reason: `unverified` for the verifier's decline.
        answer: Answer text.
        options: Number of options of an `ambiguous` answer.
        answer_id: Answer id (also the request id of an error).
        model: Answer model, from the record.
        latency_ms: Pipeline latency from the record.
        wall_ms: Round trip measured by the harness.
        model_calls: Calls to the answer model.
        verifier_retry: A second call was made after the verifier failed.
        provider_retry: A call was retried after a provider failure.
        tokens: Answer tokens per model: `{model: [input, output]}`, output with thinking.
        coverage_pending: A document of the tenant was `queued` or `processing` when the question was asked.
        status_ok: Returned status equals the expected one.
        source_ok: Some citation is on an expected page; None when it doesn't apply.
        citations: The quote check of each citation.
        ungrounded: Numbers of the answer that no cited quote (or the question) holds.
        ingestion_ok: Some expected page is indexed; None without sources.
        rank: Best fused rank of an expected page among the 30 candidates; None when absent or not applicable.
        in_prompt: An expected page was among the passages sent to the model.
        judge: The judge's outcome, when the content was judged.
        numeric: Numbers of the reference found in the answer.
        judge_review: Why the judge's verdict needs a person, if it does.
        correct: Status, content and source are right (`not_found`: status only); None when unknown.
        unsafe: Answered when it should not have, picked one case of an ambiguous question, or gave a wrong value.
        over_cautious: Declined or asked for clarification when an answer was expected.
        stage: First failing stage, in pipeline order, for a question that is not correct.
        sources_missing: Source files of the question that no file of the folder matches.
        body: The reply body as returned: the answer, or the error body.
        record: `GET /v1/answers/{id}`, or None.
    """

    set_name: str
    id: str
    repeat: int
    lang: str | None
    question: str
    expected_status: str
    expected_answer: str
    expected_sources: list[str]
    http_status: int
    error_code: str | None
    status: str | None
    reason: str | None
    answer: str | None
    options: int
    answer_id: str | None
    model: str | None
    latency_ms: float | None
    wall_ms: float
    model_calls: int
    verifier_retry: bool
    provider_retry: bool
    tokens: dict[str, list[int]]
    coverage_pending: bool
    status_ok: bool
    source_ok: bool | None
    citations: list[CitationCheck]
    ungrounded: list[str]
    ingestion_ok: bool | None
    rank: int | None
    in_prompt: bool | None
    judge: JudgeOutcome | None = None
    numeric: NumericCheck | None = None
    judge_review: str | None = None
    correct: bool | None = None
    unsafe: bool = False
    over_cautious: bool = False
    stage: str | None = None
    sources_missing: list[str] = field(default_factory=list)
    body: dict[str, object] = field(default_factory=dict, repr=False)
    record: dict[str, object] | None = field(default=None, repr=False)

    def to_json(self) -> dict[str, object]:
        """Render for `results.jsonl`, with the raw reply body (`response`) and the answer record (`record`).

        Returns:
            A JSON-ready dict.
        """
        data = asdict(self)
        data["response"] = data.pop("body")
        return data


def _citations(body: dict[str, object]) -> list[dict[str, object]]:
    """Return the top-level citations of an answer body.

    Args:
        body: Answer body.

    Returns:
        The citations.
    """
    raw = body.get("citations")
    return [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []


def ungrounded_numbers(body: dict[str, object], question: str) -> list[str]:
    """Redo the grounded-number check from the API's answer alone, with the eval's own reading of the quotes.

    `answered`: every number of the answer must be in some quote or in the question. `ambiguous`: each option's
    condition and answer against that option's quotes, read apart so that the last number of the condition and the
    first of the answer never merge into one token, and the summary against all quotes.

    Args:
        body: A 200 answer body.
        question: The question.

    Returns:
        The unsupported numeric tokens.
    """
    status = body.get("status")
    quotes = [str(c.get("quote", "")) for c in _citations(body)]
    if status == "answered":
        return ungrounded(str(body.get("answer") or ""), [*quotes, question])
    if status != "ambiguous":
        return []
    missing = ungrounded(str(body.get("answer") or ""), [*quotes, question])
    for option in body.get("options") or []:
        if not isinstance(option, dict):
            continue
        option_quotes = [str(c.get("quote", "")) for c in option.get("citations") or [] if isinstance(c, dict)]
        for text in (str(option.get("condition", "")), str(option.get("answer", ""))):
            missing.extend(ungrounded(text, [*option_quotes, question]))
    return missing


def _record_numbers(
    record: dict[str, object] | None,
) -> tuple[str | None, float | None, int, bool, bool, dict[str, list[int]]]:
    """Read the answer model, latency, call counts and tokens from the record.

    Args:
        record: `GET /v1/answers/{id}`, or None.

    Returns:
        Model, pipeline latency, model calls, verifier retry, provider retry and tokens per model.
    """
    if not record:
        return None, None, 0, False, False, {}
    versions = record.get("versions") if isinstance(record.get("versions"), dict) else {}
    totals = record.get("totals") if isinstance(record.get("totals"), dict) else {}
    calls = [c for c in record.get("calls") or [] if isinstance(c, dict)]
    tokens: dict[str, list[int]] = {}
    for call in calls:
        pair = tokens.setdefault(str(call.get("model")), [0, 0])
        pair[0] += int(call.get("input_tokens") or 0)
        pair[1] += int(call.get("output_tokens") or 0)
    latency = totals.get("latency_ms")
    return (
        str(versions.get("model")) if versions.get("model") else None,
        float(latency) if latency is not None else None,
        len(calls),
        any(c.get("purpose") == "verifier_retry" for c in calls),
        any(int(c.get("attempt") or 1) > 1 for c in calls),
        tokens,
    )


def _pending(coverage: object) -> bool:
    """Tell whether a coverage block shows pending work (DECISIONS #16: failed documents don't count).

    Args:
        coverage: The `coverage` block.

    Returns:
        True when a document was `queued` or `processing`.
    """
    if not isinstance(coverage, dict):
        return False
    incomplete = coverage.get("incomplete") or []
    return any(isinstance(d, dict) and d.get("status") in PENDING_STATUSES for d in incomplete)


def _rank(record: dict[str, object] | None, expected: set[tuple[str, int]]) -> tuple[int | None, bool | None]:
    """Find the best rank of an expected page among the record's candidates.

    Args:
        record: The answer record.
        expected: Expected `(document_id, page)` pairs.

    Returns:
        The rank (None when absent) and whether an expected page reached the prompt; both None when unknown.
    """
    if not record or not expected:
        return None, None
    ranks = [
        (int(c["rank"]), bool(c.get("sent_to_llm")))
        for c in record.get("candidates") or []
        if isinstance(c, dict) and (str(c.get("document_id")), int(c.get("page", 0))) in expected
    ]
    if not ranks:
        return None, False
    return min(r for r, _ in ranks), any(sent for _, sent in ranks)


def evaluate(
    question: Question, reply: Reply, record: dict[str, object] | None, corpus: Corpus, *, repeat: int = 1
) -> QuestionResult:
    """Score everything that doesn't need the judge.

    Args:
        question: The question.
        reply: The answer reply.
        record: Its record, or None.
        corpus: The ingested files.
        repeat: Repeat number.

    Returns:
        The result, before the judge and the final flags (`finalize`).
    """
    body = reply.body
    ok = reply.status == 200
    error = body.get("error") if isinstance(body.get("error"), dict) else {}
    answer_id = body.get("answer_id") if ok else error.get("request_id")
    model, latency, n_calls, verifier_retry, provider_retry, tokens = _record_numbers(record)
    expected = corpus.expected(question)
    status = str(body.get("status")) if ok and body.get("status") else None
    citations: list[CitationCheck] = []
    for cite in _citations(body) if ok else []:
        doc_id, page = str(cite.get("document_id")), int(cite.get("page", 0))
        transcribed = bool(cite.get("transcribed"))
        quote = str(cite.get("quote", ""))
        lines = len([line for line in quote.split("\n") if line.strip()])
        found = None
        if not transcribed:
            text = corpus.page_text(doc_id, page)
            found = quote_check(quote, text)[0] if text is not None else None
        item = corpus.by_document.get(doc_id)
        citations.append(
            CitationCheck(
                file=item.file if item else str(cite.get("filename")),
                page=page,
                transcribed=transcribed,
                lines=lines,
                lines_found=found,
                expected_page=(doc_id, page) in expected,
            )
        )
    source_ok = None
    if question.expected_status != "not_found" and expected and status in ("answered", "ambiguous"):
        source_ok = any(c.expected_page for c in citations)
    rank, in_prompt = _rank(record, expected) if question.sources else (None, None)
    coverage = body.get("coverage") if ok else (record or {}).get("coverage")
    return QuestionResult(
        set_name=question.set_name,
        id=question.id,
        repeat=repeat,
        lang=question.lang,
        question=question.question,
        expected_status=question.expected_status,
        expected_answer=question.expected_answer,
        expected_sources=sorted(f"{file}:{page}" for file, page in question.expected_pages()),
        http_status=reply.status,
        error_code=reply.error_code,
        status=status,
        reason=str(body.get("reason")) if ok and body.get("reason") else None,
        answer=str(body.get("answer")) if ok else None,
        options=len(body.get("options") or []) if ok else 0,
        answer_id=str(answer_id) if answer_id else None,
        model=model,
        latency_ms=latency,
        wall_ms=round(reply.wall_ms, 1),
        model_calls=n_calls,
        verifier_retry=verifier_retry,
        provider_retry=provider_retry,
        tokens=tokens,
        coverage_pending=_pending(coverage),
        status_ok=status == question.expected_status,
        source_ok=source_ok,
        citations=citations,
        ungrounded=ungrounded_numbers(body, question.question) if ok else [],
        ingestion_ok=corpus.any_expected_indexed(question) if question.sources else None,
        rank=rank,
        in_prompt=in_prompt,
        sources_missing=question.missing_files(),
        body=body,
        record=record,
    )


def needs_judge(result: QuestionResult) -> bool:
    """Tell whether the content of an answer is judged: the status matches and is not `not_found`.

    Args:
        result: The result.

    Returns:
        True when the judge runs.
    """
    return result.status_ok and result.expected_status != "not_found" and result.http_status == 200


def failure_stage(result: QuestionResult) -> str:
    """Name the first failing stage, in pipeline order.

    Order: `ingestion` (no expected page indexed, a source file missing from the folder included), `retrieval` (no
    expected page in the 30 candidates), `ranking` (in the 30, not in the 8 sent to the model), `provider` (an `llm_*`
    error, or `service_unavailable`), `api` (any other HTTP error), `verification` (the verifier's `unverified`
    decline), `generation` (everything else: the model's status, content or citation).

    Args:
        result: A result that is not correct.

    Returns:
        The stage.
    """
    if result.ingestion_ok is False:
        return "ingestion"
    if result.rank is None and result.in_prompt is False:
        return "retrieval"
    if result.in_prompt is False:
        return "ranking"
    if result.http_status != 200:
        code = result.error_code or ""
        return "provider" if code.startswith("llm_") or code == "service_unavailable" else "api"
    if result.reason == "unverified":
        return "verification"
    return "generation"


def finalize(result: QuestionResult, *, judged: bool) -> QuestionResult:
    """Set the headline flags and the failure stage.

    Correct: `not_found` by status alone; otherwise status, content (when judged) and a citation on an expected page.
    Without the judge, correct counts status and source only. Unsafe: answered or ambiguous when `not_found` was
    expected, answered when `ambiguous` was expected, or a contradicted fact. Over-cautious: `not_found` (the verifier's
    decline included) when an answer was expected, or `ambiguous` when `answered` was expected.

    Args:
        result: The result, with `judge` set when judged.
        judged: The run uses the judge.

    Returns:
        The same result, updated.
    """
    expected, got = result.expected_status, result.status
    if expected == "not_found":
        result.correct = got == "not_found"
    elif not result.status_ok:
        result.correct = False
    else:
        content: bool | None = True
        if judged:
            verdict = result.judge.verdict if result.judge else None
            content = None if verdict is None else verdict == "correct"
        source = result.source_ok is not False
        result.correct = None if content is None and source else (bool(content) and source)
    contradicted = bool(result.judge and result.judge.contradicted)
    result.unsafe = (
        (expected == "not_found" and got in ("answered", "ambiguous"))
        or (expected == "ambiguous" and got == "answered")
        or (result.status_ok and contradicted)
    )
    result.over_cautious = (expected in ("answered", "ambiguous") and got == "not_found") or (
        expected == "answered" and got == "ambiguous"
    )
    result.stage = failure_stage(result) if result.correct is False else None
    return result


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile.

    Args:
        values: Samples.
        q: Quantile between 0 and 1.

    Returns:
        The percentile, or None without samples.
    """
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    low = int(k)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def median(values: list[float]) -> float | None:
    """Median, or None without samples.

    Args:
        values: Samples.

    Returns:
        The median.
    """
    return statistics.median(values) if values else None
