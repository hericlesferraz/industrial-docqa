"""The Markdown report: headline first, then why each failure happened, then cost, latency and ingestion."""

from collections import Counter, defaultdict
from collections.abc import Iterable

from docqa.evaluation.harness import RunResult
from docqa.evaluation.judge import JUDGE_PROMPT_VERSION
from docqa.evaluation.metrics import STAGES, TOP_PROMPT, QuestionResult, median, percentile
from docqa.evaluation.prices import PRICES, PRICES_AS_OF, PRICES_SOURCE, cost

STATUSES = ("answered", "ambiguous", "not_found")
P95_MIN_SAMPLES = 20
ANSWER_CUT = 220


def pct(part: int, whole: int) -> str:
    """Format a count with its share.

    Args:
        part: Count.
        whole: Total.

    Returns:
        `"n (x%)"`, or `"0"` for an empty total.
    """
    return f"{part} ({100 * part / whole:.0f}%)" if whole else "0"


def ms(value: float | None) -> str:
    """Format milliseconds.

    Args:
        value: Milliseconds, or None.

    Returns:
        The value rounded, or `-`.
    """
    return "-" if value is None else f"{value:,.0f}"


def usd(value: float | None) -> str:
    """Format dollars.

    Args:
        value: US$, or None.

    Returns:
        The value with 4 decimals, or `n/a` for an unpriced model.
    """
    return "n/a" if value is None else f"{value:.4f}"


def short(text: str | None, limit: int = ANSWER_CUT) -> str:
    """Cut a text for a table or a bullet, on one line.

    Args:
        text: The text.
        limit: Maximum characters.

    Returns:
        The text on one line, cut with an ellipsis.
    """
    flat = " ".join((text or "").split()).replace("|", "/")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def headline_row(name: str, rows: list[QuestionResult]) -> str:
    """Render one headline line.

    Args:
        name: Set name or `all`.
        rows: Its results.

    Returns:
        A Markdown table row.
    """
    n = len(rows)
    correct = sum(r.correct is True for r in rows)
    unknown = sum(r.correct is None for r in rows)
    unsafe = sum(r.unsafe for r in rows)
    cautious = sum(r.over_cautious for r in rows)
    errors = sum(r.http_status != 200 for r in rows)
    extra = f" (+{unknown} unknown)" if unknown else ""
    return f"| {name} | {n} | {pct(correct, n)}{extra} | {pct(unsafe, n)} | {pct(cautious, n)} | {errors} |"


def _content(r: QuestionResult) -> str:
    """Describe the content check of one result.

    Args:
        r: The result.

    Returns:
        The judge verdict, `failed`, `-` when not applicable, or `not judged`.
    """
    if r.judge is not None:
        return r.judge.verdict or "judge failed"
    if r.status_ok and r.expected_status != "not_found" and r.http_status == 200:
        return "not judged"
    return "-"


def _yes_no(value: bool | None) -> str:
    """Render a tri-state flag.

    Args:
        value: The flag.

    Returns:
        `yes`, `no` or `-`.
    """
    return "-" if value is None else ("yes" if value else "no")


def _source(r: QuestionResult) -> str:
    """Render the source check of one result.

    Args:
        r: The result.

    Returns:
        `yes` or `no`, `file missing from folder` when no expected file was uploaded, or `-`.
    """
    if r.source_ok is None and r.sources_missing:
        return "file missing from folder"
    return _yes_no(r.source_ok)


def _got(r: QuestionResult) -> str:
    """Render what the system returned.

    Args:
        r: The result.

    Returns:
        The status (with `unverified`), or the HTTP error code.
    """
    if r.http_status != 200:
        return f"HTTP {r.http_status} {r.error_code or ''}".strip()
    return f"{r.status} (unverified)" if r.reason else str(r.status)


def section_header(run: RunResult) -> list[str]:
    """Render the run's identity.

    Args:
        run: The run.

    Returns:
        Markdown lines.
    """
    config, rows = run.config, run.results
    models = sorted({r.model for r in rows if r.model})
    judge = (
        f"`{config.judge_model}`, prompt `{JUDGE_PROMPT_VERSION}`, temperature 0 (**provisional** until the "
        "meta-evaluation against labels reviewed by a person)"
        if config.judge_model
        else "off (`--no-judge`): content not judged"
    )
    sets = Counter(q.set_name for q in run.questions)
    skipped = sum(p.skipped for p in run.input_problems)
    files = Counter(f.outcome for f in run.ingest.files)
    health = run.health.get("status", "unknown")
    reasons = run.health.get("reasons") or []
    return [
        "# Eval report",
        "",
        "| | |",
        "|---|---|",
        (
            f"| Run | {run.started_at:%Y-%m-%d %H:%M} UTC to {run.finished_at:%H:%M} UTC; "
            f"{len(run.questions)} questions × {config.repeat} repeat(s), concurrency {config.concurrency} |"
        ),
        f"| API | `{config.api_url}`, health `{health}`{'; ' + '; '.join(map(str, reasons)) if reasons else ''} |",
        f"| Answer model | {', '.join(f'`{m}`' for m in models) or 'unknown'} (from the answer records) |",
        f"| Judge | {judge} |",
        f"| PDF folder | {len(run.ingest.files)} files: "
        + ", ".join(f"{n} {k}" for k, n in sorted(files.items()))
        + " |",
        "| Question sets | "
        + ", ".join(f"`{name}` ({n})" for name, n in sets.items())
        + (f"; {skipped} line(s) skipped (see Input problems)" if skipped else "")
        + " |",
        f"| Tenant | `{run.ingest.tenant_id}` (fresh for this run) |",
        "",
    ]


def section_input(run: RunResult) -> list[str]:
    """Render the lines of the question files that were skipped or asked with part of them ignored.

    Args:
        run: The run.

    Returns:
        Markdown lines; empty when every line was used as written.
    """
    if not run.input_problems:
        return []
    skipped = [p for p in run.input_problems if p.skipped]
    kept = [p for p in run.input_problems if not p.skipped]
    lines = [
        "## Input problems",
        "",
        (
            f"{len(skipped)} line(s) skipped (not asked, not scored) and {len(kept)} caveat(s) on questions that were "
            "asked. Source files are matched by base name, ignoring case and Unicode normalization."
        ),
        "",
        "| File | Line | Outcome | Reason |",
        "|---|---|---|---|",
    ]
    for p in run.input_problems:
        outcome = "skipped" if p.skipped else "asked"
        lines.append(f"| {short(p.file, 60)} | {p.line or '-'} | {outcome} | {short(p.reason, 300)} |")
    lines.append("")
    return lines


def section_headline(run: RunResult) -> list[str]:
    """Render the headline numbers per set and overall.

    Args:
        run: The run.

    Returns:
        Markdown lines.
    """
    by_set: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in run.results:
        by_set[r.set_name].append(r)
    lines = [
        "## Headline",
        "",
        "| Set | Answers | Correct | Unsafe | Over-cautious | HTTP errors |",
        "|---|---|---|---|---|---|",
    ]
    lines += [headline_row(name, rows) for name, rows in by_set.items()]
    if len(by_set) > 1:
        lines.append(headline_row("**all**", run.results))
    lines += [
        "",
        (
            "- **Correct**: the expected status, the content right according to the judge, and a citation on an expected "
            "page. For `not_found`, the status alone."
        ),
        (
            "- **Unsafe**: an answer (or options) when the documents don't answer, one case picked when the question is "
            "ambiguous, or a fact of the reference contradicted (a wrong value or condition)."
        ),
        (
            "- **Over-cautious**, reported apart: `not_found` (the verifier's decline included) when an answer was "
            "expected, or a clarification when a direct answer was expected."
        ),
    ]
    if run.config.judge_model is None:
        lines.append(
            "- Without the judge, **correct** counts status and source only (an upper bound) and **unsafe** counts "
            "wrong statuses only (a lower bound)."
        )
    if any(r.correct is None for r in run.results):
        lines.append("- **unknown**: the judge failed on these answers; they count neither as correct nor as wrong.")
    lines.append("")
    return lines


def section_matrix(rows: list[QuestionResult]) -> list[str]:
    """Render the status matrix.

    Args:
        rows: Results.

    Returns:
        Markdown lines.
    """
    cols = [*STATUSES, "error"]
    lines = ["## Status matrix", "", "| expected ↓ / got → | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for expected in STATUSES:
        mine = [r for r in rows if r.expected_status == expected]
        if not mine:
            continue
        counts = Counter(r.status if r.http_status == 200 else "error" for r in mine)
        lines.append(f"| {expected} | " + " | ".join(str(counts.get(c, 0)) for c in cols) + " |")
    unverified = sum(r.reason == "unverified" for r in rows)
    lines += ["", f"`not_found` includes {unverified} verifier decline(s) (`reason: unverified`).", ""]
    return lines


def section_languages(rows: list[QuestionResult]) -> list[str]:
    """Render the per-language breakdown.

    Args:
        rows: Results.

    Returns:
        Markdown lines; empty when no question has `lang`.
    """
    langs: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in rows:
        langs[r.lang or "(none)"].append(r)
    if set(langs) == {"(none)"}:
        return []
    lines = ["## By language", "", "| lang | Answers | Correct | Unsafe | Over-cautious |", "|---|---|---|---|---|"]
    for lang in sorted(langs):
        mine = langs[lang]
        n = len(mine)
        lines.append(
            f"| {lang} | {n} | {pct(sum(r.correct is True for r in mine), n)} | {pct(sum(r.unsafe for r in mine), n)} "
            f"| {pct(sum(r.over_cautious for r in mine), n)} |"
        )
    lines.append("")
    return lines


def section_stages(rows: list[QuestionResult]) -> list[str]:
    """Render the failures by stage.

    Args:
        rows: Results.

    Returns:
        Markdown lines.
    """
    failed = [r for r in rows if r.stage]
    lines = ["## Failures by stage", ""]
    if not failed:
        return [*lines, "No failures.", ""]
    lines += ["| Stage | Count | Questions |", "|---|---|---|"]
    for stage in STAGES:
        mine = [r for r in failed if r.stage == stage]
        if mine:
            ids = ", ".join(f"{r.set_name}/{r.id}" + (f"#{r.repeat}" if r.repeat > 1 else "") for r in mine)
            lines.append(f"| {stage} | {len(mine)} | {ids} |")
    lines += [
        "",
        (
            "First failing stage in pipeline order: `ingestion` (no expected page indexed, or its file not in the "
            "folder), `retrieval` (no expected "
            f"page among the 30 candidates), `ranking` (among the 30, not in the {TOP_PROMPT} sent to the model), "
            "`provider` (an `llm_*` error), `api` (another HTTP error), `verification` (the verifier declined), "
            "`generation` (the model's status, content or citation)."
        ),
        "",
    ]
    return lines


def _facts(r: QuestionResult) -> str:
    """Summarize the judge's missing and contradicted facts.

    Args:
        r: The result.

    Returns:
        A short text, or an empty string.
    """
    if r.judge is None or not r.judge.facts:
        return ""
    parts = [
        f"{f.verdict}{'' if f.essential else ' (example)'}: {short(f.fact, 120)}"
        for f in r.judge.facts
        if f.verdict != "stated"
    ]
    return "; ".join(parts)


def section_questions(rows: list[QuestionResult]) -> list[str]:
    """Render one table per set, then the details of each answer that is not correct.

    Args:
        rows: Results.

    Returns:
        Markdown lines.
    """
    lines = ["## Questions", ""]
    by_set: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in rows:
        by_set[r.set_name].append(r)
    for name, mine in by_set.items():
        lines += [
            f"### {name}",
            "",
            "| id | lang | expected | got | content | source | rank | stage | latency ms |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in mine:
            rid = r.id + (f" #{r.repeat}" if r.repeat > 1 else "")
            ok = "" if r.correct else (" ⚠" if r.unsafe else " ✗")
            lines.append(
                f"| {rid}{ok} | {r.lang or '-'} | {r.expected_status} | {_got(r)} | {_content(r)} | "
                f"{_source(r)} | {r.rank if r.rank is not None else '-'} | {r.stage or ''} | "
                f"{ms(r.latency_ms)} |"
            )
        lines.append("")
        wrong = [r for r in mine if r.correct is not True]
        if wrong:
            lines.append("Not correct:")
            lines.append("")
            for r in wrong:
                facts = _facts(r)
                lines.append(
                    f"- **{r.id}**{f' #{r.repeat}' if r.repeat > 1 else ''} ({r.stage or 'unknown'}"
                    f"{', unsafe' if r.unsafe else ''}{', over-cautious' if r.over_cautious else ''}): "
                    f"expected {r.expected_status}, got {_got(r)}. Answer: «{short(r.answer)}»"
                    + (f" Judge: {facts}." if facts else "")
                    + (f" Source file not in the folder: {', '.join(r.sources_missing)}." if r.sources_missing else "")
                    + (f" Expected: «{short(r.expected_answer, 160)}»" if r.expected_answer else "")
                )
            lines.append("")
    lines.append(
        "`rank` is the best fused rank of an expected page among the 30 search candidates; ✗ not correct, ⚠ unsafe."
    )
    lines.append("")
    return lines


def section_citations(rows: list[QuestionResult]) -> list[str]:
    """Render the quote-in-PDF check and the grounded-number re-check.

    Args:
        rows: Results.

    Returns:
        Markdown lines.
    """
    cites = [(r, c) for r in rows for c in r.citations]
    layer = [(r, c) for r, c in cites if not c.transcribed]
    checked = [(r, c) for r, c in layer if c.lines_found is not None]
    full = [(r, c) for r, c in checked if c.lines_found == c.lines]
    lines_total = sum(c.lines for _, c in checked)
    lines_found = sum(c.lines_found or 0 for _, c in checked)
    transcribed = [(r, c) for r, c in cites if c.transcribed]
    answers = [r for r in rows if r.status in ("answered", "ambiguous") and r.http_status == 200]
    lines = [
        "## Citations",
        "",
        (
            f"- Citations returned: {len(cites)} in {len(answers)} answers; on an expected page: "
            f"{sum(c.expected_page for _, c in cites)}."
        ),
        f"- Text-layer pages: {len(layer)} citations; quote found in the page's text layer read by the eval with "
        f"PyMuPDF (markup and whitespace ignored, table rows cell by cell): {pct(len(full), len(checked))} "
        f"of the citations, {lines_found}/{lines_total} lines"
        + (f"; {len(layer) - len(checked)} unreadable" if len(layer) != len(checked) else "")
        + ".",
        (
            f"- Transcribed pages (`transcribed: true`): {len(transcribed)} citations, counted apart: their text comes "
            "from the page image, so the text layer can't confirm it."
        ),
    ]
    partial = [(r, c) for r, c in checked if c.lines_found != c.lines]
    if partial:
        lines.append(
            "- Quotes not fully found: "
            + "; ".join(f"{r.id} {c.file} p{c.page} ({c.lines_found}/{c.lines} lines)" for r, c in partial)
            + "."
        )
    loose = [r for r in answers if r.ungrounded]
    lines += [
        (
            f"- Grounded numbers, re-checked from the API's answer alone (every number of the answer must be in a quote "
            f"of the answer or in the question, any locale reading): {pct(len(answers) - len(loose), len(answers))} of "
            "the answers pass."
        ),
    ]
    if loose:
        lines.append(
            "- Numbers without a quote: " + "; ".join(f"{r.id}: {', '.join(r.ungrounded)}" for r in loose) + "."
        )
    lines.append("")
    return lines


def section_retrieval(rows: list[QuestionResult]) -> list[str]:
    """Render the search metrics for questions with sources.

    Args:
        rows: Results.

    Returns:
        Markdown lines.
    """
    mine = [r for r in rows if r.in_prompt is not None]
    if not mine:
        return []
    top8 = sum(bool(r.in_prompt) for r in mine)
    top30 = sum(r.rank is not None for r in mine)
    mrr = sum(1 / r.rank for r in mine if r.rank) / len(mine)
    misses = [r for r in mine if not r.in_prompt]
    lines = [
        "## Retrieval",
        "",
        (
            f"- Answers with expected pages and a record: {len(mine)}. Expected page among the {TOP_PROMPT} passages "
            f"sent to the model: {pct(top8, len(mine))}; among the 30 candidates: {pct(top30, len(mine))}; MRR "
            f"{mrr:.2f} (fused rank of the first chunk on an expected page)."
        ),
    ]
    if misses:
        lines.append(
            "- Not sent to the model: "
            + ", ".join(f"{r.id} (rank {r.rank if r.rank else '>30'})" for r in misses)
            + "."
        )
    lines.append("")
    return lines


def section_judge(run: RunResult) -> list[str]:
    """Render the judge's activity and the verdicts to review.

    Args:
        run: The run.

    Returns:
        Markdown lines.
    """
    judged = [r for r in run.results if r.judge is not None]
    lines = ["## Judge", ""]
    if run.config.judge_model is None:
        numeric = [r for r in run.results if r.numeric is not None and r.numeric.recall is not None]
        full = sum(r.numeric.recall == 1 for r in numeric if r.numeric)
        return [
            *lines,
            (
                "Off (`--no-judge`). Numeric proxy only: answers with every number of the reference: "
                f"{pct(full, len(numeric))} of {len(numeric)} answers whose reference has numbers."
            ),
            "",
        ]
    verdicts = Counter(r.judge.verdict or "failed" for r in judged if r.judge)
    cached = sum(bool(r.judge and r.judge.cached) for r in judged)
    lines += [
        (
            f"- Model `{run.config.judge_model}` (**provisional**), prompt `{JUDGE_PROMPT_VERSION}`, temperature 0, "
            "reference-guided: the model lists the facts of the reference (essential or example) and marks each "
            "`stated`, `missing` or `contradicted`; the verdict is computed by code (correct: every essential fact "
            "stated and none contradicted; partial: nothing contradicted, an essential fact missing; wrong: something "
            "contradicted or no essential fact stated)."
        ),
        f"- Judged answers: {len(judged)} (only when the status matches and is not `not_found`): "
        + ", ".join(f"{k} {v}" for k, v in sorted(verdicts.items()))
        + f"; from the cache: {cached}.",
    ]
    failed = [r for r in judged if r.judge and r.judge.error]
    if failed:
        lines.append("- Judge failures: " + "; ".join(f"{r.id}: {r.judge.error}" for r in failed if r.judge) + ".")
    review = [r for r in judged if r.judge_review]
    lines.append(
        "- Numeric cross-check (numbers of the reference that the question doesn't hold, looked up in the answer): "
        + (
            "verdicts to review: " + "; ".join(f"{r.set_name}/{r.id}: {r.judge_review}" for r in review) + "."
            if review
            else "no disagreement."
        )
    )
    lines.append("")
    return lines


def _latencies(rows: Iterable[QuestionResult]) -> tuple[list[float], list[float]]:
    """Collect pipeline and round-trip latencies.

    Args:
        rows: Results.

    Returns:
        Pipeline latencies from the records, and round trips measured by the harness.
    """
    rows = list(rows)
    return [r.latency_ms for r in rows if r.latency_ms is not None], [r.wall_ms for r in rows]


def _ids(rows: Iterable[QuestionResult]) -> str:
    """Count results and name them.

    Args:
        rows: Results.

    Returns:
        `"0"`, or the count and the ids.
    """
    names = [r.id + (f"#{r.repeat}" if r.repeat > 1 else "") for r in rows]
    return f"{len(names)} ({', '.join(names)})" if names else "0"


def section_latency(rows: list[QuestionResult]) -> list[str]:
    """Render latency and call counts.

    Args:
        rows: Results.

    Returns:
        Markdown lines.
    """
    pipeline, wall = _latencies(rows)
    p95 = len(pipeline) >= P95_MIN_SAMPLES

    def line(name: str, values: list[float]) -> str:
        """Render one latency line.

        Args:
            name: What was measured.
            values: Samples in ms.

        Returns:
            A bullet.
        """
        tail = f", p95 {ms(percentile(values, 0.95))}" if p95 else ""
        return f"- {name}: p50 {ms(median(values))} ms{tail}, max {ms(max(values) if values else None)} ms."

    return [
        "## Latency",
        "",
        line("Pipeline (from the answer record, scope check to response)", pipeline),
        line("HTTP round trip measured by the harness", wall),
        f"- Samples: {len(pipeline)}" + ("" if p95 else f" (p95 shown from {P95_MIN_SAMPLES} samples on)") + ".",
        (
            f"- Answer model calls: {sum(r.model_calls for r in rows)}; verifier retries: "
            f"{_ids(r for r in rows if r.verifier_retry)}; provider retries: "
            f"{_ids(r for r in rows if r.provider_retry)}."
        ),
        "",
    ]


def _cost_rows(run: RunResult) -> tuple[list[tuple[str, str, int, int, int]], list[tuple[str, str, int, int, int]]]:
    """Sum calls and tokens per part and model.

    Args:
        run: The run.

    Returns:
        Product rows (OCR of this run, answers) and judge rows: `(part, model, calls, input, output)`.
    """
    product: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for item in run.ingest.files:
        ocr = (item.detail or {}).get("ocr")
        by_model = ocr.get("by_model") if isinstance(ocr, dict) else None
        for model, usage in (by_model or {}).items():
            row = product[("OCR (this run)", model)]
            row[0] += int(usage.get("calls", 0))
            row[1] += int(usage.get("input_tokens", 0))
            row[2] += int(usage.get("output_tokens", 0))
    for r in run.results:
        for model, (tokens_in, tokens_out) in r.tokens.items():
            row = product[("Answers", model)]
            row[1] += tokens_in
            row[2] += tokens_out
        if r.model:
            product[("Answers", r.model)][0] += r.model_calls
    judge: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in run.results:
        if r.judge is not None and not r.judge.cached:
            row = judge[("Judge", r.judge.model)]
            row[0] += 1
            row[1] += r.judge.input_tokens
            row[2] += r.judge.output_tokens
    return (
        [(part, model, *values) for (part, model), values in product.items()],
        [(part, model, *values) for (part, model), values in judge.items()],
    )


def section_cost(run: RunResult) -> list[str]:
    """Render measured cost: product (OCR and answers) apart from the judge.

    Args:
        run: The run.

    Returns:
        Markdown lines.
    """
    product, judge = _cost_rows(run)
    lines = [
        "## Cost",
        "",
        (
            "Measured tokens (OCR from `GET /v1/documents/{id}`, answers from the answer records, judge from its calls) "
            f"times the list prices of {PRICES_AS_OF}."
        ),
        "",
        "| Part | Model | Calls | Input tokens | Output tokens | US$ |",
        "|---|---|---|---|---|---|",
    ]
    for label, rows in (("Product", product), ("Judge", judge)):
        if not rows:
            lines.append(f"| {label} | | 0 | | | 0 |")
            continue
        total: float | None = 0.0
        for part, model, calls, tokens_in, tokens_out in rows:
            value = cost(model, tokens_in, tokens_out)
            total = None if value is None or total is None else total + value
            lines.append(f"| {part} | `{model}` | {calls} | {tokens_in:,} | {tokens_out:,} | {usd(value)} |")
        lines.append(f"| **{label} total** | | | | | **{usd(total)}** |")
    answered = len(run.results)
    answer_cost = [cost(model, i, o) for part, model, _calls, i, o in product if part == "Answers"]
    known = [c for c in answer_cost if c is not None]
    cache_pages = sum(
        int(((f.detail or {}).get("ocr") or {}).get("pages_from_cache", 0)) for f in run.ingest.files if f.detail
    )
    ocr_pages = sum(
        int(((f.detail or {}).get("ocr") or {}).get("pages_transcribed", 0)) for f in run.ingest.files if f.detail
    )
    lines += [
        "",
        f"- Answers: US$ {sum(known) / answered:.5f} per question on average ({answered} answers)."
        if answered and known
        else "- Answers: no priced answer.",
        (
            f"- OCR: {ocr_pages} transcribed pages, {cache_pages} of them from the shared OCR cache (an earlier "
            "transcription of the same file: no call, no cost in this run)."
        ),
        "- Embeddings run locally (BGE-M3 int8): no per-token cost.",
        "",
        f"Prices (US$ per million tokens, input / output, thinking billed as output), {PRICES_SOURCE}, "
        f"as of {PRICES_AS_OF}: "
        + "; ".join(f"`{m}` {p.input_per_m:g} / {p.output_per_m:g}" for m, p in PRICES.items())
        + ".",
        "",
    ]
    return lines


def section_ingestion(run: RunResult) -> list[str]:
    """Render what happened to each file.

    Args:
        run: The run.

    Returns:
        Markdown lines.
    """
    ingest = run.ingest
    if ingest.complete:
        state = f"complete after {ingest.elapsed_s:.0f} s"
    elif ingest.paused:
        state = f"stopped waiting after {ingest.elapsed_s:.0f} s: every unfinished document is held by a provider pause"
    else:
        state = f"**timed out** after {ingest.elapsed_s:.0f} s; the questions were asked anyway"
    lines = [
        "## Ingestion",
        "",
        f"One batch, {len(ingest.files)} files, {state}.",
        "",
        (
            "| File | Upload | Status | Pages | Indexed | Layer / OCR | OCR: flagged / transcribed / cache | Failed pages | "
            "Error | Final after s |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for f in ingest.files:
        d = f.detail or {}
        ocr = d.get("ocr") if isinstance(d.get("ocr"), dict) else {}
        source = d.get("pages_by_source") if isinstance(d.get("pages_by_source"), dict) else {}
        failed_pages = [
            f"p{p['page']} {p['error']['code']}"
            for p in d.get("pages") or []
            if isinstance(p, dict) and p.get("state") == "failed" and isinstance(p.get("error"), dict)
        ]
        lines.append(
            f"| {short(f.file, 60)} | {f.outcome} ({f.upload_status}) | {d.get('status', '-')} | "
            f"{d.get('page_count') if d.get('page_count') is not None else '-'} | {d.get('pages_indexed', '-')} | "
            f"{source.get('layer', 0)} / {source.get('ocr', 0)} | "
            f"{ocr.get('pages_flagged', 0)} / {ocr.get('pages_transcribed', 0)} / {ocr.get('pages_from_cache', 0)} | "
            f"{', '.join(failed_pages) or '-'} | {f.error_code or '-'} | "
            f"{f'{f.final_after_s:.0f}' if f.final_after_s is not None else '-'} |"
        )
    pending = [r for r in run.results if r.coverage_pending]
    lines += [
        "",
        "Coverage: "
        + (
            f"{len(pending)} answer(s) were given while a document was still queued or processing: "
            + ", ".join(r.id for r in pending)
            + "."
            if pending
            else "no question was asked while a document was still queued or processing (failed documents don't "
            "count: they are listed above)."
        ),
        "",
    ]
    return lines


def section_repeats(rows: list[QuestionResult]) -> list[str]:
    """Render the stability of repeated questions.

    Args:
        rows: Results.

    Returns:
        Markdown lines; empty without repeats.
    """
    groups: dict[tuple[str, str], list[QuestionResult]] = defaultdict(list)
    for r in rows:
        groups[(r.set_name, r.id)].append(r)
    if all(len(g) == 1 for g in groups.values()):
        return []
    unstable = [k for k, g in groups.items() if len({(r.status, r.correct) for r in g}) > 1]
    return [
        "## Repeats",
        "",
        f"- Questions whose status or correctness changed across repeats: {len(unstable)} of {len(groups)}"
        + (": " + ", ".join(f"{s}/{i}" for s, i in unstable) if unstable else "")
        + ".",
        "",
    ]


def render(run: RunResult) -> str:
    """Render the whole report.

    Args:
        run: The run.

    Returns:
        Markdown.
    """
    rows = run.results
    parts = [
        section_header(run),
        section_input(run),
        section_headline(run),
        section_matrix(rows),
        section_languages(rows),
        section_stages(rows),
        section_questions(rows),
        section_citations(rows),
        section_retrieval(rows),
        section_judge(run),
        section_latency(rows),
        section_cost(run),
        section_ingestion(run),
        section_repeats(rows),
    ]
    return "\n".join(line for part in parts for line in part)
