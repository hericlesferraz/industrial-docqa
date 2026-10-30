"""The deterministic verifier (DECISIONS #7) with the three grounding expansions of DECISIONS #17.

Per answer, or per option of an ambiguous answer, the cited labels must exist, there must be 1 to 6 of them, and every
number of the text must have a reading in the cited units or in the question. Before judging, the cited evidence is
expanded, never with a unit that could hold another row's value:

1. a cited table row brings its table's header rows;
2. a cited starred row brings the page's footnote; a cited footnote brings only the starred cells it explains;
3. a number still ungrounded that appears in an uncited sentence (never a row) among the labels of a page already
   cited adds that sentence.

`depends_on` has no citations of its own and is grounded against the expanded sources of every option together.
Expansions 1 and 2 read the page's units from its current chunks, and are skipped when the page changed since the
labels were built.
"""

from dataclasses import dataclass
from uuid import UUID

from docqa.answer.prompt import LabelInfo
from docqa.answer.schema import MAX_CITATIONS, Ambiguous, Answered, ModelOutput
from docqa.repo.chunks import PageUnits
from docqa.text.numbers import pool, readings, ungrounded
from docqa.text.units import STARRED_CELL, Unit, is_footnote, prompt_form, starred_cells

PageKey = tuple[UUID, int]


@dataclass(frozen=True)
class Evidence:
    """A unit that backs an answer: cited by the model or attached by an expansion.

    Attributes:
        document_id: Document.
        filename: Document filename.
        page_no: 1-based page.
        text_source: `layer` or `ocr`.
        unit: The unit.
    """

    document_id: UUID
    filename: str
    page_no: int
    text_source: str
    unit: Unit


@dataclass(frozen=True)
class GroupCheck:
    """The check of one answer text: the answer, or one option of an ambiguous answer.

    Attributes:
        where: `answer` or `option <n>`.
        cited: Labels the model cited, in its order.
        added: Labels added by expansion 3.
        evidence: Units of the cited labels, then the units attached by expansions 1, 2 (footnotes only) and 3.
        problems: What failed, empty when the group passes.
    """

    where: str
    cited: list[str]
    added: list[str]
    evidence: list[Evidence]
    problems: list[str]


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of the verifier.

    Attributes:
        passed: True when no problem was found.
        problems: Every problem, for the retry message and the record.
        added: Labels added by expansion 3, over all groups.
        groups: One check per answer text, in output order; empty for `not_found`.
    """

    passed: bool
    problems: list[str]
    added: list[str]
    groups: list[GroupCheck]

    def to_json(self) -> dict[str, object]:
        """Give the form stored with each call of the answer record.

        Returns:
            `{"passed", "problems", "added_labels"}`.
        """
        return {"passed": self.passed, "problems": list(self.problems), "added_labels": list(self.added)}


def _label_evidence(info: LabelInfo) -> Evidence:
    """Wrap a label's unit as evidence.

    Args:
        info: The label.

    Returns:
        The evidence.
    """
    return Evidence(
        document_id=info.document_id,
        filename=info.filename,
        page_no=info.page_no,
        text_source=info.text_source,
        unit=info.unit,
    )


def _page_of(info: LabelInfo, page_units: dict[PageKey, PageUnits]) -> PageUnits | None:
    """Return the current units of a label's page, if the page is unchanged since the labels were built.

    Args:
        info: The label.
        page_units: Units of the cited pages, from their current chunks.

    Returns:
        The page's units, or None when they are missing or come from another text revision.
    """
    page = page_units.get((info.document_id, info.page_no))
    if page is None or page.text_rev != info.text_rev:
        return None
    return page


def _attach(found: list[Unit], page: PageUnits, filename: str, texts: list[str], evidence: list[Evidence]) -> None:
    """Add whole units of a page to the grounding texts and to the evidence.

    Args:
        found: Units to attach.
        page: The page they belong to.
        filename: Its document's filename.
        texts: Grounding texts, extended in place.
        evidence: Evidence units, extended in place.
    """
    texts.extend(prompt_form(unit) for unit in found)
    evidence.extend(
        Evidence(
            document_id=page.document_id,
            filename=filename,
            page_no=page.page_no,
            text_source=page.text_source,
            unit=unit,
        )
        for unit in found
    )


def expand(
    cites: list[str], labels: dict[str, LabelInfo], page_units: dict[PageKey, PageUnits]
) -> tuple[list[str], list[Evidence]]:
    """Apply expansions 1 and 2 to some cited labels.

    Args:
        cites: Cited labels that exist in the request.
        labels: The request's labels.
        page_units: Units of the cited pages.

    Returns:
        The texts that ground numbers (cited units, header rows, footnotes, starred cells) and the evidence units
        (cited units, header rows, footnotes; never the starred-cell fragments).
    """
    texts = [prompt_form(labels[label].unit) for label in cites]
    evidence = [_label_evidence(labels[label]) for label in cites]
    for label in cites:
        info = labels[label]
        page = _page_of(info, page_units)
        if page is None:
            continue
        unit = info.unit
        if unit.kind == "row":
            headers = [
                other
                for other in page.units
                if other.kind == "row" and other.table == unit.table and other.header and other.pos != unit.pos
            ]
            _attach(headers, page, info.filename, texts, evidence)
            if STARRED_CELL.search(unit.md):
                notes = [other for other in page.units if is_footnote(other, page.units)]
                _attach(notes, page, info.filename, texts, evidence)
        elif is_footnote(unit, page.units):
            for other in page.units:
                if other.kind == "row":
                    texts.extend(starred_cells(other.md))
    return texts, evidence


def complete(
    text: str, cites: list[str], labels: dict[str, LabelInfo], page_units: dict[PageKey, PageUnits], question: str
) -> list[str]:
    """Apply expansion 3: add uncited sentences of cited pages that hold an otherwise ungrounded number.

    Only request labels are searched, so only text the model read; table rows never qualify, so a value from another
    row of a cited table still fails.

    Args:
        text: The answer text.
        cites: Cited labels that exist in the request.
        labels: The request's labels.
        page_units: Units of the cited pages.
        question: The question, a source of numbers too.

    Returns:
        The cited labels followed by the added ones.
    """
    full = list(cites)
    pages = {(labels[label].document_id, labels[label].page_no) for label in full}
    sources, _ = expand(full, labels, page_units)
    for tok in ungrounded(text, [*sources, question]):
        wanted = readings(tok)
        for label, info in labels.items():
            if label in full or info.kind == "row" or (info.document_id, info.page_no) not in pages:
                continue
            if pool([prompt_form(info.unit)]) & wanted:
                full.append(label)
                break
    return full


def _numbers_problem(where: str, missing: list[str], whose: str) -> str:
    """Describe ungrounded numbers for the retry message.

    Args:
        where: `answer`, `option <n>` or `depends_on`.
        missing: The ungrounded tokens.
        whose: Which passages were searched.

    Returns:
        The problem sentence.
    """
    return f"{where}: the numbers {', '.join(missing)} have no source in {whose}"


def _check_group(
    where: str,
    text: str,
    cites: list[str],
    labels: dict[str, LabelInfo],
    page_units: dict[PageKey, PageUnits],
    question: str,
) -> GroupCheck:
    """Check one answer text against its citations.

    Args:
        where: `answer` or `option <n>`.
        text: The text whose numbers must be grounded.
        cites: Labels the model cited for it.
        labels: The request's labels.
        page_units: Units of the cited pages.
        question: The question.

    Returns:
        The group's check.
    """
    problems = []
    unknown = [label for label in cites if label not in labels]
    if unknown:
        problems.append(f"{where}: unknown labels {', '.join(unknown)}")
    if not 1 <= len(cites) <= MAX_CITATIONS:
        problems.append(f"{where}: {len(cites)} citations; cite 1 to {MAX_CITATIONS} passages")
    known = list(dict.fromkeys(label for label in cites if label in labels))
    full = complete(text, known, labels, page_units, question)
    sources, evidence = expand(full, labels, page_units)
    missing = ungrounded(text, [*sources, question])
    if missing:
        problems.append(_numbers_problem(where, missing, "the passages you cited"))
    return GroupCheck(where=where, cited=list(cites), added=full[len(known) :], evidence=evidence, problems=problems)


def verify(
    output: ModelOutput, labels: dict[str, LabelInfo], page_units: dict[PageKey, PageUnits], question: str
) -> VerifyResult:
    """Check an output against the request's labels, with the three expansions.

    Args:
        output: A schema-valid output.
        labels: The request's labels.
        page_units: Units of the pages of the labels, from their current chunks.
        question: The question as typed.

    Returns:
        The result; a `not_found` output always passes.
    """
    if isinstance(output, Answered):
        groups = [_check_group("answer", output.answer, list(output.citations), labels, page_units, question)]
    elif isinstance(output, Ambiguous):
        groups = [
            _check_group(
                f"option {n}",
                f"{option.condition}; {option.answer}",
                list(option.citations),
                labels,
                page_units,
                question,
            )
            for n, option in enumerate(output.options, start=1)
        ]
    else:
        return VerifyResult(passed=True, problems=[], added=[], groups=[])
    problems = [problem for group in groups for problem in group.problems]
    if isinstance(output, Ambiguous):
        every = [label for group in groups for label in [*group.cited, *group.added] if label in labels]
        sources, _ = expand(every, labels, page_units)
        missing = ungrounded(output.depends_on, [*sources, question])
        if missing:
            problems.append(_numbers_problem("depends_on", missing, "the passages the options cite"))
    added = [label for group in groups for label in group.added]
    return VerifyResult(passed=not problems, problems=problems, added=added, groups=groups)


def retry_message(problems: list[str]) -> str:
    """Write the user turn of the verifier retry (DECISIONS #17).

    Args:
        problems: The verifier's problems.

    Returns:
        The message: what failed, and what to do.
    """
    return (
        f"Verification failed: {'; '.join(problems)}. Cite the passages that contain every number, or remove the"
        " numbers. Answer again."
    )
