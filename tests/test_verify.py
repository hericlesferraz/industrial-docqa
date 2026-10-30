"""The verifier and its three expansions, with the cases of DECISIONS #7 and #17 on real page texts."""

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from docqa.answer.prompt import LabelInfo, build_passages
from docqa.answer.schema import Ambiguous, Answered, NotFound, Option
from docqa.answer.verify import PageKey, retry_message, verify
from docqa.index.chunking import chunk_page
from docqa.repo.chunks import PageUnits
from docqa.retrieval.search import Candidate
from docqa.text.units import STARRED_CELL, units

PAGES = json.loads((Path(__file__).parent / "fixtures" / "verifier_pages.json").read_text(encoding="utf-8"))
DOC = UUID(int=7)
CESTARI_TABLE = """# **Intervalos de troca de óleo**

|**Tipo de óleo**|**Temperatura**|**Intervalo**|
|---|---|---|
|CLP PG|90 °C|13.000 horas|
|CLP HC|90 °C|8.000 horas|
"""


def _page(md: str, filename: str, page_no: int, text_rev: int = 1) -> tuple[dict[str, LabelInfo], PageUnits]:
    """Label every unit of one page, as if all its chunks were retrieved.

    Args:
        md: Page text.
        filename: Document filename.
        page_no: Page number.
        text_rev: Text revision of the chunks.

    Returns:
        The labels and the page's units.
    """
    candidates = [
        Candidate(
            rank=n,
            chunk_id=uuid4(),
            document_id=DOC,
            filename=filename,
            page_no=page_no,
            ordinal=chunk.ordinal,
            text_rev=text_rev,
            text_source="layer",
            units=chunk.units,
            vec_rank=n,
            lex_rank=None,
        )
        for n, chunk in enumerate(chunk_page(filename, page_no, md), start=1)
    ]
    _, labels = build_passages(candidates)
    return labels, PageUnits(document_id=DOC, page_no=page_no, text_rev=1, text_source="layer", units=units(md))


@pytest.fixture
def lb5001() -> tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]:
    """Give the labels and units of LB5001 page 2 (relubrication intervals, footnote, grease table).

    Returns:
        The labels and the page units by `(document_id, page_no)`.
    """
    page = PAGES["lb5001_p2"]
    labels, units_ = _page(page["text"], page["filename"], page["page"])
    return labels, {(DOC, page["page"]): units_}


def _label(labels: dict[str, LabelInfo], needle: str) -> str:
    """Find the label of the unit that contains a text.

    Args:
        labels: Request labels.
        needle: Text inside the unit's stored Markdown.

    Returns:
        The first label whose unit contains it.
    """
    return next(label for label, info in labels.items() if needle in info.md)


def _answered(cites: list[str], text: str) -> Answered:
    """Build an answered output without schema validation, so a test can break the limits.

    Args:
        cites: Cited labels.
        text: Answer text.

    Returns:
        The output.
    """
    return Answered.model_construct(status="answered", citations=cites, answer=text)


def test_header_rows_ground_the_speed(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """Expansion 1: the speed sits in a header row the model didn't cite.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    row = _label(labels, "Over 210 to 280 incl.(180)|3600Hrs.")
    output = _answered([row], "9,500 hours at 1800 RPM.")
    result = verify(output, labels, page_units, "What is the relubrication interval for frame 256?")
    assert result.passed, result.problems
    attached = [evidence.unit.md for evidence in result.groups[0].evidence]
    assert "||**3600**|**1800**|**1200**|**900**|" in attached
    assert verify(output, labels, {}, "frame 256?").problems == [
        "answer: the numbers 1800 have no source in the passages you cited"
    ]


def test_starred_row_brings_its_footnote(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """Expansion 2: a cited starred row brings the footnote, whose bearing numbers then ground the answer.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    row = _label(labels, "Over 360 to 5000 incl.(300)|*2200Hrs.")
    result = verify(_answered([row], "2200 hours for 6313 or 6314 bearings."), labels, page_units, "frame 400?")
    assert result.passed, result.problems
    assert any(evidence.unit.md.startswith("*  Lubrication interval") for evidence in result.groups[0].evidence)


def test_cited_footnote_grounds_only_starred_cells(
    lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]],
) -> None:
    """Expansion 2: a cited footnote grounds the starred values, not the rest of the starred rows.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    note = _label(labels, "*  Lubrication interval")
    assert verify(_answered([note], "2200 hours."), labels, page_units, "2 pole?").passed
    failed = verify(_answered([note], "7400 hours."), labels, page_units, "2 pole?")
    assert failed.problems == ["answer: the numbers 7400 have no source in the passages you cited"]
    assert [evidence.unit.md for evidence in failed.groups[0].evidence] == [labels[note].md]


def test_bold_header_is_not_a_footnote_marker(
    lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]],
) -> None:
    """`**3600**` is bold markup, not a starred cell: citing the speed row attaches headers, never the footnote.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    speeds = _label(labels, "||**3600**|**1800**")
    assert STARRED_CELL.search(labels[speeds].md) is None
    result = verify(_answered([speeds], "3600 RPM."), labels, page_units, "speeds?")
    assert result.passed
    assert not any(evidence.unit.md.startswith("*") for evidence in result.groups[0].evidence)


def test_sentence_expansion_never_adds_a_row(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """Expansion 3 adds an uncited sentence of a cited page, but a value from another row still fails.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    row = _label(labels, "Upto 210 incl.(132)|5500Hrs.")
    halved = verify(_answered([row], "5500 hours; for roller bearings divide by 2."), labels, page_units, "?")
    note = _label(labels, "*  Lubrication interval")
    assert halved.passed and halved.added == [note]
    other_row = verify(_answered([row], "5500 or 9500 hours."), labels, page_units, "?")
    assert other_row.problems == ["answer: the numbers 9500 have no source in the passages you cited"]
    assert other_row.added == []


def test_citation_cap_and_unknown_labels(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """More than 6 citations fail, so citing everything can't ground anything; an unknown label fails.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    seven = list(labels)[:7]
    assert verify(_answered(seven, "Lubrication."), labels, page_units, "?").problems == [
        "answer: 7 citations; cite 1 to 6 passages"
    ]
    assert verify(_answered(["S999"], "Lubrication."), labels, page_units, "?").problems == [
        "answer: unknown labels S999"
    ]


def test_labels_in_the_text_are_not_numbers(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """A label written in the answer is not a number to ground.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    row = _label(labels, "Over 210 to 280 incl.(180)|3600Hrs.")
    assert verify(_answered([row], f"As {row} shows, 9500Hrs."), labels, page_units, "?").passed


def test_locale_readings_in_a_portuguese_table() -> None:
    """`13,000 hours` in an English answer is grounded by `13.000 horas` in the cited row."""
    labels, page_units = _page(CESTARI_TABLE, "cestari.pdf", 18)
    row = _label(labels, "CLP PG")
    result = verify(_answered([row], "13,000 hours."), labels, {(DOC, 18): page_units}, "CLP PG at 90 °C?")
    assert result.passed, result.problems


def test_converted_unit_is_rejected() -> None:
    """A conversion the page doesn't print fails (`48.8 m`); the printed values pass."""
    page = PAGES["mn414_p8"]
    labels, page_units = _page(page["text"], page["filename"], page["page"])
    depth = _label(labels, "160 ft (49m)")
    units_by_page = {(DOC, page["page"]): page_units}
    assert verify(_answered([depth], "160 ft (49 m)."), labels, units_by_page, "depth?").passed
    assert verify(_answered([depth], "About 48.8 m."), labels, units_by_page, "depth?").problems == [
        "answer: the numbers 48.8 have no source in the passages you cited"
    ]


def test_ambiguous_options_and_depends_on(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """Each option is checked like an answer; `depends_on` is grounded by every option's sources together.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    small = _label(labels, "Upto 210 incl.(132)|5500Hrs.")
    large = _label(labels, "Over 210 to 280 incl.(180)|3600Hrs.")
    options = [
        Option(condition="Frame up to 210", citations=[small], answer="5500 h at 3600 RPM"),
        Option(condition="Frame 210 to 280", citations=[large], answer="3600 h at 3600 RPM"),
    ]
    good = Ambiguous(status="ambiguous", depends_on="Frame size, from 132 to 180 (IEC).", options=options)
    assert verify(good, labels, page_units, "How often?").passed
    bad = Ambiguous(status="ambiguous", depends_on="Frame size, up to 5000.", options=options)
    assert verify(bad, labels, page_units, "How often?").problems == [
        "depends_on: the numbers 5000 have no source in the passages the options cite"
    ]
    wrong_option = Ambiguous(
        status="ambiguous",
        depends_on="Frame size.",
        options=[options[0], Option(condition="Frame 210 to 280", citations=[large], answer="12000 h")],
    )
    assert verify(wrong_option, labels, page_units, "How often?").problems == [
        "option 2: the numbers 12000 have no source in the passages you cited"
    ]


def test_condition_and_answer_numbers_never_merge(
    lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]],
) -> None:
    """The last number of a condition and the first of its answer stay two numbers: `210` + `5,500` is not 2105.5.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    small = _label(labels, "Upto 210 incl.(132)|5500Hrs.")
    option = Option(condition="Frame up to 210", citations=[small], answer="5,500 h at 3600 RPM")
    output = Ambiguous(status="ambiguous", depends_on="Frame size.", options=[option, option])
    assert verify(output, labels, page_units, "How often?").passed


def test_changed_page_skips_expansions(lb5001: tuple[dict[str, LabelInfo], dict[PageKey, PageUnits]]) -> None:
    """Expansions 1 and 2 read the page's current units only when the page revision is the labels' one.

    Args:
        lb5001: Labels and units of LB5001 page 2.
    """
    labels, page_units = lb5001
    row = _label(labels, "Over 210 to 280 incl.(180)|3600Hrs.")
    newer = {
        key: PageUnits(page.document_id, page.page_no, 2, page.text_source, page.units)
        for key, page in page_units.items()
    }
    assert not verify(_answered([row], "9500 hours at 1800 RPM."), labels, newer, "frame 256?").passed


def test_not_found_passes_and_retry_message() -> None:
    """`not_found` has nothing to check; the retry message lists the problems and what to do."""
    assert verify(NotFound(status="not_found", answer="No IP rating."), {}, {}, "IP?").passed
    message = retry_message(["option 2: the numbers 12000 have no source in the passages you cited"])
    assert message.startswith("Verification failed: option 2: the numbers 12000 have no source")
    assert message.endswith("Cite the passages that contain every number, or remove the numbers. Answer again.")
