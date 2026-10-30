"""Passage assembly `b2`: grouping by page, unit order, dedup by stored text, labels."""

from uuid import UUID, uuid4

from docqa.answer.prompt import ASSEMBLY_VERSION, build_passages
from docqa.index.chunking import chunk_page
from docqa.retrieval.search import Candidate
from docqa.text.units import Unit, units

DOC_A = UUID(int=1)
DOC_B = UUID(int=2)
TABLE = """# **Intervals**

||**Speed**|||
|---|---|---|---|
|Frame|3600|1800|1200|
|Upto 210|*2200Hrs.|12000Hrs.||
|Upto 210|*2200Hrs.|12000Hrs.||
|Upto 210|**2200Hrs.**|12000Hrs.||

*  Interval for 2 pole motors.
"""


def _candidate(
    rank: int, document_id: UUID, page_no: int, page_units: list[Unit], filename: str = "a.pdf"
) -> Candidate:
    """Build a candidate around some units.

    Args:
        rank: Fused rank.
        document_id: Document.
        page_no: Page.
        page_units: The chunk's units.
        filename: Document filename.

    Returns:
        The candidate.
    """
    return Candidate(
        rank=rank,
        chunk_id=uuid4(),
        document_id=document_id,
        filename=filename,
        page_no=page_no,
        ordinal=0,
        text_rev=3,
        text_source="ocr",
        units=page_units,
        vec_rank=rank,
        lex_rank=None,
    )


def test_passages_keep_cells_markers_and_lookalike_rows() -> None:
    """Rows keep empty cells, the footnote keeps `*`, identical rows appear once, look-alike rows both stay."""
    text, labels = build_passages([_candidate(1, DOC_A, 4, units(TABLE))])
    assert ASSEMBLY_VERSION == "b2"
    assert text.splitlines() == [
        "[a.pdf, page 4]",
        "S1: Intervals",
        "S2: |  | Speed |  |  |",
        "S3: | Frame | 3600 | 1800 | 1200 |",
        "S4: | Upto 210 | *2200Hrs. | 12000Hrs. |  |",
        "S5: | Upto 210 | 2200Hrs. | 12000Hrs. |  |",
        "S6: * Interval for 2 pole motors.",
    ]
    assert labels["S5"].md == "|Upto 210|**2200Hrs.**|12000Hrs.||"
    assert (labels["S4"].pos, labels["S6"].pos) == (3, 6)
    assert (labels["S2"].header, labels["S3"].header, labels["S4"].header) == (True, False, False)
    assert labels["S6"].unit.md.startswith("*  Interval")
    assert (labels["S1"].text_rev, labels["S1"].text_source) == (3, "ocr")


def test_pages_in_rank_order_units_in_page_order() -> None:
    """The best-ranked page comes first; chunks of one page merge in position order, a repeated header once."""
    rows = "\n".join(f"|Frame {n}|{n * 100}Hrs.|" for n in range(1, 100))
    page = f"|**Frame**|**Hours**|\n|---|---|\n{rows}"
    first, second = chunk_page("b.pdf", 7, page)[:2]
    other = units("# **Warranty**\n\nTwelve months.")
    text, labels = build_passages(
        [
            _candidate(1, DOC_B, 7, second.units, "b.pdf"),
            _candidate(2, DOC_A, 1, other),
            _candidate(3, DOC_B, 7, first.units, "b.pdf"),
        ]
    )
    lines = text.splitlines()
    assert lines[0] == "[b.pdf, page 7]"
    b_labels = [label for label, info in labels.items() if info.document_id == DOC_B]
    assert [labels[label].pos for label in b_labels] == sorted({u.pos for u in first.units + second.units})
    assert sum(1 for line in lines if "Frame | Hours" in line) == 1
    assert "[a.pdf, page 1]" in lines
    assert list(labels) == [f"S{n}" for n in range(1, len(labels) + 1)]


def test_no_candidates_no_passages() -> None:
    """Without candidates there is nothing to label."""
    assert build_passages([]) == ("", {})
