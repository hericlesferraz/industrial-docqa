"""Units and chunks against the prototype outputs, and the forms a unit is shown in."""

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from docqa.index.chunking import MAX_CHUNK_TOKENS, blocks, chunk_page, ntok, numeric_row, split_table, table_head
from docqa.text.units import (
    STARRED_CELL,
    Unit,
    is_footnote,
    plain,
    prompt_form,
    quote_form,
    starred_cells,
    units,
)

GOLDEN = Path(__file__).parent / "fixtures" / "chunks_golden.json.gz"
BROCHURE = "WEG-motores-eletricos-guia-de-especificacao-50032749-brochure-portuguese-web.pdf"
HEAD_REPEATED = {(BROCHURE, 27), (BROCHURE, 55)}
SKF_F2 = """|**Drive-u**|**p data for s**|**pherical rol**|**ler bearing**|
|---|---|---|---|
|**Bore**<br>**diamete**<br>d|**r**|**Reducti**<br>**radial in**|**on of**<br>**ternal**|
|over|incl.|min|max|
|mm||mm||
""" + "".join(f"|**{n}**|**{n + 20}**|0,{n:03d}|0,{n + 10:03d}|\n" for n in range(20, 1000, 20))
LB5001_P2_TABLE = """New motors should be relubricated:

||**Relubricatio**|**n Intervals**|||
|---|---|---|---|---|
|**NEMA (IEC) Frame Size**|**Rated Speed**|**(RPM)**|||
||**3600**|**1800**|**1200**|**900**|
|Upto 210 incl.(132)|5500Hrs.|12000Hrs.|18000Hrs.|22000Hrs.|
|Over 280 to 360 incl.(225)|*2200Hrs.|7400Hrs.|12000Hrs.|15000Hrs.|

*  Lubrication interval for 6313 or 6314 bearings that are used in 360 through 5000 frame, 2 pole motors.

# **Lubricant**

Baldor motors are pregreased, normally with Polyrex EM (Exxon Mobil). If other greases are preferred, ask.
"""


def _golden() -> list[dict[str, Any]]:
    """Load the prototype outputs: units with header flags and chunks of the 166 pages measured in DECISIONS #10.

    Returns:
        One entry per page.
    """
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def golden() -> list[dict[str, Any]]:
    """Give the golden pages once per module.

    Returns:
        One entry per page.
    """
    return _golden()


def test_units_match_the_prototype(golden: list[dict[str, Any]]) -> None:
    """Every page splits into the prototype's units, with the verifier's header rows.

    Args:
        golden: Prototype outputs.
    """
    for page in golden:
        got = [[unit.kind, unit.md, unit.table, unit.header] for unit in units(page["md"])]
        assert got == page["units"], (page["file"], page["page"])


def test_chunks_match_the_prototype(golden: list[dict[str, Any]]) -> None:
    """Every page cuts into the prototype's chunks: same texts, sizes and units; 533 chunks in all.

    The only change: on the two pages whose long tables are split, the later pieces also carry the table's column
    names among their units (for the answering model), never in their text (for the search).

    Args:
        golden: Prototype outputs.
    """
    total = 0
    repeated_on: set[tuple[str, int]] = set()
    for page in golden:
        page_units = units(page["md"])
        chunks = chunk_page(page["file"], page["page"], page["md"])
        assert [chunk.text for chunk in chunks] == [c["text"] for c in page["chunks"]], (page["file"], page["page"])
        assert [chunk.tokens for chunk in chunks] == [c["tokens"] for c in page["chunks"]]
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
        seen_tables: set[int | None] = set()
        for chunk, expected in zip(chunks, page["chunks"], strict=True):
            copies = []
            for unit in chunk.units:
                rows = [u for u in page_units if u.kind == "row" and u.table == unit.table]
                if unit.kind == "row" and unit.table in seen_tables and unit in table_head(rows)[1:]:
                    copies.append(unit)
            assert [u.md for u in chunk.units if u not in copies] == [page["units"][pos][1] for pos in expected["pos"]]
            assert all(page_units[unit.pos] == unit for unit in chunk.units)
            seen_tables |= {unit.table for unit in chunk.units if unit.kind == "row"}
            if copies:
                repeated_on.add((page["file"], page["page"]))
        total += len(chunks)
    assert total == 533
    assert repeated_on == HEAD_REPEATED


def test_header_flags_on_the_lb5001_tables() -> None:
    """Rows before the separator and the leading all-bold rows are headers; data rows are not."""
    rows = [unit for unit in units(LB5001_P2_TABLE) if unit.kind == "row"]
    assert [unit.header for unit in rows] == [True, True, True, False, False]
    assert {unit.table for unit in rows} == {1}


def test_header_without_separator_is_the_first_row() -> None:
    """A table without a separator line has its first row as header, then the all-bold rows."""
    md = "|a|b|\n|**x**|**y**|\n|1|2|\n|**3**|4|"
    assert [unit.header for unit in units(md)] == [True, True, False, False]


def test_quote_form_keeps_the_footnote_marker() -> None:
    """The footnote's `*` survives in the quote and the prompt; `plain` drops it, as DECISIONS #10 measured."""
    note = next(unit for unit in units(LB5001_P2_TABLE) if unit.md.startswith("*"))
    assert plain(note.md).startswith("Lubrication interval")
    assert quote_form(note).startswith("* Lubrication interval for 6313")
    assert prompt_form(note).startswith("* Lubrication interval for 6313")


def test_prompt_form_keeps_empty_cells() -> None:
    """Only one pipe is removed from each end, so leading and trailing empty cells keep the columns aligned."""
    row = Unit(pos=0, kind="row", md="||**Relubricatio**|**n Intervals**|||", table=1, header=True)
    assert prompt_form(row) == "|  | Relubricatio | n Intervals |  |  |"
    assert quote_form(row) == "Relubricatio | n Intervals"


def test_bold_number_is_not_a_starred_cell() -> None:
    """`**3600**` is bold markup, never a footnote marker; `*2200Hrs.` is (DECISIONS #17)."""
    page_units = units(LB5001_P2_TABLE)
    bold_row = next(unit for unit in page_units if "3600" in unit.md)
    starred_row = next(unit for unit in page_units if "*2200" in unit.md)
    assert not STARRED_CELL.search(bold_row.md)
    assert STARRED_CELL.search(starred_row.md)
    assert starred_cells(starred_row.md) == ["*2200Hrs."]
    assert starred_cells(bold_row.md) == []


def test_footnote_needs_a_starred_cell_on_the_page() -> None:
    """A `*` line is a footnote only when the page has a starred cell; on a transcribed page `* ` marks list items."""
    page_units = units(LB5001_P2_TABLE)
    note = next(unit for unit in page_units if unit.md.startswith("*"))
    assert is_footnote(note, page_units)
    ocr_list = units("Check:\n\n* the oil level\n* the seals\n\n|a|b|\n|---|---|\n|**1**|2|")
    item = next(unit for unit in ocr_list if unit.md.startswith("*"))
    assert not is_footnote(item, ocr_list)


def test_unit_json_round_trip() -> None:
    """The stored form rebuilds the same unit."""
    for unit in units(LB5001_P2_TABLE):
        assert Unit.from_json(json.loads(json.dumps(unit.to_json()))) == unit


def test_table_block_keeps_caption_and_starred_footnote() -> None:
    """A table, its caption and its `*` footnote form one block (DECISIONS #10)."""
    page_units = units(LB5001_P2_TABLE)
    table_block = next(block for block in blocks(page_units) if any(unit.kind == "row" for unit in block))
    assert table_block[0].md == "New motors should be relubricated:"
    assert table_block[-1].md.startswith("*  Lubrication interval")


def test_big_table_splits_by_rows_with_the_first_row_repeated() -> None:
    """A table above the budget is cut by rows; every piece repeats the first row and the caption, none crosses it."""
    rows = "\n".join(f"|Frame {n} incl. ({n * 10})|{n * 100}Hrs.|{n * 200}Hrs.|{n * 300}Hrs.|" for n in range(1, 60))
    md = f"Intervals by frame:\n\n|**Frame**|**3600**|**1800**|**1200**|\n|---|---|---|---|\n{rows}\n\nNote: grease."
    chunks = chunk_page("big.pdf", 1, md)
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.units[0].md == "Intervals by frame:"
        assert chunk.units[1].md.startswith("|**Frame**")
        assert chunk.tokens <= MAX_CHUNK_TOKENS + ntok("big — ") + 10
    assert chunks[-1].units[-1].md == "Note: grease."
    data_rows = [unit.pos for chunk in chunks for unit in chunk.units if unit.kind == "row" and not unit.header]
    assert data_rows == sorted(set(data_rows))
    assert len(data_rows) == 59


def test_context_line_has_file_stem_and_nearest_heading() -> None:
    """A chunk starts with the file stem and the nearest heading above it; a chunk that opens with a heading has none."""
    body = " ".join(f"Sentence number {n} about bearings." for n in range(60))
    md = f"# **Bearings**\n\n{body}\n\nAnother sentence."
    chunks = chunk_page("manual.v2.pdf", 3, md)
    assert chunks[0].text.startswith("manual.v2\nBearings\n")
    assert chunks[1].text.startswith("manual.v2 — Bearings\n")
    assert chunk_page("x.pdf", 1, "\n\n  \n") == []


def test_numeric_rows_and_the_repeated_head() -> None:
    """Data rows are mostly numeric; later pieces repeat the head for the reader, and the search text stays the same."""
    rows = [unit for unit in units(SKF_F2) if unit.kind == "row"]
    assert [numeric_row(unit) for unit in rows[:5]] == [False, False, False, False, True]
    assert numeric_row(Unit(0, "row", "|Upto 210 incl.(132)|5500Hrs.|12000Hrs.|18000Hrs.|"))
    head = table_head(rows)
    assert [unit.pos for unit in head] == [0, 1, 2, 3]
    pieces = split_table(rows)
    assert len(pieces) >= 2 and pieces[0][1] == []
    assert all(piece[: len(head)] == head and repeated == head[1:] for piece, repeated in pieces[1:])
    assert [u for piece, repeated in pieces for u in piece[1:] if u not in repeated] == rows[1:]
    chunks = chunk_page("skf.pdf", 407, SKF_F2)
    assert len(chunks) >= 2
    assert all("Reducti radial in" in "\n".join(plain(u.md) for u in c.units) for c in chunks)
    assert ["Reducti radial in" in c.text for c in chunks] == [True] + [False] * (len(chunks) - 1)


def test_a_text_table_keeps_its_header_rows_only() -> None:
    """Without a data row within the first rows, only the first row repeats, as before."""
    md = "|Task|Interval|\n|---|---|\n" + "".join(f"|Inspect the part number {n} and clean it|yearly|\n" for n in range(60))
    rows = [unit for unit in units(md) if unit.kind == "row"]
    assert table_head(rows) == rows[:1]
    assert all(piece[0] == rows[0] and repeated == [] for piece, repeated in split_table(rows))
