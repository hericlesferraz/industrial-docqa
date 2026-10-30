"""Citable units of a page (DECISIONS #7) and the forms they are shown in.

A deterministic splitter divides canonical page Markdown into headings, sentences or list items, and table rows. Chunks
are built from whole units, the answer prompt labels them, and a citation returns their stored text. Table header
rows are flagged for the verifier.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

ABBR = r"(?<!\bincl)(?<!\bFig)(?<!\bNo)(?<!\bex)(?<!\be\.g)(?<!\bi\.e)(?<!\bnr)(?<!\bp)"
SENTENCE_BREAK = re.compile(ABBR + r"(?<=[.!?;])\s+(?=[\"“(]?[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9])")
SEPARATOR_ROW = re.compile(r"\|[\s|:-]+\|")
FOOTNOTE_LINE = re.compile(r"^\*\s{2,}\S")
FOOTNOTE_START = re.compile(r"^\*\s")
STARRED_CELL = re.compile(r"\|\s*\*(?!\*)\S")
STARRED_VALUE = re.compile(r"\|\s*(\*(?!\*)[\d.,]+[^\s|]*)")

UnitKind = Literal["heading", "text", "row"]


@dataclass(frozen=True)
class Unit:
    """A citable piece of a page: a heading, a sentence or list item, or a table row.

    Attributes:
        pos: 0-based position in the page's unit list.
        kind: `heading`, `text` or `row`.
        md: Stored Markdown of the unit, stripped.
        table: 1-based table number on the page, for rows.
        header: True for a header row: before the separator line, then the leading all-bold rows.
    """

    pos: int
    kind: UnitKind
    md: str
    table: int | None = None
    header: bool = False

    def to_json(self) -> dict[str, object]:
        """Give the stored form of the unit.

        Returns:
            `{"pos", "kind", "md", "table", "header"}`.
        """
        return {"pos": self.pos, "kind": self.kind, "md": self.md, "table": self.table, "header": self.header}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "Unit":
        """Rebuild a unit from its stored form.

        Args:
            data: Output of `to_json`.

        Returns:
            The unit.
        """
        table = data.get("table")
        return cls(
            pos=int(str(data["pos"])),
            kind=str(data["kind"]),
            md=str(data["md"]),
            table=int(str(table)) if table is not None else None,
            header=bool(data.get("header", False)),
        )


@dataclass
class _Table:
    """Rows of one table while the page is split.

    Attributes:
        rows: Indexes, in the unit list, of the table's rows.
        before_separator: How many of the rows came before the table's last separator line.
        separator_seen: Whether the table has a separator line.
    """

    rows: list[int] = field(default_factory=list)
    before_separator: int = 0
    separator_seen: bool = False


def _all_bold(md: str) -> bool:
    """Tell whether every non-empty cell of a row is bold.

    Args:
        md: Stored Markdown of the row.

    Returns:
        True when the row has a non-empty cell and every non-empty cell starts and ends with `**`.
    """
    cells = [cell.strip() for cell in md.strip("|").split("|") if cell.strip()]
    return bool(cells) and all(cell.startswith("**") and cell.endswith("**") for cell in cells)


def _header_rows(table: _Table, kinds: list[dict[str, object]]) -> set[int]:
    """Find a table's header rows, the ones the verifier attaches to a cited row.

    The rows before the last separator line are headers; without a separator line, the first row is. Then the
    all-bold rows right after them are headers too, until the first row that isn't.

    Args:
        table: The table's rows.
        kinds: Raw units of the page (`kind`, `md`, `table`), indexed like `table.rows`.

    Returns:
        Indexes of the header rows.
    """
    if not table.rows:
        return set()
    start = table.before_separator if table.separator_seen else 1
    headers = set(table.rows[:start])
    for index in table.rows[start:]:
        if not _all_bold(str(kinds[index]["md"])):
            break
        headers.add(index)
    return headers


def units(md: str) -> list[Unit]:
    """Split page Markdown into units.

    One unit per heading line and per table row (separator lines are skipped), and one per sentence of the other
    lines. A blank line or a non-table line ends a table; the next table line starts a new table number.

    Args:
        md: Canonical page text.

    Returns:
        The units in page order.
    """
    raw: list[dict[str, object]] = []
    tables: dict[int, _Table] = {}
    table, in_table = 0, False
    for line in md.splitlines():
        stripped = line.strip()
        if not stripped:
            in_table = False
            continue
        if stripped.startswith("|"):
            if not in_table:
                table += 1
                in_table = True
                tables[table] = _Table()
            current = tables[table]
            if SEPARATOR_ROW.fullmatch(stripped):
                current.separator_seen = True
                current.before_separator = len(current.rows)
                continue
            current.rows.append(len(raw))
            raw.append({"kind": "row", "md": stripped, "table": table})
            continue
        in_table = False
        if stripped.startswith("#"):
            raw.append({"kind": "heading", "md": stripped, "table": None})
            continue
        for part in SENTENCE_BREAK.split(stripped):
            if part.strip():
                raw.append({"kind": "text", "md": part.strip(), "table": None})
    headers: set[int] = set()
    for current in tables.values():
        headers |= _header_rows(current, raw)
    return [
        Unit(
            pos=pos,
            kind=str(item["kind"]),
            md=str(item["md"]),
            table=int(str(item["table"])) if item["table"] is not None else None,
            header=pos in headers,
        )
        for pos, item in enumerate(raw)
    ]


def plain(md: str) -> str:
    """Render Markdown as plain text: no markup, cells joined with ` | `, whitespace collapsed.

    Used for chunk texts, embeddings and the full-text index (the form measured in DECISIONS #10). It drops a leading
    list or footnote marker, so a footnote loses its `*` here; `quote_form` and `prompt_form` keep it.

    Args:
        md: Markdown of a unit or a line.

    Returns:
        The plain text; empty cells are dropped.
    """
    text = re.sub(r"<br\s*/?>", " ", md)
    text = re.sub(r"</?sup>", "", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"^\s*(#+|>|[-*]\s|\d+\.\s{2,})\s*", "", text)
    if text.startswith("|"):
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        text = " | ".join(cell for cell in cells if cell)
    return re.sub(r"\s+", " ", text).strip()


def prompt_form(unit: Unit) -> str:
    """Render a unit as the answering model reads it.

    A row keeps every cell, empty ones included, so its columns stay aligned with the header: only one pipe is removed
    from each end of the line. A footnote keeps its `*` marker, which links it to the starred cells.

    Args:
        unit: The unit.

    Returns:
        The rendered text; numbers in it are what an answer is grounded against.
    """
    md = unit.md
    if unit.kind == "row":
        line = md.strip().removeprefix("|").removesuffix("|")
        cells = [re.sub(r"\*\*|__|</?sup>", "", cell).replace("<br>", " ").strip() for cell in line.split("|")]
        return "| " + " | ".join(cells) + " |"
    if FOOTNOTE_LINE.match(md):
        return "* " + plain(md[1:])
    return plain(md)


def quote_form(unit: Unit) -> str:
    """Render a unit as a citation quote: its plain text, with a footnote's `*` marker kept.

    Args:
        unit: The unit.

    Returns:
        The quote text of the unit.
    """
    if FOOTNOTE_LINE.match(unit.md):
        return "* " + plain(unit.md[1:])
    return plain(unit.md)


def starred_cells(md: str) -> list[str]:
    """Return the starred values of a row, the cells a footnote explains, each cut at its cell border.

    A starred value opens its cell and ends at the cell border. A pattern that ran on to the next
    whitespace, past the `|`, and also matched the second `*` of bold numbers (`**3600**`), so on rows without spaces
    it returned the rest of the row and the bold header cells; DECISIONS #17 grounds a cited footnote on the starred
    cells only.

    Args:
        md: Stored Markdown of a row.

    Returns:
        Values such as `*2200Hrs.`, in order.
    """
    return STARRED_VALUE.findall(md)


def is_footnote(unit: Unit, page_units: list[Unit]) -> bool:
    """Tell whether a unit is the footnote of a starred table cell on its page.

    A line that starts with `*` is a footnote only when the page also has a starred cell; on a transcribed page `* ` is
    also the list marker.

    Args:
        unit: The unit.
        page_units: Every unit of its page.

    Returns:
        True for a footnote.
    """
    if unit.kind != "text" or not FOOTNOTE_START.match(unit.md):
        return False
    return any(other.kind == "row" and STARRED_CELL.search(other.md) for other in page_units)
