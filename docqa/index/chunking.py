"""Chunks of a page (DECISIONS #10): windows of about 250 tokens made of whole units, never crossing a page.

A table, its caption and its footnotes form one block, split by rows only when the block alone exceeds the budget.
Every piece repeats the table's first row. The pieces after the first also carry the rest of the table's head (its
other header rows, then the rows up to the first mostly numeric row: column names, a min/max line, units) among their
units, which the answering model reads, but not in their text, which the search reads. A title split by the extractor
is then never the only context the model gets for a piece, while the windows of one table don't all look alike to the
search: with the column names in every window's text, the window holding the asked row of a 30-row table fell from
1st to 22nd and the column words stopped being rare terms. A heading that would close a window opens the next one
instead. Every chunk starts with a context line: the file stem and the nearest heading above the chunk.
"""

import re
from dataclasses import dataclass

from docqa.text.units import Unit, plain, quote_form, units

CHUNKER_VERSION = "win250-head4"
MAX_CHUNK_TOKENS = 250
CAPTION_MAX_CHARS = 120
HEAD_MAX_ROWS = 4
NOTE = re.compile(r"^(\*+|\(\d+\)|\d\)|[¹²³]|notes?\b|nota\b|obs\b)", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    """A window of whole units of one page.

    Attributes:
        ordinal: Position of the chunk within the page, from 0.
        text: Embedded and full-text indexed text: the context line, then the plain form of each unit except the
            repeated head rows of a split table.
        units: The chunk's units, in page order, repeated head rows included.
        tokens: Estimated size of `text` in tokens.
    """

    ordinal: int
    text: str
    units: list[Unit]
    tokens: int


def ntok(text: str) -> int:
    """Estimate the number of tokens of a text.

    Args:
        text: Any text.

    Returns:
        `len(text) // 4`, at least 1.
    """
    return max(1, len(text) // 4)


def _size(group: list[Unit]) -> int:
    """Estimate the tokens of a group of units.

    Args:
        group: Units.

    Returns:
        Estimated tokens of their plain forms joined by spaces.
    """
    return ntok(" ".join(plain(unit.md) for unit in group))


def blocks(page_units: list[Unit]) -> list[list[Unit]]:
    """Group units into blocks: a table with its caption and its footnotes is one block, any other unit is alone.

    The caption is the short unit right before the table. The footnotes are the text units right after it that start
    with a note marker; a footnote's `*` is read from its quote form, which keeps it.

    Args:
        page_units: Units of one page.

    Returns:
        Blocks in page order.
    """
    out: list[list[Unit]] = []
    i = 0
    while i < len(page_units):
        unit = page_units[i]
        if unit.kind != "row":
            out.append([unit])
            i += 1
            continue
        j = i
        while j < len(page_units) and page_units[j].kind == "row" and page_units[j].table == unit.table:
            j += 1
        block = page_units[i:j]
        if out and len(out[-1]) == 1 and out[-1][0].kind != "row" and len(plain(out[-1][0].md)) < CAPTION_MAX_CHARS:
            block = out.pop() + block
        while j < len(page_units) and page_units[j].kind == "text" and NOTE.match(quote_form(page_units[j])):
            block.append(page_units[j])
            j += 1
        out.append(block)
        i = j
    return out


def numeric_row(unit: Unit) -> bool:
    """Tell whether a table row holds data: at least half of its cells have at least as many digits as letters.

    Args:
        unit: A row.

    Returns:
        True for `100 | 120 | 0,050`, `Upto 210 incl.(132) | 5500Hrs. | 12000Hrs.`; False for `over | incl. | min`.
    """
    cells = [cell for cell in plain(unit.md).split(" | ") if cell]
    numeric = 0
    for cell in cells:
        digits = sum(ch.isdecimal() for ch in cell)
        numeric += digits > 0 and digits >= sum(ch.isalpha() for ch in cell)
    return bool(cells) and 2 * numeric >= len(cells)


def table_head(rows: list[Unit]) -> list[Unit]:
    """Choose the head of a table, the rows every piece of it starts with.

    The first row always, then the other leading header rows, then the rows before the first mostly numeric row (column
    names, a min/max line, units) when that row comes within the first `HEAD_MAX_ROWS`; a table without such a row
    (a text table) keeps its header rows only.

    Args:
        rows: The table's rows, in order.

    Returns:
        The head rows, a prefix of `rows` that leaves at least one row out.
    """
    flagged = 1
    while flagged < len(rows) - 1 and rows[flagged].header:
        flagged += 1
    end = flagged
    while end < min(len(rows) - 1, HEAD_MAX_ROWS) and not numeric_row(rows[end]):
        end += 1
    data_found = end < len(rows) - 1 and numeric_row(rows[end])
    return rows[:end] if data_found else rows[:flagged]


def split_table(block: list[Unit]) -> list[tuple[list[Unit], list[Unit]]]:
    """Split a table block that exceeds the budget into pieces, by rows.

    Every piece starts with the first row; the pieces after the first also repeat the rest of the table's head
    (`table_head`) for the reader. Sizes count what the search reads, so the repeated rows never push a data row out.
    The caption goes with every piece; the footnotes go with the last one. A head longer than half the budget is not
    repeated.

    Args:
        block: A block with at least one row.

    Returns:
        `(units, repeated)` per piece, in order: its units, and the head rows among them repeated for the reader only.
    """
    rows = [unit for unit in block if unit.kind == "row"]
    first_row = block.index(rows[0])
    last_row = block.index(rows[-1])
    head = table_head(rows)
    extra = head[1:] if _size(head) <= MAX_CHUNK_TOKENS // 2 else []
    before = [unit for n, unit in enumerate(block) if unit.kind != "row" and n < first_row]
    after = [unit for n, unit in enumerate(block) if unit.kind != "row" and n > last_row]
    parts: list[list[Unit]] = []
    current: list[Unit] = []
    for row in rows[1:]:
        if current and _size(before + rows[:1] + current + [row]) > MAX_CHUNK_TOKENS:
            parts.append(current)
            current = []
        current.append(row)
    parts.append(current)
    pieces = []
    for n, part in enumerate(parts):
        repeated = extra if n else []
        tail = after if n == len(parts) - 1 else []
        pieces.append((before + rows[:1] + repeated + part + tail, repeated))
    return pieces


def _searched(group: list[Unit], repeated: set[int]) -> list[Unit]:
    """Keep the units of a window that the search reads.

    Args:
        group: Units of a window.
        repeated: Positions of the head rows the window repeats for the reader.

    Returns:
        The units without the repeated head rows.
    """
    return [unit for unit in group if unit.pos not in repeated]


def _windows(page_units: list[Unit]) -> list[tuple[list[Unit], set[int]]]:
    """Pack the blocks of a page into windows of at most `MAX_CHUNK_TOKENS` of searched text.

    Args:
        page_units: Units of one page.

    Returns:
        The windows: each a list of whole units, and the positions of the head rows it repeats for the reader only.
    """
    groups: list[tuple[list[Unit], set[int]]] = []
    current: list[Unit] = []
    repeated: set[int] = set()
    for block in blocks(page_units):
        too_big = _size(block) > MAX_CHUNK_TOKENS and any(unit.kind == "row" for unit in block)
        for piece, extra in split_table(block) if too_big else [(block, [])]:
            searched = _searched(current, repeated) + [unit for unit in piece if unit not in extra]
            if current and _size(searched) > MAX_CHUNK_TOKENS:
                keep = len(current)
                while keep > 0 and current[keep - 1].kind == "heading":
                    keep -= 1
                if keep:
                    groups.append((current[:keep], repeated))
                    current, repeated = current[keep:], set()
            current = current + piece
            repeated = repeated | {unit.pos for unit in extra}
    if current:
        groups.append((current, repeated))
    return groups


def chunk_page(filename: str, page_no: int, md: str) -> list[Chunk]:
    """Cut one page into chunks of whole units.

    Args:
        filename: Document filename; its stem opens every chunk's context line.
        page_no: 1-based page, for the caller's bookkeeping.
        md: Canonical page text.

    Returns:
        The page's chunks in order; none for a page without units.
    """
    page_units = units(md)
    if not page_units:
        return []
    stem = filename.rsplit(".", 1)[0]
    chunks: list[Chunk] = []
    heading = ""
    for ordinal, (group, repeated) in enumerate(_windows(page_units)):
        context = "" if group[0].kind == "heading" else heading
        for unit in group:
            if unit.kind == "heading":
                heading = plain(unit.md)
        body = "\n".join(plain(unit.md) for unit in _searched(group, repeated))
        text = f"{stem} — {context}\n{body}" if context else f"{stem}\n{body}"
        chunks.append(Chunk(ordinal=ordinal, text=text, units=group, tokens=ntok(text)))
    return chunks
