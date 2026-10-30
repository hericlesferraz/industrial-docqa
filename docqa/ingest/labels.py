"""Printed page labels read from the page margins and confirmed by neighbouring pages (DECISIONS #8).

A candidate is a line in the top or bottom 10% of the page shaped like a page number (`20`, `A3`, `2-2`, `ii`). It
is accepted only if a page at most two away has a candidate in the same series at the matching distance, so a lone
number in a margin (a figure label, a part number) is never taken for a page number.

Transcribed pages add candidates from the first and last lines of their transcription. Pages finish OCR in any order,
so every `index` run recomputes the labels of the whole document from both kinds of candidates.
"""

import re
from uuid import UUID

import pymupdf
from psycopg import Connection

from docqa.repo import pages

MARGIN = 0.10
NEIGHBOURS = 2
ARABIC = re.compile(r"^([A-Z]?)(\d{1,4})$")
SECTION = re.compile(r"^(\d{1,3})-(\d{1,3})$")
ROMAN = re.compile(r"^[ivxlc]{1,6}$", re.IGNORECASE)
ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
MARKDOWN_EDGES = " \t#*_>|-`"

Series = tuple[str, str]


def roman(token: str) -> int:
    """Read a roman numeral.

    Args:
        token: Letters among `ivxlc`, any case.

    Returns:
        Its value.
    """
    low, total = token.lower(), 0
    for a, b in zip(low, low[1:] + " ", strict=True):
        value = ROMAN_VALUES[a]
        total += -value if b != " " and ROMAN_VALUES[b] > value else value
    return total


def parse(token: str) -> tuple[Series, int] | None:
    """Parse a token shaped like a page number.

    Labels in the same series must step by one per page.

    Args:
        token: A stripped margin line.

    Returns:
        `(series, number)`, or None when the token is not shaped like a page number.
    """
    if m := ARABIC.match(token):
        return ("arabic", m[1]), int(m[2])
    if m := SECTION.match(token):
        return ("section", m[1]), int(m[2])
    if ROMAN.match(token):
        return ("roman", ""), roman(token)
    return None


def margin_candidates(page: pymupdf.Page) -> list[str]:
    """List the lines in the top or bottom margin of a page that are shaped like a page number.

    Args:
        page: The page.

    Returns:
        The candidate tokens in reading order of the text blocks.
    """
    height = page.rect.height
    found = []
    for _x0, y0, _x1, y1, text, *_ in page.get_text("blocks"):
        if y0 > height * (1 - MARGIN) or y1 < height * MARGIN:
            for line in text.split("\n"):
                token = line.strip()
                if parse(token) is not None:
                    found.append(token)
    return found


def confirm_labels(candidates: list[list[str]]) -> list[str | None]:
    """Choose each page's label among its candidates, confirmed by a neighbouring page.

    Args:
        candidates: Candidate tokens per page, in page order.

    Returns:
        The confirmed label per page, or None.
    """
    parsed = [[(token, p) for token in tokens if (p := parse(token)) is not None] for tokens in candidates]
    labels: list[str | None] = []
    for i, page in enumerate(parsed):
        chosen = None
        for token, (series, number) in page:
            for j in range(max(0, i - NEIGHBOURS), min(len(parsed), i + NEIGHBOURS + 1)):
                if j != i and any(s == series and n - number == j - i for _, (s, n) in parsed[j]):
                    chosen = token
                    break
            if chosen:
                break
        labels.append(chosen)
    return labels


def edge_line_candidates(text: str) -> list[str]:
    """List the first and last non-empty lines of a transcription that are shaped like a page number.

    Markdown markers around the line (`#`, `*`, `_`, `>`, `|`, `-`, backticks) are stripped first, so `**20**` and
    `| 20 |` count as `20`.

    Args:
        text: Canonical transcription of a page.

    Returns:
        The candidate tokens, first line first, without repeats.
    """
    lines = [line.strip(MARKDOWN_EDGES) for line in text.splitlines()]
    lines = [line for line in lines if line]
    found: list[str] = []
    for token in (lines[:1] + lines[-1:]) if lines else []:
        if parse(token) is not None and token not in found:
            found.append(token)
    return found


def refresh_labels(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> int:
    """Recompute every page label of a document from the text-layer and transcription candidates.

    Args:
        conn: Connection inside the caller's transaction.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        Number of labels changed.
    """
    inputs = pages.label_inputs(conn, tenant_id=tenant_id, document_id=document_id)
    candidates = [[*page.layer, *[token for token in page.ocr if token not in page.layer]] for page in inputs]
    changed = 0
    for page, label in zip(inputs, confirm_labels(candidates), strict=True):
        if label != page.page_label:
            pages.set_page_label(conn, tenant_id=tenant_id, document_id=document_id, page_no=page.page_no, label=label)
            changed += 1
    return changed
