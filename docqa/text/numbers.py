"""Numbers in answers and sources, read in every way their format allows (DECISIONS #7).

"13.000 horas" in a Portuguese table and "13,000 hours" in an English answer are the same number, but "1.800" can be
1800 or 1.8. Each numeric token gets every reading its separators allow, and a number is grounded when any of its
readings matches any reading of the sources. The verifier and the eval share it.
"""

import re
import unicodedata

NUM = re.compile(r"\d+(?:[.,\s]\d+)*")
SEPARATOR = re.compile(r"([.,\s])")
REQUEST_LABEL = re.compile(r"\bS\d+\b")


def _joined(groups: list[str], separators: list[str]) -> set[float]:
    """Read separated digit groups as one number.

    Args:
        groups: Digit groups.
        separators: The separators between them (`.`, `,` or whitespace).

    Returns:
        Thousands grouping when every group after the first has 3 digits; a decimal mark when there is one `.` or `,`,
        or when the last separator differs from the others. Empty when no reading joins the groups.
    """
    values: set[float] = set()
    if len(set(separators)) == 1 and all(len(group) == 3 for group in groups[1:]) and len(groups[0]) <= 3:
        values.add(float("".join(groups)))
    if len(separators) == 1 and separators[0] in ".,":
        values.add(float(groups[0] + "." + groups[1]))
    if len(set(separators)) == 2 and separators[-1] in ".,":
        values.add(float("".join(groups[:-1]) + "." + groups[-1]))
    return values


def readings(tok: str) -> set[float]:
    """Return every value a numeric token can stand for.

    Args:
        tok: Digits, possibly grouped by `.`, `,` or whitespace, such as `13.000` or `1,800.5`.

    Returns:
        The joined readings (see `_joined`), or the separate groups when no joined reading applies.
    """
    tok = tok.strip()
    parts = SEPARATOR.split(tok)
    separators, groups = parts[1::2], parts[0::2]
    if not separators:
        return {float(tok)}
    return _joined(groups, separators) or {float(group) for group in groups}


def numbers(text: str) -> list[tuple[str, set[float]]]:
    """Find the numeric tokens of a text with their readings.

    The text is NFKC-normalized first, and request labels (`S12`) are not numbers. Digits inside codes count
    (`IP55`, `6205-2RS`). A token whose groups can't be read as one number is several numbers, each needing its own
    source: `280 12000` (a condition and an answer joined by a space) is 280 and 12000, never "280 or 12000".

    Args:
        text: Any text.

    Returns:
        `(token, readings)` pairs in text order.
    """
    text = unicodedata.normalize("NFKC", text)
    text = REQUEST_LABEL.sub(" ", text)
    found: list[tuple[str, set[float]]] = []
    for match in NUM.finditer(text):
        tok = match.group()
        parts = SEPARATOR.split(tok)
        separators, groups = parts[1::2], parts[0::2]
        joined = _joined(groups, separators) if separators else {float(tok)}
        if joined:
            found.append((tok, joined))
        else:
            found.extend((group, {float(group)}) for group in groups)
    return found


def pool(texts: list[str]) -> set[float]:
    """Collect every reading of every number in some texts.

    Args:
        texts: Source texts.

    Returns:
        The union of the readings.
    """
    values: set[float] = set()
    for text in texts:
        for _, found in numbers(text):
            values |= found
    return values


def ungrounded(text: str, sources: list[str]) -> list[str]:
    """Return the numbers of a text that no reading of the sources supports.

    Args:
        text: Answer text.
        sources: Texts the answer may take numbers from.

    Returns:
        The unsupported tokens, in text order.
    """
    available = pool(sources)
    return [tok for tok, found in numbers(text) if not (found & available)]
