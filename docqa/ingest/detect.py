"""Detector of unusable text layers: the signals and the page rule measured in DECISIONS #6.

A font is garbage when its text doesn't look like Portuguese, English or Spanish: too few vowels in its words, too
many rare letters, control characters, or replacement and private-use code points. A page goes to OCR when at least
20% of its letters come from garbage fonts, or when it has almost no letters but has ink (a scan, or text drawn as
outlines). A page with no letters and no ink is blank. The thresholds are the measured ones (170/170 pages).
"""

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pymupdf

VOWELS = set("aeiouáéíóúàâêôãõü")
COMMON = set("abcdefghijklmnopqrstuvwxyzáéíóúàâêôãõüçñ")
WORD = re.compile(r"[^\W\d_]{4,}")

MIN_FONT_LETTERS = 30
MIN_WORD_LETTERS = 30
MIN_WORD_VOWEL_SHARE = 0.25
MAX_RARE_SHARE = 0.2
MAX_CONTROL_SHARE = 0.05
MAX_REPLACEMENT_SHARE = 0.05
MIN_PAGE_LETTERS = 50
MIN_INK = 0.01
MIN_GARBAGE_SHARE = 0.2
INK_DPI = 36
INK_THRESHOLD = 200

TEXT_OK = "text_ok"
NEEDS_OCR = "needs_ocr"
BLANK = "blank"
NO_TEXT_LAYER = "no_text_layer"


def _is_control(ch: str) -> bool:
    """Tell whether a character is a control character other than a line break or tab.

    Args:
        ch: One character.

    Returns:
        True for control characters.
    """
    return unicodedata.category(ch) == "Cc" and ch not in "\n\t\r"


def font_stats(text: str) -> dict[str, int]:
    """Count the letter statistics of a text: the per-font and per-page signals of the detector.

    Args:
        text: Text of one font on a page, or the whole page text.

    Returns:
        Counts of characters, letters, vowels, rare letters, letters and vowels in words of 4+ letters, control
        characters, U+FFFD, private-use code points and digits.
    """
    low = text.lower()
    letters = [c for c in low if c.isalpha()]
    words = WORD.findall(low)
    word_letters = sum(map(len, words))
    return {
        "chars": len(text),
        "letters": len(letters),
        "vowels": sum(c in VOWELS for c in letters),
        "rare": sum(c not in COMMON for c in letters),
        "word_letters": word_letters,
        "word_vowels": sum(c in VOWELS for w in words for c in w),
        "ctrl": sum(_is_control(c) for c in text),
        "fffd": text.count("�"),
        "pua": sum(0xE000 <= ord(c) <= 0xF8FF for c in text),
        "digits": sum(c.isdigit() for c in text),
    }


def ink_ratio(page: pymupdf.Page) -> float:
    """Measure how much of a page is dark, on a small grayscale render.

    Args:
        page: The page.

    Returns:
        Share of pixels darker than 200 at 36 dpi, from 0 to 1.
    """
    pix = page.get_pixmap(dpi=INK_DPI, colorspace=pymupdf.csGRAY)
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    return float(np.count_nonzero(data < INK_THRESHOLD)) / data.size if data.size else 0.0


@dataclass(frozen=True)
class PageSignals:
    """The detector's inputs for one page.

    Attributes:
        page_stats: `font_stats` of the whole page text.
        fonts: `font_stats` per font name, over the page's text spans.
        ink: Ink ratio, rounded to 4 decimals as in the measurement.
    """

    page_stats: dict[str, int]
    fonts: dict[str, dict[str, int]]
    ink: float


@dataclass(frozen=True)
class Detection:
    """The detector's verdict for one page.

    Attributes:
        state: `text_ok`, `needs_ocr` or `blank`.
        reason: `garbled:<reasons>`, `no_text_layer`, or None.
        letters: Letters in the page text.
        garbage_letters: Letters in garbage fonts.
        ink: Ink ratio.
        garbage_fonts: Reasons per garbage font name.
    """

    state: str
    reason: str | None
    letters: int
    garbage_letters: int
    ink: float
    garbage_fonts: dict[str, str] = field(default_factory=dict)

    def quality(self) -> dict[str, object]:
        """Render the verdict as the `quality.detector` JSON of a page row.

        Returns:
            Letters, garbage letters and share, ink, reason and garbage fonts.
        """
        share = round(self.garbage_letters / self.letters, 4) if self.letters else 0.0
        return {
            "letters": self.letters,
            "garbage_letters": self.garbage_letters,
            "garbage_share": share,
            "ink": self.ink,
            "reason": self.reason,
            "garbage_fonts": dict(self.garbage_fonts),
        }


def page_signals(page: pymupdf.Page) -> PageSignals:
    """Collect the detector's signals for a page.

    Args:
        page: The page.

    Returns:
        Statistics of the page text and of each font's spans, and the ink ratio.
    """
    fonts: dict[str, list[str]] = {}
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                fonts.setdefault(span["font"], []).append(span["text"])
    return PageSignals(
        page_stats=font_stats(page.get_text()),
        fonts={name: font_stats("".join(texts)) for name, texts in fonts.items()},
        ink=round(ink_ratio(page), 4),
    )


def font_is_garbage(stats: dict[str, int]) -> tuple[bool, str]:
    """Decide whether one font's text is garbage.

    Args:
        stats: `font_stats` of the font's text on the page.

    Returns:
        `(garbage, reasons)`, the reasons joined by commas: `few_vowels`, `rare_letters`, `control_chars`,
        `replacement_chars`.
    """
    if stats["letters"] < MIN_FONT_LETTERS:
        return False, ""
    reasons = []
    if (
        stats["word_letters"] >= MIN_WORD_LETTERS
        and stats["word_vowels"] / stats["word_letters"] < MIN_WORD_VOWEL_SHARE
    ):
        reasons.append("few_vowels")
    if stats["rare"] / stats["letters"] > MAX_RARE_SHARE:
        reasons.append("rare_letters")
    if stats["ctrl"] / stats["chars"] > MAX_CONTROL_SHARE:
        reasons.append("control_chars")
    if (stats["fffd"] + stats["pua"]) / stats["chars"] > MAX_REPLACEMENT_SHARE:
        reasons.append("replacement_chars")
    return bool(reasons), ",".join(reasons)


def classify(signals: PageSignals) -> tuple[str, str]:
    """Apply the page rule measured in DECISIONS #6.

    Args:
        signals: The page's signals.

    Returns:
        `(state, reason)`: `("needs_ocr", "no_text_layer")`, `("needs_ocr", "garbled:<reasons>")`, `("blank", "")`
        or `("text_ok", "")`.
    """
    return _classify(signals).state_and_reason


@dataclass(frozen=True)
class _Verdict:
    """Intermediate result of the page rule.

    Attributes:
        state_and_reason: What `classify` returns.
        garbage_letters: Letters in garbage fonts.
        garbage_fonts: Reasons per garbage font.
    """

    state_and_reason: tuple[str, str]
    garbage_letters: int
    garbage_fonts: dict[str, str]


def _classify(signals: PageSignals) -> _Verdict:
    """Apply the page rule and keep the per-font evidence.

    Args:
        signals: The page's signals.

    Returns:
        The verdict with its evidence.
    """
    letters = signals.page_stats["letters"]
    if letters < MIN_PAGE_LETTERS:
        if signals.ink >= MIN_INK:
            return _Verdict((NEEDS_OCR, NO_TEXT_LAYER), 0, {})
        return _Verdict((BLANK, "") if letters == 0 else (TEXT_OK, ""), 0, {})
    garbage = 0
    why: set[str] = set()
    garbage_fonts: dict[str, str] = {}
    for name, stats in signals.fonts.items():
        bad, reason = font_is_garbage(stats)
        if bad:
            garbage += stats["letters"]
            why.add(reason)
            garbage_fonts[name] = reason
    if garbage / letters >= MIN_GARBAGE_SHARE:
        return _Verdict((NEEDS_OCR, "garbled:" + "|".join(sorted(why))), garbage, garbage_fonts)
    return _Verdict((TEXT_OK, ""), garbage, garbage_fonts)


def detect(page: pymupdf.Page) -> Detection:
    """Run the detector on a page.

    Args:
        page: The page.

    Returns:
        The verdict with the evidence stored in `quality.detector`.
    """
    signals = page_signals(page)
    verdict = _classify(signals)
    state, reason = verdict.state_and_reason
    return Detection(
        state=state,
        reason=reason or None,
        letters=signals.page_stats["letters"],
        garbage_letters=verdict.garbage_letters,
        ink=signals.ink,
        garbage_fonts=verdict.garbage_fonts,
    )
