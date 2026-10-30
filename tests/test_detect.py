"""The detector port and the page labels against the measurements of DECISIONS #6 and #8 on the 4 manuals."""

import json
from pathlib import Path

import pymupdf
import pytest

from docqa.ingest.detect import BLANK, NEEDS_OCR, NO_TEXT_LAYER, PageSignals, classify, detect, font_stats
from docqa.ingest.labels import confirm_labels, margin_candidates, parse
from tests.pdfs import CASE_DIR, CASE_PDFS, rasterized_pdf, text_pdf

FIXTURES = Path(__file__).parent / "fixtures"
STATES = {"text_ok": "text_ready", "needs_ocr": "needs_ocr", "blank": "blank"}


def test_detector_matches_g2_on_every_page() -> None:
    """All 170 pages get the measured label (53 garbled, 1 without text layer, 4 blank, 112 good) and reason."""
    expected = json.loads((FIXTURES / "detector_expected.json").read_text(encoding="utf-8"))
    right = total = 0
    mismatches = []
    by_state: dict[str, int] = {}
    for name in CASE_PDFS:
        doc = pymupdf.open(CASE_DIR / name)
        for index, page in enumerate(doc):
            detection = detect(page)
            state = STATES[detection.state]
            want = expected[name]["states"][index]
            total += 1
            right += state == want
            by_state[state] = by_state.get(state, 0) + 1
            if state != want or detection.reason != expected[name]["reasons"][index]:
                mismatches.append((name, index + 1, state, detection.reason))
    assert (right, total) == (170, 170)
    assert mismatches == []
    assert by_state == {"text_ready": 112, "needs_ocr": 54, "blank": 4}


def test_rasterized_page_goes_to_ocr() -> None:
    """A page that is only an image of text has no text layer, so it goes to OCR."""
    doc = pymupdf.open(stream=rasterized_pdf(), filetype="pdf")
    detection = detect(doc[0])
    assert (detection.state, detection.reason) == (NEEDS_OCR, NO_TEXT_LAYER)
    assert detection.ink >= 0.01


def test_blank_page_is_blank_and_short_text_is_kept() -> None:
    """No letters and no ink is blank; a short line without much ink keeps its text."""
    doc = pymupdf.open(stream=text_pdf(["", "Cover"]), filetype="pdf")
    assert detect(doc[0]).state == BLANK
    assert detect(doc[1]).state == "text_ok"


def test_garbage_font_rule() -> None:
    """A font whose words lack vowels sends the page to OCR; normal prose doesn't."""
    garbled = font_stats("7HPSHUDWXUD GH RSHUDÇÃR Pi[LPD GR PRWRU " * 3)
    prose = font_stats("Temperatura de operação máxima do motor elétrico " * 3)
    assert classify(PageSignals(page_stats=garbled, fonts={"F1": garbled}, ink=0.1)) == (
        NEEDS_OCR,
        "garbled:few_vowels",
    )
    assert classify(PageSignals(page_stats=prose, fonts={"F1": prose}, ink=0.1)) == ("text_ok", "")


def test_labels_match_g3_on_known_pages() -> None:
    """On the 147 pages whose printed number is known: 139 right, 0 wrong, 8 null."""
    expected = json.loads((FIXTURES / "labels_expected.json").read_text(encoding="utf-8"))
    tally = {"right": 0, "wrong": 0, "null": 0}
    for name in CASE_PDFS:
        doc = pymupdf.open(CASE_DIR / name)
        labels = confirm_labels([margin_candidates(page) for page in doc])
        for label, want in zip(labels, expected[name], strict=True):
            if want is None:
                continue
            tally["null" if label is None else "right" if label == want else "wrong"] += 1
    assert tally == {"right": 139, "wrong": 0, "null": 8}


@pytest.mark.parametrize(
    ("token", "parsed"),
    [
        ("20", (("arabic", ""), 20)),
        ("A3", (("arabic", "A"), 3)),
        ("2-2", (("section", "2"), 2)),
        ("xiv", (("roman", ""), 14)),
        ("Page 3", None),
    ],
)
def test_parse_page_number_shapes(token: str, parsed: tuple[tuple[str, str], int] | None) -> None:
    """Page-number shapes parse into a series and a number; other text doesn't.

    Args:
        token: Margin line.
        parsed: Expected parse.
    """
    assert parse(token) == parsed


def test_confirm_labels_needs_a_neighbour_in_step() -> None:
    """A number is a page label only if a nearby page continues its series at the right distance."""
    assert confirm_labels([["7"], ["8"], [], ["10"]]) == ["7", "8", None, "10"]
    assert confirm_labels([["7"], ["42"], ["9"]]) == ["7", None, "9"]
    assert confirm_labels([["5"]]) == [None]
    assert confirm_labels([["Fig", "3"], ["4"]]) == ["3", "4"]
