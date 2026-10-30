"""Extraction child process: open a PDF, run the detector, export each page's text layer and read its margin labels.

Usage: `python -m docqa.ingest.sandbox docqa.ingest.extract_child <pdf> <out.jsonl> <max_pages>`.

It writes JSON lines, flushed one by one so the parent can watch progress: first a header, `{"page_count": n}`, or
an error, `{"error": code, "message": ...}`; then one line per page with `page_no`, `state`, `reason`, `quality`,
`layer_text` and `label_candidates`. A page-level error ends the output with an error line. Exit code 0 means the
output says how it ended; anything else is a crash.
"""

import json
import logging
import sys
from pathlib import Path
from typing import TextIO

import pymupdf
import pymupdf4llm

from docqa.ingest.detect import BLANK, NEEDS_OCR, TEXT_OK, detect
from docqa.ingest.labels import margin_candidates

LAYER_ERROR = "layer_error"

logger = logging.getLogger("docqa.extract_child")


def _emit(out: TextIO, record: dict[str, object]) -> None:
    """Write one JSON line and flush it, so the parent sees the progress.

    Args:
        out: Output file.
        record: The record.
    """
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()


def _error(out: TextIO, code: str, message: str) -> int:
    """Write an error line.

    Args:
        out: Output file.
        code: Document error code.
        message: Detail without customer text.

    Returns:
        0, the exit code of an outcome the output describes.
    """
    _emit(out, {"error": code, "message": message})
    return 0


def _layer_text(doc: pymupdf.Document, index: int) -> str:
    """Export one page's text layer as Markdown with PyMuPDF's layout mode, its own OCR off.

    Args:
        doc: The open document.
        index: 0-based page index.

    Returns:
        The page's Markdown.
    """
    return pymupdf4llm.to_markdown(doc, pages=[index], use_ocr=False, show_progress=False)


def page_record(doc: pymupdf.Document, index: int) -> dict[str, object]:
    """Analyze one page.

    The text layer is exported for every page that is not blank, flagged pages included, as evidence. If the export
    of a page the detector found readable fails, that page goes to OCR with the reason `layer_error`.

    Args:
        doc: The open document.
        index: 0-based page index.

    Returns:
        The page line of the output.
    """
    page = doc.load_page(index)
    detection = detect(page)
    candidates = margin_candidates(page)
    quality: dict[str, object] = {"detector": detection.quality()}
    state, reason = detection.state, detection.reason
    layer_text = ""
    if state != BLANK:
        try:
            layer_text = _layer_text(doc, index)
        except MemoryError:
            raise
        except Exception as exc:
            logger.exception("page %d: the text layer export failed", index + 1)
            quality["layer_error"] = type(exc).__name__
            if state == TEXT_OK:
                state, reason = NEEDS_OCR, LAYER_ERROR
                quality["detector"] = {**detection.quality(), "reason": LAYER_ERROR}
    return {
        "page_no": index + 1,
        "state": state,
        "reason": reason,
        "quality": quality,
        "layer_text": layer_text,
        "label_candidates": candidates,
    }


def extract(pdf: Path, out: TextIO, max_pages: int) -> int:
    """Write the output lines for one PDF.

    Args:
        pdf: The PDF file.
        out: Output file.
        max_pages: Page limit of one document.

    Returns:
        The process exit code: 0 when the output says how extraction ended.
    """
    try:
        doc = pymupdf.open(pdf, filetype="pdf")
    except MemoryError:
        raise
    except Exception as exc:
        logger.exception("the file could not be opened")
        return _error(out, "pdf_corrupt", f"the file could not be opened as a PDF ({type(exc).__name__})")
    if doc.needs_pass:
        return _error(out, "pdf_encrypted", "the PDF needs a password to open")
    if doc.page_count == 0:
        if doc.is_repaired:
            return _error(out, "pdf_corrupt", "the PDF is damaged and no page could be recovered")
        return _error(out, "pdf_no_pages", "the PDF has no pages")
    if doc.page_count > max_pages:
        return _error(out, "pdf_too_many_pages", f"the PDF has {doc.page_count} pages; the limit is {max_pages}")
    _emit(out, {"page_count": doc.page_count})
    for index in range(doc.page_count):
        try:
            record = page_record(doc, index)
        except MemoryError:
            raise
        except Exception as exc:
            logger.exception("page %d could not be read", index + 1)
            return _error(out, "pdf_corrupt", f"page {index + 1} could not be read ({type(exc).__name__})")
        _emit(out, record)
    return 0


def main(argv: list[str]) -> int:
    """Run the extraction.

    Args:
        argv: `<pdf> <out.jsonl> <max_pages>`.

    Returns:
        The process exit code.
    """
    pdf, out_path, max_pages = argv
    with Path(out_path).open("w", encoding="utf-8") as out:
        return extract(Path(pdf), out, int(max_pages))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
