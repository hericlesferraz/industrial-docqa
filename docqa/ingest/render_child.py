"""Render child process: draw one PDF page as a PNG for transcription.

Usage: `python -m docqa.ingest.sandbox docqa.ingest.render_child <pdf> <page_no> <dpi> <out.png>`.

The PDF is still untrusted input, so the page is drawn here, in a sandboxed child, never in the worker. Exit code 0
means the PNG was written; anything else is a failure, with the reason on stderr.
"""

import sys
from pathlib import Path

import pymupdf


def render(pdf: Path, page_no: int, dpi: int) -> bytes:
    """Draw one page as it is drawn for every OCR call: RGB, no alpha, at the given resolution.

    Args:
        pdf: The PDF file.
        page_no: 1-based page.
        dpi: Render resolution.

    Returns:
        The PNG bytes.

    Raises:
        ValueError: The PDF needs a password or has no such page.
    """
    doc = pymupdf.open(pdf)
    if doc.needs_pass:
        raise ValueError("the PDF needs a password to open")
    if not 1 <= page_no <= doc.page_count:
        raise ValueError(f"the PDF has no page {page_no}")
    return doc[page_no - 1].get_pixmap(dpi=dpi).tobytes("png")


def main(argv: list[str]) -> int:
    """Render the page and write the PNG.

    Args:
        argv: `<pdf> <page_no> <dpi> <out.png>`.

    Returns:
        The process exit code.
    """
    pdf, page_no, dpi, out = argv
    Path(out).write_bytes(render(Path(pdf), int(page_no), int(dpi)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
