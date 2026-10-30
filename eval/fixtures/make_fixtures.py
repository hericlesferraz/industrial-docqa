"""Build the robustness files of the eval: a scanned page, a locked PDF, a truncated one, an all-blank one, non-PDFs.

Run by `make eval-robustness` inside the eval image (PyMuPDF and NumPy only), from a clean clone:
`python make_fixtures.py --source case_files/LB5001.pdf --out eval/fixtures/pdfs`. The output is deterministic for a
given PyMuPDF version (fixed noise seed) and is not committed. These files run in their own eval run and tenant, never
in the folder whose questions are scored, so that a failed document doesn't mark every scored answer's coverage
incomplete (DECISIONS #16).
"""

import argparse
from pathlib import Path

import numpy as np
import pymupdf

SCAN_DPI = 200
SCAN_ANGLE = 1.2
SCAN_CONTRAST = 0.92
SCAN_NOISE = 12.0
SCAN_JPEG_QUALITY = 35
SCAN_SEED = 7
SCAN_PAGE = 2
PASSWORD = "eval-fixture"


def scanned_page(source: Path, page_no: int) -> bytes:
    """Rasterize one page like an office scan: skewed, grey, noisy, JPEG, with no text layer.

    Args:
        source: The PDF to take the page from.
        page_no: 1-based page.

    Returns:
        A one-page PDF whose only content is the page image.
    """
    with pymupdf.open(source) as doc:
        page = doc[page_no - 1]
        zoom = SCAN_DPI / 72
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom).prerotate(SCAN_ANGLE), colorspace=pymupdf.csGRAY)
        width = page.rect.width
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(np.float32)
    blurred = arr.copy()
    blurred[1:-1, 1:-1] = (arr[:-2, 1:-1] + arr[2:, 1:-1] + arr[1:-1, :-2] + arr[1:-1, 2:] + 4 * arr[1:-1, 1:-1]) / 8
    rng = np.random.default_rng(SCAN_SEED)
    noisy = blurred * SCAN_CONTRAST + 10 + rng.normal(0, SCAN_NOISE, blurred.shape)
    gray = np.clip(noisy, 0, 255).astype(np.uint8)
    image = pymupdf.Pixmap(pymupdf.csGRAY, pix.width, pix.height, gray.tobytes(), 0)
    jpeg = image.tobytes("jpeg", jpg_quality=SCAN_JPEG_QUALITY)
    out = pymupdf.open()
    scale = width / pix.width
    target = out.new_page(width=pix.width * scale, height=pix.height * scale)
    target.insert_image(target.rect, stream=jpeg)
    data = out.tobytes(garbage=3, deflate=True, no_new_id=True)
    out.close()
    return data


def locked(source: Path) -> bytes:
    """Encrypt a PDF with an open password.

    Args:
        source: The PDF.

    Returns:
        The encrypted PDF.
    """
    with pymupdf.open(source) as doc:
        return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=PASSWORD, owner_pw=PASSWORD)


def blank() -> bytes:
    """Build a PDF whose only page is blank.

    Returns:
        The PDF.
    """
    out = pymupdf.open()
    out.new_page()
    data = out.tobytes(no_new_id=True)
    out.close()
    return data


def build(source: Path, out_dir: Path) -> list[Path]:
    """Write every robustness file.

    Args:
        source: A text PDF to derive the files from (LB5001).
        out_dir: Output folder, created if needed.

    Returns:
        The written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    files = {
        "lb5001-p2-scan.pdf": scanned_page(source, SCAN_PAGE),
        "locked.pdf": locked(source),
        "truncated.pdf": raw[: len(raw) // 2],
        "blank.pdf": blank(),
        "not-a-pdf.pdf": b"This is a plain text file with a .pdf name.\n",
        "notes.txt": b"Maintenance notes, not a PDF.\n",
        "empty.pdf": b"",
    }
    written = []
    for name, data in files.items():
        path = out_dir / name
        path.write_bytes(data)
        written.append(path)
    return written


def main() -> None:
    """Parse the arguments and write the files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="text PDF to derive the files from (LB5001.pdf)")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    args = parser.parse_args()
    for path in build(args.source, args.out):
        print(f"{path} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
