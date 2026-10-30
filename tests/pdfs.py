"""PDF files for the tests: the sample manuals, and bad or synthetic files built on the fly with PyMuPDF."""

import io
from pathlib import Path

import pymupdf

CASE_DIR = Path(__file__).resolve().parent.parent / "case_files"
LB5001 = "LB5001.pdf"
MN414 = "MN414_0224.pdf"
CESTARI = "WEG-CESTARI-manual-iom-guia-consulta-rapida-50111652-pt-en-es-web.pdf"
MOTORES = "WEG-motores-eletricos-guia-de-especificacao-50032749-brochure-portuguese-web.pdf"
CASE_PDFS = (LB5001, MN414, CESTARI, MOTORES)


def case_pdf(name: str) -> bytes:
    """Read one of the provided manuals.

    Args:
        name: File name in `case_files/`.

    Returns:
        The file's bytes.
    """
    return (CASE_DIR / name).read_bytes()


def _save(doc: pymupdf.Document, **options: object) -> bytes:
    """Save a document to bytes.

    Args:
        doc: The document.
        **options: Options of `Document.save`.

    Returns:
        The PDF bytes.
    """
    out = io.BytesIO()
    doc.save(out, **options)
    return out.getvalue()


def encrypted_pdf(user_password: str = "secret") -> bytes:
    """Encrypt LB5001 with AES-256.

    Args:
        user_password: Password needed to open the file; empty for an owner-only password.

    Returns:
        The encrypted PDF.
    """
    doc = pymupdf.open(stream=case_pdf(LB5001), filetype="pdf")
    return _save(doc, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=user_password, owner_pw="owner", permissions=0)


def truncated_pdf(cut: float) -> bytes:
    """Cut LB5001 short.

    Args:
        cut: Bytes to keep, or a fraction of the file.

    Returns:
        The truncated bytes.
    """
    data = case_pdf(LB5001)
    keep = int(len(data) * cut) if isinstance(cut, float) else int(cut)
    return data[:keep]


def garbage_pdf() -> bytes:
    """Build bytes that start like a PDF and are garbage after the header.

    Returns:
        The bytes.
    """
    return b"%PDF-1.7\n" + bytes(range(256)) * 40


def text_pdf(pages: list[str]) -> bytes:
    """Build a PDF with one line of text per page.

    Args:
        pages: Text of each page; an empty string makes a blank page.

    Returns:
        The PDF.
    """
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 144), text, fontsize=11)
    return _save(doc)


def rasterized_pdf(name: str = LB5001, page_no: int = 2, dpi: int = 100) -> bytes:
    """Turn one page of a manual into a scan: an image of the page, no text layer.

    Args:
        name: Manual in `case_files/`.
        page_no: 1-based page.
        dpi: Render resolution.

    Returns:
        A one-page PDF.
    """
    src = pymupdf.open(stream=case_pdf(name), filetype="pdf")[page_no - 1]
    pix = src.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    out = pymupdf.open()
    page = out.new_page(width=src.rect.width, height=src.rect.height)
    page.insert_image(page.rect, stream=pix.tobytes("png"))
    return _save(out)


def mixed_pdf(dpi: int = 100) -> bytes:
    """Build a two-page PDF: LB5001's first page with its text layer, then its second page as a scan.

    Args:
        dpi: Render resolution of the scanned page.

    Returns:
        The PDF.
    """
    out = pymupdf.open()
    out.insert_pdf(pymupdf.open(stream=case_pdf(LB5001), filetype="pdf"), from_page=0, to_page=0)
    out.insert_pdf(pymupdf.open(stream=rasterized_pdf(LB5001, 2, dpi), filetype="pdf"))
    return _save(out)
