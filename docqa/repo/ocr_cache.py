"""The OCR cache, shared across tenants by content: (file sha256, page, OCR version) to transcription.

It has no `tenant_id` by design (DECISIONS #5): an entry can only be reached by uploading the exact same bytes, and
whoever has the bytes already has the content. The sha256 is always computed by us from the bytes received. Only
transcriptions that ended with `STOP` are stored; a transport error never reaches the cache.
"""

from dataclasses import dataclass

from psycopg import Connection
from psycopg.rows import class_row


@dataclass(frozen=True)
class CacheRow:
    """A cached transcription.

    Attributes:
        text: Canonical transcription.
        model: Model that produced it.
    """

    text: str
    model: str


def get(conn: Connection, *, sha256: str, page_no: int, ocr_version: str) -> CacheRow | None:
    """Look up a page's transcription.

    Args:
        conn: Any connection.
        sha256: Digest of the file.
        page_no: 1-based page.
        ocr_version: OCR version of the current settings.

    Returns:
        The cached transcription, or None.
    """
    with conn.cursor(row_factory=class_row(CacheRow)) as cur:
        return cur.execute(
            "SELECT text, model FROM ocr_cache WHERE sha256 = %s AND page_no = %s AND ocr_version = %s",
            (sha256, page_no, ocr_version),
        ).fetchone()


def put(conn: Connection, *, sha256: str, page_no: int, ocr_version: str, text: str, model: str) -> None:
    """Store a page's transcription; an existing entry is kept.

    Args:
        conn: Any connection.
        sha256: Digest of the file.
        page_no: 1-based page.
        ocr_version: OCR version that produced it.
        text: Canonical transcription.
        model: Model that produced it.
    """
    conn.execute(
        "INSERT INTO ocr_cache (sha256, page_no, ocr_version, text, model) VALUES (%s, %s, %s, %s, %s)"
        " ON CONFLICT DO NOTHING",
        (sha256, page_no, ocr_version, text, model),
    )
