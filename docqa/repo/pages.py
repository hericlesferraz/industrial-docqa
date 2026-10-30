"""Page rows of a tenant's documents: created by the extraction, filled by OCR, marked by the index."""

from dataclasses import dataclass
from uuid import UUID

from psycopg import Connection
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

CANONICAL_TEXT = "CASE text_source WHEN 'ocr' THEN ocr_text ELSE layer_text END"


@dataclass(frozen=True)
class PageInit:
    """A page row as the extraction creates it.

    Attributes:
        page_no: 1-based position in the file.
        state: `needs_ocr`, `text_ready` or `blank`.
        text_source: `layer` for `text_ready` pages, else None.
        layer_text: Markdown of the text layer; empty for blank pages.
        quality: Detector signals and label candidates.
        page_label: Printed page number, when confirmed by a neighbour.
        text_rev: 1 for a page with text, 0 otherwise.
        extract_version: Version of the extractor, detector and label reader.
    """

    page_no: int
    state: str
    text_source: str | None
    layer_text: str
    quality: dict[str, object]
    page_label: str | None
    text_rev: int
    extract_version: str


@dataclass(frozen=True)
class PageText:
    """A page waiting to be indexed, with its canonical text.

    Attributes:
        page_no: 1-based position in the file.
        text_rev: Revision of the text read; the page is marked indexed only if it is unchanged.
        text_source: `layer` or `ocr`.
        text: Canonical page text (the OCR text when transcribed, else the text layer).
    """

    page_no: int
    text_rev: int
    text_source: str
    text: str


def insert_pages(conn: Connection, *, tenant_id: UUID, document_id: UUID, pages: list[PageInit]) -> None:
    """Create the page rows of a document.

    Args:
        conn: Connection inside the extraction's result transaction.
        tenant_id: Owner.
        document_id: Document id.
        pages: Every page of the file.
    """
    rows = [
        (
            tenant_id,
            document_id,
            page.page_no,
            page.state,
            page.text_source,
            page.layer_text,
            Jsonb(page.quality),
            page.page_label,
            page.text_rev,
            page.extract_version,
        )
        for page in pages
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO pages (tenant_id, document_id, page_no, state, text_source, layer_text, quality,"
            " page_label, text_rev, extract_version) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            rows,
        )


def pages_to_index(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> list[PageText]:
    """List the pages of a document that have text and are not indexed yet.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        The `text_ready` pages in page order, with their canonical text.
    """
    with conn.cursor(row_factory=class_row(PageText)) as cur:
        return cur.execute(
            f"SELECT page_no, text_rev, text_source, {CANONICAL_TEXT} AS text FROM pages"
            " WHERE tenant_id = %s AND document_id = %s AND state = 'text_ready' ORDER BY page_no",
            (tenant_id, document_id),
        ).fetchall()


def mark_indexed(
    conn: Connection, *, tenant_id: UUID, document_id: UUID, page_no: int, text_rev: int, index_version: str
) -> bool:
    """Mark a page indexed, only if its text is still the revision that was indexed.

    Args:
        conn: Connection inside the page's index transaction.
        tenant_id: Owner.
        document_id: Document id.
        page_no: Page.
        text_rev: Revision read before indexing.
        index_version: Version of the chunker and embedder.

    Returns:
        False when the text changed meanwhile (whoever wrote it enqueued another `index`).
    """
    cur = conn.execute(
        "UPDATE pages SET state = 'indexed', index_version = %s, updated_at = now()"
        " WHERE tenant_id = %s AND document_id = %s AND page_no = %s AND state = 'text_ready' AND text_rev = %s",
        (index_version, tenant_id, document_id, page_no, text_rev),
    )
    return cur.rowcount == 1


def fail_pages(
    conn: Connection, *, tenant_id: UUID, document_id: UUID, from_state: str, code: str, message: str
) -> int:
    """Mark failed every page of a document that is in a given state.

    Args:
        conn: Connection inside the `fail` transaction of a job.
        tenant_id: Owner.
        document_id: Document id.
        from_state: Pages in this state fail, for example `text_ready` when indexing gave up.
        code: Page error code.
        message: Detail without customer text.

    Returns:
        Number of pages marked failed.
    """
    cur = conn.execute(
        "UPDATE pages SET state = 'failed', error_code = %s, error_message = %s, updated_at = now()"
        " WHERE tenant_id = %s AND document_id = %s AND state = %s",
        (code, message, tenant_id, document_id, from_state),
    )
    return cur.rowcount


def page_counts(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> tuple[dict[str, int], dict[str, int]]:
    """Count a document's pages by state and by text source.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        `(by_state, by_source)`: every page state with its count, and `layer`/`ocr` with theirs.
    """
    by_state = dict.fromkeys(("needs_ocr", "text_ready", "indexed", "blank", "failed"), 0)
    by_source = dict.fromkeys(("layer", "ocr"), 0)
    rows = conn.execute(
        "SELECT state, text_source, count(*) AS n FROM pages WHERE tenant_id = %s AND document_id = %s"
        " GROUP BY state, text_source",
        (tenant_id, document_id),
    ).fetchall()
    for row in rows:
        by_state[row["state"]] += row["n"]
        if row["text_source"]:
            by_source[row["text_source"]] += row["n"]
    return by_state, by_source


@dataclass(frozen=True)
class PageForOcr:
    """What the `ocr_page` job reads about its page.

    Attributes:
        page_no: 1-based position in the file.
        state: Page state; the job only works on `needs_ocr` pages.
        text_rev: Current text revision.
    """

    page_no: int
    state: str
    text_rev: int


def get_page_for_ocr(conn: Connection, *, tenant_id: UUID, document_id: UUID, page_no: int) -> PageForOcr | None:
    """Load the state of one page.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_id: Document id.
        page_no: Page.

    Returns:
        The page, or None when it doesn't exist.
    """
    with conn.cursor(row_factory=class_row(PageForOcr)) as cur:
        return cur.execute(
            "SELECT page_no, state, text_rev FROM pages WHERE tenant_id = %s AND document_id = %s AND page_no = %s",
            (tenant_id, document_id, page_no),
        ).fetchone()


def write_ocr_text(
    conn: Connection,
    *,
    tenant_id: UUID,
    document_id: UUID,
    page_no: int,
    text: str,
    ocr_version: str,
    quality_ocr: dict[str, object],
    label_candidates: list[str] | None = None,
) -> int | None:
    """Store a page's transcription and make the page ready for indexing.

    Args:
        conn: Connection inside the job's fenced result transaction.
        tenant_id: Owner.
        document_id: Document id.
        page_no: Page.
        text: Canonical transcription.
        ocr_version: OCR version that produced it.
        quality_ocr: Evidence stored in `quality.ocr` (model, cache, recitations, fallback, finish reasons).
        label_candidates: Page-number candidates read from the transcription's edge lines, stored in
            `quality.label_candidates.ocr`.

    Returns:
        The new `text_rev`, or None when the page was no longer `needs_ocr`.
    """
    row = conn.execute(
        "UPDATE pages SET state = 'text_ready', text_source = 'ocr', ocr_text = %(text)s,"
        " text_rev = text_rev + 1, ocr_version = %(version)s, error_code = NULL, error_message = NULL,"
        " quality = quality || jsonb_build_object('ocr', %(ocr)s::jsonb, 'label_candidates',"
        " coalesce(quality->'label_candidates', '{}'::jsonb) || jsonb_build_object('ocr', %(labels)s::jsonb)),"
        " updated_at = now()"
        " WHERE tenant_id = %(tenant)s AND document_id = %(document)s AND page_no = %(page)s AND state = 'needs_ocr'"
        " RETURNING text_rev",
        {
            "text": text,
            "version": ocr_version,
            "ocr": Jsonb(quality_ocr),
            "labels": Jsonb(label_candidates or []),
            "tenant": tenant_id,
            "document": document_id,
            "page": page_no,
        },
    ).fetchone()
    return int(row["text_rev"]) if row else None


def set_page_failed(
    conn: Connection, *, tenant_id: UUID, document_id: UUID, page_no: int, code: str, message: str
) -> bool:
    """Mark a page that is waiting for OCR as failed.

    Args:
        conn: Connection inside the `fail` transaction of the page's job.
        tenant_id: Owner.
        document_id: Document id.
        page_no: Page.
        code: Page error code (`ocr_recitation`, `ocr_unavailable`, `render_failed`, ...).
        message: Detail without customer text.

    Returns:
        False when the page was no longer `needs_ocr`.
    """
    cur = conn.execute(
        "UPDATE pages SET state = 'failed', error_code = %s, error_message = %s, updated_at = now()"
        " WHERE tenant_id = %s AND document_id = %s AND page_no = %s AND state = 'needs_ocr'",
        (code, message, tenant_id, document_id, page_no),
    )
    return cur.rowcount == 1


@dataclass(frozen=True)
class LabelInput:
    """What the label refresh reads about one page.

    Attributes:
        page_no: 1-based position in the file.
        layer: Page-number candidates from the text layer's margins.
        ocr: Page-number candidates from the transcription's edge lines.
        page_label: Current label.
    """

    page_no: int
    layer: list[str]
    ocr: list[str]
    page_label: str | None


def label_inputs(conn: Connection, *, tenant_id: UUID, document_id: UUID) -> list[LabelInput]:
    """Read the label candidates of every page of a document.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        document_id: Document id.

    Returns:
        One entry per page, in page order.
    """
    with conn.cursor(row_factory=class_row(LabelInput)) as cur:
        return cur.execute(
            "SELECT page_no, coalesce(quality->'label_candidates'->'layer', '[]'::jsonb) AS layer,"
            " coalesce(quality->'label_candidates'->'ocr', '[]'::jsonb) AS ocr, page_label"
            " FROM pages WHERE tenant_id = %s AND document_id = %s ORDER BY page_no",
            (tenant_id, document_id),
        ).fetchall()


def set_page_label(conn: Connection, *, tenant_id: UUID, document_id: UUID, page_no: int, label: str | None) -> None:
    """Change one page's printed label.

    Args:
        conn: Connection inside the caller's transaction.
        tenant_id: Owner.
        document_id: Document id.
        page_no: Page.
        label: New label, or None.
    """
    conn.execute(
        "UPDATE pages SET page_label = %s, updated_at = now() WHERE tenant_id = %s AND document_id = %s AND page_no = %s",
        (label, tenant_id, document_id, page_no),
    )


def page_labels(
    conn: Connection, *, tenant_id: UUID, pairs: list[tuple[UUID, int]]
) -> dict[tuple[UUID, int], str | None]:
    """Read the printed labels of some pages, for citations.

    Args:
        conn: Any connection.
        tenant_id: Owner.
        pairs: `(document_id, page_no)` of the pages.

    Returns:
        The label of each page found, None when it has no confirmed label.
    """
    if not pairs:
        return {}
    rows = conn.execute(
        "SELECT p.document_id, p.page_no, p.page_label FROM pages p"
        " JOIN unnest(%s::uuid[], %s::int[]) AS w(document_id, page_no)"
        " ON w.document_id = p.document_id AND w.page_no = p.page_no WHERE p.tenant_id = %s",
        ([document for document, _ in pairs], [page for _, page in pairs], tenant_id),
    ).fetchall()
    return {(row["document_id"], row["page_no"]): row["page_label"] for row in rows}
