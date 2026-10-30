"""Page labels from the text layer and from the edge lines of transcriptions (DECISIONS #8)."""

import pytest
from psycopg_pool import ConnectionPool

from docqa.db import tx
from docqa.ingest.labels import edge_line_candidates, refresh_labels
from docqa.repo import pages
from docqa.repo.pages import PageInit
from tests.helpers import new_document, new_tenant
from tests.ingest_helpers import page_rows

LITE = "gemini-3.5-flash-lite"


@pytest.mark.parametrize(
    ("text", "candidates"),
    [
        ("WEG\n\n# Manual\n\nbody\n\n20", ["20"]),
        ("**14**\n\nbody\n\n| 50111652 | rev 3 |", ["14"]),
        ("2-2\nbody\n*ii*", ["2-2", "ii"]),
        ("# Garantia\nbody\nwww.weg.net", []),
        ("", []),
    ],
)
def test_edge_line_candidates(text: str, candidates: list[str]) -> None:
    """Only the first and last lines count, with Markdown markers stripped.

    Args:
        text: Transcription.
        candidates: Expected candidates.
    """
    assert edge_line_candidates(text) == candidates


def test_labels_use_transcribed_edge_lines(db: ConnectionPool) -> None:
    """A transcribed page's edge-line number joins the label evidence of the whole document.

    Args:
        db: Pool on an empty database.
    """
    tenant_id = new_tenant(db)
    document_id = new_document(db, tenant_id)
    layer = {1: ["1"], 3: ["3"], 4: []}
    rows = [
        PageInit(
            page_no=n,
            state="needs_ocr" if n == 2 else "text_ready",
            text_source=None if n == 2 else "layer",
            layer_text="",
            quality={"label_candidates": {"layer": layer.get(n, [])}},
            page_label=None,
            text_rev=0 if n == 2 else 1,
            extract_version="test",
        )
        for n in (1, 2, 3, 4)
    ]
    with tx(db) as conn:
        pages.insert_pages(conn, tenant_id=tenant_id, document_id=document_id, pages=rows)
        assert refresh_labels(conn, tenant_id=tenant_id, document_id=document_id) == 2
    assert [row["page_label"] for row in page_rows(db, document_id)] == ["1", None, "3", None]
    with tx(db) as conn:
        pages.write_ocr_text(
            conn,
            tenant_id=tenant_id,
            document_id=document_id,
            page_no=2,
            text="# Title\n\nbody\n\n2",
            ocr_version="v",
            quality_ocr={"model": LITE},
            label_candidates=["2"],
        )
        assert refresh_labels(conn, tenant_id=tenant_id, document_id=document_id) == 1
    assert [row["page_label"] for row in page_rows(db, document_id)] == ["1", "2", "3", None]
    assert page_rows(db, document_id)[1]["quality"]["label_candidates"] == {"layer": [], "ocr": ["2"]}
