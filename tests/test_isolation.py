"""Tenant isolation at the data-access layer."""

from psycopg_pool import ConnectionPool

from docqa.repo.documents import get_document
from tests.helpers import new_document, new_tenant


def test_document_of_another_tenant_is_invisible(db: ConnectionPool) -> None:
    """Tenant A can't read tenant B's document, even with its id; tenant B can.

    Args:
        db: Pool on an empty database.
    """
    tenant_a = new_tenant(db, "a")
    tenant_b = new_tenant(db, "b")
    document_id = new_document(db, tenant_b)
    with db.connection() as conn:
        assert get_document(conn, tenant_id=tenant_a, document_id=document_id) is None
        owned = get_document(conn, tenant_id=tenant_b, document_id=document_id)
    assert owned is not None
    assert owned.id == document_id
    assert owned.tenant_id == tenant_b
    assert owned.status == "queued"
    assert owned.page_count is None
