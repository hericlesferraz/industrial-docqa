"""Helpers shared by the tests: tenants, documents and direct reads of the jobs table."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient
from psycopg_pool import ConnectionPool

from docqa.db import tx
from docqa.repo import tenants

ADMIN_KEY = "test-admin-key"


@dataclass(frozen=True)
class TenantInfo:
    """A tenant and its key.

    Attributes:
        id: Tenant id.
        key: Its API key.
    """

    id: UUID
    key: str


def make_tenant(client: TestClient, name: str = "acme") -> TenantInfo:
    """Create a tenant through the admin endpoint.

    Args:
        client: HTTP client.
        name: Tenant name.

    Returns:
        The tenant id and key.
    """
    response = client.post("/v1/admin/tenants", json={"name": name}, headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    assert response.status_code == 201, response.text
    body = response.json()
    return TenantInfo(id=UUID(body["tenant_id"]), key=body["api_key"])


def new_tenant(pool: ConnectionPool, name: str = "acme") -> UUID:
    """Create a tenant directly in the database.

    Args:
        pool: Database pool.
        name: Tenant name.

    Returns:
        The tenant id.
    """
    with tx(pool) as conn:
        tenant_id, _ = tenants.create_tenant(conn, name=name)
    return tenant_id


def new_document(pool: ConnectionPool, tenant_id: UUID, sha256: str = "a" * 64) -> UUID:
    """Insert a document row directly, as the upload would.

    Args:
        pool: Database pool.
        tenant_id: Owner.
        sha256: Digest of the file.

    Returns:
        The document id.
    """
    with tx(pool) as conn:
        row = conn.execute(
            "INSERT INTO documents (tenant_id, sha256, filename, blob_key) VALUES (%s, %s, %s, %s) RETURNING id",
            (tenant_id, sha256, "manual.pdf", f"{tenant_id}/{sha256}.pdf"),
        ).fetchone()
    assert row is not None
    return row["id"]


def job_row(pool: ConnectionPool, job_id: int) -> dict[str, Any]:
    """Read a job row.

    Args:
        pool: Database pool.
        job_id: Job id.

    Returns:
        The row as a dict.
    """
    with pool.connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    return row


def expire_lease(pool: ConnectionPool, job_id: int) -> None:
    """Move a running job's lease into the past.

    Args:
        pool: Database pool.
        job_id: Job id.
    """
    with pool.connection() as conn:
        conn.execute("UPDATE jobs SET lease_expires_at = now() - interval '1 second' WHERE id = %s", (job_id,))


def make_runnable(pool: ConnectionPool, job_id: int) -> None:
    """Move a queued job's `run_after` to now, skipping its backoff.

    Args:
        pool: Database pool.
        job_id: Job id.
    """
    with pool.connection() as conn:
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,))
