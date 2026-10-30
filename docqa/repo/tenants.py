"""Tenants and their API keys; keys are stored only as sha256 hashes."""

import hashlib
import secrets
from uuid import UUID

from psycopg import Connection

KEY_PREFIX = "dq_"


def hash_key(api_key: str) -> str:
    """Hash an API key for storage and lookup.

    Args:
        api_key: The key as the client sends it.

    Returns:
        The sha256 hex digest.
    """
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_tenant(conn: Connection, *, name: str) -> tuple[UUID, str]:
    """Create a tenant and its first API key.

    Args:
        conn: Connection inside the caller's transaction.
        name: Display name of the tenant.

    Returns:
        `(tenant_id, api_key)`; the key is shown once and only its hash is stored.
    """
    api_key = KEY_PREFIX + secrets.token_urlsafe(32)
    row = conn.execute("INSERT INTO tenants (name) VALUES (%s) RETURNING id", (name,)).fetchone()
    assert row is not None
    tenant_id: UUID = row["id"]
    conn.execute("INSERT INTO api_keys (key_hash, tenant_id) VALUES (%s, %s)", (hash_key(api_key), tenant_id))
    return tenant_id, api_key


def tenant_for_key(conn: Connection, api_key: str) -> UUID | None:
    """Resolve the tenant of an API key; not filtered by tenant, because it is what finds the tenant.

    Args:
        conn: Any connection.
        api_key: The key as the client sends it.

    Returns:
        The tenant id, or None for an unknown key.
    """
    row = conn.execute("SELECT tenant_id FROM api_keys WHERE key_hash = %s", (hash_key(api_key),)).fetchone()
    return row["tenant_id"] if row else None
