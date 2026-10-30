"""Connection pool, transactions and the migration runner."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

MIGRATION_LOCK = 7310001
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def create_pool(url: str, *, max_size: int) -> ConnectionPool:
    """Open a pool of autocommit connections that return rows as dicts.

    Connections are in autocommit mode, so a statement outside `tx()` commits on its own and `tx()` opens an
    explicit transaction.

    Args:
        url: Postgres connection URL.
        max_size: Maximum number of connections.

    Returns:
        The open pool.
    """
    return ConnectionPool(
        url,
        min_size=1,
        max_size=max_size,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=True,
        name="docqa",
    )


@contextmanager
def tx(pool: ConnectionPool) -> Iterator[Connection]:
    """Borrow a connection and run a transaction on it.

    Args:
        pool: Connection pool.

    Yields:
        A connection inside a transaction that commits on exit and rolls back on an exception.
    """
    with pool.connection() as conn, conn.transaction():
        yield conn


def _migration_files() -> list[tuple[str, str]]:
    """List the SQL migrations shipped with the package.

    Returns:
        `(version, sql)` pairs in name order; the version is the file name without `.sql`.
    """
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    return [(path.stem, path.read_text(encoding="utf-8")) for path in files]


def migrate(conn: Connection) -> list[str]:
    """Apply every migration not yet recorded, one transaction each, under an advisory lock.

    Both the api and the worker call it at start; the lock makes concurrent starts safe.

    Args:
        conn: Connection in autocommit mode, outside any transaction.

    Returns:
        The versions applied by this call, in order.
    """
    applied: list[str] = []
    conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK,))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}
        for version, sql in _migration_files():
            if version in done:
                continue
            with conn.transaction():
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK,))
    return applied


def ping(pool: ConnectionPool, *, timeout_s: float = 2.0) -> bool:
    """Check that the database answers.

    Args:
        pool: Connection pool.
        timeout_s: How long to wait for a connection.

    Returns:
        True when `SELECT 1` succeeds.
    """
    try:
        with pool.connection(timeout=timeout_s) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error:
        return False
    return True
