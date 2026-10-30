"""Migrations: idempotent, extensions and objects present."""

from psycopg_pool import ConnectionPool

from docqa.db import migrate

TABLES = {
    "tenants",
    "api_keys",
    "documents",
    "batches",
    "batch_items",
    "pages",
    "ocr_cache",
    "chunks",
    "chunks_history",
    "jobs",
    "provider_state",
    "answers",
    "answer_candidates",
    "spans",
    "schema_migrations",
}


def test_migrate_twice_applies_nothing(db: ConnectionPool) -> None:
    """A second run applies no migration and the first one is recorded.

    Args:
        db: Pool on an empty database.
    """
    with db.connection() as conn:
        assert migrate(conn) == []
        versions = [row["version"] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()]
    assert versions == ["001_init", "002_search_version"]


def test_extensions_tables_and_views_exist(db: ConnectionPool) -> None:
    """The extensions, every table and both on-call views exist.

    Args:
        db: Pool on an empty database.
    """
    with db.connection() as conn:
        extensions = {row["extname"] for row in conn.execute("SELECT extname FROM pg_extension").fetchall()}
        tables = {
            row["tablename"] for row in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        }
        views = {row["viewname"] for row in conn.execute("SELECT viewname FROM pg_views WHERE schemaname = 'public'")}
        unaccented = conn.execute("SELECT docqa_unaccent('Relubrificação') AS t").fetchone()
    assert {"vector", "unaccent"} <= extensions
    assert tables >= TABLES
    assert {"v_model_calls_per_minute", "v_requests_per_minute"} <= views
    assert unaccented is not None
    assert unaccented["t"] == "Relubrificacao"
