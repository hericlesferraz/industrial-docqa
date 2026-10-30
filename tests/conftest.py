"""Shared fixtures: a fresh `docqa_test` database per session, emptied before each test that uses it."""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from docqa.api.app import create_app
from docqa.api.deps import Deps
from docqa.config import Settings, load_settings
from docqa.db import create_pool, migrate
from docqa.storage import LocalBlobStore
from docqa.telemetry import MemoryTelemetry
from docqa.worker import WorkerDeps
from tests.helpers import ADMIN_KEY, TenantInfo, make_tenant
from tests.index_helpers import FakeEmbedder

DEFAULT_TEST_URL = "postgresql://docqa:docqa@db:5432/docqa_test"


@pytest.fixture(scope="session")
def database_url() -> str:
    """Recreate the test database from scratch and apply the migrations.

    Returns:
        URL of the test database, from `DATABASE_URL`.
    """
    url = os.environ.get("DATABASE_URL", DEFAULT_TEST_URL)
    name = conninfo_to_dict(url)["dbname"]
    assert name and str(name).endswith("_test"), "tests only run against a database whose name ends in _test"
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(str(name))))
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(str(name))))
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
        migrate(conn)
    return url


@pytest.fixture(scope="session")
def pool(database_url: str) -> Iterator[ConnectionPool]:
    """Open one pool for the whole session.

    Args:
        database_url: URL of the test database.

    Yields:
        The pool.
    """
    opened = create_pool(database_url, max_size=12)
    yield opened
    opened.close()


@pytest.fixture
def db(pool: ConnectionPool) -> ConnectionPool:
    """Empty every table except `schema_migrations` before the test.

    Args:
        pool: Session pool.

    Returns:
        The same pool, on an empty database.
    """
    with pool.connection() as conn:
        tables = [
            row["tablename"]
            for row in conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> 'schema_migrations'"
            ).fetchall()
        ]
        conn.execute(
            sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(sql.SQL(", ").join(map(sql.Identifier, tables)))
        )
    return pool


@pytest.fixture
def settings(database_url: str, tmp_path: Path) -> Settings:
    """Build settings from a dict, never from the process environment; OCR calls are not spaced out.

    Args:
        database_url: URL of the test database.
        tmp_path: Per-test temp folder.

    Returns:
        Test settings.
    """
    return load_settings(
        {
            "GEMINI_API_KEY": "test-gemini-key",
            "ADMIN_API_KEY": ADMIN_KEY,
            "DATABASE_URL": database_url,
            "BLOB_STORE_PATH": str(tmp_path / "blobs"),
            "OCR_MIN_INTERVAL_S": "0",
        }
    )


@pytest.fixture
def blobs(settings: Settings) -> LocalBlobStore:
    """Give a blob store in the test's temp folder.

    Args:
        settings: Test settings.

    Returns:
        The blob store.
    """
    return LocalBlobStore(settings.blob_store_path)


@pytest.fixture
def telemetry() -> MemoryTelemetry:
    """Give an in-memory telemetry.

    Returns:
        The telemetry.
    """
    return MemoryTelemetry()


@pytest.fixture
def deps(db: ConnectionPool, settings: Settings, telemetry: MemoryTelemetry, blobs: LocalBlobStore) -> Deps:
    """Build the API dependencies with fakes, including a loaded fake embedder.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
        telemetry: In-memory telemetry.
        blobs: Blob store in a temp folder.

    Returns:
        The API dependencies.
    """
    return Deps(pool=db, settings=settings, telemetry=telemetry, blobs=blobs, embedder=FakeEmbedder())


@pytest.fixture
def worker_deps(
    db: ConnectionPool, settings: Settings, telemetry: MemoryTelemetry, blobs: LocalBlobStore
) -> WorkerDeps:
    """Build the worker dependencies with fakes.

    Args:
        db: Pool on an empty database.
        settings: Test settings.
        telemetry: In-memory telemetry.
        blobs: Blob store in a temp folder.

    Returns:
        The worker dependencies.
    """
    return WorkerDeps(pool=db, settings=settings, telemetry=telemetry, blobs=blobs)


@pytest.fixture
def client(settings: Settings, deps: Deps) -> Iterator[TestClient]:
    """Give an HTTP client over the application built with fakes.

    Args:
        settings: Test settings.
        deps: API dependencies.

    Yields:
        The client.
    """
    with TestClient(create_app(settings, deps)) as test_client:
        yield test_client


@pytest.fixture
def tenant(client: TestClient) -> TenantInfo:
    """Create a tenant.

    Args:
        client: HTTP client.

    Returns:
        The tenant id and key.
    """
    return make_tenant(client)
