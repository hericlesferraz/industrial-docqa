"""Telemetry: correlation ids, spans, events, the Postgres writer and JSON logs."""

import io
import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from docqa.api.app import create_app
from docqa.api.deps import Deps
from docqa.config import Settings
from docqa.telemetry import JsonFormatter, MemoryTelemetry, PostgresTelemetry, add_ids, bind, current_ids
from tests.helpers import ADMIN_KEY


class Failure(Exception):
    """An exception with an error code, like `ApiError` or `JobSignal`.

    Attributes:
        code: Error code.
    """

    def __init__(self, code: str) -> None:
        """Store the code.

        Args:
            code: Error code.
        """
        super().__init__(code)
        self.code = code


@pytest.fixture
def json_logs() -> Iterator[io.StringIO]:
    """Capture every log record as JSON lines.

    Yields:
        The buffer the JSON lines go to.
    """
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield buffer
    root.removeHandler(handler)
    root.setLevel(level)


def test_bind_nests_and_restores() -> None:
    """Nested binds merge ids, None removes one, and the previous ids come back on exit."""
    job_id = 7
    document_id = uuid.uuid4()
    assert current_ids() == {}
    with bind(job_id=job_id, document_id=document_id):
        with bind(page_no=3, document_id=None):
            assert current_ids() == {"job_id": job_id, "page_no": 3}
        assert current_ids() == {"job_id": job_id, "document_id": document_id}
        add_ids(tenant_id=document_id)
        assert current_ids()["tenant_id"] == document_id
    assert current_ids() == {}
    with pytest.raises(ValueError, match="unknown"), bind(user="x"):
        pass


def test_spans_and_events_record_status_ids_and_attributes() -> None:
    """Spans record duration and status, a raised code marks them failed, events are zero-duration spans."""
    telemetry = MemoryTelemetry()
    with bind(job_id=1):
        with telemetry.span("work", **{"docqa.pages": 3}) as span:
            span.set("docqa.chunks", 9)
        with pytest.raises(Failure), telemetry.span("broken"):
            raise Failure("pdf_corrupt")
        with pytest.raises(RuntimeError), telemetry.span("crashed"):
            raise RuntimeError("boom")
        telemetry.event("job.retry", error_code="transient")
        telemetry.flush()
    work, broken, crashed, event = telemetry.spans
    assert (work.name, work.status, work.ids, work.attributes) == (
        "work",
        "ok",
        {"job_id": 1},
        {"docqa.pages": 3, "docqa.chunks": 9},
    )
    assert work.duration_ms >= 0
    assert (broken.status, broken.error_code) == ("error", "pdf_corrupt")
    assert (crashed.status, crashed.error_code) == ("error", "internal_error")
    assert (event.name, event.duration_ms, event.error_code) == ("event.job.retry", 0.0, "transient")


def test_flush_takes_only_the_current_context() -> None:
    """Spans of one job are not flushed by another job's context."""
    telemetry = MemoryTelemetry()
    with bind(job_id=1):
        with telemetry.span("first"):
            pass
        with bind(job_id=2), telemetry.span("nested"):
            pass
    with bind(job_id=3):
        with telemetry.span("other"):
            pass
        telemetry.flush()
    assert [span.name for span in telemetry.spans] == ["other"]


def test_request_spans_reach_postgres_with_the_request_id(settings: Settings, deps: Deps) -> None:
    """An HTTP request writes an `http.request` span with its request id, route, status and tenant.

    Args:
        settings: Test settings.
        deps: API dependencies.
    """
    app = create_app(settings, replace(deps, telemetry=PostgresTelemetry(deps.pool)))
    with TestClient(app) as client:
        created = client.post("/v1/admin/tenants", json={"name": "a"}, headers={"Authorization": f"Bearer {ADMIN_KEY}"})
        missing = client.get("/nope")
        health = client.get("/healthz")
    with deps.pool.connection() as conn:
        rows = conn.execute(
            "SELECT request_id, tenant_id, status, attributes FROM spans ORDER BY started_at"
        ).fetchall()
    assert len(rows) == 2
    first, second = rows
    assert str(first["request_id"]) == created.headers["X-Request-ID"]
    assert str(first["tenant_id"]) == created.json()["tenant_id"]
    assert first["attributes"] == {
        "http.request.method": "POST",
        "http.route": "/v1/admin/tenants",
        "http.response.status_code": 201,
    }
    assert str(second["request_id"]) == missing.headers["X-Request-ID"]
    assert second["attributes"]["docqa.error_code"] == "not_found"
    assert "http.route" not in second["attributes"]
    assert health.status_code == 200


def test_logs_are_json_with_ids_and_never_the_key(client: TestClient, json_logs: io.StringIO) -> None:
    """Request logs are JSON lines with the request id and route, and never contain the bearer key.

    Args:
        client: HTTP client.
        json_logs: Captured JSON log lines.
    """
    response = client.post("/v1/admin/tenants", json={"name": "a"}, headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    output = json_logs.getvalue()
    assert ADMIN_KEY not in output
    assert response.json()["api_key"] not in output
    assert "Bearer" not in output
    lines = [json.loads(line) for line in output.splitlines()]
    [request] = [line for line in lines if line["msg"] == "http request"]
    assert request["request_id"] == response.headers["X-Request-ID"]
    assert request["tenant_id"] == response.json()["tenant_id"]
    assert request["route"] == "/v1/admin/tenants"
    assert request["status"] == 201
    assert {"ts", "level", "logger"} <= set(request)
