"""HTTP API: authentication, the single error body, `X-Request-ID`, health and tenant creation."""

from collections.abc import Iterator
from dataclasses import replace
from uuid import UUID

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient
from pydantic import BaseModel

from docqa.api.app import check_models, create_app
from docqa.api.deps import Deps, TenantDep
from docqa.config import Settings
from docqa.db import create_pool
from docqa.index.embed import GEMINI_EMBED_MODEL
from docqa.providers.fake import FakeLLM
from docqa.repo import tenants
from docqa.telemetry import MemoryTelemetry
from tests.helpers import ADMIN_KEY, TenantInfo, make_tenant


class Probe(BaseModel):
    """Body of the test-only route.

    Attributes:
        count: A positive number.
    """

    count: int


def _probe_router() -> APIRouter:
    """Build test-only routes: one behind tenant auth, one that fails, one with a body.

    Returns:
        The router.
    """
    router = APIRouter()

    @router.get("/test/whoami")
    def whoami(tenant_id: TenantDep) -> dict[str, str]:
        """Return the authenticated tenant.

        Args:
            tenant_id: Tenant resolved from the key.

        Returns:
            The tenant id.
        """
        return {"tenant_id": str(tenant_id)}

    @router.get("/test/boom")
    def boom() -> dict[str, str]:
        """Fail with an unexpected exception.

        Returns:
            Nothing; it always raises.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("secret detail that must not leak")

    @router.post("/test/probe")
    def probe(body: Probe) -> dict[str, int]:
        """Echo a validated body.

        Args:
            body: The body.

        Returns:
            The count.
        """
        return {"count": body.count}

    return router


@pytest.fixture
def probe_client(settings: Settings, deps: Deps) -> Iterator[TestClient]:
    """Give a client over the application plus the test-only routes.

    Args:
        settings: Test settings.
        deps: API dependencies.

    Yields:
        The client.
    """
    app = create_app(settings, deps)
    app.include_router(_probe_router())
    with TestClient(app) as test_client:
        yield test_client


def _assert_error(response: object, status: int, code: str, *, retryable: bool = False) -> dict[str, object]:
    """Check the single error body and the request id header.

    Args:
        response: The HTTP response.
        status: Expected status.
        code: Expected error code.
        retryable: Expected retryable flag.

    Returns:
        The `error` object.
    """
    assert response.status_code == status, response.text
    header = response.headers["X-Request-ID"]
    error = response.json()["error"]
    assert error["code"] == code
    assert error["retryable"] is retryable
    assert error["request_id"] == header
    assert UUID(header)
    assert error["message"]
    return error


def test_admin_creates_tenant_and_key_resolves(client: TestClient, deps: Deps) -> None:
    """The admin endpoint creates a tenant whose key resolves to it; only the hash is stored.

    Args:
        client: HTTP client.
        deps: API dependencies.
    """
    response = client.post(
        "/v1/admin/tenants", json={"name": "  plant 7 "}, headers={"Authorization": f"Bearer {ADMIN_KEY}"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "plant 7"
    assert body["api_key"].startswith("dq_")
    with deps.pool.connection() as conn:
        assert tenants.tenant_for_key(conn, body["api_key"]) == UUID(body["tenant_id"])
        stored = conn.execute("SELECT key_hash FROM api_keys").fetchall()
    assert [row["key_hash"] for row in stored] == [tenants.hash_key(body["api_key"])]


@pytest.mark.parametrize("header", [None, "Bearer wrong-key", "Basic abc", "Bearer "])
def test_admin_route_rejects_missing_or_wrong_key(client: TestClient, header: str | None) -> None:
    """Missing, wrong or malformed credentials get 401 `unauthorized`.

    Args:
        client: HTTP client.
        header: Authorization header sent, if any.
    """
    headers = {"Authorization": header} if header else {}
    response = client.post("/v1/admin/tenants", json={"name": "x"}, headers=headers)
    _assert_error(response, 401, "unauthorized")


def test_tenant_key_is_not_admin_and_admin_key_is_not_tenant(probe_client: TestClient) -> None:
    """A tenant key can't call admin routes and the admin key can't call tenant routes.

    Args:
        probe_client: Client with the test-only routes.
    """
    tenant = make_tenant(probe_client)
    as_admin = probe_client.post(
        "/v1/admin/tenants", json={"name": "x"}, headers={"Authorization": f"Bearer {tenant.key}"}
    )
    _assert_error(as_admin, 401, "unauthorized")
    as_tenant = probe_client.get("/test/whoami", headers={"Authorization": f"Bearer {ADMIN_KEY}"})
    _assert_error(as_tenant, 401, "unauthorized")
    missing = probe_client.get("/test/whoami")
    _assert_error(missing, 401, "unauthorized")
    ok = probe_client.get("/test/whoami", headers={"Authorization": f"Bearer {tenant.key}"})
    assert ok.status_code == 200
    assert ok.json() == {"tenant_id": str(tenant.id)}


def test_each_tenant_key_resolves_to_its_own_tenant(probe_client: TestClient) -> None:
    """Two tenants' keys resolve to two different tenants.

    Args:
        probe_client: Client with the test-only routes.
    """
    first: TenantInfo = make_tenant(probe_client, "a")
    second: TenantInfo = make_tenant(probe_client, "b")
    seen = {
        probe_client.get("/test/whoami", headers={"Authorization": f"Bearer {t.key}"}).json()["tenant_id"]
        for t in (first, second)
    }
    assert seen == {str(first.id), str(second.id)}


def test_error_bodies_for_404_405_422_and_500(probe_client: TestClient) -> None:
    """Unknown routes, wrong methods, invalid bodies and crashes share the error body and carry a request id.

    Args:
        probe_client: Client with the test-only routes.
    """
    _assert_error(probe_client.get("/nope"), 404, "not_found")
    _assert_error(probe_client.get("/v1/admin/tenants"), 405, "method_not_allowed")
    invalid = _assert_error(probe_client.post("/test/probe", json={"count": "many"}), 422, "invalid_request")
    assert invalid["details"]["errors"][0]["loc"] == ["body", "count"]
    assert "many" not in str(invalid)
    crashed = probe_client.get("/test/boom")
    _assert_error(crashed, 500, "internal_error", retryable=True)
    assert "secret detail" not in crashed.text


def test_incoming_request_id_is_ignored(client: TestClient) -> None:
    """The server always generates the request id.

    Args:
        client: HTTP client.
    """
    sent = "11111111-1111-4111-8111-111111111111"
    response = client.get("/healthz", headers={"X-Request-ID": sent})
    assert response.headers["X-Request-ID"] != sent
    assert UUID(response.headers["X-Request-ID"])


def test_healthz_ok(client: TestClient) -> None:
    """`/healthz` answers 200 with the database and embedder checks and no auth.

    Args:
        client: HTTP client.
    """
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok", "embedder": "ok"}, "reasons": []}


def test_healthz_503_when_database_is_down(settings: Settings, deps: Deps) -> None:
    """`/healthz` answers 503 `service_unavailable` when the database is unreachable.

    Args:
        settings: Test settings.
        deps: API dependencies.
    """
    dead = create_pool("postgresql://docqa:docqa@127.0.0.1:1/none?connect_timeout=1", max_size=1)
    try:
        app = create_app(settings, replace(deps, pool=dead))
        with TestClient(app) as down:
            error = _assert_error(down.get("/healthz"), 503, "service_unavailable", retryable=True)
        assert error["details"]["checks"] == {"database": "unreachable", "embedder": "ok"}
    finally:
        dead.close(timeout=0)


def test_healthz_503_without_the_embedder(settings: Settings, deps: Deps) -> None:
    """`/healthz` answers 503 while the embedder is not loaded: search can't run without it.

    Args:
        settings: Test settings.
        deps: API dependencies.
    """
    with TestClient(create_app(settings, replace(deps, embedder=None))) as client:
        error = _assert_error(client.get("/healthz"), 503, "service_unavailable", retryable=True)
    assert error["details"]["checks"] == {"database": "ok", "embedder": "not_loaded"}
    assert error["details"]["reasons"] == ["embedder: not_loaded"]


def test_openapi_lists_routes_and_bearer_schemes(client: TestClient) -> None:
    """The OpenAPI document has the admin route, both bearer schemes and the error body.

    Args:
        client: HTTP client.
    """
    spec = client.get("/openapi.json").json()
    assert "/v1/admin/tenants" in spec["paths"]
    assert {"TenantKey", "AdminKey"} >= set(spec["components"]["securitySchemes"])
    assert "AdminKey" in spec["components"]["securitySchemes"]
    assert "ErrorBody" in spec["components"]["schemas"]


def _with_provider(settings: Settings, provider: str) -> Settings:
    """Change the embedding provider of the test settings.

    Args:
        settings: Test settings.
        provider: `local` or `gemini`.

    Returns:
        The changed settings.
    """
    return replace(settings, embedding_provider=provider)


def test_model_checks_include_the_gemini_embedding_model(settings: Settings) -> None:
    """With `EMBEDDING_PROVIDER=gemini` the start-up checks also fetch the embedding model's metadata.

    Args:
        settings: Test settings.
    """
    with_gemini = FakeLLM()
    check_models(with_gemini, _with_provider(settings, "gemini"), MemoryTelemetry())
    assert with_gemini.checked[-1] == GEMINI_EMBED_MODEL
    local = FakeLLM()
    check_models(local, _with_provider(settings, "local"), MemoryTelemetry())
    assert GEMINI_EMBED_MODEL not in local.checked
