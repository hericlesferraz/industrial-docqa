"""Dependencies of the HTTP API: the container of process objects and the authentication checks."""

import hmac
from dataclasses import dataclass, field
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg_pool import ConnectionPool

from docqa.config import Settings
from docqa.errors import UNAUTHORIZED, ApiError
from docqa.index.embed import Embedder
from docqa.providers.llm import LLM
from docqa.repo import tenants
from docqa.storage import BlobStore
from docqa.telemetry import Telemetry, add_ids


@dataclass
class Health:
    """Results of the start-up checks that `/healthz` reports.

    Attributes:
        models: Status per configured model, filled by the model checks.
        reasons: Why the service is degraded, if it is.
    """

    models: dict[str, str] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class Deps:
    """Process objects shared by the routes.

    Attributes:
        pool: Database pool.
        settings: Process settings.
        telemetry: Telemetry.
        blobs: File store.
        llm: Provider gateway, or None in tests that don't call a model.
        embedder: Embedder, or None in tests that don't search.
        health: Start-up check results.
    """

    pool: ConnectionPool
    settings: Settings
    telemetry: Telemetry
    blobs: BlobStore
    llm: LLM | None = None
    embedder: Embedder | None = None
    health: Health = field(default_factory=Health)


tenant_bearer = HTTPBearer(
    auto_error=False, scheme_name="TenantKey", description="API key of a tenant, created by the admin endpoint."
)
admin_bearer = HTTPBearer(auto_error=False, scheme_name="AdminKey", description="The `ADMIN_API_KEY`.")


def get_deps(request: Request) -> Deps:
    """Return the process objects stored on the application.

    Args:
        request: The current request.

    Returns:
        The dependencies container.
    """
    return request.app.state.deps


DepsDep = Annotated[Deps, Depends(get_deps)]


def _unauthorized() -> ApiError:
    """Build the 401 error.

    Returns:
        The error, with the `WWW-Authenticate` header.
    """
    return ApiError(401, UNAUTHORIZED, "Missing or invalid API key.", headers={"WWW-Authenticate": "Bearer"})


def require_tenant(
    deps: DepsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(tenant_bearer)],
) -> UUID:
    """Resolve the tenant of the request from its bearer key.

    Args:
        deps: Process objects.
        credentials: The parsed `Authorization: Bearer` header, if any.

    Returns:
        The tenant id.

    Raises:
        ApiError: 401 when the key is missing or unknown; the admin key is not a tenant key.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized()
    with deps.pool.connection() as conn:
        tenant_id = tenants.tenant_for_key(conn, credentials.credentials)
    if tenant_id is None:
        raise _unauthorized()
    add_ids(tenant_id=tenant_id)
    return tenant_id


def require_admin(
    deps: DepsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(admin_bearer)],
) -> None:
    """Check the admin key of the request in constant time.

    Args:
        deps: Process objects.
        credentials: The parsed `Authorization: Bearer` header, if any.

    Raises:
        ApiError: 401 when the key is missing or is not `ADMIN_API_KEY`.
    """
    if credentials is None or not credentials.credentials:
        raise _unauthorized()
    given = credentials.credentials.encode("utf-8")
    if not hmac.compare_digest(given, deps.settings.admin_api_key.encode("utf-8")):
        raise _unauthorized()


TenantDep = Annotated[UUID, Depends(require_tenant)]
AdminDep = Annotated[None, Depends(require_admin)]
