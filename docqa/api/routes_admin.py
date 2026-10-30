"""Admin routes, protected by `ADMIN_API_KEY`."""

from fastapi import APIRouter

from docqa.api.deps import AdminDep, DepsDep
from docqa.api.models import TenantCreate, TenantCreated
from docqa.db import tx
from docqa.errors import ErrorBody
from docqa.repo import tenants
from docqa.telemetry import add_ids

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.post(
    "/tenants",
    status_code=201,
    response_model=TenantCreated,
    responses={401: {"model": ErrorBody}, 422: {"model": ErrorBody}},
    summary="Create a tenant and its API key",
)
def create_tenant(body: TenantCreate, deps: DepsDep, _admin: AdminDep) -> TenantCreated:
    """Create a tenant; the key in the response is shown once.

    Args:
        body: Tenant name.
        deps: Process objects.
        _admin: Admin authentication.

    Returns:
        The tenant id, its name and its API key.
    """
    with tx(deps.pool) as conn:
        tenant_id, api_key = tenants.create_tenant(conn, name=body.name)
    add_ids(tenant_id=tenant_id)
    return TenantCreated(tenant_id=tenant_id, name=body.name, api_key=api_key)
