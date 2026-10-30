"""Command line for operators: `python -m docqa.cli create-tenant --name NAME` (used by `make tenant`)."""

import argparse
import sys

import httpx

from docqa.config import ConfigError, Settings, load_settings

REQUEST_TIMEOUT_S = 30.0


def create_tenant(settings: Settings, name: str) -> dict[str, str]:
    """Create a tenant through the admin endpoint of the running API.

    Args:
        settings: Settings with `ADMIN_API_KEY` and `EVAL_API_URL`.
        name: Display name of the tenant.

    Returns:
        The `TenantCreated` body.

    Raises:
        RuntimeError: The API answered with an error; the message carries its error body.
    """
    response = httpx.post(
        f"{settings.eval_api_url}/v1/admin/tenants",
        json={"name": name},
        headers={"Authorization": f"Bearer {settings.admin_api_key}"},
        timeout=REQUEST_TIMEOUT_S,
    )
    if response.status_code != 201:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    return response.json()


def main(argv: list[str] | None = None) -> int:
    """Run a command.

    Args:
        argv: Command-line arguments; `sys.argv[1:]` when None.

    Returns:
        Process exit code: 0 on success, 1 on an API error, 2 on a configuration error.
    """
    parser = argparse.ArgumentParser(prog="python -m docqa.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create-tenant", help="create a tenant and print its API key")
    create.add_argument("--name", required=True, help="display name of the tenant")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(require_database=False)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        created = create_tenant(settings, args.name)
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"create-tenant failed: {exc}", file=sys.stderr)
        return 1
    print(f"tenant_id={created['tenant_id']}")
    print(f"api_key={created['api_key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
