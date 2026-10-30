"""Guard of the data-access rule: SQL only in the data layer, and the layer filters by tenant.

The allowlists below are the only exceptions; each one carries its reason here, never in a code comment.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "docqa"

SQL_MODULES = {
    "db.py": "pool, transactions and migrations",
    "queue.py": "the job queue serves every tenant; jobs carry tenant_id",
    "telemetry.py": "span rows carry tenant_id as data, written by flush",
    "providers/pacing.py": "per-model call slots and pauses, not customer data",
    "retrieval/search.py": "vector and rare-term search; every function takes tenant_id",
}
SQL_PACKAGES = ("repo/",)
TENANT_CHECKED = ("repo/", "retrieval/")
TENANT_EXEMPT = {
    "repo/ocr_cache.py": "the OCR cache is keyed by file content, shared across tenants by design [#5]",
    "repo/maintenance.py": "cleanup and index reconcile run across tenants",
    "repo/tenants.py:create_tenant": "creates the tenant",
    "repo/tenants.py:tenant_for_key": "resolves the tenant from its key",
}
SQL_CALLS = {"execute", "executemany", "cursor", "copy"}


def _modules() -> list[tuple[str, ast.Module]]:
    """Parse every module of the package.

    Returns:
        `(path relative to the package, tree)` pairs.
    """
    return [
        (path.relative_to(PACKAGE).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(PACKAGE.rglob("*.py"))
    ]


def _sql_calls(node: ast.AST) -> list[int]:
    """Find calls of `.execute`, `.executemany`, `.cursor` or `.copy` under a node.

    Args:
        node: Any AST node.

    Returns:
        Line numbers of the calls.
    """
    return [
        call.lineno
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in SQL_CALLS
    ]


def test_sql_only_in_the_data_layer() -> None:
    """No module outside the allowlist runs SQL."""
    offenders = [
        f"docqa/{name}:{line}"
        for name, tree in _modules()
        if name not in SQL_MODULES and not name.startswith(SQL_PACKAGES)
        for line in _sql_calls(tree)
    ]
    assert offenders == []


def test_data_layer_functions_take_tenant_id() -> None:
    """Every function of the tenant-checked modules that runs SQL takes a keyword-only `tenant_id`."""
    offenders = []
    for name, tree in _modules():
        if not name.startswith(TENANT_CHECKED) or name in TENANT_EXEMPT:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or f"{name}:{node.name}" in TENANT_EXEMPT:
                continue
            own_body = ast.Module(body=node.body, type_ignores=[])
            if _sql_calls(own_body) and "tenant_id" not in {arg.arg for arg in node.args.kwonlyargs}:
                offenders.append(f"docqa/{name}:{node.lineno} {node.name}")
    assert offenders == []


def test_allowlists_point_at_real_code() -> None:
    """A function exemption names a function that exists once its module exists, so stale entries don't linger."""
    modules = _modules()
    names = {name for name, _ in modules}
    functions = {
        f"{name}:{node.name}" for name, tree in modules for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    for entry in TENANT_EXEMPT:
        module, _, function = entry.partition(":")
        if module in names and function:
            assert entry in functions, entry
