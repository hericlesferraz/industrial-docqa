"""Style checks ruff can't do: no `#` comments, and a docstring on every function, method and class.

Usage: `python tools/check_style.py PATH...`. Tool pragmas (`# noqa`, `# type: ignore`) are allowed and listed.
"""

import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterator
from pathlib import Path

PRAGMA = re.compile(r"^#\s*(noqa|type:\s*ignore)\b")
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def python_files(paths: list[str]) -> Iterator[Path]:
    """Expand paths into the Python files they contain.

    Args:
        paths: Files or folders.

    Yields:
        Every `.py` file, in sorted order, skipping `__pycache__`.
    """
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(p for p in path.rglob("*.py") if "__pycache__" not in p.parts)


def check_source(source: str, name: str) -> tuple[list[str], list[str]]:
    """Check one file's source.

    Args:
        source: File content.
        name: Path shown in the messages.

    Returns:
        `(findings, pragmas)`: `path:line: message` lines that fail the check, and the allowed pragmas found.
    """
    findings: list[str] = []
    pragmas: list[str] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        return [f"{name}:1: cannot tokenize: {exc}"], []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line = token.start[0]
        if PRAGMA.match(token.string):
            pragmas.append(f"{name}:{line}: {token.string}")
        elif line == 1 and token.string.startswith("#!"):
            continue
        else:
            findings.append(f"{name}:{line}: comment not allowed, move it to a docstring: {token.string[:60]}")
    try:
        tree = ast.parse(source, filename=name)
    except SyntaxError as exc:
        return [*findings, f"{name}:{exc.lineno or 1}: syntax error: {exc.msg}"], pragmas
    for node in ast.walk(tree):
        if isinstance(node, DEFINITIONS) and ast.get_docstring(node) is None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            findings.append(f"{name}:{node.lineno}: {kind} {node.name!r} has no docstring")
    return findings, pragmas


def main(argv: list[str]) -> int:
    """Check every Python file under the given paths.

    Args:
        argv: Files or folders to check.

    Returns:
        0 when every file passes, 1 otherwise.
    """
    findings: list[str] = []
    pragmas: list[str] = []
    for path in python_files(argv):
        found, allowed = check_source(path.read_text(encoding="utf-8"), str(path))
        findings.extend(found)
        pragmas.extend(allowed)
    for line in pragmas:
        print(f"pragma: {line}")
    for line in findings:
        print(line)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
