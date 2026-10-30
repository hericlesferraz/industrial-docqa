"""The style checker of `make lint`."""

from tools.check_style import check_source

DOCUMENTED = '''"""Module."""


def visible(x: int) -> int:
    """Return x.

    Args:
        x: A number.

    Returns:
        The number.
    """
    return x
'''
COMMENT = "# explain"
PRAGMA = "# noqa: E501"


def test_clean_source_passes() -> None:
    """A documented file without comments passes."""
    assert check_source(DOCUMENTED, "ok.py") == ([], [])


def test_comment_fails() -> None:
    """A `#` comment is reported with its line."""
    findings, _ = check_source(f"{DOCUMENTED}\nvalue = 1  {COMMENT}\n", "bad.py")
    assert len(findings) == 1
    assert findings[0].startswith("bad.py:15: comment not allowed")


def test_private_and_nested_functions_need_docstrings() -> None:
    """Private functions, nested functions and classes without a docstring are reported."""
    source = (
        DOCUMENTED + "\n\ndef _hidden() -> None:\n    def inner() -> None:\n        pass\n\n\nclass Thing:\n    pass\n"
    )
    findings, _ = check_source(source, "bad.py")
    messages = " ".join(findings)
    assert "'_hidden' has no docstring" in messages
    assert "'inner' has no docstring" in messages
    assert "'Thing' has no docstring" in messages


def test_pragmas_pass_and_are_listed() -> None:
    """Tool pragmas are allowed and listed in the output."""
    findings, pragmas = check_source(f"{DOCUMENTED}\nvalue = 1  {PRAGMA}\n", "ok.py")
    assert findings == []
    assert pragmas == [f"ok.py:15: {PRAGMA}"]
