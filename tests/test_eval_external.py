"""The external eval set: its manifest, `eval/external/fetch.py` (against `file://` URLs) and the `make eval` defaults."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "eval" / "external"
FIELDS = {"file", "title", "url", "source_page", "sha256", "bytes", "pages", "languages", "scanned", "license", "tests"}


def _load_fetch() -> ModuleType:
    """Import the fetch script, which is not part of a package.

    Returns:
        The module.
    """
    spec = importlib.util.spec_from_file_location("external_fetch", EXTERNAL / "fetch.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch = _load_fetch()


def _pdf(pages: int, tag: str) -> bytes:
    """Build a small PDF whose pages say their number.

    Args:
        pages: Page count.
        tag: Text that makes the file unique.

    Returns:
        The PDF bytes.
    """
    doc = pymupdf.open()
    for n in range(1, pages + 1):
        doc.new_page().insert_text((72, 72), f"{tag} page {n}")
    data = doc.tobytes(no_new_id=True)
    doc.close()
    return data


def _sha(data: bytes) -> str:
    """Hash bytes.

    Args:
        data: The bytes.

    Returns:
        The sha256 hex digest.
    """
    return hashlib.sha256(data).hexdigest()


class Remote:
    """A folder that stands for the download servers, reached through `file://` URLs."""

    def __init__(self, root: Path) -> None:
        """Create the folder.

        Args:
            root: Where to keep the served files.
        """
        self.root = root
        root.mkdir()

    def serve(self, name: str, data: bytes) -> str:
        """Publish a file.

        Args:
            name: File name.
            data: Content.

        Returns:
            Its URL.
        """
        path = self.root / name
        path.write_bytes(data)
        return path.as_uri()


def _entry(name: str, url: str, data: bytes, **extra: object) -> dict[str, object]:
    """Build a manifest entry.

    Args:
        name: File name.
        url: Download URL.
        data: The content the manifest records.
        **extra: Fields to add or override.

    Returns:
        The entry.
    """
    return {"file": name, "url": url, "sha256": _sha(data), "bytes": len(data), **extra}


def _run(tmp_path: Path, entries: list[dict[str, object]]) -> tuple[int, Path, Path]:
    """Write a manifest and run the script's `main` on it.

    Args:
        tmp_path: Test folder.
        entries: Manifest entries.

    Returns:
        `(exit code, eval folder, originals folder)`.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": entries}), encoding="utf-8")
    out, originals = tmp_path / "pdfs", tmp_path / "originals"
    code = fetch.main(["--manifest", str(manifest), "--out", str(out), "--originals", str(originals)])
    return code, out, originals


def _visible(folder: Path) -> list[str]:
    """List the files the eval would upload.

    Args:
        folder: The eval folder.

    Returns:
        Visible file names, sorted.
    """
    return sorted(p.name for p in folder.iterdir() if not p.name.startswith("."))


def test_downloads_checks_and_skips_present_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Files are downloaded once; a second run finds them by hash and doesn't touch the URL."""
    remote = Remote(tmp_path / "remote")
    a, b = _pdf(2, "a"), _pdf(3, "b")
    entries = [_entry("a.pdf", remote.serve("a.pdf", a), a), _entry("b.pdf", remote.serve("b.pdf", b), b)]
    code, out, _ = _run(tmp_path, entries)
    assert code == 0
    assert (out / "a.pdf").read_bytes() == a and (out / "b.pdf").read_bytes() == b
    assert "downloaded a.pdf: sha256 ok" in capsys.readouterr().out
    for served in (tmp_path / "remote").iterdir():
        served.unlink()
    code, out, _ = _run(tmp_path, entries)
    assert code == 0
    printed = capsys.readouterr().out
    assert "present    a.pdf: sha256 ok" in printed and "2 of 2 files available" in printed


def test_failures_are_listed_and_the_rest_goes_on(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A changed file, a dead URL and an oversized file fail with a reason; the good file still arrives."""
    remote = Remote(tmp_path / "remote")
    good, recorded, changed = _pdf(1, "good"), _pdf(1, "old"), _pdf(1, "new")
    big = _pdf(4, "big")
    entries = [
        _entry("changed.pdf", remote.serve("changed.pdf", changed), recorded),
        _entry("gone.pdf", (tmp_path / "remote" / "nothing.pdf").as_uri(), good),
        _entry("big.pdf", remote.serve("big.pdf", big), big, bytes=len(big) - 1),
        _entry("good.pdf", remote.serve("good.pdf", good), good),
    ]
    code, out, _ = _run(tmp_path, entries)
    printed = capsys.readouterr().out
    assert code == 0
    assert _visible(out) == ["good.pdf"]
    assert list(out.iterdir()) == [out / "good.pdf"]
    assert f"failed     changed.pdf: sha256 mismatch: got {_sha(changed)}, expected {_sha(recorded)}" in printed
    assert "failed     gone.pdf: download failed: URLError" in printed
    assert "failed     big.pdf: larger than the recorded" in printed
    assert "1 of 4 files available" in printed


def test_exit_code_is_1_only_when_nothing_is_available(tmp_path: Path) -> None:
    """Every file failing is the only case that exits non-zero."""
    data = _pdf(1, "x")
    code, out, _ = _run(tmp_path, [_entry("x.pdf", (tmp_path / "missing.pdf").as_uri(), data)])
    assert code == 1
    assert _visible(out) == []


def test_stale_file_is_replaced(tmp_path: Path) -> None:
    """A file already in the folder with another hash is downloaded again."""
    remote = Remote(tmp_path / "remote")
    data = _pdf(2, "fresh")
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "a.pdf").write_bytes(b"stale")
    code, out, _ = _run(tmp_path, [_entry("a.pdf", remote.serve("a.pdf", data), data)])
    assert code == 0
    assert (out / "a.pdf").read_bytes() == data


def test_excerpt_keeps_the_page_range_and_the_original_out_of_the_eval_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only the excerpt reaches the eval folder: the recorded pages, the same bytes on every cut, found by hash."""
    remote = Remote(tmp_path / "remote")
    whole = _pdf(5, "manual")
    excerpt = {"file": "manual_p2-3.pdf", "first_page": 2, "last_page": 3, "sha256": "0" * 64}
    entry = _entry("manual.pdf", remote.serve("manual.pdf", whole), whole, excerpt=excerpt)
    code, out, originals = _run(tmp_path, [entry])
    assert code == 0
    assert _visible(out) == ["manual_p2-3.pdf"]
    assert (originals / "manual.pdf").read_bytes() == whole
    with pymupdf.open(out / "manual_p2-3.pdf") as doc:
        assert [page.get_text().strip() for page in doc] == ["manual page 2", "manual page 3"]
    first = (out / "manual_p2-3.pdf").read_bytes()
    assert "differs from the recorded" in capsys.readouterr().out
    excerpt["sha256"] = _sha(first)
    (out / "manual_p2-3.pdf").unlink()
    code, out, _ = _run(tmp_path, [entry])
    assert "cut        manual_p2-3.pdf: pages 2-3 of the verified original, sha256 ok" in capsys.readouterr().out
    assert (out / "manual_p2-3.pdf").read_bytes() == first
    code, out, _ = _run(tmp_path, [entry])
    assert "present    manual_p2-3.pdf: sha256 ok" in capsys.readouterr().out


def test_excerpt_past_the_last_page_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A range beyond the original is a listed failure, not a crash."""
    remote = Remote(tmp_path / "remote")
    whole = _pdf(2, "short")
    excerpt = {"file": "short_p1-9.pdf", "first_page": 1, "last_page": 9, "sha256": "0" * 64}
    code, out, _ = _run(tmp_path, [_entry("short.pdf", remote.serve("short.pdf", whole), whole, excerpt=excerpt)])
    assert code == 1
    assert "the original has 2 pages, the excerpt needs 9" in capsys.readouterr().out
    assert _visible(out) == []


def test_extra_files_in_the_folder_are_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A visible file the manifest doesn't name is pointed out, since the eval would upload it."""
    remote = Remote(tmp_path / "remote")
    data = _pdf(1, "a")
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "pdfs" / "mine.pdf").write_bytes(b"x")
    _run(tmp_path, [_entry("a.pdf", remote.serve("a.pdf", data), data)])
    assert "extra      mine.pdf: not in the manifest" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["../up.pdf", "sub/a.pdf", ".hidden.pdf", "C:\\x.pdf", ""])
def test_manifest_rejects_unsafe_file_names(tmp_path: Path, name: str) -> None:
    """A file name that leaves the folder or hides from the eval makes the manifest unreadable (exit 2)."""
    code, _, _ = _run(tmp_path, [{"file": name, "url": "file:///x", "sha256": "0" * 64, "bytes": 1}])
    assert code == 2


def test_real_manifest_is_complete() -> None:
    """Every manual of the committed manifest has its documented fields, a full hash and a unique eval file name."""
    documents = json.loads((EXTERNAL / "manifest.json").read_text(encoding="utf-8"))["documents"]
    assert len(documents) == 10
    for document in documents:
        assert FIELDS <= set(document), document["file"]
        assert re.fullmatch(r"[0-9a-f]{64}", document["sha256"])
        assert document["url"].startswith("https://")
    manuals = fetch.load_manifest(EXTERNAL / "manifest.json")
    assert len({m.eval_file for m in manuals}) == 10
    cut = [m for m in manuals if m.excerpt]
    assert [(m.file, m.excerpt.first_page, m.excerpt.last_page) for m in cut] == [
        ("tm5-5060_leroi_air_compressor_1942_scan.pdf", 1, 60)
    ]


def test_make_eval_defaults_stay_on_the_case_files() -> None:
    """`make eval` without arguments keeps the sample PDFs and public questions; only the external targets name the set."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^PDFS \?= case_files$", makefile, re.MULTILINE)
    assert re.search(r"^QUESTIONS \?= eval/public_questions\.jsonl$", makefile, re.MULTILINE)
    recipes = dict(re.findall(r"^([\w-]+):[^\n=]*\n((?:\t[^\n]*\n)*)", makefile, re.MULTILINE))
    users = sorted(name for name, body in recipes.items() if "EXTERNAL" in body or "external" in body)
    assert users == ["eval-external", "eval-external-fetch"]
    assert "QUESTIONS=$(EXTERNAL)/questions_external.jsonl" in recipes["eval-external"]
