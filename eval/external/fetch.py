"""Download the external eval set listed in `manifest.json` and check each file against its recorded sha256.

Run by `make eval-external` in the eval image, or with any Python 3.12 that has PyMuPDF:
`python -I eval/external/fetch.py`. Each manual is downloaded from its original URL into `pdfs/` next to this script
(git-ignored); a manual with an `excerpt` is downloaded into `originals/` instead, and only its page range is written
to `pdfs/`. A file already there with the right sha256 is not downloaded again.

The PDFs are untrusted: this script only downloads them, hashes them and copies a verified file's pages with PyMuPDF;
it never runs anything they contain. A failed download or a wrong hash is listed with its reason, the file is kept out
of `pdfs/`, and the other files go on. The exit code is 1 only when no file of the manifest ends up available, 2 when
the manifest can't be read.
"""

import argparse
import hashlib
import http.client
import json
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
USER_AGENT = "docqa-eval-fetch/1.0"
CHUNK_BYTES = 1 << 20
ATTEMPTS = 2
OK_STATES = ("present", "downloaded", "cut")


class FetchError(Exception):
    """A file could not be made available; the message says why."""


@dataclass(frozen=True)
class Excerpt:
    """A page range of a downloaded manual that the eval uses in place of the whole file.

    Attributes:
        file: Name of the excerpt in the eval folder.
        first_page: First 1-based page kept.
        last_page: Last 1-based page kept.
        sha256: What the copy rule writes with the PyMuPDF version of the image.
    """

    file: str
    first_page: int
    last_page: int
    sha256: str


@dataclass(frozen=True)
class Manual:
    """One entry of the manifest.

    Attributes:
        file: Name of the downloaded file.
        url: Direct download URL.
        sha256: Hex digest of the file as downloaded when the set was built.
        size: Size in bytes; a download is cut off past it.
        excerpt: The page range used in place of the whole file, if any.
    """

    file: str
    url: str
    sha256: str
    size: int
    excerpt: Excerpt | None = None

    @property
    def eval_file(self) -> str:
        """Name of the file this manual puts in the eval folder.

        Returns:
            The excerpt's name when there is one, else the manual's.
        """
        return self.excerpt.file if self.excerpt else self.file


@dataclass(frozen=True)
class Outcome:
    """What happened to one manual.

    Attributes:
        file: Name of the file in the eval folder.
        state: `present`, `downloaded`, `cut` or `failed`.
        detail: Reason of a failure, or a note.
    """

    file: str
    state: str
    detail: str

    @property
    def available(self) -> bool:
        """Whether the eval folder holds the file.

        Returns:
            True unless the manual failed.
        """
        return self.state in OK_STATES


def safe_name(name: object) -> str:
    """Check that a manifest file name is a plain, visible file name.

    Args:
        name: The value from the manifest.

    Returns:
        The name.

    Raises:
        ValueError: The name is not a string, has a path separator or starts with a dot.
    """
    if not isinstance(name, str) or not name or name != Path(name).name or name.startswith(".") or "\\" in name:
        raise ValueError(f"bad file name {name!r}")
    return name


def load_manifest(path: Path) -> list[Manual]:
    """Read the manifest.

    Args:
        path: The JSON file.

    Returns:
        Its manuals, in order.

    Raises:
        ValueError: The file is not valid JSON or an entry lacks a field.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    manuals = []
    for entry in data["documents"]:
        cut = entry.get("excerpt")
        excerpt = None
        if cut is not None:
            excerpt = Excerpt(safe_name(cut["file"]), int(cut["first_page"]), int(cut["last_page"]), str(cut["sha256"]))
            if not 1 <= excerpt.first_page <= excerpt.last_page:
                raise ValueError(f"bad page range in the excerpt of {entry['file']!r}")
        manuals.append(Manual(safe_name(entry["file"]), str(entry["url"]), str(entry["sha256"]), int(entry["bytes"]),
                              excerpt))
    names = [m.eval_file for m in manuals]
    if len(set(names)) != len(names):
        raise ValueError("two manuals share a file name")
    return manuals


def sha256_of(path: Path) -> str:
    """Hash a file.

    Args:
        path: The file.

    Returns:
        Its sha256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def partial_path(target: Path) -> Path:
    """Name of the temporary file a download goes to; hidden, so the eval never uploads a leftover one.

    Args:
        target: The final path.

    Returns:
        `.<name>.part` in the same folder.
    """
    return target.with_name(f".{target.name}.part")


def download_once(url: str, part: Path, limit: int, timeout: float) -> str:
    """Stream one URL into a file, hashing it on the way.

    Args:
        url: The URL.
        part: Where to write.
        limit: Largest size accepted, in bytes.
        timeout: Socket timeout in seconds.

    Returns:
        The sha256 hex digest of what was written.

    Raises:
        FetchError: The server answered with more bytes than `limit`.
        OSError: The connection or the HTTP request failed (`urllib.error.URLError` is one).
        http.client.HTTPException: The response was cut or malformed.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            size += len(chunk)
            if size > limit:
                raise FetchError(f"larger than the recorded {limit} bytes")
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def download(manual: Manual, target: Path, timeout: float) -> None:
    """Download a manual to `target` and keep it only if its sha256 matches the manifest.

    Args:
        manual: The manual.
        target: Final path.
        timeout: Socket timeout in seconds.

    Raises:
        FetchError: Every attempt failed, or the file differs from the recorded one.
    """
    part = partial_path(target)
    error = ""
    try:
        for _ in range(ATTEMPTS):
            try:
                digest = download_once(manual.url, part, manual.size, timeout)
            except (OSError, http.client.HTTPException, ValueError) as exc:
                error = f"download failed: {type(exc).__name__}: {exc}"
                continue
            if digest != manual.sha256:
                raise FetchError(f"sha256 mismatch: got {digest}, expected {manual.sha256} (not the recorded file)")
            os.replace(part, target)
            return
        raise FetchError(error)
    finally:
        part.unlink(missing_ok=True)


def ensure(manual: Manual, target: Path, timeout: float) -> str:
    """Make sure `target` holds the manual, downloading it if needed.

    Args:
        manual: The manual.
        target: Where it must be.
        timeout: Socket timeout in seconds.

    Returns:
        `present` when the right file was already there, `downloaded` otherwise.

    Raises:
        FetchError: The file could not be made available; a stale file at `target` is removed.
    """
    if target.is_file():
        if sha256_of(target) == manual.sha256:
            return "present"
        target.unlink()
    download(manual, target, timeout)
    return "downloaded"


def cut_excerpt(original: Path, excerpt: Excerpt, target: Path) -> str:
    """Copy a page range of a verified PDF into a new one, page numbers unchanged.

    Args:
        original: The whole manual, already checked against its sha256.
        excerpt: The range and output name.
        target: Output path.

    Returns:
        The sha256 hex digest of the written excerpt.

    Raises:
        FetchError: The original has fewer pages than the range needs.
    """
    with pymupdf.open(original, filetype="pdf") as source:
        if source.page_count < excerpt.last_page:
            raise FetchError(f"the original has {source.page_count} pages, the excerpt needs {excerpt.last_page}")
        out = pymupdf.open()
        out.insert_pdf(source, from_page=excerpt.first_page - 1, to_page=excerpt.last_page - 1)
        data = out.tobytes(garbage=3, deflate=True, no_new_id=True)
        out.close()
    part = partial_path(target)
    part.write_bytes(data)
    os.replace(part, target)
    return hashlib.sha256(data).hexdigest()


def fetch_manual(manual: Manual, out_dir: Path, originals_dir: Path, timeout: float) -> Outcome:
    """Make one manual available in the eval folder.

    Args:
        manual: The manual.
        out_dir: The eval folder.
        originals_dir: Where whole manuals that only lend an excerpt are kept.
        timeout: Socket timeout in seconds.

    Returns:
        What happened; never raises for a download or hash problem.
    """
    excerpt = manual.excerpt
    if excerpt is None:
        try:
            state = ensure(manual, out_dir / manual.file, timeout)
        except FetchError as exc:
            return Outcome(manual.file, "failed", str(exc))
        return Outcome(manual.file, state, "sha256 ok")
    target = out_dir / excerpt.file
    if target.is_file() and sha256_of(target) == excerpt.sha256:
        return Outcome(excerpt.file, "present", "sha256 ok")
    target.unlink(missing_ok=True)
    try:
        ensure(manual, originals_dir / manual.file, timeout)
        digest = cut_excerpt(originals_dir / manual.file, excerpt, target)
    except FetchError as exc:
        return Outcome(excerpt.file, "failed", f"original {manual.file}: {exc}")
    pages = f"pages {excerpt.first_page}-{excerpt.last_page} of the verified original"
    if digest == excerpt.sha256:
        return Outcome(excerpt.file, "cut", f"{pages}, sha256 ok")
    return Outcome(excerpt.file, "cut", f"{pages}; sha256 {digest} differs from the recorded {excerpt.sha256} "
                                        "(another PyMuPDF version writes other bytes; the pages are the same)")


def strays(out_dir: Path, manuals: list[Manual]) -> list[str]:
    """List the visible files of the eval folder that the manifest doesn't name; the eval would upload them too.

    Args:
        out_dir: The eval folder.
        manuals: The manifest.

    Returns:
        Their names, sorted.
    """
    known = {m.eval_file for m in manuals}
    return sorted(p.name for p in out_dir.iterdir() if p.is_file() and not p.name.startswith(".") and p.name not in known)


def main(argv: list[str]) -> int:
    """Fetch every manual of the manifest and print one line per file.

    Args:
        argv: Command-line arguments, without the program name.

    Returns:
        0 when at least one file is available, 1 when none is, 2 when the manifest can't be read.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json", help="manifest JSON")
    parser.add_argument("--out", type=Path, default=HERE / "pdfs", help="eval folder (default: pdfs/ next to this file)")
    parser.add_argument("--originals", type=Path, default=HERE / "originals",
                        help="where whole manuals that only lend an excerpt are kept (default: originals/)")
    parser.add_argument("--timeout", type=float, default=60.0, help="socket timeout in seconds")
    args = parser.parse_args(argv)
    try:
        manuals = load_manifest(args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"cannot read the manifest {args.manifest}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    args.originals.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for manual in manuals:
        outcome = fetch_manual(manual, args.out, args.originals, args.timeout)
        outcomes.append(outcome)
        print(f"{outcome.state:<10} {outcome.file}: {outcome.detail}", flush=True)
    for name in strays(args.out, manuals):
        print(f"{'extra':<10} {name}: not in the manifest; the eval uploads it too", flush=True)
    available = sum(o.available for o in outcomes)
    print(f"{available} of {len(outcomes)} files available in {args.out}", flush=True)
    return 0 if available else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
