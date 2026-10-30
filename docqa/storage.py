"""Uploaded PDFs stored by key, behind an interface shaped for object storage."""

import os
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from uuid import UUID

KEY_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/[0-9a-f]{64}\.pdf$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class BlobStore(Protocol):
    """Stores uploaded PDFs by key; the database keeps keys, never paths."""

    def new_temp(self) -> Path:
        """Create an empty temp file on the same filesystem as the stored files.

        Returns:
            Path of the new temp file, under `{root}/tmp/`.
        """
        ...

    def commit_temp(self, tmp: Path, tenant_id: UUID, sha256: str) -> str:
        """Move a finished temp file to its final key with an atomic rename.

        Args:
            tmp: Temp file written by the upload.
            tenant_id: Owner of the document.
            sha256: Hex digest computed while streaming.

        Returns:
            The key `{tenant_id}/{sha256}.pdf`. If the file already exists, the temp file is deleted.
        """
        ...

    def open(self, key: str) -> AbstractContextManager[Path]:
        """Give a local path to read a stored file.

        Args:
            key: Key returned by `commit_temp`.

        Returns:
            A context manager yielding a local path: the file itself here, a downloaded temp copy for S3.
        """
        ...

    def cleanup_temp(self, older_than: timedelta) -> int:
        """Delete abandoned temp files.

        Args:
            older_than: Minimum age of the files to delete.

        Returns:
            Number of files deleted.
        """
        ...


def blob_key(tenant_id: UUID, sha256: str) -> str:
    """Build the key of a stored PDF.

    Args:
        tenant_id: Owner of the document.
        sha256: Hex digest of the file.

    Returns:
        `{tenant_id}/{sha256}.pdf`.

    Raises:
        ValueError: The digest is not 64 lowercase hex characters.
    """
    if not SHA256_PATTERN.match(sha256):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    return f"{tenant_id}/{sha256}.pdf"


class LocalBlobStore:
    """`BlobStore` over a local folder, normally a named volume shared by the api and the worker.

    Attributes:
        root: Root folder; stored files live under `{root}/{tenant_id}/`, temp files under `{root}/tmp/`.
    """

    def __init__(self, root: Path) -> None:
        """Create the folders if needed.

        Args:
            root: Root folder.
        """
        self.root = root
        self._tmp = root / "tmp"
        self._tmp.mkdir(parents=True, exist_ok=True)

    def new_temp(self) -> Path:
        """Create an empty temp file on the same filesystem as the stored files.

        Returns:
            Path of the new temp file, under `{root}/tmp/`.
        """
        path = self._tmp / f"{uuid.uuid4()}.part"
        path.touch(exist_ok=False)
        return path

    def commit_temp(self, tmp: Path, tenant_id: UUID, sha256: str) -> str:
        """Move a finished temp file to its final key with an atomic rename.

        Args:
            tmp: Temp file written by the upload.
            tenant_id: Owner of the document.
            sha256: Hex digest computed while streaming.

        Returns:
            The key `{tenant_id}/{sha256}.pdf`. If the file already exists, the temp file is deleted.
        """
        key = blob_key(tenant_id, sha256)
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            tmp.unlink(missing_ok=True)
        else:
            os.replace(tmp, target)
        return key

    @contextmanager
    def _open(self, key: str) -> Iterator[Path]:
        """Implement `open`.

        Args:
            key: Key returned by `commit_temp`.

        Yields:
            The stored file's path.

        Raises:
            KeyError: The key is malformed or the file doesn't exist.
        """
        if not KEY_PATTERN.match(key):
            raise KeyError(key)
        path = self.root / key
        if not path.is_file():
            raise KeyError(key)
        yield path

    def open(self, key: str) -> AbstractContextManager[Path]:
        """Give a local path to read a stored file.

        Args:
            key: Key returned by `commit_temp`.

        Returns:
            A context manager yielding the stored file's path; it raises `KeyError` on entry when the key is
            malformed or the file doesn't exist.
        """
        return self._open(key)

    def cleanup_temp(self, older_than: timedelta) -> int:
        """Delete abandoned temp files.

        Args:
            older_than: Minimum age of the files to delete.

        Returns:
            Number of files deleted.
        """
        cutoff = time.time() - older_than.total_seconds()
        deleted = 0
        for path in self._tmp.glob("*.part"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except FileNotFoundError:
                continue
        return deleted
