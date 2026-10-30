"""The local blob store."""

import os
import time
import uuid
from datetime import timedelta

import pytest

from docqa.storage import LocalBlobStore, blob_key


def test_commit_open_and_duplicate(blobs: LocalBlobStore) -> None:
    """A temp file is renamed to its key; a second commit of the same content drops the temp file.

    Args:
        blobs: Blob store in a temp folder.
    """
    tenant_id = uuid.uuid4()
    digest = "b" * 64
    tmp = blobs.new_temp()
    tmp.write_bytes(b"%PDF-1.7 first")
    key = blobs.commit_temp(tmp, tenant_id, digest)
    assert key == f"{tenant_id}/{digest}.pdf"
    assert not tmp.exists()
    again = blobs.new_temp()
    again.write_bytes(b"%PDF-1.7 second")
    assert blobs.commit_temp(again, tenant_id, digest) == key
    assert not again.exists()
    with blobs.open(key) as path:
        assert path.read_bytes() == b"%PDF-1.7 first"


@pytest.mark.parametrize("key", ["../etc/passwd", "x/y.pdf", f"{uuid.uuid4()}/{'c' * 64}.pdf"])
def test_open_rejects_bad_or_missing_keys(blobs: LocalBlobStore, key: str) -> None:
    """Malformed keys and missing files raise `KeyError`.

    Args:
        blobs: Blob store in a temp folder.
        key: Key to open.
    """
    with pytest.raises(KeyError), blobs.open(key):
        pass


def test_blob_key_validates_digest() -> None:
    """The key builder refuses anything but a lowercase hex sha256."""
    with pytest.raises(ValueError, match="sha256"):
        blob_key(uuid.uuid4(), "../../x")


def test_cleanup_temp_deletes_only_old_files(blobs: LocalBlobStore) -> None:
    """Temp files older than the grace period are deleted; fresh ones stay.

    Args:
        blobs: Blob store in a temp folder.
    """
    old = blobs.new_temp()
    fresh = blobs.new_temp()
    past = time.time() - 7200
    os.utime(old, (past, past))
    assert blobs.cleanup_temp(timedelta(hours=1)) == 1
    assert not old.exists()
    assert fresh.exists()
