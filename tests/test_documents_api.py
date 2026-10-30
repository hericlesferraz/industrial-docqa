"""Upload, batches and document views through the HTTP API, with tenant isolation."""

import hashlib
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docqa.api.app import create_app
from docqa.api.deps import Deps
from docqa.storage import LocalBlobStore
from docqa.worker import WorkerDeps
from tests.helpers import TenantInfo, make_tenant
from tests.ingest_helpers import drain, jobs_of
from tests.pdfs import LB5001, MN414, case_pdf, encrypted_pdf, garbage_pdf, truncated_pdf

BOUNDARY = "docqa-test-boundary"
MIB = 2**20


def _auth(tenant: TenantInfo) -> dict[str, str]:
    """Build the bearer header of a tenant.

    Args:
        tenant: The tenant.

    Returns:
        The header.
    """
    return {"Authorization": f"Bearer {tenant.key}"}


def _upload(
    client: TestClient, tenant: TenantInfo, filename: str, data: bytes, batch_id: str | None = None
) -> httpx.Response:
    """Upload one file as multipart part `file`.

    Args:
        client: HTTP client.
        tenant: Uploading tenant.
        filename: Client filename.
        data: File bytes.
        batch_id: Batch to record the file in.

    Returns:
        The response.
    """
    params = {"batch_id": batch_id} if batch_id else None
    return client.post(
        "/v1/documents", params=params, files={"file": (filename, data, "application/pdf")}, headers=_auth(tenant)
    )


def _multipart(parts: list[tuple[str, str | None, bytes]], *, close: bool = True) -> bytes:
    """Build a raw multipart body.

    Args:
        parts: `(name, filename, data)` per part; a None filename omits the parameter.
        close: Whether to write the closing boundary.

    Returns:
        The body.
    """
    body = b""
    for name, filename, data in parts:
        disposition = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename is not None else "")
        body += f"--{BOUNDARY}\r\nContent-Disposition: {disposition}\r\nContent-Type: application/pdf\r\n\r\n".encode()
        body += data + b"\r\n"
    if close:
        body += f"--{BOUNDARY}--\r\n".encode()
    return body


def _temp_files(blobs: LocalBlobStore) -> list[str]:
    """List the upload temp files left behind.

    Args:
        blobs: Blob store.

    Returns:
        Their names.
    """
    return sorted(path.name for path in (blobs.root / "tmp").iterdir())


def test_upload_then_duplicate(client: TestClient, tenant: TenantInfo, blobs: LocalBlobStore, deps: Deps) -> None:
    """A new file answers 202 and queues one `extract`; the same bytes again answer 200 with `duplicate: true`.

    Args:
        client: HTTP client.
        tenant: The tenant.
        blobs: Blob store.
        deps: API dependencies.
    """
    data = case_pdf(LB5001)
    first = _upload(client, tenant, "LB5001.pdf", data)
    assert first.status_code == 202, first.text
    body = first.json()
    assert (body["filename"], body["status"], body["duplicate"]) == ("LB5001.pdf", "queued", False)
    sha256 = hashlib.sha256(data).hexdigest()
    assert (blobs.root / f"{tenant.id}/{sha256}.pdf").read_bytes() == data
    again = _upload(client, tenant, "renamed.pdf", data)
    assert again.status_code == 200
    assert again.json() == {**body, "duplicate": True}
    assert [job["kind"] for job in jobs_of(deps.pool, body["document_id"])] == ["extract"]
    assert _temp_files(blobs) == []


def test_concurrent_identical_uploads_make_one_document(client: TestClient, tenant: TenantInfo) -> None:
    """Four simultaneous uploads of the same bytes: one 202, three duplicates, one document.

    Args:
        client: HTTP client.
        tenant: The tenant.
    """
    data = case_pdf(LB5001)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda n: _upload(client, tenant, f"copy{n}.pdf", data), range(4)))
    assert sorted(r.status_code for r in responses) == [200, 200, 200, 202]
    assert len({r.json()["document_id"] for r in responses}) == 1
    listing = client.get("/v1/documents", headers=_auth(tenant)).json()
    assert listing["total"] == 1


def test_same_bytes_in_another_tenant_is_a_new_document(client: TestClient, tenant: TenantInfo) -> None:
    """Deduplication is per tenant: another tenant's upload of the same bytes is its own document.

    Args:
        client: HTTP client.
        tenant: The first tenant.
    """
    other = make_tenant(client, "other")
    data = case_pdf(LB5001)
    first = _upload(client, tenant, "a.pdf", data).json()
    second = _upload(client, other, "a.pdf", data)
    assert second.status_code == 202
    assert second.json()["document_id"] != first["document_id"]


@pytest.mark.parametrize(
    ("data", "code"),
    [(b"", "empty_file"), (b"PK\x03\x04 a zip, not a PDF" * 100, "not_a_pdf"), (b"x" * 2000 + b"%PDF-", "not_a_pdf")],
    ids=["empty", "zip", "magic-too-late"],
)
def test_rejected_files(client: TestClient, tenant: TenantInfo, blobs: LocalBlobStore, data: bytes, code: str) -> None:
    """Empty files and files without `%PDF-` in the first 1 KiB are refused with 422 and leave no temp file.

    Args:
        client: HTTP client.
        tenant: The tenant.
        blobs: Blob store.
        data: File bytes.
        code: Expected error code.
    """
    response = _upload(client, tenant, "bad.pdf", data)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]
    assert _temp_files(blobs) == []


@pytest.mark.parametrize(
    ("body", "content_type"),
    [
        (b'{"file": "x"}', "application/json"),
        (_multipart([("document", "m.pdf", b"%PDF-1.4")]), f"multipart/form-data; boundary={BOUNDARY}"),
        (_multipart([("file", None, b"%PDF-1.4")]), f"multipart/form-data; boundary={BOUNDARY}"),
        (_multipart([("file", "..", b"%PDF-1.4")]), f"multipart/form-data; boundary={BOUNDARY}"),
        (
            _multipart([("file", "a.pdf", b"%PDF-1.4"), ("file", "b.pdf", b"%PDF-1.4")]),
            f"multipart/form-data; boundary={BOUNDARY}",
        ),
        (_multipart([("file", "a.pdf", b"%PDF-1.4")], close=False), f"multipart/form-data; boundary={BOUNDARY}"),
        (b"", "multipart/form-data"),
    ],
    ids=["json", "wrong-name", "no-filename", "dot-filename", "two-parts", "unterminated", "no-boundary"],
)
def test_malformed_uploads(
    client: TestClient, tenant: TenantInfo, blobs: LocalBlobStore, body: bytes, content_type: str
) -> None:
    """Anything but one part named `file` with a filename is a 422 `invalid_upload`.

    Args:
        client: HTTP client.
        tenant: The tenant.
        blobs: Blob store.
        body: Raw body.
        content_type: Content-Type header.
    """
    response = client.post("/v1/documents", content=body, headers={**_auth(tenant), "Content-Type": content_type})
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "invalid_upload"
    assert _temp_files(blobs) == []


@pytest.mark.parametrize(
    ("sent", "kept"),
    [
        ("../../etc/pass\x07wd.pdf", "passwd.pdf"),
        ("C:\\docs\\manual.pdf", "manual.pdf"),
        ("a" * 300 + ".pdf", "a" * 255),
    ],
)
def test_filename_is_reduced_to_a_safe_basename(client: TestClient, tenant: TenantInfo, sent: str, kept: str) -> None:
    """Paths and control characters are stripped from the filename, which is cut to 255 characters.

    Args:
        client: HTTP client.
        tenant: The tenant.
        sent: Filename as sent.
        kept: Filename stored.
    """
    body = _multipart([("file", sent, case_pdf(LB5001))])
    response = client.post(
        "/v1/documents",
        content=body,
        headers={**_auth(tenant), "Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    assert response.status_code == 202, response.text
    assert response.json()["filename"] == kept


@pytest.fixture
def small_app(deps: Deps) -> Iterator[tuple[FastAPI, TestClient]]:
    """Build an application with a 1 MiB upload limit.

    Args:
        deps: API dependencies.

    Yields:
        The application and a client over it.
    """
    settings = replace(deps.settings, max_upload_mb=1)
    app = create_app(settings, replace(deps, settings=settings))
    with TestClient(app) as test_client:
        yield app, test_client


def _asgi_upload(app: FastAPI, key: str, chunks: list[bytes], content_length: int | None) -> tuple[int, Any, int]:
    """Send an upload straight to the ASGI application, counting how many body chunks it reads.

    Args:
        app: The application.
        key: Tenant key.
        chunks: Body chunks.
        content_length: Declared size, or None to send without `Content-Length`.

    Returns:
        `(status, body, chunks read)`.
    """
    read = 0
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        """Hand the next body chunk to the application.

        Returns:
            An ASGI request message.
        """
        nonlocal read
        if read < len(chunks):
            read += 1
            return {"type": "http.request", "body": chunks[read - 1], "more_body": read < len(chunks)}
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        """Collect the response.

        Args:
            message: An ASGI response message.
        """
        messages.append(message)

    headers = [
        (b"authorization", f"Bearer {key}".encode()),
        (b"content-type", f"multipart/form-data; boundary={BOUNDARY}".encode()),
    ]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/documents",
        "raw_path": b"/v1/documents",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("test", 1),
        "server": ("test", 80),
    }

    async def call() -> None:
        """Run the request."""
        await app(scope, receive, send)

    anyio.run(call)
    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return start["status"], json.loads(body), read


def _big_body_chunks(size: int) -> list[bytes]:
    """Split a multipart body holding a PDF-looking file of a given size into 256 KiB chunks.

    Args:
        size: File size.

    Returns:
        The chunks; the first one holds the part headers.
    """
    body = _multipart([("file", "big.pdf", b"%PDF-1.7\n" + b"0" * (size - 9))])
    return [body[i : i + 256 * 1024] for i in range(0, len(body), 256 * 1024)]


def test_oversized_upload_is_refused_before_the_body_is_read(
    small_app: tuple[FastAPI, TestClient], blobs: LocalBlobStore
) -> None:
    """With `Content-Length` over the limit, the 413 comes after the part headers, without reading the rest.

    Args:
        small_app: Application with a 1 MiB limit.
        blobs: Blob store.
    """
    app, client = small_app
    tenant = make_tenant(client)
    chunks = _big_body_chunks(3 * MIB)
    status, body, read = _asgi_upload(app, tenant.key, chunks, sum(map(len, chunks)))
    assert status == 413
    assert body["error"]["code"] == "file_too_large"
    assert body["error"]["details"] == {"max_bytes": MIB}
    assert read == 1 < len(chunks)
    assert _temp_files(blobs) == []


def test_oversized_upload_without_length_is_cut_while_streaming(
    small_app: tuple[FastAPI, TestClient], blobs: LocalBlobStore
) -> None:
    """Without `Content-Length`, the upload stops as soon as the file passes the limit.

    Args:
        small_app: Application with a 1 MiB limit.
        blobs: Blob store.
    """
    app, client = small_app
    tenant = make_tenant(client)
    chunks = _big_body_chunks(3 * MIB)
    status, body, read = _asgi_upload(app, tenant.key, chunks, None)
    assert (status, body["error"]["code"]) == (413, "file_too_large")
    assert read < len(chunks)
    assert _temp_files(blobs) == []
    within = _big_body_chunks(MIB - 1024)
    status, body, read = _asgi_upload(app, tenant.key, within, None)
    assert status == 202, body
    assert read == len(within)


def test_upload_needs_a_tenant_key(client: TestClient) -> None:
    """Without a key the upload is a 401 before anything is read.

    Args:
        client: HTTP client.
    """
    response = client.post("/v1/documents", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")})
    assert response.status_code == 401


def test_batch_tracks_every_outcome(
    client: TestClient, tenant: TenantInfo, worker_deps: WorkerDeps, blobs: LocalBlobStore
) -> None:
    """A batch lists accepted, duplicate and rejected files, counts them, and completes when documents are final.

    Args:
        client: HTTP client.
        tenant: The tenant.
        worker_deps: Worker dependencies, on the same database and blob store.
        blobs: Blob store.
    """
    created = client.post("/v1/batches", headers=_auth(tenant))
    assert created.status_code == 201
    batch_id = created.json()["batch_id"]
    assert client.get(f"/v1/batches/{batch_id}", headers=_auth(tenant)).json()["complete"] is True
    assert _upload(client, tenant, "LB5001.pdf", case_pdf(LB5001), batch_id).status_code == 202
    assert _upload(client, tenant, "copy.pdf", case_pdf(LB5001), batch_id).status_code == 200
    assert _upload(client, tenant, "notes.txt", b"plain text", batch_id).status_code == 422
    assert _upload(client, tenant, "empty.pdf", b"", batch_id).status_code == 422
    view = client.get(f"/v1/batches/{batch_id}", headers=_auth(tenant)).json()
    assert [(i["filename"], i["outcome"], i["error_code"]) for i in view["items"]] == [
        ("LB5001.pdf", "accepted", None),
        ("copy.pdf", "duplicate", None),
        ("notes.txt", "rejected", "not_a_pdf"),
        ("empty.pdf", "rejected", "empty_file"),
    ]
    assert view["counts"] == {
        "items": 4,
        "accepted": 1,
        "duplicate": 1,
        "rejected": 2,
        "documents": {"queued": 1, "processing": 0, "ready": 0, "partial": 0, "failed": 0},
    }
    assert view["complete"] is False
    assert view["items"][2]["document"] is None
    drain(worker_deps)
    view = client.get(f"/v1/batches/{batch_id}", headers=_auth(tenant)).json()
    assert view["complete"] is True
    assert view["counts"]["documents"]["ready"] == 1
    assert view["items"][0]["document"]["pages_indexed"] == 2


def test_unknown_or_foreign_batch_is_not_found(client: TestClient, tenant: TenantInfo) -> None:
    """Another tenant's batch and an unknown id both answer 404, for reading and for uploading.

    Args:
        client: HTTP client.
        tenant: The tenant.
    """
    other = make_tenant(client, "other")
    foreign = client.post("/v1/batches", headers=_auth(other)).json()["batch_id"]
    unknown = "00000000-0000-4000-8000-000000000000"
    for batch_id in (foreign, unknown):
        assert client.get(f"/v1/batches/{batch_id}", headers=_auth(tenant)).json()["error"]["code"] == "batch_not_found"
        response = _upload(client, tenant, "a.pdf", case_pdf(LB5001), batch_id)
        assert (response.status_code, response.json()["error"]["code"]) == (404, "batch_not_found")


def test_document_views_and_isolation(client: TestClient, tenant: TenantInfo, worker_deps: WorkerDeps) -> None:
    """The list and the document view show states, sources and labels; another tenant sees nothing of them.

    Args:
        client: HTTP client.
        tenant: The tenant.
        worker_deps: Worker dependencies.
    """
    document_id = _upload(client, tenant, "MN414.pdf", case_pdf(MN414)).json()["document_id"]
    drain(worker_deps)
    listing = client.get("/v1/documents", headers=_auth(tenant)).json()
    assert listing["total"] == 1
    assert listing["documents"][0]["status"] == "ready"
    detail = client.get(f"/v1/documents/{document_id}", headers=_auth(tenant)).json()
    assert (detail["page_count"], detail["pages_indexed"], detail["error"]) == (16, 15, None)
    assert detail["pages_by_state"] == {"needs_ocr": 0, "text_ready": 0, "indexed": 15, "blank": 1, "failed": 0}
    assert detail["pages_by_source"] == {"layer": 15, "ocr": 0}
    assert len(detail["pages"]) == 16
    assert {p["flag_reason"] for p in detail["pages"]} == {None}
    other = make_tenant(client, "other")
    assert client.get("/v1/documents", headers=_auth(other)).json() == {"documents": [], "total": 0}
    response = client.get(f"/v1/documents/{document_id}", headers=_auth(other))
    assert (response.status_code, response.json()["error"]["code"]) == (404, "document_not_found")


def test_list_paging(client: TestClient, tenant: TenantInfo) -> None:
    """The list is newest first and pages with `limit` and `offset`; out-of-range values are a 422.

    Args:
        client: HTTP client.
        tenant: The tenant.
    """
    names = [f"doc{n}.pdf" for n in range(3)]
    for n, name in enumerate(names):
        _upload(client, tenant, name, b"%PDF-1.4\n" + bytes([n]) * 10)
    page = client.get("/v1/documents", params={"limit": 2, "offset": 1}, headers=_auth(tenant)).json()
    assert page["total"] == 3
    assert [d["filename"] for d in page["documents"]] == ["doc1.pdf", "doc0.pdf"]
    assert client.get("/v1/documents", params={"limit": 0}, headers=_auth(tenant)).status_code == 422


def test_batch_of_good_and_bad_files_ends_with_the_right_outcomes(
    client: TestClient, tenant: TenantInfo, worker_deps: WorkerDeps
) -> None:
    """One batch mixing manuals and broken files: every item gets its outcome and the batch completes.

    Args:
        client: HTTP client.
        tenant: The tenant.
        worker_deps: Worker dependencies.
    """
    files = {
        "LB5001.pdf": case_pdf(LB5001),
        "MN414.pdf": case_pdf(MN414),
        "locked.pdf": encrypted_pdf(),
        "owner-only.pdf": encrypted_pdf(user_password=""),
        "cut-1k.pdf": truncated_pdf(1024),
        "cut-half.pdf": truncated_pdf(0.5),
        "garbage.pdf": garbage_pdf(),
        "notes.txt": b"not a pdf at all",
        "empty.pdf": b"",
    }
    batch_id = client.post("/v1/batches", headers=_auth(tenant)).json()["batch_id"]
    for name, data in files.items():
        _upload(client, tenant, name, data, batch_id)
    drain(worker_deps)
    view = client.get(f"/v1/batches/{batch_id}", headers=_auth(tenant)).json()
    outcome = {
        item["filename"]: (item["outcome"], item["error_code"])
        if item["document"] is None
        else (item["document"]["status"], item["document"]["error_code"])
        for item in view["items"]
    }
    assert outcome == {
        "LB5001.pdf": ("ready", None),
        "MN414.pdf": ("ready", None),
        "locked.pdf": ("failed", "pdf_encrypted"),
        "owner-only.pdf": ("ready", None),
        "cut-1k.pdf": ("failed", "pdf_corrupt"),
        "cut-half.pdf": ("failed", "pdf_corrupt"),
        "garbage.pdf": ("failed", "pdf_corrupt"),
        "notes.txt": ("rejected", "not_a_pdf"),
        "empty.pdf": ("rejected", "empty_file"),
    }
    assert view["complete"] is True
