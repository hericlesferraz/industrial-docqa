"""The eval's HTTP client: the only way the harness reaches the system, through the same API clients use."""

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

REQUEST_TIMEOUT_S = 30.0
UPLOAD_TIMEOUT_S = 600.0
QUESTION_TIMEOUT_S = 150.0


class HarnessError(Exception):
    """The harness can't go on: the API is unreachable or refused a call the run depends on."""


@dataclass(frozen=True)
class Reply:
    """An HTTP reply the harness keeps, success or error.

    Attributes:
        status: HTTP status.
        body: Parsed JSON body, or `{}` when the body is not JSON.
        wall_ms: Round-trip time measured by the client.
    """

    status: int
    body: dict[str, object]
    wall_ms: float

    @property
    def error_code(self) -> str | None:
        """Return the error code of a non-2xx reply.

        Returns:
            `error.code` of the body, or None.
        """
        error = self.body.get("error")
        return str(error.get("code")) if isinstance(error, dict) and error.get("code") else None


def _json(response: httpx.Response) -> dict[str, object]:
    """Parse a response body as a JSON object.

    Args:
        response: The response.

    Returns:
        The object, or `{}` when the body is not a JSON object.
    """
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


class EvalClient:
    """Calls the API as an admin (to create the run's tenant) and then as that tenant."""

    def __init__(self, base_url: str, admin_key: str, *, transport: httpx.BaseTransport | None = None) -> None:
        """Open the HTTP client.

        Args:
            base_url: API base URL, such as `http://api:8000`.
            admin_key: `ADMIN_API_KEY`; never printed.
            transport: Replacement transport, for tests.
        """
        self._admin_key = admin_key
        self._tenant_key: str | None = None
        self._http = httpx.Client(base_url=base_url, transport=transport, timeout=REQUEST_TIMEOUT_S)

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def _headers(self) -> dict[str, str]:
        """Build the tenant's auth header.

        Returns:
            The header.

        Raises:
            HarnessError: No tenant was created yet.
        """
        if self._tenant_key is None:
            raise HarnessError("no tenant: call create_tenant first")
        return {"Authorization": f"Bearer {self._tenant_key}"}

    def _send(self, method: str, path: str, *, timeout: float = REQUEST_TIMEOUT_S, **kwargs: object) -> Reply:
        """Send a request and keep the reply, whatever its status.

        Args:
            method: HTTP method.
            path: Path under the base URL.
            timeout: Request timeout in seconds.
            **kwargs: Passed to `httpx.Client.request`.

        Returns:
            The reply.

        Raises:
            HarnessError: The API could not be reached.
        """
        started = time.perf_counter()
        try:
            response = self._http.request(method, path, timeout=timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise HarnessError(f"{method} {path}: {type(exc).__name__}: {exc}") from None
        return Reply(response.status_code, _json(response), (time.perf_counter() - started) * 1000)

    def health(self) -> Reply:
        """Read `GET /healthz`.

        Returns:
            The reply.
        """
        return self._send("GET", "/healthz")

    def create_tenant(self, name: str) -> str:
        """Create the run's tenant through the admin endpoint and use its key from now on.

        Args:
            name: Tenant name.

        Returns:
            The tenant id.

        Raises:
            HarnessError: The admin endpoint refused.
        """
        reply = self._send(
            "POST", "/v1/admin/tenants", json={"name": name}, headers={"Authorization": f"Bearer {self._admin_key}"}
        )
        if reply.status != 201:
            raise HarnessError(f"creating the tenant failed: HTTP {reply.status} {reply.error_code}")
        self._tenant_key = str(reply.body["api_key"])
        return str(reply.body["tenant_id"])

    def create_batch(self) -> str:
        """Create a batch.

        Returns:
            The batch id.

        Raises:
            HarnessError: The API refused.
        """
        reply = self._send("POST", "/v1/batches", headers=self._headers())
        if reply.status != 201:
            raise HarnessError(f"creating the batch failed: HTTP {reply.status} {reply.error_code}")
        return str(reply.body["batch_id"])

    def upload(self, path: Path, batch_id: str) -> Reply:
        """Upload one file of the folder, whatever its type: rejecting non-PDFs is the system's job.

        Args:
            path: The file.
            batch_id: Batch to add it to.

        Returns:
            The reply: 202 new, 200 duplicate, or an error body with the rejection code.
        """
        with path.open("rb") as handle:
            return self._send(
                "POST",
                "/v1/documents",
                timeout=UPLOAD_TIMEOUT_S,
                params={"batch_id": batch_id},
                files={"file": (path.name, handle, "application/pdf")},
                headers=self._headers(),
            )

    def batch(self, batch_id: str) -> dict[str, object]:
        """Read `GET /v1/batches/{id}`.

        Args:
            batch_id: The batch.

        Returns:
            The batch view.

        Raises:
            HarnessError: The API refused.
        """
        reply = self._send("GET", f"/v1/batches/{batch_id}", headers=self._headers())
        if reply.status != 200:
            raise HarnessError(f"reading the batch failed: HTTP {reply.status} {reply.error_code}")
        return reply.body

    def document(self, document_id: str) -> dict[str, object] | None:
        """Read `GET /v1/documents/{id}`.

        Args:
            document_id: The document.

        Returns:
            The document detail, or None when the API answered with an error.
        """
        reply = self._send("GET", f"/v1/documents/{document_id}", headers=self._headers())
        return reply.body if reply.status == 200 else None

    def ask(self, question: str) -> Reply:
        """Send `POST /v1/questions` over the whole tenant.

        Args:
            question: The question.

        Returns:
            The reply: 200 with the answer, or the error body (whose `request_id` is the answer id).
        """
        return self._send(
            "POST", "/v1/questions", timeout=QUESTION_TIMEOUT_S, json={"question": question}, headers=self._headers()
        )

    def answer_record(self, answer_id: str) -> dict[str, object] | None:
        """Read `GET /v1/answers/{id}`.

        Args:
            answer_id: The answer id.

        Returns:
            The record, or None when there is none (questions refused before the pipeline ran).
        """
        reply = self._send("GET", f"/v1/answers/{answer_id}", headers=self._headers())
        return reply.body if reply.status == 200 else None
