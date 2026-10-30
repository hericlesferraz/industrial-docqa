"""The single error body of the API, the exception that produces it, and the code catalogue."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

UNAUTHORIZED = "unauthorized"
NOT_FOUND = "not_found"
METHOD_NOT_ALLOWED = "method_not_allowed"
DOCUMENT_NOT_FOUND = "document_not_found"
BATCH_NOT_FOUND = "batch_not_found"
ANSWER_NOT_FOUND = "answer_not_found"
FILE_TOO_LARGE = "file_too_large"
INVALID_REQUEST = "invalid_request"
INVALID_UPLOAD = "invalid_upload"
EMPTY_FILE = "empty_file"
NOT_A_PDF = "not_a_pdf"
INTERNAL_ERROR = "internal_error"
LLM_INVALID_OUTPUT = "llm_invalid_output"
LLM_INVALID_REQUEST = "llm_invalid_request"
LLM_REFUSED = "llm_refused"
LLM_UNAVAILABLE = "llm_unavailable"
LLM_RATE_LIMITED = "llm_rate_limited"
LLM_BILLING = "llm_billing"
LLM_MISCONFIGURED = "llm_misconfigured"
LLM_TIMEOUT = "llm_timeout"
SERVICE_UNAVAILABLE = "service_unavailable"

logger = logging.getLogger("docqa.errors")


class ErrorDetail(BaseModel):
    """Content of an error response.

    Attributes:
        code: Stable error code from the catalogue.
        message: Human-readable explanation; never customer text.
        request_id: Id of the request, also in the `X-Request-ID` header.
        retryable: Whether the same request may succeed later.
        details: Extra structured data, omitted when empty.
    """

    code: str
    message: str
    request_id: str
    retryable: bool
    details: dict[str, object] | None = Field(default=None)


class ErrorBody(BaseModel):
    """Body of every non-2xx response.

    Attributes:
        error: The error content.
    """

    error: ErrorDetail


class ApiError(Exception):
    """An error the API returns with the single error body.

    Attributes:
        status: HTTP status code.
        code: Error code from the catalogue.
        message: Human-readable explanation.
        retryable: Whether the same request may succeed later.
        details: Extra structured data.
        headers: Extra response headers, such as `Retry-After`.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Store the error.

        Args:
            status: HTTP status code.
            code: Error code from the catalogue.
            message: Human-readable explanation; never customer text.
            retryable: Whether the same request may succeed later.
            details: Extra structured data.
            headers: Extra response headers.
        """
        super().__init__(code, message)
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details
        self.headers = headers


def error_body(
    code: str, message: str, request_id: str, *, retryable: bool, details: dict[str, object] | None = None
) -> dict[str, object]:
    """Build the JSON error body.

    Args:
        code: Error code.
        message: Human-readable explanation.
        request_id: Id of the request.
        retryable: Whether the same request may succeed later.
        details: Extra structured data, omitted when empty.

    Returns:
        The body as a plain dict.
    """
    detail = ErrorDetail(code=code, message=message, request_id=request_id, retryable=retryable, details=details)
    return {"error": detail.model_dump(exclude_none=True)}


def request_id_of(request: Request) -> str:
    """Read the request id the middleware stored in the request state.

    Args:
        request: The current request.

    Returns:
        The request id, or an empty string outside the middleware.
    """
    return str(getattr(request.state, "request_id", ""))


def error_response(request: Request, error: ApiError) -> JSONResponse:
    """Render an `ApiError` and note its code for the request span.

    Args:
        request: The current request.
        error: The error to render.

    Returns:
        The JSON response.
    """
    request.state.error_code = error.code
    body = error_body(
        error.code, error.message, request_id_of(request), retryable=error.retryable, details=error.details
    )
    return JSONResponse(body, status_code=error.status, headers=error.headers)


async def _api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle `ApiError`.

    Args:
        request: The current request.
        exc: The raised `ApiError`.

    Returns:
        The JSON error response.
    """
    assert isinstance(exc, ApiError)
    return error_response(request, exc)


async def _validation_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn a request validation error into a 422 `invalid_request`, without echoing the input.

    Args:
        request: The current request.
        exc: The raised `RequestValidationError`.

    Returns:
        The JSON error response.
    """
    assert isinstance(exc, RequestValidationError)
    errors = [{"loc": list(item.get("loc", ())), "msg": str(item.get("msg", ""))} for item in exc.errors()]
    error = ApiError(422, INVALID_REQUEST, "The request is invalid.", details={"errors": errors})
    return error_response(request, error)


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn Starlette's HTTP exceptions (unknown route, wrong method) into the error body.

    Args:
        request: The current request.
        exc: The raised `HTTPException`.

    Returns:
        The JSON error response.
    """
    assert isinstance(exc, StarletteHTTPException)
    if exc.status_code == 404:
        error = ApiError(404, NOT_FOUND, "Not found.")
    elif exc.status_code == 405:
        error = ApiError(405, METHOD_NOT_ALLOWED, "Method not allowed.", headers=exc.headers)
    elif exc.status_code >= 500:
        error = ApiError(exc.status_code, INTERNAL_ERROR, "Internal error.", retryable=True)
    else:
        error = ApiError(exc.status_code, INVALID_REQUEST, "The request is invalid.")
    return error_response(request, error)


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers that render every handled error with the single body.

    Unhandled exceptions are rendered by the request middleware, which also adds `X-Request-ID` to them.

    Args:
        app: The application.
    """
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
