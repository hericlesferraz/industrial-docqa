"""The FastAPI application: request middleware, error handlers, `/healthz` and the routers."""

import logging
import sys
import time
import uuid

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from docqa.api import routes_admin, routes_documents, routes_questions
from docqa.api.deps import Deps, DepsDep, Health
from docqa.api.models import HealthView
from docqa.config import ConfigError, Settings, load_settings
from docqa.db import create_pool, migrate, ping
from docqa.errors import (
    INTERNAL_ERROR,
    SERVICE_UNAVAILABLE,
    ApiError,
    ErrorBody,
    error_body,
    error_response,
    install_error_handlers,
)
from docqa.index.embed import GEMINI_EMBED_MODEL, get_embedder
from docqa.providers.gemini import GeminiGateway
from docqa.providers.llm import LLM, ProviderError
from docqa.storage import LocalBlobStore
from docqa.telemetry import PostgresTelemetry, Telemetry, bind, configure_logging

API_POOL_SIZE = 16
HEALTH_PATH = "/healthz"

logger = logging.getLogger("docqa.api")


def _route_template(scope: Scope) -> str | None:
    """Return the path template of the route that handled the request, never the raw path.

    Args:
        scope: ASGI scope after routing.

    Returns:
        The template, such as `/v1/documents/{document_id}`, or None when no route matched.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return str(path) if path else None


class RequestContextMiddleware:
    """Gives every HTTP request an id, a root span, one log line and the error body for unhandled errors.

    The id is generated here (an incoming `X-Request-ID` is ignored), returned in the `X-Request-ID` header of
    every response and bound to the telemetry context. `/healthz` answers with 200 are not recorded.
    """

    def __init__(self, app: ASGIApp, telemetry: Telemetry) -> None:
        """Wrap an ASGI application.

        Args:
            app: The wrapped application.
            telemetry: Where the request spans go.
        """
        self.app = app
        self.telemetry = telemetry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI call.

        Args:
            scope: ASGI scope.
            receive: ASGI receive channel.
            send: ASGI send channel.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4()
        state = scope.setdefault("state", {})
        state["request_id"] = str(request_id)
        response: dict[str, int | bool] = {"status": 500, "started": False}

        async def send_with_id(message: Message) -> None:
            """Add `X-Request-ID` to the response start and note the status.

            Args:
                message: ASGI message.
            """
            if message["type"] == "http.response.start":
                response["status"] = int(message["status"])
                response["started"] = True
                MutableHeaders(scope=message)["X-Request-ID"] = str(request_id)
            await send(message)

        started = time.perf_counter()
        recorded = scope.get("path") != HEALTH_PATH
        with bind(request_id=request_id):
            if recorded:
                with self.telemetry.span("http.request", **{"http.request.method": scope["method"]}) as span:
                    await self._call_app(scope, receive, send_with_id, response)
                    route = _route_template(scope)
                    if route:
                        span.set("http.route", route)
                    span.set("http.response.status_code", int(response["status"]))
                    code = state.get("error_code")
                    if code:
                        span.set("docqa.error_code", str(code))
                    if int(response["status"]) >= 500:
                        span.error(str(code or INTERNAL_ERROR))
            else:
                await self._call_app(scope, receive, send_with_id, response)
            if recorded or response["status"] != 200:
                fields = {
                    "method": scope["method"],
                    "route": _route_template(scope),
                    "status": response["status"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "error_code": state.get("error_code"),
                }
                logger.info("http request", extra={"fields": fields})
            if recorded:
                await anyio.to_thread.run_sync(self._flush)

    async def _call_app(self, scope: Scope, receive: Receive, send: Send, response: dict[str, int | bool]) -> None:
        """Call the application and turn an unhandled exception into a 500 with the error body.

        Args:
            scope: ASGI scope.
            receive: ASGI receive channel.
            send: ASGI send channel that adds the request id.
            response: Status holder updated by `send`.
        """
        try:
            await self.app(scope, receive, send)
        except Exception:
            logger.exception("unhandled error")
            scope["state"]["error_code"] = INTERNAL_ERROR
            if response["started"]:
                return
            body = error_body(INTERNAL_ERROR, "Internal error.", scope["state"]["request_id"], retryable=True)
            await JSONResponse(body, status_code=500)(scope, receive, send)

    def _flush(self) -> None:
        """Write the request's spans; a failure is logged and never reaches the client."""
        try:
            self.telemetry.flush()
        except Exception:
            logger.exception("telemetry flush failed")


def create_app(settings: Settings, deps: Deps) -> FastAPI:
    """Build the application around given dependencies; tests pass fakes.

    Args:
        settings: Process settings.
        deps: Process objects.

    Returns:
        The application.
    """
    app = FastAPI(
        title="docqa",
        version="0.1.0",
        description="Ask questions about industrial PDF manuals and get answers with verifiable citations.",
    )
    app.state.settings = settings
    app.state.deps = deps
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, telemetry=deps.telemetry)
    app.include_router(routes_admin.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_questions.router)

    @app.get(
        HEALTH_PATH,
        response_model=HealthView,
        responses={503: {"model": ErrorBody}},
        tags=["health"],
        summary="Service health",
    )
    def healthz(request: Request, deps: DepsDep) -> HealthView | JSONResponse:
        """Report the database status, the embedder and the start-up checks.

        Args:
            request: The current request.
            deps: Process objects.

        Returns:
            200 with the checks, or 503 `service_unavailable` with the same body in `details` when the database is
            unreachable or the embedder is not loaded (search needs both).
        """
        database = "ok" if ping(deps.pool) else "unreachable"
        embedder = "ok" if deps.embedder is not None else "not_loaded"
        checks: dict[str, str | dict[str, str]] = {"database": database, "embedder": embedder}
        if deps.health.models:
            checks["models"] = dict(deps.health.models)
        reasons = list(deps.health.reasons)
        view = HealthView(status="degraded" if reasons else "ok", checks=checks, reasons=reasons)
        failed = [
            f"{name}: {state}" for name, state in (("database", database), ("embedder", embedder)) if state != "ok"
        ]
        if not failed:
            return view
        details = view.model_dump()
        details["reasons"] = [*reasons, *failed]
        message = "The database is unreachable." if database != "ok" else "The embedder is not loaded."
        error = ApiError(503, SERVICE_UNAVAILABLE, message, retryable=True, details=details)
        return error_response(request, error)

    return app


def describe_provider_error(error: ProviderError) -> str:
    """Say in a few words why a model check failed.

    Args:
        error: The classified error.

    Returns:
        A short reason for `/healthz`, without the provider's message.
    """
    if error.kind == "misconfigured":
        return "provider rejected the API key" if error.account_wide else "provider rejected the model"
    reasons = {
        "billing": "billing: prepaid balance or spend cap exhausted",
        "quota_exhausted": "daily quota exhausted",
        "rate_limited": "rate limited",
        "timeout": "provider did not answer in time",
        "transient": "provider unreachable",
    }
    return reasons.get(error.kind, error.kind)


def check_models(llm: LLM, settings: Settings, telemetry: Telemetry) -> Health:
    """Check every configured model once with a metadata call (no generation, no cost).

    The embedding model is checked too when `EMBEDDING_PROVIDER=gemini`.

    A failure only marks the service `degraded`: search keeps working without the provider (DECISIONS #15).

    Args:
        llm: Provider gateway.
        settings: Process settings, for the answer and OCR models.
        telemetry: Where the check spans go; flushed here.

    Returns:
        Status per model and the reasons for a degraded status.
    """
    health = Health()
    configured = [settings.gemini_answer_model, settings.gemini_ocr_model, settings.gemini_ocr_fallback_model]
    if settings.embedding_provider == "gemini":
        configured.append(GEMINI_EMBED_MODEL)
    models = dict.fromkeys(configured)
    with bind():
        for model in models:
            try:
                llm.check_model(model)
            except ProviderError as error:
                health.models[model] = error.kind
                health.reasons.append(f"{model}: {describe_provider_error(error)}")
                logger.warning("model check failed", extra={"fields": {"model": model, "kind": error.kind}})
            else:
                health.models[model] = "ok"
        try:
            telemetry.flush()
        except Exception:
            logger.exception("telemetry flush failed")
    return health


def create_app_from_env() -> FastAPI:
    """Build the application from the environment: fail fast on configuration, migrate, load the embedder, check models.

    The embedder is loaded (and run once) before the application exists, so `/healthz` turns healthy only after it.

    Returns:
        The application.
    """
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        sys.exit(2)
    configure_logging(settings.log_level)
    pool = create_pool(settings.database_url, max_size=API_POOL_SIZE)
    with pool.connection() as conn:
        applied = migrate(conn)
    if applied:
        logger.info("migrations applied", extra={"fields": {"versions": applied}})
    telemetry = PostgresTelemetry(pool)
    llm = GeminiGateway(settings.gemini_api_key, telemetry)
    deps = Deps(
        pool=pool,
        settings=settings,
        telemetry=telemetry,
        blobs=LocalBlobStore(settings.blob_store_path),
        llm=llm,
        embedder=get_embedder(settings, llm),
        health=check_models(llm, settings, telemetry),
    )
    return create_app(settings, deps)
