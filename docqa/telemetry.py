"""Own telemetry: spans buffered per request or job and written to Postgres, and JSON logs on stdout.

Span names and attributes follow the OpenTelemetry conventions (GenAI ones for model calls), so an
OpenTelemetry exporter can implement the same `Telemetry` interface without changing call sites.
"""

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from docqa.db import tx

AttrValue = str | int | float | bool | list[str]
IdValue = UUID | int | None

ID_NAMES = ("request_id", "tenant_id", "job_id", "document_id", "page_no")

logger = logging.getLogger("docqa.telemetry")


@dataclass
class SpanRecord:
    """A finished span, ready to be written.

    Attributes:
        span_id: Id of the span.
        name: Span name; events are named `event.<name>`.
        started_at: Start time, UTC.
        duration_ms: Duration in milliseconds; 0 for events.
        status: `ok` or `error`.
        error_code: Error code when the status is `error`.
        ids: Correlation ids bound when the span ended.
        attributes: Span attributes; never customer text or keys.
    """

    span_id: UUID
    name: str
    started_at: datetime
    duration_ms: float
    status: str
    error_code: str | None
    ids: dict[str, UUID | int]
    attributes: dict[str, AttrValue]


@dataclass
class _Scope:
    """Correlation ids of a request or a job, and the spans it has buffered.

    Attributes:
        ids: Bound correlation ids.
        spans: Finished spans waiting for `flush`; shared by nested scopes.
    """

    ids: dict[str, UUID | int]
    spans: list[SpanRecord] = field(default_factory=list)


_scope: ContextVar[_Scope | None] = ContextVar("docqa_scope", default=None)
_unscoped = _Scope(ids={})


def _active_scope() -> _Scope:
    """Return the scope of the current context, or the process-wide one outside any `bind`.

    Returns:
        The active scope.
    """
    return _scope.get() or _unscoped


def _clean_ids(ids: dict[str, IdValue]) -> dict[str, IdValue]:
    """Check that only known correlation ids are given.

    Args:
        ids: Ids to check.

    Returns:
        The same ids.

    Raises:
        ValueError: An id name is not one of `ID_NAMES`.
    """
    unknown = set(ids) - set(ID_NAMES)
    if unknown:
        raise ValueError(f"unknown correlation ids: {sorted(unknown)}")
    return ids


@contextmanager
def _bind(ids: dict[str, IdValue]) -> Iterator[None]:
    """Implement `bind`.

    Args:
        ids: Ids to bind; None removes an inherited id.

    Yields:
        Nothing; the previous ids come back on exit.
    """
    parent = _scope.get()
    merged: dict[str, UUID | int] = dict(parent.ids) if parent else {}
    for name, value in _clean_ids(ids).items():
        if value is None:
            merged.pop(name, None)
        else:
            merged[name] = value
    scope = _Scope(ids=merged, spans=parent.spans if parent else [])
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


def bind(**ids: IdValue) -> AbstractContextManager[None]:
    """Bind correlation ids to the current context for spans and logs.

    A top-level `bind` (a request or a job) starts a new span buffer; a nested one shares its parent's.

    Args:
        **ids: Any of `request_id`, `tenant_id`, `job_id`, `document_id`, `page_no`.

    Returns:
        A context manager that restores the previous ids on exit.
    """
    return _bind(ids)


def add_ids(**ids: IdValue) -> None:
    """Add ids learned in the middle of a request, such as the tenant after authentication.

    The ids are added to the current scope in place, so the enclosing request span and its log line see them
    even when the code runs in a worker thread with a copy of the context.

    Args:
        **ids: Any of `request_id`, `tenant_id`, `job_id`, `document_id`, `page_no`.
    """
    scope = _scope.get()
    if scope is None:
        return
    for name, value in _clean_ids(ids).items():
        if value is not None:
            scope.ids[name] = value


def current_ids() -> dict[str, UUID | int]:
    """Return the correlation ids bound to the current context.

    Returns:
        A copy of the bound ids.
    """
    return dict(_active_scope().ids)


class Span(Protocol):
    """An open span."""

    def set(self, key: str, value: AttrValue) -> None:
        """Set an attribute.

        Args:
            key: Attribute name.
            value: Attribute value; never customer text or a key.
        """

    def error(self, code: str) -> None:
        """Mark the span as failed.

        Args:
            code: Error code.
        """


class Telemetry(Protocol):
    """Records spans and events; replaceable by an OpenTelemetry exporter."""

    def span(self, name: str, **attributes: AttrValue) -> AbstractContextManager[Span]:
        """Open a span.

        Args:
            name: Span name.
            **attributes: Initial attributes.

        Returns:
            A context manager yielding the span; it records duration and status on exit.
        """
        ...

    def event(self, name: str, **attributes: AttrValue) -> None:
        """Record an event as a zero-duration span named `event.<name>`.

        Args:
            name: Event name.
            **attributes: Event attributes.
        """

    def flush(self, conn: Connection | None = None) -> None:
        """Write the buffered spans of the current context.

        Args:
            conn: Transaction to write in; None opens a short transaction of its own.
        """


class _OpenSpan:
    """Mutable state of a span while it is open.

    Attributes:
        span_id: Id of the span.
        attributes: Attributes set so far.
        error_code: Error code once marked as failed.
    """

    def __init__(self, attributes: dict[str, AttrValue]) -> None:
        """Start the span state.

        Args:
            attributes: Initial attributes.
        """
        self.span_id = uuid.uuid4()
        self.attributes = dict(attributes)
        self.error_code: str | None = None

    def set(self, key: str, value: AttrValue) -> None:
        """Set an attribute.

        Args:
            key: Attribute name.
            value: Attribute value; never customer text or a key.
        """
        self.attributes[key] = value

    def error(self, code: str) -> None:
        """Mark the span as failed.

        Args:
            code: Error code.
        """
        self.error_code = code


class BufferedTelemetry:
    """Span and event recording shared by the Postgres and in-memory implementations."""

    @contextmanager
    def _span(self, name: str, attributes: dict[str, AttrValue]) -> Iterator[Span]:
        """Implement `span`.

        Args:
            name: Span name.
            attributes: Initial attributes.

        Yields:
            The open span.

        Raises:
            BaseException: Whatever the body raises, after marking the span as failed.
        """
        scope = _active_scope()
        span = _OpenSpan(attributes)
        started_at = datetime.now(UTC)
        start = time.perf_counter()
        try:
            yield span
        except BaseException as exc:
            if span.error_code is None:
                span.error(str(getattr(exc, "code", "") or "internal_error"))
            raise
        finally:
            scope.spans.append(
                SpanRecord(
                    span_id=span.span_id,
                    name=name,
                    started_at=started_at,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    status="error" if span.error_code else "ok",
                    error_code=span.error_code,
                    ids=current_ids(),
                    attributes=span.attributes,
                )
            )

    def span(self, name: str, **attributes: AttrValue) -> AbstractContextManager[Span]:
        """Open a span.

        Args:
            name: Span name.
            **attributes: Initial attributes.

        Returns:
            A context manager yielding the span; it records duration and status on exit.
        """
        return self._span(name, attributes)

    def event(self, name: str, **attributes: AttrValue) -> None:
        """Record an event as a zero-duration span named `event.<name>`.

        Args:
            name: Event name.
            **attributes: Event attributes.
        """
        code = attributes.get("error_code")
        _active_scope().spans.append(
            SpanRecord(
                span_id=uuid.uuid4(),
                name=f"event.{name}",
                started_at=datetime.now(UTC),
                duration_ms=0.0,
                status="ok",
                error_code=str(code) if code else None,
                ids=current_ids(),
                attributes=dict(attributes),
            )
        )

    @staticmethod
    def _take() -> list[SpanRecord]:
        """Remove and return the spans buffered by the current context.

        Returns:
            The buffered spans, oldest first.
        """
        spans = _active_scope().spans
        taken = spans[:]
        del spans[: len(taken)]
        return taken


class MemoryTelemetry(BufferedTelemetry):
    """Telemetry that keeps flushed spans in a list, for tests.

    Attributes:
        spans: Every flushed span, in flush order.
    """

    def __init__(self) -> None:
        """Start with no spans."""
        self.spans: list[SpanRecord] = []

    def flush(self, conn: Connection | None = None) -> None:
        """Move the buffered spans of the current context to `spans`.

        Args:
            conn: Ignored.
        """
        self.spans.extend(self._take())

    def named(self, name: str) -> list[SpanRecord]:
        """Return the flushed spans with a given name.

        Args:
            name: Span name.

        Returns:
            The matching spans.
        """
        return [span for span in self.spans if span.name == name]


INSERT_SPAN = (
    "INSERT INTO spans (span_id, name, started_at, duration_ms, status, error_code,"
    " tenant_id, request_id, job_id, document_id, page_no, attributes)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


class PostgresTelemetry(BufferedTelemetry):
    """Telemetry that writes spans to the `spans` table at flush time."""

    def __init__(self, pool: ConnectionPool) -> None:
        """Keep the pool used by flushes without a connection.

        Args:
            pool: Database pool.
        """
        self._pool = pool

    def flush(self, conn: Connection | None = None) -> None:
        """Write the buffered spans of the current context with one `executemany`.

        Args:
            conn: Transaction to write in; None opens a short transaction of its own.
        """
        spans = self._take()
        if not spans:
            return
        rows = [
            (
                span.span_id,
                span.name,
                span.started_at,
                span.duration_ms,
                span.status,
                span.error_code,
                span.ids.get("tenant_id"),
                span.ids.get("request_id"),
                span.ids.get("job_id"),
                span.ids.get("document_id"),
                span.ids.get("page_no"),
                Jsonb(span.attributes),
            )
            for span in spans
        ]
        if conn is not None:
            conn.cursor().executemany(INSERT_SPAN, rows)
            return
        with tx(self._pool) as own:
            own.cursor().executemany(INSERT_SPAN, rows)


class JsonFormatter(logging.Formatter):
    """Formats log records as one JSON object per line, with the bound correlation ids."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record.

        Args:
            record: The log record; structured fields go in `extra={"fields": {...}}`.

        Returns:
            One JSON line.
        """
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({name: str(value) for name, value in current_ids().items()})
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """Send every log record, uvicorn's included, to stdout as JSON lines.

    uvicorn's access log is turned off: the request middleware writes one line per request with the route
    template and the correlation ids instead.

    Args:
        level: Logging level name.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error"):
        named = logging.getLogger(name)
        named.handlers = []
        named.propagate = True
    access = logging.getLogger("uvicorn.access")
    access.handlers = []
    access.propagate = False
    access.disabled = True
    for name in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))
