"""The contract between the worker and the job handlers: the context a handler receives and the signals it raises.

It lives apart from `docqa.worker` so that handler modules can import it while the worker imports the handlers.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from psycopg import Connection
from psycopg_pool import ConnectionPool

from docqa.config import Settings
from docqa.index.embed import Embedder
from docqa.providers.llm import LLM
from docqa.queue import Job
from docqa.storage import BlobStore
from docqa.telemetry import Telemetry


class JobSignal(Exception):
    """Base of the exceptions a handler raises to steer the queue.

    Attributes:
        code: Stored in `jobs.last_error_code`.
        message: Stored in `jobs.last_error`.
    """

    def __init__(self, code: str, message: str = "") -> None:
        """Store the code and the message.

        Args:
            code: Error code.
            message: Detail without customer text or keys.
        """
        super().__init__(code, message)
        self.code = code
        self.message = message


class Retry(JobSignal):
    """Transient failure: the attempt is spent and the job comes back after `BACKOFF_S`."""


class PermanentFailure(JobSignal):
    """The work can't succeed: the worker calls `queue.fail`, which runs `Handler.on_failure`."""


class Reschedule(JobSignal):
    """Rate limit or provider pause: the job comes back at `run_after` without spending an attempt.

    Attributes:
        run_after: When the job may be claimed again.
    """

    def __init__(self, code: str, run_after: datetime) -> None:
        """Store the reason and the new start time.

        Args:
            code: Reason stored in `jobs.last_error_code`.
            run_after: When the job may be claimed again.
        """
        super().__init__(code)
        self.run_after = run_after


@dataclass(frozen=True)
class JobContext:
    """What a handler receives.

    Attributes:
        job: The claimed job.
        pool: Database pool; the handler opens its own short transactions.
        settings: Process settings.
        telemetry: Telemetry bound to the job's ids.
        blobs: File store.
        llm: Provider gateway, or None in tests that don't call a model.
        embedder: Embedder, or None in tests that don't embed.
    """

    job: Job
    pool: ConnectionPool
    settings: Settings
    telemetry: Telemetry
    blobs: BlobStore
    llm: LLM | None
    embedder: Embedder | None


class Handler(Protocol):
    """Runs one kind of job and records its permanent failure."""

    def run(self, ctx: JobContext) -> None:
        """Do the job's work and, on success, call `queue.complete` inside the fenced result transaction.

        Args:
            ctx: The claimed job and the process dependencies.

        Raises:
            Retry: Transient failure; the attempt is spent and the job comes back after `BACKOFF_S`.
            Reschedule: Rate limit or provider pause; the job comes back at `run_after`, no attempt spent.
            PermanentFailure: The work can't succeed; the worker calls `queue.fail`, which runs `on_failure`.
            LeaseLost: The lease moved to another owner; the worker drops the result.
        """

    def on_failure(self, conn: Connection, job: Job, code: str, message: str) -> None:
        """Record the failure on the job's subject (document or page) inside the fenced `fail` transaction.

        Args:
            conn: Connection inside the `fail` transaction.
            job: The failing job.
            code: Error code.
            message: Detail without customer text or keys.
        """
