"""The worker process: one claim loop per job kind, a lease heartbeat and periodic maintenance.

Run with `python -m docqa.worker`; `python -m docqa.worker --healthcheck` is the container health check.
"""

import argparse
import logging
import os
import signal
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from psycopg import Connection
from psycopg_pool import ConnectionPool

from docqa import queue
from docqa.config import ConfigError, Settings, load_settings
from docqa.db import create_pool, migrate, tx
from docqa.handler import Handler, JobContext, JobSignal, PermanentFailure, Reschedule, Retry
from docqa.index.embed import Embedder, get_embedder, index_version
from docqa.index.jobs import IndexHandler
from docqa.ingest.jobs import ExtractHandler, OcrPageHandler
from docqa.providers.gemini import GeminiGateway
from docqa.providers.llm import LLM
from docqa.queue import HEARTBEAT_S, Job, LeaseLost
from docqa.repo import maintenance
from docqa.storage import BlobStore, LocalBlobStore
from docqa.telemetry import PostgresTelemetry, Telemetry, bind, configure_logging

ALIVE_FILE = Path("/tmp/docqa-worker.alive")
ALIVE_MAX_AGE_S = 30.0
IDLE_SLEEP_S = 1.0
MAINTENANCE_EVERY_S = 3600.0
STOP_GRACE_S = 25.0

__all__ = [
    "HANDLERS",
    "Handler",
    "JobContext",
    "JobSignal",
    "PermanentFailure",
    "Reschedule",
    "Retry",
    "Worker",
    "WorkerDeps",
    "default_kinds",
    "execute",
    "healthcheck",
    "main",
    "run_maintenance",
    "run_one",
    "startup",
]

logger = logging.getLogger("docqa.worker")


@dataclass(frozen=True)
class WorkerDeps:
    """Dependencies of the worker process.

    Attributes:
        pool: Database pool.
        settings: Process settings.
        telemetry: Telemetry.
        blobs: File store.
        llm: Provider gateway, or None in tests that don't call a model.
        embedder: Embedder, or None in tests that don't embed.
    """

    pool: ConnectionPool
    settings: Settings
    telemetry: Telemetry
    blobs: BlobStore
    llm: LLM | None = None
    embedder: Embedder | None = None


class NoopHandler:
    """A job that does nothing but complete; exercises the queue end to end."""

    def run(self, ctx: JobContext) -> None:
        """Complete the job in a fenced transaction.

        Args:
            ctx: The claimed job and the process dependencies.

        Raises:
            LeaseLost: The lease moved to another owner.
        """
        with tx(ctx.pool) as conn:
            queue.assert_lease(conn, ctx.job)
            queue.complete(conn, ctx.job)

    def on_failure(self, conn: Connection, job: Job, code: str, message: str) -> None:
        """Nothing to record: a noop job has no subject.

        Args:
            conn: Connection inside the `fail` transaction.
            job: The failing job.
            code: Error code.
            message: Detail.
        """


HANDLERS: dict[str, Handler] = {
    "noop": NoopHandler(),
    "extract": ExtractHandler(),
    "ocr_page": OcrPageHandler(),
    "index": IndexHandler(),
}


def default_kinds(settings: Settings) -> dict[str, int]:
    """Return the claim threads per job kind of a production worker.

    Args:
        settings: Process settings.

    Returns:
        Threads per kind: one `extract`, one `index`, `OCR_MAX_CONCURRENCY` `ocr_page`.
    """
    return {"extract": 1, "index": 1, "ocr_page": settings.ocr_max_concurrency}


def owner_name() -> str:
    """Name the current thread as a lease owner.

    Returns:
        `"<hostname>:<pid>:<thread>"`.
    """
    return f"{socket.gethostname()}:{os.getpid()}:{threading.current_thread().name}"


def _claim(deps: WorkerDeps, kind: str, owner: str) -> Job | None:
    """Requeue expired leases and claim one job of a kind.

    Args:
        deps: Worker dependencies.
        kind: Job kind.
        owner: Lease owner name.

    Returns:
        The claimed job, or None.
    """
    ocr = kind == "ocr_page"
    with deps.pool.connection() as conn:
        queue.requeue_expired(conn)
        return queue.claim(
            conn,
            kind=kind,
            owner=owner,
            ocr_cap=deps.settings.ocr_max_concurrency if ocr else None,
            ocr_model=deps.settings.gemini_ocr_model if ocr else None,
        )


def _retry(deps: WorkerDeps, job: Job, code: str, message: str) -> None:
    """Put a job back in the queue after a transient failure and record the event.

    Args:
        deps: Worker dependencies.
        job: The held job.
        code: Error code.
        message: Detail without customer text or keys.
    """
    with tx(deps.pool) as conn:
        queue.retry(conn, job, code=code, message=message)
    deps.telemetry.event("job.retry", **{"docqa.job.kind": job.kind, "error_code": code})


def _run_job(deps: WorkerDeps, job: Job, handler: Handler) -> tuple[str, str | None]:
    """Run a claimed job and apply the queue transition its outcome asks for.

    Args:
        deps: Worker dependencies.
        job: The claimed job.
        handler: Handler of the job's kind.

    Returns:
        `(outcome, error_code)`, the outcome being `done`, `retry`, `reschedule` or `failed`.

    Raises:
        LeaseLost: The lease moved to another owner while the job ran or while its outcome was written.
    """
    if job.attempts > job.max_attempts:
        code = job.last_error_code or "attempts_exhausted"
        with tx(deps.pool) as conn:
            queue.fail(conn, job, code=code, message="attempts exhausted", on_failure=handler.on_failure)
        return "failed", code
    ctx = JobContext(
        job=job,
        pool=deps.pool,
        settings=deps.settings,
        telemetry=deps.telemetry,
        blobs=deps.blobs,
        llm=deps.llm,
        embedder=deps.embedder,
    )
    try:
        handler.run(ctx)
    except LeaseLost:
        raise
    except Reschedule as signal_:
        with tx(deps.pool) as conn:
            queue.reschedule(conn, job, run_after=signal_.run_after, code=signal_.code)
        deps.telemetry.event("job.rescheduled", **{"docqa.job.kind": job.kind, "error_code": signal_.code})
        return "reschedule", signal_.code
    except Retry as signal_:
        _retry(deps, job, signal_.code, signal_.message)
        return "retry", signal_.code
    except PermanentFailure as signal_:
        with tx(deps.pool) as conn:
            queue.fail(conn, job, code=signal_.code, message=signal_.message, on_failure=handler.on_failure)
        return "failed", signal_.code
    except Exception as exc:
        logger.exception("job raised an unexpected exception", extra={"fields": {"kind": job.kind}})
        _retry(deps, job, "internal_error", type(exc).__name__)
        return "retry", "internal_error"
    return "done", None


def execute(deps: WorkerDeps, job: Job, handler: Handler) -> str:
    """Run a claimed job inside its telemetry context, then log it and flush its spans.

    Args:
        deps: Worker dependencies.
        job: The claimed job.
        handler: Handler of the job's kind.

    Returns:
        The outcome: `done`, `retry`, `reschedule`, `failed` or `lease_lost`.
    """
    started = time.perf_counter()
    with bind(job_id=job.id, tenant_id=job.tenant_id, document_id=job.document_id, page_no=job.page_no):
        with deps.telemetry.span("job.run", **{"docqa.job.kind": job.kind, "docqa.job.attempt": job.attempts}) as span:
            try:
                outcome, code = _run_job(deps, job, handler)
            except LeaseLost:
                outcome, code = "lease_lost", "lease_lost"
                deps.telemetry.event("job.lease_lost", **{"docqa.job.kind": job.kind, "error_code": code})
            span.set("docqa.job.outcome", outcome)
            if outcome in ("retry", "failed", "lease_lost") and code:
                span.error(code)
        fields = {
            "kind": job.kind,
            "outcome": outcome,
            "attempt": job.attempts,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "error_code": code,
        }
        logger.info("job finished", extra={"fields": fields})
        try:
            deps.telemetry.flush()
        except Exception:
            logger.exception("telemetry flush failed")
    return outcome


def run_one(deps: WorkerDeps, kind: str, *, owner: str = "test", handlers: Mapping[str, Handler] | None = None) -> bool:
    """Claim and run one job of a kind synchronously, without a heartbeat; for tests.

    Args:
        deps: Worker dependencies.
        kind: Job kind.
        owner: Lease owner name.
        handlers: Handler registry; `HANDLERS` when None.

    Returns:
        True when a job was claimed and run.
    """
    job = _claim(deps, kind, owner)
    if job is None:
        return False
    execute(deps, job, (handlers or HANDLERS)[kind])
    return True


def run_maintenance(deps: WorkerDeps) -> None:
    """Run the periodic cleanups: abandoned upload temp files, old finished jobs, unreferenced chunk history.

    Args:
        deps: Worker dependencies.
    """
    temp_files = deps.blobs.cleanup_temp(maintenance.TMP_GRACE)
    with deps.pool.connection() as conn:
        jobs = maintenance.delete_done_jobs(conn, keep=maintenance.DONE_JOBS_KEEP)
        history = maintenance.cleanup_history(conn, grace=maintenance.HISTORY_GRACE)
    fields = {"temp_files_deleted": temp_files, "jobs_deleted": jobs, "history_deleted": history}
    logger.info("maintenance done", extra={"fields": fields})


def startup(deps: WorkerDeps) -> int:
    """Run the start-up steps: send back to indexing the pages indexed under another index version.

    That covers the pages `NullIndexer` marked before search existed and a switch of embedding provider: the pages are
    reindexed from their stored text, without paying for OCR again.

    Args:
        deps: Worker dependencies; without an embedder there is nothing to reconcile.

    Returns:
        Number of pages sent back to indexing.
    """
    if deps.embedder is None:
        return 0
    current = index_version(deps.embedder)
    with tx(deps.pool) as conn:
        reset = maintenance.reconcile_index_version(conn, current=current)
    logger.info("index version reconciled", extra={"fields": {"index_version": current, "pages_reset": reset}})
    return reset


class Worker:
    """Claim loops per job kind, a heartbeat for held leases and a maintenance thread.

    Attributes:
        deps: Worker dependencies.
        kinds: Claim threads per job kind.
        handlers: Handler per job kind.
    """

    def __init__(self, deps: WorkerDeps, *, kinds: dict[str, int], handlers: Mapping[str, Handler]) -> None:
        """Prepare the worker; nothing runs until `run`.

        Args:
            deps: Worker dependencies.
            kinds: Claim threads per job kind.
            handlers: Handler per job kind; every kind in `kinds` needs one.

        Raises:
            ValueError: A kind has no handler.
        """
        missing = set(kinds) - set(handlers)
        if missing:
            raise ValueError(f"no handler for job kinds {sorted(missing)}")
        self.deps = deps
        self.kinds = kinds
        self.handlers = handlers
        self._stop = threading.Event()
        self._held: dict[int, Job] = {}
        self._held_lock = threading.Lock()

    def stop(self) -> None:
        """Ask every loop to stop; running jobs get `STOP_GRACE_S` to finish."""
        self._stop.set()

    def run(self, alive_file: Path = ALIVE_FILE) -> None:
        """Start the threads and block until `stop`, touching the liveness file while every thread is alive.

        Args:
            alive_file: File whose age `--healthcheck` reads.
        """
        claimers = [
            threading.Thread(target=self._claim_loop, args=(kind,), name=f"{kind}-{n}", daemon=True)
            for kind, count in self.kinds.items()
            for n in range(count)
        ]
        helpers = [
            threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True),
            threading.Thread(target=self._maintenance_loop, name="maintenance", daemon=True),
        ]
        for thread in claimers + helpers:
            thread.start()
        logger.info("worker started", extra={"fields": {"kinds": self.kinds}})
        while not self._stop.is_set():
            if all(thread.is_alive() for thread in claimers + helpers):
                alive_file.touch()
            self._stop.wait(IDLE_SLEEP_S)
        deadline = time.monotonic() + STOP_GRACE_S
        for thread in claimers:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        logger.info("worker stopped")

    def _claim_loop(self, kind: str) -> None:
        """Claim and run jobs of one kind until stopped.

        Args:
            kind: Job kind.
        """
        owner = owner_name()
        handler = self.handlers[kind]
        while not self._stop.is_set():
            try:
                job = _claim(self.deps, kind, owner)
            except Exception:
                logger.exception("claim failed", extra={"fields": {"kind": kind}})
                self._stop.wait(IDLE_SLEEP_S)
                continue
            if job is None:
                self._stop.wait(IDLE_SLEEP_S)
                continue
            with self._held_lock:
                self._held[job.id] = job
            try:
                execute(self.deps, job, handler)
            except Exception:
                logger.exception("job outcome could not be recorded", extra={"fields": {"kind": kind}})
            finally:
                with self._held_lock:
                    self._held.pop(job.id, None)

    def _heartbeat_loop(self) -> None:
        """Extend the leases of held jobs every `HEARTBEAT_S` and record the ones lost."""
        while not self._stop.wait(HEARTBEAT_S):
            with self._held_lock:
                held = list(self._held.values())
            for job in held:
                try:
                    with self.deps.pool.connection() as conn:
                        alive = queue.heartbeat(conn, job)
                except Exception:
                    logger.exception("heartbeat failed")
                    continue
                if alive:
                    continue
                with self._held_lock:
                    self._held.pop(job.id, None)
                with bind(job_id=job.id, tenant_id=job.tenant_id, document_id=job.document_id, page_no=job.page_no):
                    self.deps.telemetry.event(
                        "job.lease_lost", **{"docqa.job.kind": job.kind, "error_code": "lease_lost"}
                    )
                    logger.warning("lease lost", extra={"fields": {"kind": job.kind}})
                    try:
                        self.deps.telemetry.flush()
                    except Exception:
                        logger.exception("telemetry flush failed")

    def _maintenance_loop(self) -> None:
        """Run the cleanups at start and every `MAINTENANCE_EVERY_S`."""
        while not self._stop.is_set():
            try:
                run_maintenance(self.deps)
            except Exception:
                logger.exception("maintenance failed")
            self._stop.wait(MAINTENANCE_EVERY_S)


def healthcheck(alive_file: Path = ALIVE_FILE, max_age_s: float = ALIVE_MAX_AGE_S) -> int:
    """Check that the worker touched its liveness file recently.

    Args:
        alive_file: File touched by `Worker.run`.
        max_age_s: Maximum age in seconds.

    Returns:
        0 when healthy, 1 otherwise.
    """
    try:
        age = time.time() - alive_file.stat().st_mtime
    except FileNotFoundError:
        return 1
    return 0 if age < max_age_s else 1


def main(argv: list[str] | None = None) -> int:
    """Run the worker process, or the health check with `--healthcheck`.

    Args:
        argv: Command-line arguments; `sys.argv[1:]` when None.

    Returns:
        Process exit code: 2 on a configuration error.
    """
    parser = argparse.ArgumentParser(prog="python -m docqa.worker")
    parser.add_argument("--healthcheck", action="store_true", help="exit 0 if the worker loop is alive")
    args = parser.parse_args(argv)
    if args.healthcheck:
        return healthcheck()
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)
    kinds = {kind: count for kind, count in default_kinds(settings).items() if kind in HANDLERS}
    pool = create_pool(settings.database_url, max_size=sum(kinds.values()) + 4)
    with pool.connection() as conn:
        applied = migrate(conn)
    if applied:
        logger.info("migrations applied", extra={"fields": {"versions": applied}})
    telemetry = PostgresTelemetry(pool)
    llm = GeminiGateway(settings.gemini_api_key, telemetry)
    deps = WorkerDeps(
        pool=pool,
        settings=settings,
        telemetry=telemetry,
        blobs=LocalBlobStore(settings.blob_store_path),
        llm=llm,
        embedder=get_embedder(settings, llm),
    )
    startup(deps)
    worker = Worker(deps, kinds=kinds, handlers=HANDLERS)

    def _on_signal(signum: int, frame: FrameType | None) -> None:
        """Stop the worker on SIGTERM or SIGINT.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        logger.info("stop requested", extra={"fields": {"signal": signum}})
        worker.stop()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    worker.run()
    pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
