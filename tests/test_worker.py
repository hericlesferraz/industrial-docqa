"""The worker: running jobs end to end, outcomes, liveness and maintenance."""

import os
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from psycopg import Connection

from docqa import queue
from docqa.db import tx
from docqa.queue import Job
from docqa.telemetry import MemoryTelemetry, PostgresTelemetry
from docqa.worker import (
    JobContext,
    PermanentFailure,
    Reschedule,
    Retry,
    Worker,
    WorkerDeps,
    healthcheck,
    run_maintenance,
    run_one,
)
from tests.helpers import expire_lease, job_row, make_runnable, new_tenant


class ScriptedHandler:
    """A handler that raises a given exception, or completes when there is none.

    Attributes:
        error: Exception to raise from `run`, or None to complete.
        failures: Calls of `on_failure` as `(job id, code, message)`.
    """

    def __init__(self, error: Exception | None = None) -> None:
        """Store the scripted outcome.

        Args:
            error: Exception to raise from `run`, or None to complete.
        """
        self.error = error
        self.failures: list[tuple[int, str, str]] = []

    def run(self, ctx: JobContext) -> None:
        """Raise the scripted exception or complete the job.

        Args:
            ctx: The job context.

        Raises:
            Exception: The scripted one.
        """
        if self.error is not None:
            raise self.error
        with tx(ctx.pool) as conn:
            queue.assert_lease(conn, ctx.job)
            queue.complete(conn, ctx.job)

    def on_failure(self, conn: Connection, job: Job, code: str, message: str) -> None:
        """Record the failure.

        Args:
            conn: Connection of the `fail` transaction.
            job: The failing job.
            code: Error code.
            message: Detail.
        """
        self.failures.append((job.id, code, message))


def _enqueue_noop(deps: WorkerDeps) -> int:
    """Enqueue a `noop` job for a new tenant.

    Args:
        deps: Worker dependencies.

    Returns:
        The job id.
    """
    tenant_id = new_tenant(deps.pool)
    with tx(deps.pool) as conn:
        job_id = queue.enqueue(conn, tenant_id=tenant_id, kind="noop")
    assert job_id is not None
    return job_id


def test_noop_job_runs_end_to_end(worker_deps: WorkerDeps, telemetry: MemoryTelemetry) -> None:
    """`run_one` claims the job, runs it, completes it and records a `job.run` span with the job id.

    Args:
        worker_deps: Worker dependencies.
        telemetry: In-memory telemetry.
    """
    job_id = _enqueue_noop(worker_deps)
    assert run_one(worker_deps, "noop") is True
    assert run_one(worker_deps, "noop") is False
    assert job_row(worker_deps.pool, job_id)["state"] == "done"
    [span] = telemetry.named("job.run")
    assert span.ids["job_id"] == job_id
    assert span.attributes["docqa.job.outcome"] == "done"
    assert span.attributes["docqa.job.kind"] == "noop"
    assert span.status == "ok"


def test_job_span_row_is_written_to_postgres(worker_deps: WorkerDeps) -> None:
    """With the Postgres telemetry, the `job.run` span lands in `spans` with the job and tenant ids.

    Args:
        worker_deps: Worker dependencies.
    """
    deps = replace(worker_deps, telemetry=PostgresTelemetry(worker_deps.pool))
    job_id = _enqueue_noop(deps)
    tenant_id = job_row(deps.pool, job_id)["tenant_id"]
    assert run_one(deps, "noop") is True
    with deps.pool.connection() as conn:
        rows = conn.execute("SELECT name, status, tenant_id, attributes FROM spans WHERE job_id = %s", (job_id,))
        [row] = rows.fetchall()
    assert row["name"] == "job.run"
    assert row["status"] == "ok"
    assert row["tenant_id"] == tenant_id
    assert row["attributes"]["docqa.job.outcome"] == "done"


@pytest.mark.parametrize(
    ("error", "state", "attempts", "code"),
    [
        (Retry("transient", "503"), "queued", 1, "transient"),
        (Reschedule("rate_limited", datetime.now(UTC) + timedelta(minutes=1)), "queued", 0, "rate_limited"),
        (PermanentFailure("pdf_corrupt", "cannot open"), "failed", 1, "pdf_corrupt"),
        (ValueError("customer text must not be stored"), "queued", 1, "internal_error"),
    ],
)
def test_handler_outcomes_steer_the_queue(
    worker_deps: WorkerDeps, telemetry: MemoryTelemetry, error: Exception, state: str, attempts: int, code: str
) -> None:
    """Each handler signal leads to its queue transition; an unexpected exception is a retry.

    Args:
        worker_deps: Worker dependencies.
        telemetry: In-memory telemetry.
        error: Exception raised by the handler.
        state: Expected job state.
        attempts: Expected attempts after the transition.
        code: Expected `last_error_code`.
    """
    job_id = _enqueue_noop(worker_deps)
    handler = ScriptedHandler(error)
    assert run_one(worker_deps, "noop", handlers={"noop": handler}) is True
    row = job_row(worker_deps.pool, job_id)
    assert (row["state"], row["attempts"], row["last_error_code"]) == (state, attempts, code)
    assert "customer text" not in str(row["last_error"])
    assert handler.failures == ([(job_id, code, "cannot open")] if state == "failed" else [])
    [span] = telemetry.named("job.run")
    assert span.attributes["docqa.job.outcome"] in {"retry", "reschedule", "failed"}


def test_attempts_exhausted_fails_with_last_code(worker_deps: WorkerDeps) -> None:
    """After `max_attempts` retries, the next claim fails the job with the last code and runs `on_failure`.

    Args:
        worker_deps: Worker dependencies.
    """
    job_id = _enqueue_noop(worker_deps)
    handler = ScriptedHandler(Retry("transient", "503"))
    for _ in range(queue.MAX_ATTEMPTS["noop"]):
        assert run_one(worker_deps, "noop", handlers={"noop": handler}) is True
        make_runnable(worker_deps.pool, job_id)
    assert job_row(worker_deps.pool, job_id)["state"] == "queued"
    assert run_one(worker_deps, "noop", handlers={"noop": handler}) is True
    row = job_row(worker_deps.pool, job_id)
    assert row["state"] == "failed"
    assert row["last_error_code"] == "transient"
    assert handler.failures == [(job_id, "transient", "attempts exhausted")]


def test_lost_lease_drops_the_result(worker_deps: WorkerDeps, telemetry: MemoryTelemetry) -> None:
    """A job whose lease was taken over while it ran can't write its result; the new owner keeps the job.

    Args:
        worker_deps: Worker dependencies.
        telemetry: In-memory telemetry.
    """
    job_id = _enqueue_noop(worker_deps)
    stolen: list[Job] = []

    class SlowHandler(ScriptedHandler):
        """A handler whose lease expires and is reclaimed by another worker before it completes."""

        def run(self, ctx: JobContext) -> None:
            """Let another owner take the job, then try to complete it.

            Args:
                ctx: The job context.
            """
            expire_lease(ctx.pool, ctx.job.id)
            with ctx.pool.connection() as conn:
                queue.requeue_expired(conn)
                taken = queue.claim(conn, kind="noop", owner="other")
            assert taken is not None
            stolen.append(taken)
            super().run(ctx)

    assert run_one(worker_deps, "noop", handlers={"noop": SlowHandler()}) is True
    row = job_row(worker_deps.pool, job_id)
    assert row["state"] == "running"
    assert row["lease_owner"] == "other"
    assert row["lease_token"] == stolen[0].lease_token
    [span] = telemetry.named("job.run")
    assert span.attributes["docqa.job.outcome"] == "lease_lost"
    assert telemetry.named("event.job.lease_lost")


def test_worker_threads_run_jobs_and_touch_the_alive_file(worker_deps: WorkerDeps, tmp_path: Path) -> None:
    """A running `Worker` drains the queue with several threads, keeps the liveness file fresh and stops.

    Args:
        worker_deps: Worker dependencies.
        tmp_path: Per-test temp folder.
    """
    job_ids = [_enqueue_noop(worker_deps) for _ in range(6)]
    alive = tmp_path / "alive"
    worker = Worker(worker_deps, kinds={"noop": 2}, handlers={"noop": ScriptedHandler()})
    thread = threading.Thread(target=worker.run, args=(alive,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if all(job_row(worker_deps.pool, job_id)["state"] == "done" for job_id in job_ids):
            break
        time.sleep(0.1)
    worker.stop()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert all(job_row(worker_deps.pool, job_id)["state"] == "done" for job_id in job_ids)
    assert healthcheck(alive) == 0


def test_worker_needs_a_handler_per_kind(worker_deps: WorkerDeps) -> None:
    """A worker can't be built for a kind without a handler.

    Args:
        worker_deps: Worker dependencies.
    """
    with pytest.raises(ValueError, match="no handler"):
        Worker(worker_deps, kinds={"extract": 1}, handlers={})


def test_healthcheck_reads_the_age_of_the_alive_file(tmp_path: Path) -> None:
    """The health check passes for a fresh file and fails for a stale or missing one.

    Args:
        tmp_path: Per-test temp folder.
    """
    alive = tmp_path / "alive"
    assert healthcheck(alive) == 1
    alive.touch()
    assert healthcheck(alive) == 0
    past = time.time() - 120
    os.utime(alive, (past, past))
    assert healthcheck(alive) == 1


def test_maintenance_deletes_old_done_jobs_only(worker_deps: WorkerDeps) -> None:
    """Old `done` jobs are deleted; recent ones and `failed` ones stay.

    Args:
        worker_deps: Worker dependencies.
    """
    old_done, recent_done, old_failed = (_enqueue_noop(worker_deps) for _ in range(3))
    with worker_deps.pool.connection() as conn:
        conn.execute(
            "UPDATE jobs SET state = 'done', finished_at = now() - interval '8 days' WHERE id = %s", (old_done,)
        )
        conn.execute("UPDATE jobs SET state = 'done', finished_at = now() WHERE id = %s", (recent_done,))
        conn.execute(
            "UPDATE jobs SET state = 'failed', finished_at = now() - interval '30 days' WHERE id = %s", (old_failed,)
        )
        run_maintenance(worker_deps)
        left = {row["id"] for row in conn.execute("SELECT id FROM jobs").fetchall()}
    assert left == {recent_done, old_failed}
