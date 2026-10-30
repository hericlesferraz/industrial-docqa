"""The sandbox: progress watchdog, total time limit, memory limit, crashes and the child's environment."""

from pathlib import Path

import pytest

from docqa.ingest.sandbox import EXIT_MEMORY, SandboxResult, run_sandboxed

CHILD = "tests.sandbox_children"


def _run(
    mode: str, out: Path, *, timeout_s: float = 30, memory_mb: int = 1024, stall_s: float | None = 2
) -> SandboxResult:
    """Run a fake child.

    Args:
        mode: Behaviour of the fake child.
        out: Its progress file.
        timeout_s: Total time limit.
        memory_mb: Memory limit.
        stall_s: Progress watchdog.

    Returns:
        The sandbox result.
    """
    return run_sandboxed(
        CHILD, [mode, str(out)], timeout_s=timeout_s, memory_mb=memory_mb, progress_path=out, stall_s=stall_s
    )


def test_child_that_finishes(tmp_path: Path) -> None:
    """A child that ends normally reports exit 0, its wall time and its peak memory.

    Args:
        tmp_path: Temp folder.
    """
    result = _run("ok", tmp_path / "out")
    assert result.returncode == 0
    assert not result.killed_by_parent
    assert result.crash_signal is None
    assert result.peak_rss_mb > 0
    assert (tmp_path / "out").read_text() == "start\ndone\n"


def test_peak_memory_is_the_childs_own(tmp_path: Path) -> None:
    """A parent holding hundreds of MiB (the worker holds the embedding model) doesn't inflate the child's peak.

    Args:
        tmp_path: Temp folder.
    """
    ballast = bytearray(300 * 2**20)
    for i in range(0, len(ballast), 4096):
        ballast[i] = 1
    result = _run("ok", tmp_path / "out")
    assert 0 < result.peak_rss_mb < 200
    assert len(ballast) == 300 * 2**20


def test_peak_memory_of_a_killed_child_comes_from_the_samples(tmp_path: Path) -> None:
    """A child killed at the memory limit can't report its peak; the parent's samples give it.

    Args:
        tmp_path: Temp folder.
    """
    result = _run("memory", tmp_path / "out", memory_mb=200)
    assert result.memory_exceeded
    assert result.peak_rss_mb > 200


def test_child_without_progress_is_killed(tmp_path: Path) -> None:
    """A child whose output stops growing is killed after the stall time, well before the total limit.

    Args:
        tmp_path: Temp folder.
    """
    result = _run("stall", tmp_path / "out", timeout_s=30, stall_s=1)
    assert result.stalled
    assert not result.timed_out
    assert result.crash_signal is None
    assert result.wall_s < 10


def test_child_that_progresses_runs_until_the_total_limit(tmp_path: Path) -> None:
    """A child that keeps writing is not a stall; it is stopped by the total time limit.

    Args:
        tmp_path: Temp folder.
    """
    result = _run("slow", tmp_path / "out", timeout_s=2, stall_s=1)
    assert result.timed_out
    assert not result.stalled
    assert 2 <= result.wall_s < 10


def test_child_over_the_memory_limit_is_killed(tmp_path: Path) -> None:
    """A child whose resident memory passes the limit is killed and reported as such.

    Args:
        tmp_path: Temp folder.
    """
    result = _run("memory", tmp_path / "out", memory_mb=200, stall_s=None)
    assert result.memory_exceeded
    assert result.killed_by_parent


@pytest.mark.parametrize(("mode", "code"), [("crash", 1), ("memory_error", EXIT_MEMORY)])
def test_child_exit_codes(tmp_path: Path, mode: str, code: int) -> None:
    """An exception exits 1 with the traceback in stderr; a `MemoryError` exits with `EXIT_MEMORY`.

    Args:
        tmp_path: Temp folder.
        mode: Behaviour of the fake child.
        code: Expected exit code.
    """
    result = _run(mode, tmp_path / "out")
    assert result.returncode == code
    assert result.crash_signal is None
    if mode == "crash":
        assert "RuntimeError: broken" in result.stderr_tail


def test_child_killed_by_a_signal(tmp_path: Path) -> None:
    """A child killed by a signal the parent didn't send reports that signal.

    Args:
        tmp_path: Temp folder.
    """
    result = _run("signal", tmp_path / "out")
    assert result.returncode == -9
    assert result.crash_signal == 9


def test_child_gets_no_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The child's environment is an allowlist: no API keys, no database URL.

    Args:
        tmp_path: Temp folder.
        monkeypatch: Sets secrets in the parent's environment.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "secret-gemini")
    monkeypatch.setenv("ADMIN_API_KEY", "secret-admin")
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    result = _run("env", tmp_path / "out")
    names = set((tmp_path / "out").read_text().split())
    assert result.returncode == 0
    assert {"GEMINI_API_KEY", "ADMIN_API_KEY", "DATABASE_URL"}.isdisjoint(names)
    assert "MALLOC_ARENA_MAX" in names
