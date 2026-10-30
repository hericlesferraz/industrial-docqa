"""Run untrusted-input work in a child process with a memory limit, a progress watchdog and a total time limit.

`run_sandboxed` starts `python -m docqa.ingest.sandbox <module> <args...>`. This module's entry point turns core dumps
off in the child before the target module (and PyMuPDF) is imported, then runs the module; doing it in the child
avoids `preexec_fn`, which can deadlock a multi-threaded parent such as the worker. The child gets a minimal
environment, without the API keys or the database URL.

The memory limit is enforced by the parent, which samples the child's resident memory and kills it above the limit.
An address-space limit (`RLIMIT_AS`) was the first design and was dropped: importing PyMuPDF's layout mode starts
ONNX Runtime thread pools that reserve about 2.4 GiB of address space with 140 MiB resident on a 16-CPU machine, the
reservation grows with the CPU count, and a child that hits the limit fails at import or hangs instead of raising
`MemoryError`. `MALLOC_ARENA_MAX=2` halves that reservation and keeps resident memory compact.

The child's peak memory is its own high-water mark (`VmHWM`), which the child writes to a file as it ends and the
parent also samples. `wait4`'s `ru_maxrss` is not used: the child is spawned from the worker, which holds the
embedding model, and Linux carries the parent's resident size into the child's `ru_maxrss` at `exec`.
"""

import os
import resource
import runpy
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

EXIT_MEMORY = 3
POLL_S = 0.2
STDERR_TAIL_BYTES = 2048
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
CHILD_ENV_FIXED = {"MALLOC_ARENA_MAX": "2"}
PEAK_FILE_ENV = "DOCQA_SANDBOX_PEAK_FILE"
CHILD_ENV_KEYS = (
    "PATH",
    "PYTHONPATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
)


@dataclass(frozen=True)
class SandboxResult:
    """How a sandboxed child ended.

    Attributes:
        returncode: Exit code; negative when the child died from a signal (the signal number, negated).
        timed_out: The parent killed the child at the total time limit.
        stalled: The parent killed the child because its progress file stopped growing.
        memory_exceeded: The parent killed the child because its resident memory passed the limit.
        stderr_tail: The last 2 KiB of the child's stderr.
        wall_s: Wall time of the child in seconds.
        peak_rss_mb: Peak resident memory of the child in MiB (`VmHWM`), or 0 when it was never read.
    """

    returncode: int
    timed_out: bool
    stalled: bool
    memory_exceeded: bool
    stderr_tail: str
    wall_s: float
    peak_rss_mb: float

    @property
    def killed_by_parent(self) -> bool:
        """Tell whether the parent killed the child (time limit, stall or memory).

        Returns:
            True when `timed_out`, `stalled` or `memory_exceeded`.
        """
        return self.timed_out or self.stalled or self.memory_exceeded

    @property
    def crash_signal(self) -> int | None:
        """Return the signal that killed the child, unless the parent sent it.

        Returns:
            The signal number, or None when the child exited or the parent killed it.
        """
        if self.returncode < 0 and not self.killed_by_parent:
            return -self.returncode
        return None


def _child_env() -> dict[str, str]:
    """Build the child's environment from an allowlist, so secrets never reach code that parses untrusted files.

    Returns:
        The allowed variables of the current environment, plus `MALLOC_ARENA_MAX`.
    """
    return {**{key: os.environ[key] for key in CHILD_ENV_KEYS if key in os.environ}, **CHILD_ENV_FIXED}


def _rss_mb(pid: int) -> float:
    """Read a process's resident memory.

    Args:
        pid: Process id.

    Returns:
        Resident memory in MiB, or 0 when the process is gone.
    """
    try:
        resident = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return 0.0
    return resident * PAGE_SIZE / 2**20


def _hwm_mb(pid: int | str) -> float:
    """Read a process's peak resident memory since its `exec`.

    Args:
        pid: Process id, or `self`.

    Returns:
        `VmHWM` in MiB, or 0 when the process is gone or the field is missing.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return 0.0


def _read_peak(path: Path) -> float:
    """Read the peak the child wrote as it ended.

    Args:
        path: The file named by `PEAK_FILE_ENV`.

    Returns:
        Peak in MiB, or 0 when the child didn't write it (killed, or crashed hard).
    """
    try:
        return float(path.read_text().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0.0


def _kill_group(pid: int) -> None:
    """Kill a child and everything it started.

    Args:
        pid: Child pid, which is also its process group id.
    """
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _tail(path: Path) -> str:
    """Read the end of a file.

    Args:
        path: The file.

    Returns:
        Its last `STDERR_TAIL_BYTES` bytes, decoded leniently.
    """
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - STDERR_TAIL_BYTES))
        return fh.read().decode("utf-8", errors="replace")


def _size(path: Path | None) -> int:
    """Return a file's size, or -1 when it doesn't exist yet.

    Args:
        path: The file, or None.

    Returns:
        Size in bytes, or -1.
    """
    if path is None:
        return -1
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return -1


def run_sandboxed(
    module: str,
    args: list[str],
    *,
    timeout_s: float,
    memory_mb: int,
    progress_path: Path | None = None,
    stall_s: float | None = None,
) -> SandboxResult:
    """Run `python -m <module> <args>` in a child process with a wall-clock and a memory limit.

    The parent samples the child's resident memory every `POLL_S` and kills the child's process group above
    `memory_mb`, at `timeout_s`, or when `progress_path` hasn't grown for `stall_s`. The child runs without core dumps.

    Args:
        module: Module to run with `python -m`.
        args: Command-line arguments for the module.
        timeout_s: Wall-clock limit in seconds.
        memory_mb: Resident memory limit of the child in MiB.
        progress_path: File the child appends to as it progresses; None disables the stall watchdog.
        stall_s: Longest time without growth of `progress_path` before the child is killed.

    Returns:
        Exit code, whether the parent killed it, the last 2 KiB of stderr, the wall time and the peak RSS.
    """
    command = [sys.executable, "-m", "docqa.ingest.sandbox", module, *args]
    with tempfile.TemporaryDirectory(prefix="docqa-sandbox-") as scratch:
        stderr_path = Path(scratch) / "stderr"
        peak_path = Path(scratch) / "peak"
        with stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                env={**_child_env(), PEAK_FILE_ENV: str(peak_path)},
                start_new_session=True,
                close_fds=True,
            )
        started = time.monotonic()
        last_size, last_progress = _size(progress_path), started
        timed_out = stalled = memory_exceeded = False
        sampled_peak = 0.0
        while True:
            pid, status = os.waitpid(proc.pid, os.WNOHANG)
            if pid:
                break
            now = time.monotonic()
            size = _size(progress_path)
            if size != last_size:
                last_size, last_progress = size, now
            sampled_peak = max(sampled_peak, _hwm_mb(proc.pid))
            if _rss_mb(proc.pid) > memory_mb:
                memory_exceeded = True
            elif now - started > timeout_s:
                timed_out = True
            elif progress_path is not None and stall_s is not None and now - last_progress > stall_s:
                stalled = True
            if timed_out or stalled or memory_exceeded:
                _kill_group(proc.pid)
                pid, status = os.waitpid(proc.pid, 0)
                break
            time.sleep(POLL_S)
        wall_s = time.monotonic() - started
        proc.returncode = os.waitstatus_to_exitcode(status)
        return SandboxResult(
            returncode=proc.returncode,
            timed_out=timed_out,
            stalled=stalled,
            memory_exceeded=memory_exceeded,
            stderr_tail=_tail(stderr_path),
            wall_s=wall_s,
            peak_rss_mb=max(sampled_peak, _read_peak(peak_path)),
        )


def _write_peak() -> None:
    """Child side: write the child's own peak resident memory to the file the parent named, if any."""
    target = os.environ.get(PEAK_FILE_ENV)
    if not target:
        return
    try:
        Path(target).write_text(f"{_hwm_mb('self'):.1f}")
    except OSError:
        return


def main(argv: list[str]) -> None:
    """Child entry point: turn core dumps off, then run the target module as `__main__`.

    A `MemoryError` that escapes the module ends the process with `EXIT_MEMORY`. The child's peak memory is written
    for the parent on the way out.

    Args:
        argv: `<module> <args...>`.
    """
    module, *args = argv
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.argv = [module, *args]
    out_of_memory = False
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except MemoryError:
        out_of_memory = True
    finally:
        _write_peak()
    if out_of_memory:
        os._exit(EXIT_MEMORY)


if __name__ == "__main__":
    main(sys.argv[1:])
