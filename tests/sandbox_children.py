"""Fake sandboxed children for the sandbox tests.

Usage: `python -m docqa.ingest.sandbox tests.sandbox_children <mode> <out>`.
"""

import os
import signal
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    """Behave as the mode says.

    Args:
        argv: `<mode> <out>`; modes: `ok`, `stall`, `slow`, `signal`, `crash`, `memory`, `memory_error`, `env`.

    Returns:
        The exit code.

    Raises:
        RuntimeError: In mode `crash`.
        MemoryError: In mode `memory_error`.
    """
    mode, out = argv[0], Path(argv[1])
    with out.open("a", encoding="utf-8") as fh:
        fh.write("start\n")
        fh.flush()
        if mode == "ok":
            fh.write("done\n")
            return 0
        if mode == "stall":
            time.sleep(60)
        if mode == "slow":
            for _ in range(600):
                fh.write("page\n")
                fh.flush()
                time.sleep(0.1)
        if mode == "signal":
            os.kill(os.getpid(), signal.SIGKILL)
        if mode == "crash":
            raise RuntimeError("broken")
        if mode == "memory":
            hog = bytearray(400 * 2**20)
            for i in range(0, len(hog), 4096):
                hog[i] = 1
            time.sleep(60)
        if mode == "memory_error":
            raise MemoryError
        if mode == "env":
            fh.write("\n".join(sorted(os.environ)) + "\n")
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
