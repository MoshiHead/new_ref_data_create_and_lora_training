"""proc_utils.py — small subprocess-streaming helper shared by the multi-GPU
dataset-generation runner and the multi-GPU training launcher, so a
long-running child process's output shows up live in the notebook cell
instead of only appearing after it exits.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Optional


def _pump(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{prefix} {line.rstrip()}", flush=True)


def run_streaming(
    cmd: list[str], env: Optional[dict] = None, cwd: Optional[str] = None, prefix: str = "",
) -> int:
    """Runs `cmd` to completion, streaming its combined stdout/stderr live
    (each line prefixed with `prefix`), and returns its exit code."""
    proc = subprocess.Popen(
        cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    t = threading.Thread(target=_pump, args=(proc, prefix), daemon=True)
    t.start()
    proc.wait()
    t.join(timeout=5)
    return proc.returncode
