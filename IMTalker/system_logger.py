"""system_logger.py — the system-information log for the live PersonaPlex +
IMTalker server.

This is the deployment-facing half of the logging pair. Its counterpart,
`conversation_logger.py`, records what was SAID; this file records what the
process IS: every model that was loaded, the exact path or Hugging Face repo it
came from, every LoRA adapter and whether it merged, the resolved runtime
configuration, the video/render parameters, GPU state, how long each startup
stage took, and any component that failed to come up.

Why it is separate from the conversation log:

  * different lifetimes -- system facts are written once at startup and then
    only on state changes, while conversation events arrive continuously. Mixed
    into one file, a single startup fact becomes unfindable after ten minutes
    of conversation traffic;
  * different readers -- "which LoRA is actually live on this pod?" is a
    deployment question answered by grepping one short file, not by replaying a
    conversation.

Outputs (all append-only; each server run gets its own timestamped session id,
so restarting never overwrites a previous run's evidence):

  1. Console (stdout, captured into live_server.log by the launch notebook).
  2. `system_<session>.log`  -- human-readable, aligned, one line per fact.
  3. `system_<session>.jsonl` -- the same records as JSON, for tooling.

Every line carries a millisecond timestamp (`YYYY-mm-dd HH:MM:SS.mmm`). That
precision is not decoration: startup stages and per-turn stage costs are
correlated ACROSS the two log files by timestamp, and second-resolution
timestamps cannot separate events inside one 80 ms pipeline tick.

Thread-safety: the file writes take a lock and `logging` is internally
thread-safe, so this is safe to call from the GPU thread, the search thread,
and the asyncio loop alike. Nothing here ever raises into the caller --
a logger that can crash the pipeline is worse than no logger.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["SystemLogger", "configure", "get"]


def _stamp() -> str:
    """`YYYY-mm-dd HH:MM:SS.mmm` — full date plus milliseconds.

    The date is included (unlike the conversation log's time-only stamps)
    because this file is the one people read days later when asking what a
    given pod was actually running."""
    now = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int(now % 1 * 1000):03d}"


class SystemLogger:
    """Structured system/deployment logger. Construct once per process via
    `configure()`; retrieve anywhere with `get()`."""

    _FIELD_W = 22

    def __init__(self, log_dir: str = "", session_id: str = "") -> None:
        self.session_id = session_id or time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = str(log_dir or "")
        self._jsonl_path: str | None = None
        self._text_path: str | None = None
        self._lock = threading.Lock()
        self._t_start = time.perf_counter()
        # Startup stage costs, accumulated so the final banner can print the
        # whole load in one big-to-small breakdown instead of forcing a reader
        # to subtract timestamps by hand.
        self._stages: dict[str, float] = {}

        self.logger = logging.getLogger(f"system.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(
                logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [SYS] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(console)

            if self.log_dir:
                try:
                    os.makedirs(self.log_dir, exist_ok=True)
                    self._text_path = os.path.join(self.log_dir, f"system_{self.session_id}.log")
                    handler = logging.FileHandler(self._text_path, encoding="utf-8")
                    handler.setFormatter(
                        logging.Formatter(
                            "[%(asctime)s.%(msecs)03d] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                        )
                    )
                    self.logger.addHandler(handler)
                    self._jsonl_path = os.path.join(self.log_dir, f"system_{self.session_id}.jsonl")
                    print(
                        f"[system_logger] system information -> {self._text_path} and "
                        f"{self._jsonl_path}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[system_logger] file logging disabled ({e!r}); console only", flush=True)
                    self._text_path = None
                    self._jsonl_path = None

    # -- low level ---------------------------------------------------------

    def _write_jsonl(self, record: dict[str, Any]) -> None:
        if not self._jsonl_path:
            return
        record = dict(record)
        record.setdefault("ts", time.time())
        record.setdefault("ts_text", _stamp())
        record.setdefault("session_id", self.session_id)
        record.setdefault("uptime_s", round(time.perf_counter() - self._t_start, 3))
        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            with self._lock, open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[system_logger] failed to write jsonl record: {e!r}", flush=True)

    def event(self, kind: str, summary: str = "", **fields: Any) -> None:
        line = f"{kind:<{self._FIELD_W}} | {summary}" if summary else kind
        extra = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
        if extra:
            line = f"{line}  {extra}" if summary else f"{line} | {extra}"
        self.logger.info(line)
        self._write_jsonl({"kind": kind, "summary": summary, **fields})

    def section(self, title: str) -> None:
        self.logger.info("-" * 72)
        self.logger.info(title.upper())
        self._write_jsonl({"kind": "section", "summary": title})

    # -- startup facts -----------------------------------------------------

    def process_start(self, argv: list[str] | None = None) -> None:
        self.section("process")
        self.event(
            "process_start",
            f"pid={os.getpid()} python={platform.python_version()} host={platform.node()}",
            pid=os.getpid(),
            python=platform.python_version(),
            platform=platform.platform(),
            cwd=os.getcwd(),
            argv=list(argv if argv is not None else sys.argv),
        )
        for var in (
            "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH",
            "PYTORCH_CUDA_ALLOC_CONF",
            "IMTALKER_CACHED_ENGINE",
            "IMTALKER_PROMPT_STATE_CACHE",
            "HF_HOME",
        ):
            value = os.environ.get(var)
            if value:
                self.event("env", f"{var}={value}", var=var, value=value)

    def torch_info(self) -> None:
        """Torch/CUDA identity and the live GPU. Recorded because a silent
        driver or wheel change is the usual explanation when a pod that worked
        yesterday is slow or broken today."""
        try:
            import torch

            self.event(
                "torch",
                f"version={torch.__version__} cuda={torch.version.cuda} "
                f"available={torch.cuda.is_available()}",
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                cuda_available=bool(torch.cuda.is_available()),
                cudnn_version=(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
            )
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.event(
                    "gpu",
                    f"{props.name} vram={props.total_memory / (1 << 30):.1f}GB "
                    f"sm={props.major}.{props.minor}",
                    device_name=props.name,
                    total_vram_gb=round(props.total_memory / (1 << 30), 2),
                    capability=f"{props.major}.{props.minor}",
                    device_count=torch.cuda.device_count(),
                )
        except Exception as e:
            self.event("torch", f"probe failed: {e!r}")

    def config(self, args: Any, keys: list[str] | None = None) -> None:
        """Record the resolved runtime configuration.

        Defaults matter as much as overrides here: "which chunk size was this
        run actually using?" is a latency question, and reading it off the
        launch script is guesswork once environment overrides are in play."""
        self.section("configuration")
        namespace = vars(args) if hasattr(args, "__dict__") else dict(args)
        selected = keys if keys is not None else sorted(namespace)
        payload: dict[str, Any] = {}
        for key in selected:
            if key not in namespace:
                continue
            value = namespace[key]
            # Never echo a credential into a log file that gets shared in bug
            # reports. The presence of the key is the useful fact, not its value.
            if any(secret in key.lower() for secret in ("api_key", "token", "secret", "password")):
                value = "<set>" if value else "<unset>"
            payload[key] = value
            self.event("config", f"{key} = {value}", key=key, value=value)
        self._write_jsonl({"kind": "config_snapshot", "config": payload})

    def path(self, role: str, path: str | Path | None, required: bool = True) -> bool:
        """Record one on-disk artifact with its existence and size.

        Returns whether it exists, so callers can branch on the same fact they
        just logged instead of stat-ing the file twice."""
        if not path:
            self.event("path", f"{role:<28} <not configured>", role=role, path=None, exists=False)
            return False
        p = Path(path)
        exists = p.exists()
        size_mb = None
        if exists and p.is_file():
            try:
                size_mb = round(p.stat().st_size / (1 << 20), 2)
            except OSError:
                size_mb = None
        detail = f"{role:<28} {p}"
        if size_mb is not None:
            detail += f"  ({size_mb} MB)"
        if not exists:
            detail += "  [MISSING]" if required else "  [absent, optional]"
        self.event(
            "path", detail, role=role, path=str(p), exists=exists,
            size_mb=size_mb, required=required,
        )
        return exists

    def model_loaded(
        self,
        name: str,
        source: str = "",
        *,
        device: str = "",
        dtype: str = "",
        params_b: float | None = None,
        load_s: float | None = None,
        quantization: str = "",
        **extra: Any,
    ) -> None:
        """One model came up. `source` is the local path or HF repo it was read
        from -- always record where, not just what: two pods running "the
        generator" from different checkpoints is the single most common cause
        of "it behaves differently here"."""
        bits = [f"{name}"]
        if source:
            bits.append(f"from={source}")
        if device:
            bits.append(f"device={device}")
        if dtype:
            bits.append(f"dtype={dtype}")
        if quantization:
            bits.append(f"quant={quantization}")
        if params_b is not None:
            bits.append(f"params={params_b:.2f}B")
        if load_s is not None:
            bits.append(f"load={load_s:.2f}s")
            self._stages[f"load:{name}"] = float(load_s)
        self.event(
            "model_loaded", "  ".join(bits), model=name, source=source, device=device,
            dtype=dtype, quantization=quantization, params_b=params_b, load_s=load_s,
            **extra,
        )

    def lora_loaded(
        self,
        name: str,
        path: str,
        *,
        merged: bool = False,
        rank: Any = None,
        alpha: Any = None,
        target: str = "",
        **extra: Any,
    ) -> None:
        """A LoRA adapter was applied. `merged` is recorded explicitly because
        merged and unmerged adapters behave identically until you try to unload
        one, and because an unmerged QLoRA adapter costs forward-time compute
        that a merged one does not."""
        self.event(
            "lora_loaded",
            f"{name} path={path} merged={merged} rank={rank} alpha={alpha} "
            f"target={target or 'n/a'}",
            lora=name, path=path, merged=bool(merged), rank=rank, alpha=alpha,
            applied_to=target, **extra,
        )

    def component_failed(self, name: str, exc: BaseException, traceback_text: str = "") -> None:
        """A component did not come up. Every optional subsystem here degrades
        rather than aborts, so without this line a disabled feature looks
        exactly like a feature that was never configured."""
        self.event(
            "component_failed", f"{name}: {exc!r}", component=name,
            error=repr(exc), traceback=traceback_text[-4000:],
        )

    def stage(self, name: str, elapsed_s: float) -> None:
        self._stages[name] = float(elapsed_s)
        self.event("stage", f"{name:<28} {elapsed_s:.3f}s", stage=name, elapsed_s=round(float(elapsed_s), 3))

    def video_config(self, **fields: Any) -> None:
        """Video/render parameters: resolution, fps, chunk size, sub-batch,
        JPEG quality, codec. These are the knobs that trade latency against
        quality, so they belong in the record of every run."""
        self.section("video / render")
        for key, value in fields.items():
            self.event("video", f"{key:<28} {value}", key=key, value=value)
        self._write_jsonl({"kind": "video_config", "video": dict(fields)})

    def gpu_memory(self, label: str = "") -> None:
        try:
            import torch

            if not torch.cuda.is_available():
                return
            allocated = torch.cuda.memory_allocated() / (1 << 30)
            reserved = torch.cuda.memory_reserved() / (1 << 30)
            self.event(
                "gpu_memory",
                f"{label or 'now':<28} allocated={allocated:.2f}GB reserved={reserved:.2f}GB",
                label=label, allocated_gb=round(allocated, 3), reserved_gb=round(reserved, 3),
            )
        except Exception:
            pass

    def active_work(self, what: str, **fields: Any) -> None:
        """What the server is doing right now (session opened, generating,
        searching, rendering, idle). Distinct from `stage`, which is one-shot
        startup timing."""
        self.event("active_work", what, **fields)

    def ready(self, url: str = "") -> None:
        """Close the startup record with a big-to-small breakdown of the load,
        so the slowest startup stage is visible without adding timestamps up."""
        total = time.perf_counter() - self._t_start
        ordered = sorted(self._stages.items(), key=lambda kv: kv[1], reverse=True)
        breakdown = "  ".join(f"{n}={s:.2f}s" for n, s in ordered)
        self.section("ready")
        self.event(
            "server_ready",
            f"startup={total:.2f}s {('| ' + breakdown) if breakdown else ''}"
            + (f" | {url}" if url else ""),
            startup_s=round(total, 3), url=url or None,
            stages={n: round(s, 3) for n, s in self._stages.items()},
        )


# ── Process-wide instance ───────────────────────────────────────────────────
#
# A module-level singleton rather than a constructor argument threaded through
# every class: the engines, the loaders and the request handlers all need to
# log system facts, and plumbing a logger through all of them would be a much
# larger and riskier change than the logging is worth. `get()` always returns a
# usable logger, so a caller never has to null-check.

_INSTANCE: SystemLogger | None = None
_INSTANCE_LOCK = threading.Lock()


def configure(log_dir: str = "", session_id: str = "") -> SystemLogger:
    """Create (or return) the process-wide system logger. Safe to call more
    than once; the first call wins so a late caller cannot silently redirect
    the log to a different directory mid-run."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = SystemLogger(log_dir=log_dir, session_id=session_id)
        return _INSTANCE


def get() -> SystemLogger:
    """The process-wide system logger, creating a console-only one on demand.

    Console-only is the right fallback: a module that logs before `configure()`
    has run must still produce output rather than either crashing or silently
    dropping the fact."""
    if _INSTANCE is None:
        return configure()
    return _INSTANCE
