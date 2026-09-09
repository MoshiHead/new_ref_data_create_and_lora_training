"""launch_training.py — launches `train_worker.py` via `torchrun` so LoRA
training scales across every visible GPU. Each GPU runs its own worker
process with a full 4-bit copy of the model and its own shard of the
training data; gradients are averaged across GPUs every optimizer step via
PyTorch DistributedDataParallel. This is the correct kind of parallelism for
training THROUGHPUT -- it's what drives every GPU toward ~100% utilization
and makes N GPUs roughly N times faster, unlike `device_map="auto"`-style
model sharding (splits one model's layers across GPUs to fit a model that's
too big for one GPU; does not speed up training and isn't needed here since
the 4-bit 7B model comfortably fits on a single modern GPU).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .multi_gpu_runner import detect_gpu_count
from .proc_utils import run_streaming


def _print_shm_size() -> None:
    """Prints /dev/shm's size, purely informational.

    Initially suspected as the cause of a DistributedDataParallel ALLGATHER
    hang (small /dev/shm is a common Docker default and NCCL does use it for
    intra-node buffers), but a real run showed 352GB free, ruling that out --
    kept here since it's a cheap, useful fact to have visible in the log for
    any future NCCL issue, not because it's currently a known problem."""
    try:
        result = subprocess.run(["df", "-h", "/dev/shm"], capture_output=True, text=True, timeout=10)
        print(f"[launch_training] /dev/shm:\n{result.stdout}", flush=True)
    except Exception as e:
        print(f"[launch_training] could not check /dev/shm size ({e!r})", flush=True)


def check_gpus_clean(n_gpus: int, max_used_mib: int = 1500) -> None:
    """Raises if any of the first `n_gpus` GPUs already has significant
    memory in use before a fresh multi-GPU launch. A prior run repeatedly
    crashed with rank 0 (which always maps to GPU 0) aborting during
    DistributedDataParallel setup, and that pod's `nvidia-smi` showed GPU 0
    already holding ~2.2GB (later ~11.7GB) while every other GPU showed ~0 --
    almost certainly a stale allocation left behind by an earlier in-notebook
    model load (e.g. the Section 6 single-GPU contract check) or a previous
    crashed torchrun launch that didn't clean up. Launching a fresh 5-way DDP
    run onto a GPU already in that state is exactly the kind of thing that
    produces confusing, rank-specific failures. This is a cheap, fast check
    that turns that into a clear error with an actionable fix instead of
    another multi-minute failed run.

    `max_used_mib` defaults to 1500, not near-zero: once a process has ever
    used a GPU via CUDA, that process keeps a baseline context footprint on
    it (typically 300-600 MiB) for as long as the process lives -- this is
    the CUDA runtime/driver context itself, not cached-but-reusable
    allocator memory, so `torch.cuda.empty_cache()` cannot release it; only
    the process exiting (e.g. a kernel restart) does. A notebook that has
    ever loaded a model for the Section 6 contract check will show this
    baseline on GPU 0 forever after, and it is completely harmless to launch
    training alongside (a few hundred MB against a 40GB+ GPU). The threshold
    is set well above that normal baseline and well below what even a
    fraction of a leaked 4-bit 7B model would leave behind (multiple GB), so
    it only fires on the actual problem case, not on ordinary CUDA context
    overhead."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except Exception as e:
        print(f"[launch_training] could not run nvidia-smi to pre-check GPU memory ({e!r}); "
              "skipping this check.", flush=True)
        return

    dirty = []
    for line in result.stdout.strip().splitlines():
        idx_str, used_str = (p.strip() for p in line.split(","))
        idx, used_mib = int(idx_str), int(used_str)
        if idx < n_gpus and used_mib > max_used_mib:
            dirty.append((idx, used_mib))

    if dirty:
        listing = ", ".join(f"GPU {i}: {m} MiB used" for i, m in dirty)
        raise RuntimeError(
            f"Refusing to launch a {n_gpus}-GPU training run: {listing} already has significant "
            f"memory allocated (threshold {max_used_mib} MiB) before this run even started. This "
            "is very likely a stale allocation from an earlier in-notebook model load (the "
            "Section 6 contract-check cell) or a half-torn-down process from a previous crashed "
            "torchrun launch -- launching onto a GPU already in that state is exactly what "
            "produced rank-specific crashes before.\n"
            "Fix: restart the Jupyter kernel (this releases any GPU memory this notebook process "
            "itself is holding), then run `!nvidia-smi` again to confirm every GPU shows ~0 MiB "
            "used. If memory is still stuck after a kernel restart, a process from a previous "
            "crashed run is orphaned -- find its PID in `nvidia-smi`'s process list and kill it "
            "(or restart the pod if it won't die)."
        )
    print(f"[launch_training] GPU pre-check OK: no GPU among the first {n_gpus} has more than "
          f"{max_used_mib} MiB already in use.", flush=True)


def launch_ddp_training(
    *,
    moshi_root: str,
    mimi_hf_repo: str,
    quantize_4bit: bool,
    num_codebooks: int,
    lora_rank: int,
    lora_alpha: Optional[int],
    lora_dropout: float,
    lora_target_modules: Optional[list[str]],
    train_path: str,
    val_path: str,
    max_seq_len: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    num_epochs: int,
    log_every: int,
    eval_every: int,
    save_every: int,
    output_ref_lora_dir: str,
    n_gpus: Optional[int] = None,
    hf_token: str = "",
) -> None:
    n_gpus = n_gpus or detect_gpu_count()
    check_gpus_clean(n_gpus)
    _print_shm_size()
    worker_script = Path(__file__).resolve().parent / "train_worker.py"
    project_root = Path(__file__).resolve().parents[2]

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={n_gpus}",
        str(worker_script),
        "--moshi_root", str(moshi_root),
        "--mimi_hf_repo", str(mimi_hf_repo),
        "--num_codebooks", str(num_codebooks),
        "--lora_rank", str(lora_rank),
        "--lora_dropout", str(lora_dropout),
        "--train_path", str(train_path),
        "--max_seq_len", str(max_seq_len),
        "--batch_size", str(batch_size),
        "--grad_accum_steps", str(grad_accum_steps),
        "--lr", str(lr),
        "--num_epochs", str(num_epochs),
        "--log_every", str(log_every),
        "--eval_every", str(eval_every),
        "--save_every", str(save_every),
        "--output_ref_lora_dir", str(output_ref_lora_dir),
    ]
    if quantize_4bit:
        cmd.append("--quantize_4bit")
    if lora_alpha is not None:
        cmd += ["--lora_alpha", str(lora_alpha)]
    if lora_target_modules:
        cmd += ["--lora_target_modules", ",".join(lora_target_modules)]
    if val_path:
        cmd += ["--val_path", str(val_path)]
    if hf_token:
        cmd += ["--hf_token", hf_token]

    # Two earlier fixes were tried and both are now known NOT to be the cause:
    # /dev/shm had 352GB free (ruling out NCCL_SHM_DISABLE's original reason
    # for existing), and stacking NCCL_SHM_DISABLE=1 together with
    # NCCL_P2P_DISABLE=1 turned out to be actively HARMFUL, confirmed by
    # NCCL_DEBUG=INFO output: with BOTH of NCCL's only same-node grouping
    # mechanisms (P2P and shared memory) disabled at once, NCCL had no way
    # left to recognize that all 5 GPUs share one host, and its own log showed
    # `nNodes 5 localRanks 1` for every rank -- i.e. it was treating each GPU
    # as an isolated single-GPU "node" forced to communicate over a socket,
    # which is what actually produced the hang, not whatever the original
    # problem was. Lesson: disabling both at once was never a real fix, it
    # replaced the mystery with a self-inflicted one. Only NCCL_IB_DISABLE=1
    # is kept (there is no InfiniBand fabric on a single-node pod regardless,
    # so this is inert either way) plus NCCL_DEBUG=INFO so NCCL's own
    # transport-selection log is visible with its NORMAL same-node transports
    # (P2P, SHM) left intact and able to actually group the ranks correctly.
    env = os.environ.copy()
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_DEBUG", "INFO")

    print(
        f"[launch_training] {n_gpus} GPU(s) -> torchrun --nproc_per_node={n_gpus} "
        f"(NCCL_IB_DISABLE={env['NCCL_IB_DISABLE']} NCCL_DEBUG={env['NCCL_DEBUG']})",
        flush=True,
    )
    code = run_streaming(cmd, env=env, cwd=str(project_root), prefix="[torchrun]")
    if code != 0:
        raise RuntimeError(
            f"training exited with code {code} -- scroll up in this cell's output for the "
            "traceback (each worker rank's own prints are interleaved above, unprefixed by "
            "rank except for the rank-0-only [train_worker] lines)."
        )
    print("[launch_training] training finished successfully.", flush=True)
