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

import sys
from pathlib import Path
from typing import Optional

from .multi_gpu_runner import detect_gpu_count
from .proc_utils import run_streaming


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

    print(f"[launch_training] {n_gpus} GPU(s) -> torchrun --nproc_per_node={n_gpus}", flush=True)
    code = run_streaming(cmd, cwd=str(project_root), prefix="[torchrun]")
    if code != 0:
        raise RuntimeError(
            f"training exited with code {code} -- scroll up in this cell's output for the "
            "traceback (each worker rank's own prints are interleaved above, unprefixed by "
            "rank except for the rank-0-only [train_worker] lines)."
        )
    print("[launch_training] training finished successfully.", flush=True)
