"""multi_gpu_runner.py — splits QA rows across every visible GPU and runs one
`generation_worker.py` subprocess per GPU, so dataset generation scales with
however many GPUs the RunPod pod has (1, 2, 3, 4, 5+) instead of using only
one.

Data-parallel, not model-parallel: each GPU loads its own full copy of the
(4-bit) generator model and works through its own shard of rows
independently. That's the right shape for this workload -- many short,
independent, stateless generations -- and it's what actually drives every
GPU to ~100% utilization, unlike splitting one model's layers thinly across
GPUs (`device_map="auto"`-style model parallelism), which helps a single
huge model or a single huge generation, not throughput on many small ones.

Each worker runs in its own OS process (not a Python thread or
`multiprocessing` fork) with `CUDA_VISIBLE_DEVICES` set before the
interpreter even starts, so every worker gets a completely fresh, isolated
CUDA context pinned to exactly one physical GPU -- there is no shared
process/CUDA-context state between them to get subtly wrong.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from .dataset_builder import read_jsonl
from .ref_format import Episode


def detect_gpu_count() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def _stream_output(proc: subprocess.Popen, prefix: str) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{prefix} {line.rstrip()}", flush=True)


def run_sharded_generation(
    qa_rows: list[dict],
    local_model_id: str,
    local_4bit: bool,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
    source_tag: str,
    output_dir: Path,
    n_gpus: Optional[int] = None,
    hf_token: str = "",
) -> list[Episode]:
    """Shards `qa_rows` contiguously across `n_gpus` (auto-detected if not
    given), launches one worker subprocess per non-empty shard, streams their
    combined stdout live (each line prefixed by which GPU produced it), waits
    for all of them, and returns the merged episodes. Raises if any worker
    process exits non-zero -- scroll up in the notebook cell's output for
    that worker's own traceback, printed by `generation_worker.py` /
    `LLMGenerator.preload()`.

    Safe to re-run: each shard's output file is itself resumable (workers
    skip episode ids already present in their own shard's output), same as
    the single-GPU path used to be.
    """
    n_gpus = n_gpus or detect_gpu_count()
    print(f"[multi_gpu_runner] using {n_gpus} GPU worker(s) for {len(qa_rows)} rows", flush=True)

    work_dir = Path(output_dir) / "_shards"
    work_dir.mkdir(parents=True, exist_ok=True)
    sysprompt_path = work_dir / "system_prompt.txt"
    sysprompt_path.write_text(system_prompt, encoding="utf-8")

    # Contiguous shards, tagged with each row's GLOBAL index up front, so
    # episode ids (source_tag-{idx:06d}) stay stable and resumable no matter
    # how many GPUs a given run happens to use.
    indexed_rows = [dict(r, row_index_global=i) for i, r in enumerate(qa_rows)]
    shard_size = max(1, (len(indexed_rows) + n_gpus - 1) // n_gpus)
    shards = [indexed_rows[i:i + shard_size] for i in range(0, len(indexed_rows), shard_size)]
    while len(shards) < n_gpus:
        shards.append([])  # fewer rows than GPUs -- extra workers just exit immediately

    worker_script = Path(__file__).resolve().parent / "generation_worker.py"
    project_root = Path(__file__).resolve().parents[2]

    procs: list[subprocess.Popen] = []
    threads: list[threading.Thread] = []
    shard_outputs: list[Path] = []
    launched_gpu_ids: list[int] = []

    for gpu_id, shard in enumerate(shards):
        shard_path = work_dir / f"shard_{gpu_id}.json"
        shard_path.write_text(json.dumps(shard), encoding="utf-8")
        out_path = work_dir / f"shard_{gpu_id}_out.jsonl"
        shard_outputs.append(out_path)
        if not shard:
            print(f"[multi_gpu_runner] GPU {gpu_id}: no rows assigned, skipping", flush=True)
            continue

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if hf_token:
            env["HF_TOKEN"] = hf_token
        cmd = [
            sys.executable, str(worker_script),
            "--shard_path", str(shard_path), "--output_path", str(out_path),
            "--local_model_id", local_model_id, "--temperature", str(temperature),
            "--max_tokens", str(max_tokens), "--system_prompt_path", str(sysprompt_path),
            "--source_tag", source_tag, "--worker_id", str(gpu_id),
        ]
        if local_4bit:
            cmd.append("--local_4bit")

        print(f"[multi_gpu_runner] launching worker for GPU {gpu_id}: {len(shard)} rows", flush=True)
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(project_root),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        t = threading.Thread(target=_stream_output, args=(proc, f"[gpu {gpu_id}]"), daemon=True)
        t.start()
        procs.append(proc)
        threads.append(t)
        launched_gpu_ids.append(gpu_id)

    failed_gpu_ids = []
    for gpu_id, proc in zip(launched_gpu_ids, procs):
        proc.wait()
        if proc.returncode != 0:
            failed_gpu_ids.append(gpu_id)
    for t in threads:
        t.join(timeout=5)

    if failed_gpu_ids:
        raise RuntimeError(
            f"generation worker(s) for GPU(s) {failed_gpu_ids} exited with a non-zero return "
            "code -- scroll up in this cell's output for that worker's own traceback "
            "(each line is prefixed '[gpu N]')."
        )

    # De-duplicated by id when merging: if N_GPUS changes between a run and a
    # resumed run, a row can shift from one shard file to another and get
    # regenerated there (wasteful but harmless) -- this just stops it from
    # being counted/written twice in the merged result.
    by_id: dict[str, Episode] = {}
    for out_path in shard_outputs:
        if out_path.exists():
            for ep in read_jsonl(out_path):
                by_id[ep.id] = ep
    all_episodes = list(by_id.values())
    print(f"[multi_gpu_runner] all workers finished: {len(all_episodes)} episodes total", flush=True)
    return all_episodes
