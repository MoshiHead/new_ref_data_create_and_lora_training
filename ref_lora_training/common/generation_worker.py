"""generation_worker.py — standalone single-GPU worker process for dataset
generation.

Launched as a subprocess (one per GPU) by `multi_gpu_runner.run_sharded_generation`
so each worker gets its own fresh CUDA context pinned to exactly one physical
GPU via `CUDA_VISIBLE_DEVICES`, set by the parent process before this script
starts. This is data parallelism -- each worker loads its own full copy of
the (4-bit) generator model and works through its own shard of QA rows
independently -- which is the right way to scale many independent, short,
stateless generations across multiple GPUs (as opposed to model-parallel
sharding of one model's layers, which would help a single huge generation,
not many small independent ones).

Can also be run directly for debugging one shard in isolation:

    CUDA_VISIBLE_DEVICES=0 python -m ref_lora_training.common.generation_worker \\
        --shard_path shard_0.json --output_path shard_0_out.jsonl \\
        --local_model_id Qwen/Qwen2.5-14B-Instruct --local_4bit \\
        --system_prompt_path sysprompt.txt --source_tag financial-qa-10K \\
        --temperature 0.4 --max_tokens 600
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--local_model_id", required=True)
    parser.add_argument("--local_4bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=600)
    parser.add_argument("--system_prompt_path", required=True)
    parser.add_argument("--source_tag", required=True)
    parser.add_argument("--worker_id", type=int, default=0)
    args = parser.parse_args()

    # This file lives at <project_root>/ref_lora_training/common/generation_worker.py
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from ref_lora_training.common.llm_backends import LLMGenerator, BackendConfig
    from ref_lora_training.common.dataset_builder import generate_grounded_episode, write_jsonl, read_jsonl

    system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8")
    shard: list[dict] = json.loads(Path(args.shard_path).read_text(encoding="utf-8"))

    out_path = Path(args.output_path)
    already = {ep.id for ep in read_jsonl(out_path)} if out_path.exists() else set()
    print(f"[worker {args.worker_id}] {len(shard)} rows in this shard, {len(already)} already done", flush=True)

    if not shard:
        print(f"[worker {args.worker_id}] empty shard, nothing to do", flush=True)
        return

    backend_cfg = BackendConfig(
        local_model_id=args.local_model_id, local_4bit=bool(args.local_4bit),
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    generator = LLMGenerator(backend_cfg)
    generator.preload()  # fails loudly, with a full traceback, if this worker's GPU/env is broken

    episodes = list(read_jsonl(out_path)) if out_path.exists() else []
    n_failed = 0
    for item in shard:
        idx = item["row_index_global"]
        ep_id = f"{args.source_tag}-{idx:06d}"
        if ep_id in already:
            continue
        ep = generate_grounded_episode(item, idx, generator, system_prompt, args.source_tag)
        if ep is None:
            n_failed += 1
            continue
        episodes.append(ep)
        if len(episodes) % 10 == 0:
            write_jsonl(episodes, out_path)
            print(f"[worker {args.worker_id}] {len(episodes)} done ({n_failed} failed so far)", flush=True)

    write_jsonl(episodes, out_path)
    print(f"[worker {args.worker_id}] FINISHED: {len(episodes)} episodes, {n_failed} failed conversions", flush=True)


if __name__ == "__main__":
    main()
