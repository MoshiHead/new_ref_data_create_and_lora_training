"""train_worker.py — the actual LoRA training loop, run once per GPU under
`torchrun` (launched by `common/launch_training.launch_ddp_training`, one
process per visible GPU, each with its own full 4-bit copy of the model,
gradients averaged across GPUs every optimizer step via
DistributedDataParallel). This is data-parallel training, not model
sharding: it's what actually drives every GPU toward ~100% utilization and
scales throughput with GPU count, unlike `device_map="auto"`-style layer
sharding (which only helps a model that doesn't fit on one GPU, and does not
speed up training).

Can be run directly for debugging a single GPU without torchrun:
    python -m ref_lora_training.common.train_worker --moshi_root ... [...]
(RANK/WORLD_SIZE/LOCAL_RANK env vars absent -> runs single-process, no DDP.)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def setup_distributed():
    """Returns (rank, world_size, local_rank, distributed). torchrun sets
    RANK/WORLD_SIZE/LOCAL_RANK in the environment before this process starts;
    their absence means "run as a single plain process, no DDP" (e.g. when
    debugging this script directly, or when torchrun was launched with
    --nproc_per_node=1 -- DDP is skipped there too, see `distributed` use
    below, since world_size=1 gains nothing from it)."""
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank, world_size > 1
    return 0, 1, 0, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moshi_root", required=True)
    parser.add_argument("--mimi_hf_repo", required=True)
    parser.add_argument("--quantize_4bit", action="store_true")
    parser.add_argument("--num_codebooks", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", type=str, default="")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--output_ref_lora_dir", required=True)
    parser.add_argument("--hf_token", type=str, default="")
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import torch

    rank, world_size, local_rank, distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    def log(*a, **kw):
        if rank == 0:
            print(*a, **kw, flush=True)

    log(f"[train_worker] world_size={world_size} distributed={distributed}")
    print(f"[train_worker rank {rank}] using device {device}", flush=True)

    from ref_lora_training.common.model_adapter import (
        load_base_model, attach_lora, run_contract_check, resolve_vocab_size,
        compute_text_loss, save_adapter,
    )
    from ref_lora_training.common.dataset_builder import read_jsonl
    from ref_lora_training.common.batching import tokenize_episode, resolve_text_pad_id, build_batch

    base = load_base_model(
        moshi_root=args.moshi_root, mimi_hf_repo=args.mimi_hf_repo, device=device,
        quantize_4bit=args.quantize_4bit, num_codebooks=args.num_codebooks, load_mimi=False,
    )
    lm, tokenizer = base.lm, base.tokenizer
    log(f"[train_worker] model_type={base.model_type}  "
        f"params(B)={sum(p.numel() for p in lm.parameters()) / 1e9:.2f}")

    target_modules = args.lora_target_modules.split(",") if args.lora_target_modules else None
    peft_model = attach_lora(
        lm, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    vocab_size_guess = resolve_vocab_size(tokenizer)
    # Every rank validates its OWN model instance independently -- cheap (one
    # forward+backward on a tiny synthetic batch), and if any rank's contract
    # check fails it raises in that process, which torchrun reports as a
    # failed worker rather than hanging the others waiting on a DDP allreduce
    # that will never come.
    forward_attempt = run_contract_check(peft_model, args.num_codebooks, vocab_size_guess, device=device)
    log(f"[train_worker] contract check passed on every rank, using: {forward_attempt}")

    if distributed:
        # find_unused_parameters=True: defensive default. If a LoRA target
        # module ends up not exercised on every single forward (depends on
        # the exact architecture graph), DDP would otherwise raise "mark
        # variable ready twice" / hang waiting on a bucket that never fills.
        # The extra overhead is small relative to a 7B forward/backward pass.
        #
        # broadcast_buffers=False: every rank already independently loaded
        # the identical pretrained 4-bit checkpoint from the same source, so
        # there is nothing meaningful for DDP's per-forward buffer broadcast
        # to synchronize -- and bitsandbytes' custom quantized-weight
        # parameter types are a known-fragile area to push through DDP's
        # buffer-broadcast path, so skipping it removes a class of
        # version-dependent breakage on top of being free speed.
        peft_model = torch.nn.parallel.DistributedDataParallel(
            peft_model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False,
        )

    train_episodes = read_jsonl(args.train_path)
    val_episodes = read_jsonl(args.val_path) if args.val_path and os.path.exists(args.val_path) else []
    assert train_episodes, f"no training episodes found at {args.train_path}"

    if distributed:
        # Every rank computes this identically (pure function of N and
        # world_size, no communication needed) so every rank ends up with
        # EXACTLY the same number of batches per epoch -- required for DDP,
        # since every rank must call backward() the same number of times in
        # the same order for its gradient allreduce to line up. A shard
        # imbalance of +/-1 episode would otherwise desync the last batch of
        # every epoch and hang.
        n_total = len(train_episodes)
        shard_counts = [len(range(r, n_total, world_size)) for r in range(world_size)]
        min_count = min(shard_counts)
        my_train = train_episodes[rank::world_size][:min_count]
        log(f"[train_worker] {n_total} total train episodes, {min_count} used per rank "
            f"({world_size} ranks, trimmed for equal shard sizes)")
    else:
        my_train = train_episodes
        log(f"[train_worker] {len(my_train)} train episodes (single process)")

    text_pad_id = resolve_text_pad_id(tokenizer)

    def make_batches(episodes, batch_size, shuffle, seed=0):
        import random
        idxs = list(range(len(episodes)))
        if shuffle:
            random.Random(seed).shuffle(idxs)
        for start in range(0, len(idxs), batch_size):
            chunk = [episodes[i] for i in idxs[start:start + batch_size]]
            tokenized = [tokenize_episode(ep, tokenizer, max_len=args.max_seq_len) for ep in chunk]
            yield build_batch(
                tokenized, num_audio_codebooks=args.num_codebooks,
                text_pad_id=text_pad_id, device=device,
            )

    trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    @torch.no_grad()
    def eval_loss(episodes, n_batches=10):
        if not episodes:
            return float("nan")
        peft_model.eval()
        losses = []
        for i, (codes, loss_mask) in enumerate(make_batches(episodes, args.batch_size, shuffle=False)):
            if i >= n_batches:
                break
            loss, _ = compute_text_loss(peft_model, codes, loss_mask, forward_attempt_name=forward_attempt)
            losses.append(loss.item())
        peft_model.train()
        return sum(losses) / max(1, len(losses))

    def unwrap(m):
        return m.module if distributed else m

    peft_model.train()
    step = 0
    t_start = time.perf_counter()
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        for micro_step, (codes, loss_mask) in enumerate(
            make_batches(my_train, args.batch_size, shuffle=True, seed=epoch)
        ):
            loss, _ = compute_text_loss(peft_model, codes, loss_mask, forward_attempt_name=forward_attempt)
            (loss / args.grad_accum_steps).backward()

            if (micro_step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.perf_counter() - t_start
                    log(f"epoch {epoch} step {step}: train_loss={loss.item():.4f} ({elapsed:.0f}s elapsed)")
                if step % args.eval_every == 0 and rank == 0:
                    v = eval_loss(val_episodes)
                    log(f"  -- val_loss={v:.4f} at step {step}")
                if step % args.save_every == 0 and rank == 0:
                    save_adapter(unwrap(peft_model), args.output_ref_lora_dir)
        if distributed:
            import torch.distributed as dist
            dist.barrier()

    if rank == 0:
        save_adapter(unwrap(peft_model), args.output_ref_lora_dir)
        log("[train_worker] training complete.")

    if distributed:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
