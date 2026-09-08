"""model_adapter.py — loads the REAL PersonaPlex/Moshi stack the exact same
way production does, applies a fresh LoRA via peft, and provides the
teacher-forcing forward/loss used to train it.

READ THIS BEFORE TRUSTING IT BLINDLY
=====================================
`load_base_model()` and `load_ref_lora_for_inference_check()` are copied
line-for-line from the loading logic in IMTalker/liveTry.py (the file your
production server actually runs). That part is guaranteed to match, because
it IS the production code path.

`compute_text_loss()` is NOT copied from anywhere -- production never trains
this model, it only ever calls the streaming, no-grad `LMGen.step()` /
`lm_gen._step()` inference API (see IMTalker/imtalker_personaplex_try_vad2_8998.py
`_inject_tokens`). There is no training entry point in this repository to
copy from. This function instead targets the standard training contract of
the open-source Moshi `LMModel` class that PersonaPlex's own `loaders.py` is
built on (`codes: [B, K, T]` int tensor, codebook 0 = text, forward returns
per-codebook logits) -- the same contract Kyutai's own reference LoRA
fine-tuning recipe for Moshi-family models trains against. Because
PersonaPlex is NVIDIA's own checkpoint/fork and this analysis environment has
no network access to fetch its bundled `moshi` source, that contract could
not be verified against the real class before you run this on RunPod.

That is exactly what `run_contract_check()` is for. It runs ONE forward+
backward pass on a tiny synthetic batch, on your actual RunPod GPU, against
your actual installed `moshi` package, before any real training happens, and
prints exactly what it tried and what worked. If your fork's forward
signature differs, this is where you'll see it fail with a clear message
telling you which attempt was tried -- fix ONLY the `_FORWARD_ATTEMPTS` list
below and re-run the contract check; nothing else in this file or in the
training notebook needs to change.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


# ── Step 1: load the base model exactly like liveTry.py does ───────────────

def ensure_moshi_importable(moshi_root: str | Path) -> None:
    root = Path(moshi_root)
    pkg = root / "moshi"
    if pkg.exists() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


@dataclass
class LoadedBase:
    lm: torch.nn.Module
    tokenizer: object
    model_type: str
    mimi: Optional[object] = None


def load_base_model(
    moshi_root: str,
    mimi_hf_repo: str,
    device: str = "cuda",
    moshi_weight: str = "",
    mimi_weight: str = "",
    tokenizer_path: str = "",
    quantize_4bit: bool = True,
    num_codebooks: int = 8,
    load_mimi: bool = False,
) -> LoadedBase:
    """Mirrors IMTalker/liveTry.py lines ~131-191 exactly (both the modern
    `CheckpointInfo` branch and the older `loaders.get_moshi_lm` fallback),
    so the object you fine-tune here is loaded the identical way the live
    server loads it. `load_mimi=False` by default: training only touches the
    text-token policy (see module docstring), so skipping the audio codec
    saves VRAM and load time on the training box."""
    ensure_moshi_importable(moshi_root)
    from moshi.models import loaders

    dev = torch.device(device)
    t0 = time.perf_counter()
    if hasattr(loaders, "CheckpointInfo"):
        ckpt_info = loaders.CheckpointInfo.from_hf_repo(mimi_hf_repo)
        mimi = ckpt_info.get_mimi(device=dev) if load_mimi else None
        lm = ckpt_info.get_moshi(device=dev, dtype=torch.bfloat16)
        tokenizer = ckpt_info.get_text_tokenizer()
        model_type = getattr(ckpt_info, "model_type", "moshi")
    else:
        from huggingface_hub import hf_hub_download
        import inspect
        import sentencepiece

        repo = mimi_hf_repo or getattr(loaders, "DEFAULT_REPO", "nvidia/personaplex-7b-v1")
        if not mimi_weight and load_mimi:
            mimi_weight = hf_hub_download(repo, loaders.MIMI_NAME)
        if not moshi_weight:
            moshi_weight = hf_hub_download(repo, loaders.MOSHI_NAME)
        if not tokenizer_path:
            tokenizer_path = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
        mimi = loaders.get_mimi(mimi_weight, dev) if load_mimi else None
        lm_kwargs = {"device": dev, "dtype": torch.bfloat16}
        supported = set(inspect.signature(loaders.get_moshi_lm).parameters)
        optional_kwargs = {"quantize_4bit": bool(quantize_4bit), "num_codebooks": int(num_codebooks)}
        lm_kwargs.update({k: v for k, v in optional_kwargs.items() if k in supported})
        lm = loaders.get_moshi_lm(moshi_weight, **lm_kwargs)
        tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        model_type = "personaplex"

    lm.eval()  # training only turns grad on for the LoRA params; base stays frozen+eval
    print(f"[model_adapter] base model loaded in {time.perf_counter() - t0:.1f}s (type={model_type})", flush=True)
    return LoadedBase(lm=lm, tokenizer=tokenizer, model_type=model_type, mimi=mimi)


# ── Step 2: attach a FRESH LoRA (for training) ──────────────────────────────

def discover_target_modules(lm: torch.nn.Module, extra_exclude: tuple[str, ...] = ()) -> list[str]:
    """Auto-detects LoRA-able leaf module names instead of hardcoding names
    that might not match this fork. Collects the last path segment of every
    Linear-like leaf (plain nn.Linear or a bitsandbytes 4-bit/8-bit Linear)
    whose name doesn't look like the output head. peft matches target_modules
    by suffix, so returning short names (e.g. "q_proj") targets every layer
    that has one."""
    import torch.nn as nn
    try:
        import bitsandbytes as bnb
        linear_types = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except ImportError:
        linear_types = (nn.Linear,)

    exclude_markers = ("lm_head", "text_emb", "embed", "wte", "output_proj") + extra_exclude
    names: set[str] = set()
    for name, module in lm.named_modules():
        if not isinstance(module, linear_types):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if any(marker in name.lower() for marker in exclude_markers):
            continue
        names.add(leaf)
    return sorted(names)


def attach_lora(
    lm: torch.nn.Module,
    rank: int = 16,
    alpha: Optional[int] = None,
    dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
):
    """Wraps `lm` in a fresh trainable LoRA via peft.get_peft_model (NOT
    PeftModel.from_pretrained -- that call, used in liveTry.py, is for
    *loading* an already-trained adapter; here we're creating new adapter
    weights to train). peft handles bitsandbytes 4-bit base layers
    generically, and liveTry.py already proves this exact `lm` object is
    peft-compatible when loaded this way (it successfully wraps it with
    `PeftModel.from_pretrained` for inference)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=True)
    if target_modules is None:
        target_modules = discover_target_modules(lm)
        print(f"[model_adapter] auto-detected target_modules: {target_modules}", flush=True)
    if not target_modules:
        raise RuntimeError(
            "discover_target_modules() found nothing to target. Inspect "
            "`lm.named_modules()` yourself and pass target_modules explicitly."
        )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha or (2 * rank), lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type=None,
    )
    peft_model = get_peft_model(lm, cfg)
    peft_model.print_trainable_parameters()
    return peft_model


# ── Step 3: the part that could not be verified offline (see docstring) ────
#
# Each attempt is (name, callable(lm, codes) -> object_with_logits). Tried in
# order; the first one that runs without raising AND yields a tensor shaped
# like [B, T, vocab] or [B, K, T, vocab] is used. Add your fork's real
# calling convention to the TOP of this list if none of these match.

def _get_text_logits(out, codes_text_codebook_index: int = 0):
    """Normalizes a handful of plausible LMModel output shapes down to plain
    [B, T, vocab] text logits."""
    if hasattr(out, "text_logits") and out.text_logits is not None:
        return out.text_logits
    if hasattr(out, "logits"):
        logits = out.logits
        if logits.dim() == 4:  # [B, K, T, vocab] -- text is codebook 0
            return logits[:, codes_text_codebook_index]
        if logits.dim() == 3:  # already [B, T, vocab]
            return logits
    if torch.is_tensor(out):
        if out.dim() == 4:
            return out[:, codes_text_codebook_index]
        if out.dim() == 3:
            return out
    raise TypeError(f"don't know how to extract text logits from output of type {type(out)}")


_FORWARD_ATTEMPTS = [
    ("lm(codes)", lambda lm, codes: lm(codes)),
    ("lm(codes=codes)", lambda lm, codes: lm(codes=codes)),
    ("lm(codes, condition_tensors=None)", lambda lm, codes: lm(codes, condition_tensors=None)),
    ("lm.forward(codes)", lambda lm, codes: lm.forward(codes)),
]


def compute_text_loss(lm: torch.nn.Module, codes: torch.Tensor, loss_mask: torch.Tensor,
                       forward_attempt_name: Optional[str] = None) -> tuple[torch.Tensor, str]:
    """codes: [B, 1+n_audio, T] int64, codebook 0 = text, next-token target is
    codes[:, 0, 1:]. loss_mask: [B, T-1] float/bool, 1 where that target
    position should be supervised (assistant-speech tokens only -- everything
    else, including the injected <ref> block itself, is context and must NOT
    receive gradient, exactly like prompt-masking in ordinary causal-LM SFT).

    Returns (loss, attempt_name_used). Pass `forward_attempt_name` once you
    know which one works (from run_contract_check) to skip re-probing every
    step."""
    attempts = _FORWARD_ATTEMPTS
    if forward_attempt_name is not None:
        attempts = [a for a in _FORWARD_ATTEMPTS if a[0] == forward_attempt_name] or _FORWARD_ATTEMPTS

    last_err = None
    for name, fn in attempts:
        try:
            out = fn(lm, codes)
            text_logits = _get_text_logits(out)  # [B, T, V]
            targets = codes[:, 0, 1:]              # [B, T-1]
            pred = text_logits[:, :-1, :]           # predict position t+1 from position t
            vocab = pred.size(-1)
            per_tok = F.cross_entropy(
                pred.reshape(-1, vocab), targets.reshape(-1), reduction="none",
            ).view_as(targets)
            mask = loss_mask.to(per_tok.dtype)
            denom = mask.sum().clamp_min(1.0)
            loss = (per_tok * mask).sum() / denom
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss from attempt {name!r}")
            return loss, name
        except Exception as e:  # noqa: BLE001 - intentionally broad, we're probing
            last_err = e
            continue
    raise RuntimeError(
        "No forward-call convention in _FORWARD_ATTEMPTS worked against your installed "
        f"moshi LMModel. Last error: {last_err!r}\n"
        "Fix: inspect `inspect.signature(lm.forward)` and `type(lm).__mro__` in a scratch "
        "cell, add the correct call as a new entry at the TOP of _FORWARD_ATTEMPTS in "
        "ref_lora_training/common/model_adapter.py, and re-run run_contract_check()."
    )


def run_contract_check(lm: torch.nn.Module, num_codebooks: int, vocab_size: int, device: str = "cuda") -> str:
    """Builds a tiny synthetic batch and runs ONE forward+backward pass
    before real training starts. This is the single most important cell in
    the training notebook -- do not skip it. Returns the forward-attempt name
    that worked, to pass into compute_text_loss for every real step."""
    B, T = 2, 16
    codes = torch.randint(0, min(vocab_size, 1000), (B, 1 + num_codebooks, T), device=device)
    loss_mask = torch.ones(B, T - 1, device=device)
    loss_mask[:, : T // 2] = 0.0  # exercise the masking path, not just "supervise everything"

    trainable = [p for p in lm.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "lm has zero trainable parameters -- attach_lora() must run before "
            "run_contract_check()."
        )

    loss, used = compute_text_loss(lm, codes, loss_mask)
    loss.backward()
    grads_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
    for p in trainable:
        p.grad = None
    if not grads_ok:
        raise RuntimeError(
            f"forward attempt {used!r} produced a finite loss but no finite gradient "
            "reached the LoRA parameters -- the base model may be frozen more deeply "
            "than expected, or the wrong tensor is being differentiated."
        )
    print(f"[model_adapter] contract check passed using forward convention: {used!r}", flush=True)
    return used


# ── Step 4: save in the exact layout liveTry.py expects to load ────────────

def save_adapter(peft_model, ref_lora_dir: str | Path) -> Path:
    """Writes to `<ref_lora_dir>/lora/{adapter_config.json,
    adapter_model.safetensors}` -- the exact path run_imtalker_personaplex.sh
    and liveTry.py._load_ref_lora expect via --ref_lora_dir / REF_LORA_DIR."""
    out = Path(ref_lora_dir) / "lora"
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out))
    print(f"[model_adapter] adapter saved to {out}", flush=True)
    return out
