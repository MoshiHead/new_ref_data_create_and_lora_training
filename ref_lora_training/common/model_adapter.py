"""model_adapter.py — loads the REAL PersonaPlex/Moshi stack the exact same
way production does, applies a fresh LoRA via peft, and provides the
teacher-forcing forward/loss used to train it.

READ THIS BEFORE TRUSTING IT BLINDLY
=====================================
`load_base_model()` and `attach_lora()`'s model-loading half are copied
line-for-line from the loading logic in IMTalker/liveTry.py (the file your
production server actually runs). That part is guaranteed to match, because
it IS the production code path.

`compute_text_loss()` calls `LMModel.forward_train(codes)` -- CONFIRMED
against the real class by reading its actual source
(`moshi/models/lm.py` inside the downloaded
`brianmatzelle/personaplex-7b-v1-bnb-4bit` checkpoint) directly on a live
RunPod pod, not guessed. `LMModel` has no plain `forward` at all (calling
`lm(codes)` raises `NotImplementedError('Module [LMModel] is missing the
required "forward" function')` -- production never calls it that way either;
it only ever drives the model through the streaming, no-grad `LMGen.step()`
inference API, see `IMTalker/imtalker_personaplex_try_vad2_8998.py`'s
`_inject_tokens`). `forward_train` is the method whose name and signature
(`codes: torch.Tensor` of shape `[B, K, T]`) make it unambiguously the
training entry point, and its source shows it already handles the model's
internal per-codebook delay pattern and realigns its output logits back to
the SAME time indices as the input -- see `compute_text_loss`'s own
docstring for what that means for how targets are computed here (no manual
shift-by-one).

`run_contract_check()` still exists and still matters: it runs one real
forward+backward pass on a tiny synthetic batch, on your actual GPU, before
any real training time is spent, so a bad LoRA-target-module choice or a
frozen-gradient bug is caught in seconds rather than after a training run
appears to proceed but produces a useless adapter.
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
    use_gradient_checkpointing: bool = False,
):
    """Wraps `lm` in a fresh trainable LoRA via peft.get_peft_model (NOT
    PeftModel.from_pretrained -- that call, used in liveTry.py, is for
    *loading* an already-trained adapter; here we're creating new adapter
    weights to train). peft handles bitsandbytes 4-bit base layers
    generically, and liveTry.py already proves this exact `lm` object is
    peft-compatible when loaded this way (it successfully wraps it with
    `PeftModel.from_pretrained` for inference).

    `use_gradient_checkpointing` defaults to False on purpose: peft's
    gradient-checkpointing setup (inside `prepare_model_for_kbit_training`)
    calls `model.gradient_checkpointing_enable()` / `model.get_input_embeddings()`
    unconditionally -- both are `transformers.PreTrainedModel` APIs that a raw
    `moshi.models.lm.LMModel` does not implement, so turning this on would
    raise an AttributeError before training even starts. Flip it on only if
    you've confirmed your installed moshi fork's LMModel actually exposes
    that API (check with `hasattr(lm, "gradient_checkpointing_enable")`)."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=use_gradient_checkpointing)
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

def resolve_num_codebooks(lm: torch.nn.Module, fallback: int) -> int:
    """`lm.num_codebooks` (K in the [B, K, T] codes tensor forward_train
    expects: 1 text codebook + N audio codebooks) read directly off the
    loaded model -- confirmed via source inspection
    (`moshi/models/lm.py`'s `embed_codes`: `assert K == self.num_codebooks`).
    peft's `PeftModel.__getattr__` delegates unknown attributes to the
    wrapped base model, so this works whether `lm` is the raw LMModel or a
    peft-wrapped one. Falls back to the caller's guess only if the attribute
    is genuinely missing (e.g. a future fork that renames it)."""
    val = getattr(lm, "num_codebooks", None)
    if isinstance(val, int) and val > 0:
        return val
    print(
        f"[model_adapter] lm.num_codebooks not found or not an int (got {val!r}); "
        f"falling back to {fallback}. Verify this against your loaded model "
        f"(`raw_lm.num_codebooks`) if training shapes look wrong.",
        flush=True,
    )
    return fallback


def resolve_zero_token_id(lm: torch.nn.Module, fallback: int = 0) -> int:
    """`lm.zero_token_id` -- the model's OWN sentinel for "no token here" on
    the codes grid, used internally by `forward_train` to decide which
    positions are valid (`codes[..] != self.zero_token_id`). This is the
    correct fill value for the silent placeholder audio channels (see
    batching.py's module docstring) and for text-side batch padding -- NOT
    an arbitrary 0, and NOT the text tokenizer's own pad id (a different,
    unrelated concept: that's about the SentencePiece vocabulary, this is
    about the multistream codes grid)."""
    val = getattr(lm, "zero_token_id", None)
    if isinstance(val, int):
        return val
    print(
        f"[model_adapter] lm.zero_token_id not found (got {val!r}); defaulting to {fallback}.",
        flush=True,
    )
    return fallback


def compute_text_loss(
    lm: torch.nn.Module, codes: torch.Tensor, loss_mask: torch.Tensor,
    forward_attempt_name: Optional[str] = None,
) -> tuple[torch.Tensor, str]:
    """codes: [B, K, T] int64, K = lm.num_codebooks, codebook 0 = text.
    loss_mask: [B, T] float/bool, 1 where codes[:, 0, t] should be supervised
    (assistant-speech tokens only -- everything else, including the injected
    <ref> block itself, is context and must NOT receive gradient, exactly
    like prompt-masking in ordinary causal-LM SFT).

    Calls the confirmed-correct training entry point,
    `LMModel.forward_train(codes)` (found by reading
    moshi/models/lm.py directly -- `LMModel` has no plain `forward`, so
    `lm(codes)` always raised `NotImplementedError`). `forward_train` already
    handles the model's internal delay pattern and prepends its own initial
    token before computing logits, then re-aligns (`_undelay_sequence`) the
    result back to the SAME time indices as the input `codes` -- so
    `text_logits[:, t]` is already the (causally correct) prediction FOR
    `codes[:, 0, t]` itself. No additional manual shift-by-one belongs here;
    an earlier version of this function applied one on top, which would have
    silently trained against off-by-one targets.

    `forward_train` deliberately fills logits at delay-invalid positions with
    NaN (confirmed in its source) and returns a companion validity mask
    marking which positions are real. Both `loss_mask` (your supervision
    mask) and that validity mask are combined, and the NaN entries are
    zeroed out before the cross-entropy call -- multiplying a NaN by a 0 mask
    does NOT produce 0, it produces NaN, so this must happen before, not
    after, the loss is computed.

    `LMOutput` is a plain `@dataclass` (not a NamedTuple, so it is NOT
    subscriptable -- `out[3]` raises `TypeError`), and its field names do not
    all match the local variable names used where it's constructed in
    `forward_train`'s source (`return LMOutput(logits, logits_mask,
    text_logits, text_logits_mask)`): the 3rd positional field genuinely is
    named `text_logits`, confirmed live, but the 4th is not necessarily named
    `text_logits_mask`. Rather than guess its real name too, this reads
    `LMOutput`'s fields by POSITION via `dataclasses.fields()`, which is
    exactly as reliable as reading the source's positional constructor call
    and does not depend on knowing the name at all."""
    import dataclasses

    out = lm.forward_train(codes)
    if dataclasses.is_dataclass(out) and not isinstance(out, type):
        values = [getattr(out, f.name) for f in dataclasses.fields(out)]
    elif isinstance(out, tuple):
        values = list(out)
    else:
        raise TypeError(
            f"forward_train returned {type(out)!r}, which is neither a dataclass instance "
            "nor a tuple/NamedTuple -- inspect it directly (`dataclasses.fields(out)` or "
            "`out._fields`) and adjust compute_text_loss accordingly."
        )
    # Positional order per forward_train's source: (logits, logits_mask, text_logits, text_logits_mask)
    text_logits, valid_mask = values[2], values[3]
    text_logits = text_logits[:, 0]  # [B, 1, T, V] -> [B, T, V]
    valid_mask = valid_mask[:, 0].to(loss_mask.dtype)  # [B, 1, T] -> [B, T]

    targets = codes[:, 0, :]  # [B, T] -- direct correspondence, see docstring
    combined_mask = loss_mask.to(text_logits.dtype) * valid_mask.to(text_logits.dtype)

    safe_logits = torch.nan_to_num(text_logits, nan=0.0)
    vocab = safe_logits.size(-1)
    per_tok = F.cross_entropy(
        safe_logits.reshape(-1, vocab), targets.reshape(-1).long(), reduction="none",
    ).view_as(targets)
    denom = combined_mask.sum().clamp_min(1.0)
    loss = (per_tok * combined_mask).sum() / denom
    if not torch.isfinite(loss):
        raise RuntimeError(
            "compute_text_loss produced a non-finite loss even after NaN-safe masking -- "
            "check that `loss_mask` actually has at least one supervised position with a "
            "delay-valid target (combined_mask.sum() might be 0)."
        )
    return loss, "forward_train"


def resolve_vocab_size(tokenizer, default: int = 32000) -> int:
    """Best-effort vocabulary size across tokenizer implementations.

    A naive `getattr(tokenizer, "vocab_size", None) or
    getattr(tokenizer, "get_piece_size", lambda: default)()` looks reasonable
    but is wrong for a sentencepiece `SentencePieceProcessor` (the tokenizer
    `load_base_model` actually returns for PersonaPlex): there, `vocab_size`
    is a METHOD, not an int property. A bound method is truthy, so the `or`
    short-circuits on the method object itself without ever calling it,
    handing a method (not an int) to whatever expects a vocab size -- which
    is exactly what raised `TypeError: '<' not supported between instances
    of 'int' and 'method'` inside `run_contract_check`'s `min(vocab_size,
    1000)`. This checks each candidate attribute and calls it only when it's
    actually callable (mirrors the same pattern already used for this reason
    in `search_helpers.describe_tokenizer`)."""
    for attr in ("vocab_size", "get_piece_size", "__len__"):
        try:
            val = getattr(tokenizer, attr)
        except AttributeError:
            continue
        try:
            val = val() if callable(val) else val
        except Exception:
            continue
        if isinstance(val, int) and val > 0:
            return val
    print(
        f"[model_adapter] could not resolve a vocab size from the tokenizer "
        f"(type={type(tokenizer).__name__}); defaulting to {default}",
        flush=True,
    )
    return default


def run_contract_check(lm: torch.nn.Module, vocab_size: int, device: str = "cuda",
                        num_codebooks: Optional[int] = None) -> str:
    """Builds a tiny synthetic batch and runs ONE forward+backward pass
    before real training starts. This is the single most important cell in
    the training notebook -- do not skip it.

    `num_codebooks` is only a fallback for `resolve_num_codebooks`; the real
    value is read directly off `lm` when available (see its docstring), so
    passing nothing is normally fine and safer than hardcoding a guess here.

    Values are drawn from a tiny range (1-3), not `vocab_size` -- this is a
    synthetic smoke test that only needs to exercise every codebook's
    embedding lookup and the loss computation without going out of bounds of
    whichever cardinality that particular codebook actually has (text and
    audio codebooks generally have different vocab sizes); it deliberately
    avoids 0 in case that collides with the model's own zero_token_id
    sentinel, which would trivially zero out the whole loss mask."""
    K = resolve_num_codebooks(lm, fallback=(1 + num_codebooks) if num_codebooks else 9)
    B, T = 2, 16
    codes = torch.randint(1, 4, (B, K, T), device=device)
    loss_mask = torch.ones(B, T, device=device)
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
