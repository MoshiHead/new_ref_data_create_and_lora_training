"""batching.py — turns Episodes into the `codes` / `loss_mask` tensors
`model_adapter.compute_text_loss` expects.

Design choice (documented, not accidental): every text token occupies one
consecutive "frame" slot with all audio codebooks held at a constant silence
id, for the WHOLE episode (system prompt + prior turns + <ref> block +
reply). This mirrors exactly how `_inject_tokens` already forces context into
the live model in production
(IMTalker/imtalker_personaplex_try_vad2_8998.py:763-774: one `lm_gen.step()`
per text token, zeroed audio frame, no padding between tokens) rather than
trying to reconstruct realistic word-per-second speech timing, which we have
no paired audio to justify anyway. This trains the TEXT-conditioning policy
(does the model correctly use/ignore a <ref> block) -- the specific behavior
that's broken -- not prosody or pacing.

Only assistant-speech tokens are supervised (loss_mask=1); the system
prompt, the user's words, and the <ref>/<lookup> block itself are context
and receive gradient exactly as they would in ordinary prompt-masked causal
LM SFT.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ref_format import Episode, wrap_chatml_system, wrap_with_ref_tags


@dataclass
class TokenizedEpisode:
    ids: list[int]
    target_mask: list[int]  # len(ids), 1 where ids[i] is an assistant-speech token


def tokenize_episode(
    ep: Episode, tokenizer, include_system_prompt: bool = True, max_len: int = 512,
) -> TokenizedEpisode:
    ids: list[int] = []
    mask: list[int] = []

    def _add(text: str, supervised: bool) -> None:
        piece_ids = tokenizer.encode(text)
        ids.extend(piece_ids)
        mask.extend([1 if supervised else 0] * len(piece_ids))

    if include_system_prompt and ep.system_prompt:
        _add(wrap_chatml_system(ep.system_prompt), supervised=False)
    for turn in ep.turns:
        _add(turn.user_text.strip(), supervised=False)
        if turn.ref_fact:
            _add(wrap_with_ref_tags(turn.ref_fact), supervised=False)
        _add(turn.assistant_text.strip(), supervised=True)

    if max_len and len(ids) > max_len:
        # Truncate from the FRONT (drop oldest context) so the most recent
        # turn -- the one carrying the actual supervised reply -- always
        # survives. Long-context episodes are rare in this dataset (one or
        # two turns), so this mainly protects against a runaway system prompt.
        ids = ids[-max_len:]
        mask = mask[-max_len:]
    return TokenizedEpisode(ids=ids, target_mask=mask)


def resolve_text_pad_id(tokenizer) -> int:
    """Best-effort pad id for the text-token stream, tried in the order most
    likely to exist across a sentencepiece object or an HF tokenizer. Prints
    what it found so a wrong guess is visible immediately rather than baked
    silently into every batch."""
    for attr, is_method in (("pad_id", True), ("pad_token_id", False)):
        try:
            val = getattr(tokenizer, attr)
            val = val() if is_method and callable(val) else val
            if isinstance(val, int) and val >= 0:
                print(f"[batching] using tokenizer.{attr} = {val} as the text pad id", flush=True)
                return val
        except Exception:
            continue
    print(
        "[batching] could not find a pad id on the tokenizer; defaulting to 0. "
        "Verify this against your fork (e.g. `tokenizer.encode(' ')`, or check "
        "`lm_gen.zero_text_code` in a loaded live session) and override "
        "TEXT_PAD_ID in the training notebook's config cell if 0 is wrong.",
        flush=True,
    )
    return 0


def build_batch(
    tokenized: list[TokenizedEpisode],
    num_audio_codebooks: int,
    text_pad_id: int,
    audio_pad_id: int = 0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads a list of TokenizedEpisode to the batch's max length and returns
    (codes, loss_mask):
      codes:      [B, 1 + num_audio_codebooks, T] int64
      loss_mask:  [B, T-1] float -- aligned to `codes[:, 0, 1:]` targets
    """
    max_len = max(len(t.ids) for t in tokenized)
    B = len(tokenized)
    text_row = torch.full((B, max_len), text_pad_id, dtype=torch.long)
    target_mask_full = torch.zeros((B, max_len), dtype=torch.float32)
    for i, t in enumerate(tokenized):
        n = len(t.ids)
        text_row[i, :n] = torch.tensor(t.ids, dtype=torch.long)
        target_mask_full[i, :n] = torch.tensor(t.target_mask, dtype=torch.float32)

    audio_rows = torch.full((B, num_audio_codebooks, max_len), audio_pad_id, dtype=torch.long)
    codes = torch.cat([text_row.unsqueeze(1), audio_rows], dim=1).to(device)
    loss_mask = target_mask_full[:, 1:].to(device)  # shift to align with next-token targets
    return codes, loss_mask
