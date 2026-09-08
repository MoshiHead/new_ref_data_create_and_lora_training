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

`loss_mask` is aligned DIRECTLY with `codes[:, 0, :]` (no shift-by-one):
`LMModel.forward_train` (confirmed via source inspection -- see
model_adapter.py's module docstring) already applies its own internal delay
pattern and re-aligns its output logits back to the same time indices as the
input codes, so `text_logits[:, t]` is already the causally-correct
prediction FOR `codes[:, 0, t]` itself. An earlier version of this file
shifted the mask by one on the assumption the caller had to do that
manually, which would have silently trained against off-by-one targets.
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


def build_batch(
    tokenized: list[TokenizedEpisode],
    num_audio_codebooks: int,
    text_pad_id: int,
    audio_pad_id: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads a list of TokenizedEpisode to the batch's max length and returns
    (codes, loss_mask):
      codes:      [B, 1 + num_audio_codebooks, T] int64
      loss_mask:  [B, T] float -- aligned DIRECTLY with codes[:, 0, :] (see
                  module docstring for why no shift is applied here)

    `text_pad_id` and `audio_pad_id` should both normally be the model's own
    `zero_token_id` (see `model_adapter.resolve_zero_token_id`) -- the
    sentinel `forward_train` itself uses to decide which positions are real,
    not an arbitrary 0 or the text tokenizer's own (unrelated) pad id.
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
    loss_mask = target_mask_full.to(device)
    return codes, loss_mask
