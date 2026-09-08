"""ref_format.py — single source of truth for the <ref>/<lookup>/<system>
context-injection format used by both training notebooks in this folder.

This module deliberately imports the wrapping helpers straight from the
production file (`IMTalker/search_helpers.py`) instead of re-implementing
them. The whole point of this training pipeline is that the LoRA is taught to
consume EXACTLY what the live pipeline injects -- any drift between "what we
trained on" and "what production emits" reproduces the same
train/inference-mismatch failure the ref LoRA already suffers from. If you
change the tag format, change it in `IMTalker/search_helpers.py` and this
module picks it up automatically; do not hardcode the tags a second time here.

Format contract (must match IMTalker/search_helpers.py and
IMTalker/liveTry.py exactly):

  * A ref block is a CLOSED pair: "<ref> fact sentence. </ref>"
    (older logs and an earlier version of this pipeline used an unclosed
    "<ref> ... <ref>" pair -- both delimiters the same string. That ambiguity
    is a likely cause of the silence / tag-echo / stale-context failures
    documented in this project's conversation logs, and has been fixed in
    search_helpers.wrap_with_ref_tags. Train ONLY on the closed form.)
  * The lookup filler is also closed: "<lookup> Please wait a minute. </lookup>"
  * The one-time system prompt is wrapped ChatML-style:
    "<|im_start|>system\n{prompt}<|im_end|>\n"
  * Everything else (ordinary user/assistant exchange) carries NO role
    markers at all -- PersonaPlex is a full-duplex speech model; turns are
    delimited by voice activity, not by text tokens. Do not invent
    "<|im_start|>user" turns anywhere except the one-time system prompt: the
    production pipeline never emits them, so training on them would teach a
    format the model will never see live.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Wire up to the production helpers (no re-implementation, no drift) ──────
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_IMTALKER_DIR = _PROJECT_ROOT / "IMTalker"
if str(_IMTALKER_DIR) not in sys.path:
    sys.path.insert(0, str(_IMTALKER_DIR))

from search_helpers import (  # noqa: E402
    wrap_with_ref_tags,
    wrap_with_lookup_tags,
    wrap_with_system_tags,
    strip_injected_tags,
    clean_web_text,
)
from speech_text import normalize_for_speech, PLAIN_TEXT_RULES  # noqa: E402

__all__ = [
    "wrap_with_ref_tags", "wrap_with_lookup_tags", "wrap_with_system_tags",
    "strip_injected_tags", "clean_web_text", "normalize_for_speech", "PLAIN_TEXT_RULES",
    "DEFAULT_SYSTEM_PROMPT", "NO_CONTEXT_FALLBACK_FACT", "wrap_chatml_system",
    "TurnExample", "ExampleType", "load_default_system_prompt",
]

# Exact fallback string the live pipeline injects when a search genuinely
# found nothing usable -- copied verbatim from
# imtalker_personaplex_try_vad2_8998.py (both the "no hits" path and the
# search_max_filler_sec timeout path use this same sentence). Training on
# this exact string, not a paraphrase, matters: it is one of the concrete
# <ref> payloads the model will see live.
NO_CONTEXT_FALLBACK_FACT = (
    "There's no specific information available on this, so answer from general knowledge."
)

_DEFAULT_PROMPT_PATH = _IMTALKER_DIR / "prompts" / "Robert_8998_default.txt"


def load_default_system_prompt() -> str:
    """The same system-prompt text the live server loads by default. Falls
    back to a short inline copy if the prompt file is not present in this
    checkout (e.g. a stripped-down training-only clone)."""
    try:
        return _DEFAULT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return (
            "You are Robert, an assistant from RB Labs. You have knowledge about "
            "crypto, finance, investment, technology, and general topics. Answer "
            "every question completely, descriptively, and with useful detail."
        )


DEFAULT_SYSTEM_PROMPT = load_default_system_prompt()


def wrap_chatml_system(prompt_text: str) -> str:
    """Exact wrapper liveTry.py._apply_system_prompts uses when the
    tokenizer's vocabulary supports ChatML markers (checked there via
    `tokenizer.encode('<|im_start|>')`; the training notebook performs the
    same check against the real tokenizer before relying on this)."""
    return f"<|im_start|>system\n{prompt_text.strip()}<|im_end|>\n"


# ── Training example schema ─────────────────────────────────────────────────
#
# One JSONL row == one ExampleType.GROUNDED / SCOPE_NEGATIVE / NO_CONTEXT
# episode. Kept intentionally flat and dependency-free (plain dataclass, no
# pydantic) so both notebooks -- and any ad hoc inspection script -- can load
# it with nothing more than `json`.

class ExampleType:
    GROUNDED = "grounded"                # <ref> present, answer must use it
    NO_CONTEXT = "no_context"            # fallback <ref> text, answer hedges honestly
    SCOPE_NEGATIVE = "scope_negative"    # multi-turn: ref used once, then must NOT bleed
    NO_REF_BASELINE = "no_ref_baseline"  # ordinary Q&A, no <ref> at all -- keeps general
                                          # conversational ability from regressing


@dataclass
class Turn:
    """One exchange inside an episode. `ref_fact` is None for a turn that
    carries no injected context (the router decided no search was needed)."""
    user_text: str
    assistant_text: str
    ref_fact: Optional[str] = None       # plain sentence, pre-wrap (or None)
    is_fallback: bool = False            # True for the NO_CONTEXT_FALLBACK_FACT case
    note: str = ""                       # free-text provenance, not used in training


@dataclass
class Episode:
    """A full training example: zero or more turns of prior context, then the
    turn(s) that actually matter for this example's `example_type`."""
    id: str
    example_type: str
    turns: list = field(default_factory=list)   # list[Turn]
    system_prompt: str = ""
    source: str = ""                            # e.g. "financial-qa-10K#1234"

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "example_type": self.example_type,
            "system_prompt": self.system_prompt,
            "source": self.source,
            "turns": [
                {
                    "user_text": t.user_text,
                    "assistant_text": t.assistant_text,
                    "ref_fact": t.ref_fact,
                    "is_fallback": t.is_fallback,
                    "note": t.note,
                }
                for t in self.turns
            ],
        }

    @staticmethod
    def from_json(d: dict) -> "Episode":
        turns = [Turn(**t) for t in d.get("turns", [])]
        return Episode(
            id=d["id"], example_type=d["example_type"], turns=turns,
            system_prompt=d.get("system_prompt", ""), source=d.get("source", ""),
        )


# Backward/forward-compat alias used by a couple of early-draft cells.
TurnExample = Turn


def render_episode_text(ep: Episode, include_system_prompt: bool = True) -> str:
    """Render one episode as the flat text stream PersonaPlex actually
    consumes: an optional one-time ChatML system block, then each turn as
    plain user words followed by an (optional) closed <ref> block followed by
    plain assistant words -- with NO role markers anywhere else, matching
    `render_episode_text`'s use in the training notebook's tokenization step.

    This is a *reference* renderer for eyeballing/QA of the dataset (printed
    by the dataset-generation notebook's validation cell); the training
    notebook tokenizes each field separately so it can mask the loss to
    assistant spans only, rather than tokenizing this concatenated string.
    """
    parts: list[str] = []
    if include_system_prompt and ep.system_prompt:
        parts.append(wrap_chatml_system(ep.system_prompt))
    for t in ep.turns:
        parts.append(t.user_text.strip())
        if t.ref_fact:
            parts.append(wrap_with_ref_tags(t.ref_fact))
        parts.append(t.assistant_text.strip())
    return "\n".join(p for p in parts if p)


def validate_episode(ep: Episode) -> list[str]:
    """Cheap, dependency-free sanity checks run by the generation notebook
    before an episode is written to the dataset file. Returns a list of
    problem descriptions; empty list means "looks fine". Catches the classes
    of bad example that would actively teach the wrong behavior rather than
    just being low quality."""
    problems: list[str] = []
    if not ep.turns:
        problems.append("episode has zero turns")
    for i, t in enumerate(ep.turns):
        if not t.assistant_text.strip():
            problems.append(f"turn {i}: empty assistant_text")
        low = t.assistant_text.lower()
        if "<ref" in low or "<lookup" in low or "<system" in low:
            problems.append(f"turn {i}: assistant_text leaks a tag literal")
        if t.ref_fact and t.ref_fact.strip().lower() not in ("",) and not t.is_fallback:
            # Grounding check mirroring ContextCompressor's own overlap gate:
            # the reply should share real content words with the fact it was
            # supposedly grounded on, or the example is teaching the model to
            # ignore <ref> content rather than use it.
            fact_words = {w.strip(".,!?").lower() for w in t.ref_fact.split() if len(w) > 2}
            reply_words = {w.strip(".,!?").lower() for w in t.assistant_text.split() if len(w) > 2}
            if fact_words and not (fact_words & reply_words):
                problems.append(
                    f"turn {i}: assistant_text shares no content words with ref_fact "
                    f"(likely ignored the injected context)"
                )
        if ep.example_type == ExampleType.SCOPE_NEGATIVE and i > 0:
            prev = ep.turns[i - 1]
            if prev.ref_fact and not prev.is_fallback:
                prev_fact_words = {
                    w.strip(".,!?$%").lower() for w in prev.ref_fact.split()
                    if len(w) > 2 and any(c.isdigit() for c in w)
                }
                reply_words = {w.strip(".,!?$%").lower() for w in t.assistant_text.split()}
                leaked = prev_fact_words & reply_words
                if leaked and not t.ref_fact:
                    problems.append(
                        f"turn {i}: reused numeric fact {leaked!r} from the previous "
                        f"turn's <ref> even though this turn has no ref of its own "
                        f"(this is exactly the cross-turn contamination we're trying "
                        f"to train AWAY from)"
                    )
    return problems
