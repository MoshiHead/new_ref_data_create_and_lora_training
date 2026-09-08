"""dataset_builder.py — turns raw QA rows into ref-injection training
episodes. Called from 01_Dataset_Generation.ipynb; kept out of the notebook
itself so the notebook stays a short, readable orchestration script.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Optional

from .distractors import DISTRACTOR_TURNS
from .generation_prompts import build_conversion_prompt
from .llm_backends import LLMGenerator
from .ref_format import (
    Episode, ExampleType, Turn, NO_CONTEXT_FALLBACK_FACT, validate_episode,
)


# ── Step 1: pull QA rows from a Hugging Face dataset ────────────────────────

def load_qa_rows(
    dataset_id: str,
    split: str,
    question_col: str,
    answer_col: str,
    context_col: str = "",
    limit: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Loads `dataset_id` via `datasets.load_dataset` and returns a plain list
    of {"question", "answer", "context"} dicts, shuffled before truncation to
    `limit` so a small run still samples across the whole dataset rather than
    just its first rows."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    if limit:
        idxs = idxs[:limit]
    rows = []
    for i in idxs:
        row = ds[i]
        q = str(row.get(question_col, "")).strip()
        a = str(row.get(answer_col, "")).strip()
        c = str(row.get(context_col, "")).strip() if context_col else ""
        if not q or not a:
            continue
        rows.append({"question": q, "answer": a, "context": c, "row_index": i})
    return rows


# ── Step 2: one QA row -> one grounded episode, via the LLM ─────────────────

def generate_grounded_episode(
    row: dict, idx: int, generator: LLMGenerator, system_prompt: str, source_tag: str,
) -> Optional[Episode]:
    system_p, user_p = build_conversion_prompt(row["question"], row["answer"], row.get("context", ""))
    parsed = generator.generate_json(system_p, user_p)
    if not parsed:
        return None
    spoken_q = str(parsed.get("spoken_question", "")).strip()
    ref_fact = str(parsed.get("ref_fact", "")).strip()
    spoken_a = str(parsed.get("spoken_answer", "")).strip()
    if not spoken_q or not ref_fact or not spoken_a:
        return None
    ep = Episode(
        id=f"{source_tag}-{idx:06d}",
        example_type=ExampleType.GROUNDED,
        system_prompt=system_prompt,
        source=f"{source_tag}#{row.get('row_index', idx)}",
        turns=[Turn(user_text=spoken_q, assistant_text=spoken_a, ref_fact=ref_fact)],
    )
    return ep


# ── Step 3: pair grounded episodes with a distractor to teach scope ─────────

def build_scope_negative_episode(base_ep: Episode, idx: int, rng: random.Random) -> Episode:
    """Turn 1 = the grounded exchange (ref used correctly). Turn 2 = an
    unrelated no-search-needed question (see common/distractors.py) whose
    reply must not reuse turn 1's injected fact. This directly targets the
    cross-turn contamination bug documented in the project's conversation
    log (a stock-price <ref> bleeding into two unrelated follow-up turns)."""
    distractor = rng.choice(DISTRACTOR_TURNS)
    ep = Episode(
        id=f"{base_ep.id}-scopeneg",
        example_type=ExampleType.SCOPE_NEGATIVE,
        system_prompt=base_ep.system_prompt,
        source=base_ep.source,
        turns=[
            base_ep.turns[0],
            Turn(
                user_text=distractor["user_text"],
                assistant_text=distractor["assistant_text"],
                ref_fact=None,
            ),
        ],
    )
    return ep


# ── Step 4: template-based "search found nothing" examples (no LLM needed) ──

_NO_CONTEXT_QUESTIONS = [
    "What's the exchange rate for the Icelandic krona right now?",
    "What's today's closing price for a stock that doesn't exist?",
    "What's the current score of a game that isn't being played?",
    "What is the latest reading for an obscure regional price index?",
]

_NO_CONTEXT_REPLIES = [
    "I don't have a specific figure for that right now, but I'm happy to help with "
    "something related if that's useful.",
    "I wasn't able to find a current number for that one. Is there something nearby "
    "I can help with instead?",
]


def build_no_context_episode(idx: int, system_prompt: str, rng: random.Random) -> Episode:
    return Episode(
        id=f"nocontext-{idx:06d}",
        example_type=ExampleType.NO_CONTEXT,
        system_prompt=system_prompt,
        source="template",
        turns=[
            Turn(
                user_text=rng.choice(_NO_CONTEXT_QUESTIONS),
                assistant_text=rng.choice(_NO_CONTEXT_REPLIES),
                ref_fact=NO_CONTEXT_FALLBACK_FACT,
                is_fallback=True,
            )
        ],
    )


# ── Step 5: no-ref baseline episodes (keep ordinary chat ability intact) ────

def build_no_ref_baseline_episode(idx: int, system_prompt: str, rng: random.Random) -> Episode:
    d1, d2 = rng.sample(DISTRACTOR_TURNS, 2)
    return Episode(
        id=f"noref-{idx:06d}",
        example_type=ExampleType.NO_REF_BASELINE,
        system_prompt=system_prompt,
        source="template",
        turns=[
            Turn(user_text=d1["user_text"], assistant_text=d1["assistant_text"], ref_fact=None),
            Turn(user_text=d2["user_text"], assistant_text=d2["assistant_text"], ref_fact=None),
        ],
    )


# ── I/O + reporting ──────────────────────────────────────────────────────────

def write_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[Episode]:
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Episode.from_json(json.loads(line)))
    return out


def validate_all(episodes: list[Episode]) -> dict:
    """Runs ref_format.validate_episode over the whole set and returns a
    summary dict. Does not drop anything itself -- the notebook decides
    whether to keep, fix, or discard flagged rows."""
    report = {"total": len(episodes), "clean": 0, "flagged": []}
    for ep in episodes:
        problems = validate_episode(ep)
        if problems:
            report["flagged"].append({"id": ep.id, "problems": problems})
        else:
            report["clean"] += 1
    return report


def train_val_split(episodes: list[Episode], val_ratio: float, seed: int = 42) -> tuple[list[Episode], list[Episode]]:
    eps = list(episodes)
    random.Random(seed).shuffle(eps)
    n_val = max(1, int(len(eps) * val_ratio)) if eps else 0
    return eps[n_val:], eps[:n_val]
