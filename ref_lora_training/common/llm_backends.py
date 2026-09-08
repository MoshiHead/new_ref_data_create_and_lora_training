"""llm_backends.py — runs a local Hugging Face instruct model (on the RunPod
GPU itself) to convert QA rows into training episodes, through a single
`generate_json(...)` call.

Local-only by design: no OpenAI/Anthropic API dependency, no API key, nothing
that leaves the pod. Default model is Qwen/Qwen2.5-14B-Instruct.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort JSON extraction: strips ```json fences if present, then
    falls back to grabbing the largest {...} span. Returns None on failure
    rather than raising, so the caller can retry or skip the row."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class BackendConfig:
    local_model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    local_device: str = "cuda"
    local_4bit: bool = True
    temperature: float = 0.4
    max_tokens: int = 400


class LLMGenerator:
    """Lazy-loads the local model on first call."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[llm_backends] loading {self.cfg.local_model_id} on {self.cfg.local_device} "
              f"(4bit={self.cfg.local_4bit}) ...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.local_model_id, token=os.getenv("HF_TOKEN"))
        kwargs = dict(token=os.getenv("HF_TOKEN"), device_map=self.cfg.local_device)
        if self.cfg.local_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.local_model_id, **kwargs)
        self._model.eval()
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e9
        print(f"[llm_backends] ready — {n_params:.2f}B params", flush=True)

    def _call_raw(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_tokens,
                do_sample=self.cfg.temperature > 0,
                temperature=max(self.cfg.temperature, 1e-5),
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate_json(self, system_prompt: str, user_prompt: str, retries: int = 2) -> Optional[dict]:
        last_err = None
        for attempt in range(retries + 1):
            try:
                raw = self._call_raw(system_prompt, user_prompt)
            except Exception as e:  # noqa: BLE001 - batch job, log and retry/skip
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            parsed = _extract_json(raw)
            if parsed is not None:
                return parsed
            last_err = ValueError(f"could not parse JSON from model output: {raw[:200]!r}")
        print(f"[llm_backends] giving up after {retries + 1} attempts: {last_err!r}", flush=True)
        return None
