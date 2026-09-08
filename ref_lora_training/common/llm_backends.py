"""llm_backends.py — one small wrapper so the dataset-generation notebook can
call an OpenAI model, an Anthropic model, or a local Hugging Face model on the
RunPod GPU itself, through the same `generate_json(...)` call.

Kept intentionally minimal: a handful of retries and a tolerant JSON
extractor, nothing else. This is offline batch dataset generation, not a
production code path, so it does not need the defensive depth of
search_helpers.py.
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
    backend: str = "openai"          # "openai" | "anthropic" | "local_hf"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.4
    max_tokens: int = 400
    # local_hf only:
    local_model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    local_device: str = "cuda"
    local_4bit: bool = True


class LLMGenerator:
    """Lazy-inits whichever backend is selected on first call, so importing
    this module never requires every SDK to be installed."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self._client = None
        self._local_pipe = None

    def _ensure_client(self):
        if self._client is not None or self._local_pipe is not None:
            return
        if self.cfg.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.cfg.api_key or os.getenv("OPENAI_API_KEY"))
        elif self.cfg.backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.cfg.api_key or os.getenv("ANTHROPIC_API_KEY"))
        elif self.cfg.backend == "local_hf":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.cfg.local_model_id, token=os.getenv("HF_TOKEN"))
            kwargs = dict(token=os.getenv("HF_TOKEN"), device_map=self.cfg.local_device)
            if self.cfg.local_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["torch_dtype"] = torch.bfloat16
            model = AutoModelForCausalLM.from_pretrained(self.cfg.local_model_id, **kwargs)
            model.eval()
            self._local_pipe = (tok, model)
        else:
            raise ValueError(f"unknown backend {self.cfg.backend!r}")

    def _call_raw(self, system_prompt: str, user_prompt: str) -> str:
        self._ensure_client()
        if self.cfg.backend == "openai":
            resp = self._client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return resp.choices[0].message.content or ""
        if self.cfg.backend == "anthropic":
            resp = self._client.messages.create(
                model=self.cfg.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            return "".join(b.text for b in resp.content if hasattr(b, "text"))
        if self.cfg.backend == "local_hf":
            import torch
            tok, model = self._local_pipe
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=self.cfg.max_tokens, do_sample=self.cfg.temperature > 0,
                    temperature=max(self.cfg.temperature, 1e-5), pad_token_id=tok.eos_token_id,
                )
            return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        raise AssertionError("unreachable")

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
