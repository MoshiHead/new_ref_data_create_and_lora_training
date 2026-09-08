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
import traceback
from dataclasses import dataclass
from typing import Optional

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


def _extract_json(cleaned: str) -> Optional[dict]:
    """Best-effort JSON extraction from already-fence-stripped text: tries a
    direct parse, then falls back to grabbing the largest {...} span. Returns
    None on failure rather than raising, so the caller can retry or fall back
    to `_extract_fields_lenient`."""
    if not cleaned:
        return None
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


def _extract_fields_lenient(text: str, fields: list[str]) -> Optional[dict]:
    """Fallback used when the model's output isn't valid JSON at all --
    observed in practice from the smaller/quantized local generator model
    forgetting to wrap one field's value in quotes (e.g. producing
    `"ref_fact": The company reported ...` instead of `"ref_fact": "The
    company reported ..."`), which breaks `json.loads` even though every
    field is still present and in order.

    Finds each expected field by name (in the order given) and takes
    everything between it and the start of the next expected field (or the
    end of the text, for the last one) as that field's value -- no strict
    JSON syntax required. `fields` must be given in the same order the
    prompt asked the model to produce them in."""
    if not text or not fields:
        return None
    result: dict[str, str] = {}
    for i, field in enumerate(fields):
        m_start = re.search(rf'"{re.escape(field)}"\s*:\s*"?', text)
        if not m_start:
            return None
        value_start = m_start.end()
        end_pos = len(text)
        for other in fields[i + 1:]:
            m_next = re.search(rf'"{re.escape(other)}"\s*:', text[value_start:])
            if m_next:
                end_pos = value_start + m_next.start()
                break
        raw_value = text[value_start:end_pos]
        # Strip trailing JSON furniture a truncated/malformed blob leaves
        # behind: an optional closing quote, comma, closing brace, whitespace.
        raw_value = re.sub(r'"?\s*,?\s*\}?\s*$', "", raw_value.rstrip())
        value = raw_value.strip().replace('\\"', '"').replace("\\n", " ").replace("\\'", "'")
        if not value:
            return None
        result[field] = value
    return result


@dataclass
class BackendConfig:
    local_model_id: str = "Qwen/Qwen2.5-14B-Instruct"
    local_device: str = "cuda"
    local_4bit: bool = True
    temperature: float = 0.4
    max_tokens: int = 600


class LLMGenerator:
    """Loads the local model once (via `preload()`, called explicitly by the
    notebook before the generation loop) and reuses it for every row.

    A load failure is cached and re-raised immediately on every subsequent
    call rather than retried -- retrying a broken environment does not fix
    it, it just re-attempts the same multi-second `from_pretrained` call
    (and re-prints the same "loading..." line) for every single row and
    every retry-within-a-row, which is what made an earlier version of this
    file look like it was hanging/looping on a load error instead of failing
    once, loudly, with the real underlying traceback."""

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self._tokenizer = None
        self._model = None
        self._load_error: Optional[BaseException] = None

    def preload(self) -> None:
        """Call this once, explicitly, before the generation loop. Loads the
        model and raises immediately (with the full original traceback) on
        failure -- see the module docstring for why this must not be folded
        into the per-row retry path."""
        if self._model is not None:
            return
        if self._load_error is not None:
            raise self._load_error

        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(
            f"[llm_backends] transformers={transformers.__version__} torch={torch.__version__}",
            flush=True,
        )
        print(f"[llm_backends] loading {self.cfg.local_model_id} on {self.cfg.local_device} "
              f"(4bit={self.cfg.local_4bit}) ...", flush=True)
        try:
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
        except Exception as e:
            self._load_error = e
            print(
                "[llm_backends] MODEL LOAD FAILED -- this is almost always a "
                "transformers/tokenizers/accelerate version mismatch (Qwen2.5 needs "
                "transformers>=4.46; this project's own IMTalker/prepare_imtalker_personaplex.sh "
                "pins transformers==4.52.4 for the same Qwen2.5 family and is known-good). "
                "Full traceback:",
                flush=True,
            )
            traceback.print_exc()
            raise
        self._model.eval()
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e9
        print(f"[llm_backends] ready — {n_params:.2f}B params", flush=True)

    def _call_raw(self, system_prompt: str, user_prompt: str) -> str:
        import torch

        self.preload()
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

    def generate_json(
        self, system_prompt: str, user_prompt: str, retries: int = 2,
        expected_fields: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """`expected_fields`, if given, is tried as a lenient name-based
        fallback (see `_extract_fields_lenient`) whenever strict JSON parsing
        fails -- pass the field names in the same order the prompt asked for
        them."""
        last_err = None
        for attempt in range(retries + 1):
            try:
                raw = self._call_raw(system_prompt, user_prompt)
            except Exception as e:  # noqa: BLE001 - batch job, log and retry/skip
                if e is self._load_error:
                    # The model never loaded -- every subsequent call will fail
                    # identically. Retrying a broken environment 2 more times
                    # per row (and for every row after this one) wastes GPU
                    # time and buries the real traceback; preload() already
                    # printed it once above.
                    raise
                last_err = e
                time.sleep(1.5 * (attempt + 1))
                continue
            cleaned = _strip_code_fences(raw)
            parsed = _extract_json(cleaned)
            if parsed is None and expected_fields:
                parsed = _extract_fields_lenient(cleaned, expected_fields)
            if parsed is not None:
                return parsed
            last_err = ValueError(f"could not parse JSON from model output: {raw[:300]!r}")
        print(f"[llm_backends] giving up after {retries + 1} attempts: {last_err!r}", flush=True)
        return None
