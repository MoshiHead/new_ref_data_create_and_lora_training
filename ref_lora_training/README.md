# ref_lora_training

Trains the PersonaPlex reference LoRA -- the adapter that teaches the model to
correctly consume `<ref>...</ref>` / `<lookup>...</lookup>` context injected by
the web-search pipeline in `IMTalker/imtalker_personaplex_try_vad2_8998.py`.

This exists because the current adapter is unreliable: the same conversation
log (`logs/detailed_20260908_054904.log`) that drove this design shows three
distinct failures from the same `<ref>` injection -- a turn that produced no
speech at all for 24 seconds, a turn that spoke fragments of the tag markup
aloud, and a stock-price fact that bled into two unrelated turns several
exchanges later. The dataset and training design here target exactly those
three failure modes.

## What changed in production alongside this folder

`IMTalker/search_helpers.py`'s `wrap_with_ref_tags` / `wrap_with_system_tags` /
`wrap_with_lookup_tags` were fixed to emit a **closed** tag pair
(`<ref> ... </ref>`) instead of the previous unclosed pair
(`<ref> ... <ref>`, both delimiters the same string). This is a likely
contributor to the failures above -- the model had no unambiguous signal for
where grounding ends and its own reply should begin. **This dataset and this
LoRA are built against the closed format.** If you revert that change in
`search_helpers.py`, this adapter will be trained on a format production no
longer emits.

## Workflow

```
QA dataset (e.g. financial-qa-10K on HF)
        |
        v
01_Dataset_Generation.ipynb   -- LLM rewrites QA rows into spoken-style
        |                         (question, <ref> fact, grounded answer)
        |                         triples, plus synthetic scope-negative /
        |                         no-context / no-ref-baseline episodes
        v
ref_lora_training/dataset_out/{train,val}.jsonl
        |
        v
02_LoRA_Training.ipynb        -- loads PersonaPlex exactly like liveTry.py
        |                         does, attaches a fresh LoRA, trains it
        v
ref_lora_training/checkpoints_out/rag_lora/lora/
    adapter_config.json
    adapter_model.safetensors
        |
        v
Deploy: point REF_LORA_DIR at the parent of that lora/ folder and
launch run_imtalker_personaplex.sh with ENABLE_SEARCH=1
```

Run both notebooks on RunPod. Notebook 1 loads a local instruct model
(default `Qwen/Qwen2.5-14B-Instruct`, 4-bit) on the pod's own GPU to do the
conversion -- no external API, no API key, nothing leaves the pod. Notebook 2
needs a GPU capable of loading the PersonaPlex 7B model (4-bit) plus LoRA
training overhead -- an RTX 4090/5090 or A100/L40S-class pod is comfortable
for both.

## Folder layout

```
ref_lora_training/
  01_Dataset_Generation.ipynb   run this first
  02_LoRA_Training.ipynb        run this second
  common/
    ref_format.py        the <ref>/<lookup>/<system> format contract --
                          imports the wrapping functions directly from
                          IMTalker/search_helpers.py so there is exactly one
                          place that format can be defined, not two that can
                          drift apart
    distractors.py        fixed pool of "no search needed" follow-up turns,
                          used to build the scope-negative examples
    generation_prompts.py the prompt sent to the conversion LLM
    llm_backends.py       loads a local instruct model (Qwen2.5-14B-Instruct
                          by default) on the pod's GPU and calls it -- no
                          external API
    dataset_builder.py    QA rows -> episodes -> validated JSONL
    model_adapter.py      loads PersonaPlex like liveTry.py does, attaches
                          peft LoRA, and the training forward/loss -- READ
                          THE MODULE DOCSTRING before trusting this file
    batching.py           episodes -> the codes/loss_mask tensors used to
                          train
  dataset_out/            01's output lands here (train.jsonl, val.jsonl,
                          raw_generated.jsonl)
  checkpoints_out/        02's output lands here
```

## The one thing you must verify before trusting a training run

`common/model_adapter.py`'s model-loading code (`load_base_model`,
`attach_lora`) is copied directly from `IMTalker/liveTry.py` -- it is
guaranteed to match production because it *is* production's own loading
logic.

The training forward/loss call (`compute_text_loss`) is not copied from
anywhere, because production never trains this model -- it only ever calls
the streaming, no-grad inference API. That function targets the standard
Moshi-family `LMModel` training contract (`codes: [B, K, T]`, codebook 0 =
text), which is what PersonaPlex's own loader API is built on, but PersonaPlex
is NVIDIA's own checkpoint/fork and its exact source could not be fetched or
verified from the environment this was written in.

`02_LoRA_Training.ipynb` Section 4 runs `run_contract_check()` specifically to
catch this: one real forward+backward pass, on your actual RunPod GPU,
against your actual installed `moshi` package, before any real training
starts. If it fails, the fix is entirely contained to the `_FORWARD_ATTEMPTS`
list at the top of `model_adapter.py` -- nothing else needs to change.

## Design choices worth knowing about

- **Silent-audio training.** Every training example is represented as
  text tokens one-per-frame with the audio codebooks held at a constant
  silence value for the whole sequence -- this mirrors exactly how
  `_inject_tokens` already forces `<ref>` content into the live model
  (zeroed audio frame, one token per step), and trains the text-conditioning
  policy without needing paired speech audio, which we don't have. It does
  not train prosody or timing, only whether the model correctly uses or
  ignores injected context -- which is the specific behavior that's broken.
- **Loss is masked to assistant-speech tokens only.** The system prompt, the
  user's words, and the `<ref>` block itself get no gradient, same as
  ordinary prompt-masked causal-LM SFT.
- **Scope-negative examples directly target the observed contamination bug.**
  Each pairs a real grounded exchange with a second, unrelated turn (drawn
  from the same question shapes `search_helpers.rule_route_explain` already
  classifies as "no search needed") whose answer must not reuse the first
  turn's fact. `common/ref_format.validate_episode` checks this automatically
  during generation.
- **Default QA source is `virattt/financial-qa-10K`**, a real HF dataset of
  (question, answer, context) triples derived from 10-K filings -- matches
  the finance/crypto/investment persona in
  `IMTalker/prompts/Robert_8998_default.txt`. Swap `DATASET_HF_ID` and the
  three column names in notebook 1's config cell for any other QA dataset.

## After training

Loading the adapter is not the finish line. Load it into a real running
server (`ENABLE_SEARCH=1 REF_LORA_DIR=... WEB_SEARCH_API_KEY=... ./run_imtalker_personaplex.sh`)
and replay, by voice, the exact turns from `logs/detailed_20260908_054904.log`
("What is today's gold market rate?", "What is today's Tesla stock market
rate?", then the unrelated follow-ups) -- checking specifically for the three
original failure modes: silence after injection, tag/markup leaking into
speech, and a fact bleeding into unrelated later turns. That log is a
ready-made regression test; use it as one before calling any new adapter
better than the last.
