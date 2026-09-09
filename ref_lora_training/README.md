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
downloads everything it needs itself (PersonaPlex's 4-bit weights and its
bundled `moshi` source -- just the pieces training needs, not the full
avatar/renderer/voice stack `prepare_imtalker_personaplex.sh` also fetches
for the live server) and needs a GPU capable of loading the PersonaPlex 7B
model (4-bit) plus LoRA training overhead -- an RTX 4090/5090 or A100/L40S-
class pod is comfortable for both. Both notebooks use every GPU visible to
the pod automatically.

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
                          by default) and calls it -- no external API. Also
                          has the lenient JSON fallback parser (see below)
    generation_worker.py  standalone single-GPU worker process (one QA-row
                          shard in, one JSONL of episodes out) -- launched as
                          a subprocess, one per GPU
    multi_gpu_runner.py   shards QA rows across every visible GPU and runs
                          one generation_worker.py per GPU in parallel
    dataset_builder.py    QA rows -> episodes -> validated JSONL
    model_adapter.py      loads PersonaPlex like liveTry.py does, attaches
                          peft LoRA, and the training forward/loss -- READ
                          THE MODULE DOCSTRING before trusting this file
    batching.py           episodes -> the codes/loss_mask tensors used to
                          train
    train_worker.py       the actual training loop, run once per GPU under
                          torchrun (DistributedDataParallel)
    launch_training.py    builds and launches the torchrun command for
                          train_worker.py across every visible GPU
    proc_utils.py         shared subprocess-streaming helper used by both
                          multi_gpu_runner.py and launch_training.py
  dataset_out/            01's output lands here (train.jsonl, val.jsonl,
                          raw_generated.jsonl)
  checkpoints_out/        02's output lands here
```

## The training forward pass, confirmed against the real source

`common/model_adapter.py`'s model-loading code (`load_base_model`,
`attach_lora`) is copied directly from `IMTalker/liveTry.py` -- it is
guaranteed to match production because it *is* production's own loading
logic.

`compute_text_loss()` calls `LMModel.forward_train(codes)`. This was
originally a best-guess against the standard Moshi-family training contract
(this environment had no way to fetch PersonaPlex's exact bundled `moshi`
source), and the first real run surfaced the guess was wrong in its calling
convention: `LMModel` has no plain `forward()` at all (`lm(codes)` raises
`NotImplementedError('Module [LMModel] is missing the required "forward"
function')`). Reading the actual source
(`checkpoints/personaplex_bnb4/moshi/moshi/models/lm.py`, downloaded by
Section 4) on a live pod resolved it: `forward_train(codes)` is the real
training entry point, confirmed by name, signature, and behavior. Its source
also revealed that it already handles the model's internal per-codebook
delay pattern and re-aligns its output logits back to the same time indices
as the input `codes` -- so `compute_text_loss` does NOT apply a manual
shift-by-one on top (an earlier version did, which would have silently
trained against off-by-one targets even once the call itself worked). See
`model_adapter.py`'s module docstring and `compute_text_loss`'s own
docstring for the full detail, including how the NaN-filled invalid
positions `forward_train` returns are handled.

`resolve_num_codebooks()` and `resolve_zero_token_id()` read the real
codebook count and the model's own "empty" sentinel directly off the loaded
model (`lm.num_codebooks`, `lm.zero_token_id`) rather than trusting the
notebook's `NUM_CODEBOOKS` config guess or an arbitrary 0 -- both confirmed
attributes on the real class.

`02_LoRA_Training.ipynb` Section 6 still runs `run_contract_check()`: one
real forward+backward pass, on your actual GPU, before any real training
starts, so a bad LoRA-target-module choice or a frozen-gradient bug is caught
in seconds. The full multi-GPU training run (Section 8) repeats this same
check independently on every GPU it uses, since each one is a separate
process with its own model copy.

## Multi-GPU dataset generation

Notebook 1's conversion step (Section 4) auto-detects every GPU visible to
the pod (`N_GPUS = None` in the config cell) and launches one worker
subprocess per GPU via `common/multi_gpu_runner.py`, each loading its own
full copy of the local model and working through its own slice of the QA
rows. This is data parallelism, not model parallelism -- the right choice
for many independent, short generations -- so with N GPUs you get roughly an
N x speedup and every GPU should show ~100% utilization in `nvidia-smi`
while it runs. Each worker's output streams live into the notebook cell,
prefixed `[gpu 0]`, `[gpu 1]`, etc. Set `N_GPUS` to an int to cap how many
GPUs are used instead of using all of them.

## Multi-GPU training

Notebook 2's training step (Section 8) auto-detects every GPU visible to the
pod and launches `common/train_worker.py` under `torchrun`
(`common/launch_training.py`), one process per GPU. Each process loads its
own full 4-bit copy of PersonaPlex plus its own LoRA, and gradients are
averaged across every GPU each optimizer step via PyTorch
`DistributedDataParallel` -- standard data-parallel training, which is what
actually drives every GPU toward ~100% utilization and scales throughput
with GPU count (as opposed to `device_map="auto"`-style model sharding,
which only helps a model that doesn't fit on one GPU and doesn't speed up
training). The training data is split evenly across ranks so every GPU runs
the exact same number of optimizer steps per epoch, which DDP requires.
Output from every rank streams live into the cell; only rank 0 saves
checkpoints and prints the periodic loss/eval lines to keep the log
readable.

**Known container gotcha, still being narrowed down.** A 5-GPU run
repeatedly hit `DistributedDataParallel`'s constructor-time ALLGATHER timing
out after the full configured window. The debugging trail so far:

1. First it looked like one random rank (0, then 4, then 1) reported as
   having "0 params" while every rank's own diagnostic print showed the
   correct count -- suspected as `/dev/shm` being too small for NCCL's
   intra-node buffers (a common Docker default). Ruled out: a real run showed
   352GB free.
2. Adding `NCCL_P2P_DISABLE=1` *alongside* the (unnecessary) `NCCL_SHM_DISABLE=1`
   made things WORSE, not better -- `NCCL_DEBUG=INFO` output showed NCCL
   reporting `nNodes 5 localRanks 1` for a run with all 5 GPUs on one host.
   P2P and shared memory are NCCL's only two mechanisms for recognizing GPUs
   share a node; disabling both left it no way to group the ranks, so it fell
   back to treating each GPU as an isolated single-GPU "node" forced to
   communicate over a socket -- a self-inflicted hang, not the original
   problem. Lesson logged here so it isn't repeated: don't stack untested
   NCCL transport-disabling env vars: change one at a time.
3. With only `NCCL_IB_DISABLE=1` and `NCCL_DEBUG=INFO` set, a follow-up run's
   log showed the communicator setup succeeding COMPLETELY and CORRECTLY:
   `nNodes 1 localRanks 5` (right topology), every rank reaching "Init
   COMPLETE", rings connected. So GPU-to-GPU communication is not
   structurally broken. The ALLGATHER still hung for the full timeout anyway
   -- and it's moving a single element per rank (`NumelIn=1`), which should
   complete in microseconds on a genuinely working channel. The channel log
   showed `via P2P/CUMEM`: NCCL's newer CUDA-VMM-based peer memory mapping,
   which has a known class of bug on some driver/virtualization combinations
   where the mapping handshake succeeds but real data transfer through it
   hangs. Current state: `NCCL_CUMEM_ENABLE=0` added (alone, not stacked)
   to force the older, more broadly-compatible legacy P2P/IPC memory path.

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
  ordinary prompt-masked causal-LM SFT. The mask is aligned directly with
  each input position (no shift-by-one) -- `forward_train` already applies
  its own internal delay/shift and hands back logits realigned to the
  original positions, and combines that with its own NaN-marked
  delay-invalid positions, which `compute_text_loss` excludes via
  `torch.nan_to_num` before the loss rather than after (multiplying a NaN by
  a zero mask still produces NaN, not zero).
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
- **`LORA_TARGET_MODULES` defaults to `["proj", "fc1", "out_proj", "fc2",
  "linear", "in_proj"]`**, not auto-discovery -- these are the exact module
  names from the *currently-deployed* reference LoRA's own
  `adapter_config.json` (embedded in `prepare_imtalker_personaplex.sh`),
  confirmed-correct for this exact architecture. Set it to `None` to fall
  back to `common/model_adapter.discover_target_modules`'s heuristic instead.
- **Gradient checkpointing is off by default** (`attach_lora`'s
  `use_gradient_checkpointing=False`). peft's gradient-checkpointing setup
  calls `transformers.PreTrainedModel`-only APIs
  (`gradient_checkpointing_enable`, `get_input_embeddings`) that a raw
  `moshi.models.lm.LMModel` doesn't implement -- turning it on would raise
  before training starts unless your specific fork happens to expose that API.

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
