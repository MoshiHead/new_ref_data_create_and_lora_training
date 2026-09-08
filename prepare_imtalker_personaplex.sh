#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMTALKER_DIR="$ROOT/IMTalker"
CHECKPOINT_DIR="$ROOT/checkpoints"
PERSONAPLEX_DIR="$CHECKPOINT_DIR/personaplex_bnb4"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
WITH_SEARCH="${WITH_SEARCH:-0}"
# On-disk name kept as "rag_lora" for continuity with existing pods; only
# its role was renamed (it is the <lookup>/<ref> context adapter, not a
# retrieval index).
REF_LORA_DIR="${REF_LORA_DIR:-$CHECKPOINT_DIR/rag_lora}"
STT_PKG_DIR="${STT_PKG_DIR:-$CHECKPOINT_DIR/stt}"
# Step count differs between the two modes; keep the banners honest.
if [[ "$WITH_SEARCH" == "1" ]]; then STEPS=9; else STEPS=7; fi
usage() {
  cat <<'EOF'
Usage: ./prepare_imtalker_personaplex.sh --hf-token TOKEN [--with-search]

  --hf-token TOKEN  Use TOKEN to download the required Hugging Face assets.
  --with-search     Also install and download everything the online-search
                    feature needs: the peft/transformers pins, an ISOLATED copy
                    of the upstream Kyutai moshi package for the STT submodel,
                    and the reference LoRA adapter that teaches PersonaPlex to
                    consume injected <lookup>/<ref> context. Without this flag
                    the server still runs, but only with ENABLE_SEARCH=0.
                    Equivalent to WITH_SEARCH=1 in the environment.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      [[ $# -ge 2 && -n "$2" ]] || {
        echo "--hf-token requires a token value." >&2
        usage >&2
        exit 2
      }
      HF_TOKEN="$2"
      export HF_TOKEN
      shift
      ;;
    --with-search)
      WITH_SEARCH=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

[[ -f "$IMTALKER_DIR/requirement.txt" ]] || {
  echo "Run this script from a complete speech2avatar clone." >&2
  exit 1
}

if [[ "$(id -u)" -eq 0 ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-venv \
    ffmpeg git git-lfs htop tmux curl ca-certificates build-essential
  git lfs install
else
  echo "Not root: skipping apt packages. Python 3.11, ffmpeg, git-lfs, and build tools must already exist."
fi

command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing $PYTHON_BIN." >&2
  exit 1
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.9.0"
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$IMTALKER_DIR/requirement.txt"
python -m pip install \
  "huggingface_hub[cli]==0.36.2" \
  hf_transfer tensorboard \
  "sphn==0.2.1" einops sentencepiece \
  "aiohttp==3.14.3" "av==17.1.0" "aiortc==1.15.0" \
  "bitsandbytes==0.50.0"

if [[ "$WITH_SEARCH" == "1" ]]; then
  # transformers is installed LAST and deliberately overrides whatever
  # IMTalker/requirement.txt pinned: the STT submodel and the Qwen
  # router/compressor both need a modern release. IMTalker's own use of
  # transformers is Wav2Vec2FeatureExtractor, a long-stable API.
  # peft is what applies the unmerged reference LoRA on top of the 4-bit base.
  python -m pip install "peft>=0.19,<0.20" "transformers==4.52.4"
fi

if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face access is required for the gated PersonaPlex assets." >&2
  echo "Run again with: ./prepare_imtalker_personaplex.sh --hf-token TOKEN" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p \
  "$IMTALKER_DIR/checkpoints/wav2vec2-base-960h" \
  "$CHECKPOINT_DIR/fullgen_static_2s_6400_resume" \
  "$CHECKPOINT_DIR/personaplex_unitalk_strict2s_2gpu_15k" \
  "$CHECKPOINT_DIR/lora" \
  "$CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/stats" \
  "$PERSONAPLEX_DIR"

echo "[1/$STEPS] IMTalker renderer and Wav2Vec files"
for file in \
  renderer.ckpt \
  wav2vec2-base-960h/config.json \
  wav2vec2-base-960h/pytorch_model.bin \
  wav2vec2-base-960h/preprocessor_config.json \
  wav2vec2-base-960h/feature_extractor_config.json; do
  hf download cbsjtu01/IMTalker "$file" --local-dir "$IMTALKER_DIR/checkpoints"
done

echo "[2/$STEPS] Two-second IMTalker generator and adapter"
hf download niloy629/hdtf_preprocess \
  live_winner/fullgen_static_2s_6400_resume/last.ckpt \
  live_winner/adapters/personaplex_unitalk_strict2s_2gpu_15k_last.pt \
  --repo-type dataset --local-dir "$CHECKPOINT_DIR"
ln -sfn \
  "$CHECKPOINT_DIR/live_winner/fullgen_static_2s_6400_resume/last.ckpt" \
  "$CHECKPOINT_DIR/fullgen_static_2s_6400_resume/last.ckpt"
ln -sfn \
  "$CHECKPOINT_DIR/live_winner/adapters/personaplex_unitalk_strict2s_2gpu_15k_last.pt" \
  "$CHECKPOINT_DIR/personaplex_unitalk_strict2s_2gpu_15k/last.pt"

echo "[3/$STEPS] Blink motion and silence Helium seed"
hf download niloy629/hdtf_preprocess \
  lora/3robert_audio3_ditto_static_motion.pt \
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  --repo-type dataset --local-dir "$CHECKPOINT_DIR"

echo "[4/$STEPS] PersonaPlex bnb4 package and weights"
hf download brianmatzelle/personaplex-7b-v1-bnb-4bit \
  --local-dir "$PERSONAPLEX_DIR"

echo "[5/$STEPS] PersonaPlex Mimi and tokenizer"
hf download nvidia/personaplex-7b-v1 \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  tokenizer_spm_32k_3.model \
  --local-dir "$PERSONAPLEX_DIR"

echo "[6/$STEPS] PersonaPlex voices and bundled Robert_5 voice"
hf download nvidia/personaplex-7b-v1 voices.tgz --local-dir "$PERSONAPLEX_DIR"
tar --no-same-owner -xzf "$PERSONAPLEX_DIR/voices.tgz" -C "$PERSONAPLEX_DIR"
install -m 0644 "$ROOT/bundled_assets/Robert_5.pt" "$PERSONAPLEX_DIR/voices/Robert_5.pt"

if [[ "$WITH_SEARCH" == "1" ]]; then
  echo "[7/$STEPS] Reference LoRA for <lookup>/<ref> context injection"
  # This adapter is what makes context injection work at all: the base
  # PersonaPlex model does not reliably act on text injected mid-stream, and
  # this LoRA was trained to consume exactly the <lookup>/<ref> tags the search
  # path emits. It is applied UNMERGED (QLoRA-style) on top of the 4-bit base.
  mkdir -p "$REF_LORA_DIR/lora"
  hf download Darknsu/helium_lora_v1 \
    adapter_model.safetensors \
    --repo-type dataset \
    --local-dir "$REF_LORA_DIR/lora"
  # The dataset repo publishes only the weights, so the matching config is
  # written here by hand. These are the exact values the adapter was trained
  # and saved with -- if it is ever retrained, update this block (or publish an
  # adapter_config.json alongside the weights) or PEFT will build the wrong
  # shapes and the load will fail.
  cat > "$REF_LORA_DIR/lora/adapter_config.json" <<'JSON'
{
  "alora_invocation_tokens": null,
  "alpha_pattern": {},
  "arrow_config": null,
  "auto_mapping": null,
  "base_model_name_or_path": null,
  "bias": "none",
  "corda_config": null,
  "ensure_weight_tying": false,
  "eva_config": null,
  "exclude_modules": null,
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 256.0,
  "lora_bias": false,
  "lora_dropout": 0.05,
  "lora_ga_config": null,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "peft_version": "0.19.1",
  "qalora_group_size": 16,
  "r": 128,
  "rank_pattern": {},
  "revision": null,
  "target_modules": ["proj", "fc1", "out_proj", "fc2", "linear", "in_proj"],
  "target_parameters": null,
  "task_type": "FEATURE_EXTRACTION",
  "trainable_token_indices": null,
  "use_bdlora": null,
  "use_dora": false,
  "use_qalora": false,
  "use_rslora": false
}
JSON

  echo "[8/$STEPS] Isolated upstream Kyutai moshi package (STT submodel only)"
  # PersonaPlex ships a FORK of moshi under the same top-level import name, and
  # the STT submodel needs the genuine upstream package
  # (moshi.models.loaders.CheckpointInfo, which the fork does not define). Two
  # packages cannot both own sys.modules["moshi"], so the upstream one is
  # installed into its own directory and loaded under a private alias
  # (moshi_stt) by search_helpers.load_upstream_moshi_stt. Installing it into
  # site-packages instead WOULD collide with the fork.
  mkdir -p "$STT_PKG_DIR"
  python -m pip install --no-deps --target "$STT_PKG_DIR" moshi

  [[ -f "$ROOT/assets/ai-thinking-sound.wav" ]] || {
    echo "Missing $ROOT/assets/ai-thinking-sound.wav (played while an online search runs)." >&2
    exit 1
  }
  echo "  reference LoRA: $REF_LORA_DIR/lora"
  echo "  STT package:    $STT_PKG_DIR"
  echo "  thinking sound: $ROOT/assets/ai-thinking-sound.wav"
fi

echo "[$STEPS/$STEPS] Install bundled Moshi and verify deployment"
[[ -f "$PERSONAPLEX_DIR/moshi/pyproject.toml" ]] || {
  echo "PersonaPlex download is missing bundled moshi source." >&2
  exit 1
}
python -m pip install -e "$PERSONAPLEX_DIR/moshi" --no-deps

SPEECH2AVATAR_ROOT="$ROOT" VENV_DIR="$VENV_DIR" \
  "$ROOT/run_imtalker_personaplex.sh" --check-only

if [[ "$WITH_SEARCH" == "1" ]]; then
  SPEECH2AVATAR_ROOT="$ROOT" VENV_DIR="$VENV_DIR" ENABLE_SEARCH=1 \
    "$ROOT/run_imtalker_personaplex.sh" --check-only
fi

echo
echo "Preparation complete. Start the server with:"
echo "  cd $ROOT && bash run_imtalker_personaplex.sh"
if [[ "$WITH_SEARCH" == "1" ]]; then
  echo
  echo "To start it with online search enabled:"
  echo "  cd $ROOT && ENABLE_SEARCH=1 WEB_SEARCH_API_KEY=<tavily-key> bash run_imtalker_personaplex.sh"
fi
echo
echo "Logs are written to $ROOT/logs (override with LOG_DIR):"
echo "  system_<session>.log        models, sources, LoRA paths, video config, startup timing"
echo "  detailed_<session>.log      per-turn report: question, search, summary, injection, reply, timings"
echo "  conversation_<session>.log  the same events as a compact one-line-per-stage trace"
