#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IM="$ROOT/IMTalker"
BNB="$ROOT/checkpoints/personaplex_bnb4"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PORT="${PORT:-8998}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
VOICE_PROMPT="${VOICE_PROMPT:-Robert_5.pt}"
DEFAULT_PROMPT_FILE="$IM/prompts/Robert_8998_default.txt"
TEXT_PROMPT_FILE="${TEXT_PROMPT_FILE:-$DEFAULT_PROMPT_FILE}"
PROMPT_CACHE="${PROMPT_CACHE:-0}"
CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1
if [[ -n "${TEXT_PROMPT:-}" ]]; then
  TEXT_PROMPT_VALUE="$TEXT_PROMPT"
elif [[ -f "$TEXT_PROMPT_FILE" ]]; then
  TEXT_PROMPT_VALUE="$(<"$TEXT_PROMPT_FILE")"
else
  echo "Missing text prompt file: $TEXT_PROMPT_FILE" >&2
  exit 1
fi
case "$PROMPT_CACHE" in
  0|false|no|off) PROMPT_CACHE=0 ;;
  1|true|yes|on) PROMPT_CACHE=1 ;;
  *) echo "PROMPT_CACHE must be 0 or 1." >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------
# Structured logging (always on)
# ---------------------------------------------------------------------------
# Two files per server run, both with millisecond timestamps:
#   system_<session>.log        what the process IS -- every model, the exact
#                               path/repo it came from, LoRA paths and whether
#                               they merged, resolved config, video/render
#                               parameters, GPU state, per-stage startup timing.
#   conversation_<session>.log  what was SAID -- plus detailed_<session>.log,
#   + detailed_<session>.log    a per-turn report covering the question, the
#                               search decision and why, the query, the results,
#                               the summary, exactly what was injected, the
#                               reply, and the duration of every stage.
# Set LOG_DIR="" to keep console-only logging.
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

# ---------------------------------------------------------------------------
# Latency controls
# ---------------------------------------------------------------------------
# MAX_INPUT_BUFFER_SEC is the single most important one. The GPU producer is
# rate-limited to exactly real time by frame_q backpressure, so it consumes one
# second of microphone audio per second of wall clock and can NEVER drain a
# backlog. Without this cap, any transient stall -- model warmup, one slow
# render, a client hiccup -- becomes a permanent, session-long reply delay.
# That is how a pipeline whose floor is ~2-4s ends up answering 10-12s late.
# With the cap, the oldest audio is dropped (and the drop logged) so the delay
# is bounded no matter what happens. Set 0 for the old unbounded behaviour.
# 2.0s, matching the old pipeline. Raising it to 6.0 to absorb chunk-boundary
# stalls was a mistake: the cap is not just a drop threshold, it is the ceiling
# on how STALE the audio the model answers is allowed to be. At 6.0 the model
# can be replying to a question that finished six seconds ago while the user has
# already moved on, and measured first-word latency reached 16.1s and 47.6s.
# Raise it only if you would rather lose the start of a sentence than answer late.
MAX_INPUT_BUFFER_SEC="${MAX_INPUT_BUFFER_SEC:-2.0}"
# 8, matching the old pipeline's proven start_winner_live.sh. This is a VRAM
# setting first and a latency setting second: the renderer's cross-attention at
# resolution 64 allocates batch x 8 heads x 4096 x 4096 x 4 bytes, so 10 frames
# is a single 5.00 GiB block and 8 frames is 4.00 GiB. With the search models
# resident (STT 1B + Qwen 1.5B + LoRA, ~3.4 GB) the 10-frame block no longer fit
# and the render thread died of an OOM, which silenced the avatar for the rest
# of the session. Raise it only if you have measured the headroom.
RENDER_SUB_BATCH="${RENDER_SUB_BATCH:-8}"
# 90 is visually indistinguishable from 82 here but encodes and transfers
# noticeably more slowly. Raise it back to 90 if you prefer the extra headroom.
JPEG_QUALITY="${JPEG_QUALITY:-82}"
# Deliberately left at 1. Setting it to 0 looks like a free latency win but is
# not: prebuffer_ready is what gates _get_media_epoch(), and the media epoch is
# the anchor the audio sender paces against (target_t = epoch + idx * 80ms).
# Releasing it before the first chunk exists anchors that clock seconds ahead of
# the first packet, so every early packet computes a target already in the past
# and the sender burst-sends until the anti-burst spacing catches up. It is also
# a one-time session-start cost, not a per-turn one, so there is nothing to win.
PREBUFFER_CHUNKS="${PREBUFFER_CHUNKS:-1}"
# Opt-in: publish each render sub-batch as soon as it is encoded instead of
# holding the whole 2s chunk. Removes render time from the reply path -- the
# largest remaining win after MAX_INPUT_BUFFER_SEC -- but gives up the atomic
# A/V publication guarantee. Enable only after confirming render time is stable.
INCREMENTAL_PUBLISH="${INCREMENTAL_PUBLISH:-0}"

# ---------------------------------------------------------------------------
# STT + query routing + online search (opt-in, additive)
# ---------------------------------------------------------------------------
# ENABLE_SEARCH=0 (the default) appends no search flags at all and reproduces
# the plain conversational launch exactly: no STT model, no router, no extra
# VRAM, and no per-chunk cost.
#
# With ENABLE_SEARCH=1 the flow is:
#   speech -> STT transcript -> router decides "does this need live data?"
#     no  -> the model answers from its own knowledge, nothing is injected,
#            nothing is played, and the reply starts in a few hundred ms;
#     yes -> thinking sound starts, web search runs, results are summarized
#            into ONE spoken-form sentence, and that sentence is injected as a
#            <ref> block into the live context (the conversation is NOT reset).
ENABLE_SEARCH="${ENABLE_SEARCH:-0}"
SEARCH_ARGS=()
if [[ "$ENABLE_SEARCH" == "1" ]]; then
  # The reference LoRA is what teaches PersonaPlex to consume the injected
  # <lookup>/<ref> tags. Without it the base model does not reliably act on
  # injected context, which is exactly the problem it was trained to solve --
  # so a missing adapter is a hard error here rather than a silent degradation.
  REF_LORA_DIR="${REF_LORA_DIR:-$ROOT/checkpoints/rag_lora}"
  STT_PKG_DIR="${STT_PKG_DIR:-$ROOT/checkpoints/stt}"
  THINKING_SOUND_PATH="${THINKING_SOUND_PATH:-$ROOT/assets/ai-thinking-sound.wav}"
  SEARCH_ARGS=(
    --ref_lora_dir "$REF_LORA_DIR"
    --stt_hf_repo "${STT_HF_REPO:-kyutai/stt-1b-en_fr-candle}"
    --stt_pkg_dir "$STT_PKG_DIR"
    --vad_threshold "${VAD_THRESHOLD:-0.5}"
    # The bundled STT model is English/French only, so a transcript in another
    # script is decode garbage, not a language surprise. Dropping it stops a
    # question nobody asked from being routed and searched.
    --stt_reject_foreign_script "${STT_REJECT_FOREIGN_SCRIPT:-1}"
    --stt_max_non_latin_ratio "${STT_MAX_NON_LATIN_RATIO:-0.15}"
    # Also drop Latin-script transcripts that are not English: the STT model is
    # bilingual and hallucinates fluent Spanish/French on unclear audio.
    --stt_require_english "${STT_REQUIRE_ENGLISH:-1}"
    # Hold the model silent for the whole search instead of only muting its
    # audio. Muting alone lets it compose an invented figure behind the filler
    # and finish that sentence even after the real <ref> arrives.
    --suppress_text_during_search "${SUPPRESS_TEXT_DURING_SEARCH:-1}"
    # Silence appended after the system prompt. Default 0: forcing runs of
    # silence into the model's own text stream biases it toward staying silent
    # and suppresses the opening greeting. Raise only in small steps.
    --prompt_settle_sec "${PROMPT_SETTLE_SEC:-0.0}"
    # ONE small instruct model does double duty: it routes every transcript
    # (search / no search) AND compresses web results into one spoken sentence.
    # Sharing it means routing costs no extra VRAM and no extra load time.
    --compressor_model "${COMPRESSOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    --compressor_device "${COMPRESSOR_DEVICE:-cuda}"
    # Router bias, below 0.5 on purpose: an unnecessary search costs a couple of
    # seconds of thinking sound and is recoverable, while a missed search
    # produces a confidently wrong answer spoken aloud.
    --router_threshold "${ROUTER_THRESHOLD:-0.40}"
    # 1 = run the instant regex pre-pass first, so obvious cases never pay for a
    # model forward pass. This is what keeps the no-search path fast.
    --router_rules "${ROUTER_RULES:-1}"
    --thinking_sound_path "$THINKING_SOUND_PATH"
    # Real search + compression regularly takes 2.5-3.7s end to end, so a ~2s
    # cap discards a correctly-computed answer; 6.0s leaves comfortable margin.
    --search_max_filler_sec "${SEARCH_MAX_FILLER_SEC:-6.0}"
    # Relevance floor. Search engines always return something, so this is the
    # only thing between an unrelated page and the assistant's spoken answer.
    --web_search_min_score "${WEB_SEARCH_MIN_SCORE:-0.15}"
    --max_ref_tokens "${MAX_REF_TOKENS:-250}"
    # STT/VAD runs on its own thread and CUDA stream. Inline (the old placement)
    # put a 1B forward plus two device syncs inside the 80ms real-time budget
    # of the thread that also runs the 7B model and the renderer, which pushed
    # the producer below real time and silenced the avatar. Set STT_INLINE=1
    # only to reproduce that for comparison.
    --stt_queue_frames "${STT_QUEUE_FRAMES:-64}"
  )
  [[ "${STT_INLINE:-0}" == "1" ]] && SEARCH_ARGS+=(--stt_inline)
  # SPOKEN_FORM_NUMBERS=1 spells numbers out in words in the injected summary
  # ("three hundred nine dollars" instead of "$309.32"). OFF by default, and
  # that default is deliberate: PersonaPlex already reads digits aloud
  # correctly, while spelled-out numbers force it to re-encode the value and it
  # drops magnitudes -- $309.32 came back as $39.32 and a euro price came back
  # in dollars. Markdown, brackets and citation markers are stripped either way.
  [[ "${SPOKEN_FORM_NUMBERS:-0}" == "1" ]] && SEARCH_ARGS+=(--spoken_form_numbers)
  # WAIT_FOR_USER=0 lets the model speak before the user has said anything. On
  # by default: otherwise it free-runs from the system prompt at session start
  # and reads fragments of it aloud, and arrives at the first real question
  # already committed to a topic.
  if [[ "${WAIT_FOR_USER:-1}" == "1" ]]; then
    SEARCH_ARGS+=(--wait_for_user)
  else
    SEARCH_ARGS+=(--no-wait_for_user)
  fi
  # The reference LoRA runs UNMERGED, exactly as the old pipeline does -- it
  # ships the same adapter with merge disabled and holds real time on the same
  # GPU. Merging is also impossible against a bnb-4bit base (peft adds a dense
  # delta to a packed quantized blob and raises a shape mismatch), so
  # MERGE_REF_LORA=1 only takes the loud fallback path and ends up unmerged.
  if [[ "${MERGE_REF_LORA:-0}" == "1" ]]; then
    SEARCH_ARGS+=(--merge_ref_lora)
  else
    SEARCH_ARGS+=(--no-merge_ref_lora)
  fi
  # Web search is what "needs live data" resolves to, so default it ON whenever
  # a key is available. Without a key the router still runs and still decides --
  # turns that need live data just fall back to the model's own knowledge.
  if [[ -n "${WEB_SEARCH_API_KEY:-}" ]]; then
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-1}"
  else
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-0}"
  fi
  if [[ "$WEB_SEARCH_ENABLED" == "1" ]]; then
    SEARCH_ARGS+=(
      --web_search_enabled
      --web_search_api_key "${WEB_SEARCH_API_KEY:?set WEB_SEARCH_API_KEY when WEB_SEARCH_ENABLED=1}"
      --web_search_provider "${WEB_SEARCH_PROVIDER:-tavily}"
      --web_search_max_results "${WEB_SEARCH_MAX_RESULTS:-3}"
      --web_search_timeout "${WEB_SEARCH_TIMEOUT:-3.0}"
    )
  else
    echo "[warn] ENABLE_SEARCH=1 but no WEB_SEARCH_API_KEY -- the router will still run," >&2
    echo "       but questions needing live data will fall back to the model's own knowledge." >&2
  fi
fi

EXTRA_ARGS=()
[[ "$INCREMENTAL_PUBLISH" == "1" ]] && EXTRA_ARGS+=(--incremental_chunk_publish)

required=(
 "$VENV_DIR/bin/python"
 "$IM/imtalker_personaplex_try_vad2_8998.py" "$IM/seedvc_runtime.py" "$IM/liveTry.py" "$IM/liveTry_cached.py" "$IM/ws_av_binary_codec.py"
 "$IM/search_helpers.py" "$IM/conversation_logger.py" "$IM/system_logger.py" "$IM/speech_text.py"
 "$IM/experiments/original_pod_8998/FM.py" "$IM/experiments/original_pod_8998/FMT.py"
 "$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html" "$IM/static/assets/robert_idle_10s.mp4" "$IM/static/assets/audio-processor-aj-nodrop.js"
 "$IM/static/assets/decoderWorker.min.js" "$IM/static/assets/decoderWorker.min.wasm" "$IM/static/assets/encoderWorker.min-DpsJ02BN.js"
 "$IM/assets/3robert.jpeg" "$IM/checkpoints/renderer.ckpt" "$IM/checkpoints/wav2vec2-base-960h/config.json"
 "$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt" "$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt"
 "$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt"
 "$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"
 "$BNB/model_bnb_4bit.pt" "$BNB/tokenizer-e351c8d8-checkpoint125.safetensors" "$BNB/tokenizer_spm_32k_3.model" "$BNB/voices/$VOICE_PROMPT"
)
for path in "${required[@]}"; do [[ -e "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }; done

if [[ "$ENABLE_SEARCH" == "1" ]]; then
  for required_search in \
    "$REF_LORA_DIR/lora/adapter_config.json" \
    "$REF_LORA_DIR/lora/adapter_model.safetensors" \
    "$STT_PKG_DIR/moshi/__init__.py" \
    "$THINKING_SOUND_PATH"; do
    [[ -e "$required_search" ]] || {
      echo "Missing required search asset: $required_search" >&2
      echo "Re-run ./prepare_imtalker_personaplex.sh --hf-token TOKEN, or launch with ENABLE_SEARCH=0." >&2
      exit 1
    }
  done
fi

declare -A hashes=(
 ["$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html"]="5cf3981351668e0366b7b4adf2f36c7e43f5ab0c672f6616a343a72817582fa6"
 ["$IM/static/assets/robert_idle_10s.mp4"]="6bdfb847fb3dd2a76d42278a138e26e2729bf5ed938f6733a3b428768a9e7916"
 ["$IM/experiments/original_pod_8998/FM.py"]="8620d6cad2b945276a792a1d63159369654cbb83f9114ab5788f93a3d8daf5d9"
 ["$IM/experiments/original_pod_8998/FMT.py"]="286eb512e710926b0a88d1bc47f14aef5cfc3ef6fc0987fc3cf0d9e7bd004c5d"
 ["$IM/seedvc_runtime.py"]="fe46773af65e010e3d6f41732f0fa1c3e3cf6a8221d9c68718e15561062337f7"
 ["$IM/ws_av_binary_codec.py"]="c090b6a5a076743055f1dd34301662405a28d5cb1636556e9de4c895ddffe4d3"
 ["$BNB/voices/Robert_5.pt"]="a9684503d2a9d37f527341c9a0385b9ed0943eac955b40159bc34f4796563c3d"
 ["$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt"]="000d595124516f6437e218213a31c2ede2350ebfda7bb121a957ef5d52b0e88e"
 ["$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt"]="c9c86d108f81fbdef57e1548ca403b78a68acc32c5a37dab12265d72654f55b9"
 ["$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt"]="e29a41ff004b228d7efee15cad0f32f4d4bc5466563709e2ba78b158d4e340bb"
 ["$IM/checkpoints/renderer.ckpt"]="ca1686c1157b8ef5de43eabdeb846db4612694f5f74012be38742b0871808755"
 ["$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"]="20a6d6eb58608d6d202bac46958e595e243635fdeeb8f04eb1afbe2ac7f2f16d"
)
# NOTE: the thinking sound is deliberately NOT pinned. THINKING_SOUND_PATH is a
# documented override, so replacing the clip is a supported action, and a pinned
# hash would turn it into a launch failure. It is filler audio played while a
# search runs -- it cannot affect a single word the model says -- and SHA256SUMS
# still covers the copy that ships with the repo.
# NOTE: the server sources (imtalker_personaplex_try_vad2_8998.py, liveTry.py,
# liveTry_cached.py, search_helpers.py, conversation_logger.py,
# system_logger.py, speech_text.py) are deliberately NOT pinned here any more.
# They are the files this deployment actively develops, and a pinned hash on a
# file you edit turns every legitimate change into a launch failure. Their
# integrity is covered by SHA256SUMS at the repo level; the entries kept above
# are the immutable model/asset/vendored files, which is what the check was
# actually protecting.
checksum_failures=0
for path in "${!hashes[@]}"; do
  expected="${hashes[$path]}"
  [[ "$expected" == __*__ ]] && continue
  if [[ ! -f "$path" ]]; then
    echo "Checksum check: MISSING  $path" >&2
    echo "    re-run ./prepare_imtalker_personaplex.sh to download it" >&2
    checksum_failures=$((checksum_failures + 1))
    continue
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum check: MODIFIED $path" >&2
    echo "    expected $expected" >&2
    echo "    actual   $actual" >&2
    checksum_failures=$((checksum_failures + 1))
  fi
done
if (( checksum_failures > 0 )); then
  echo "" >&2
  echo "$checksum_failures protected file(s) failed verification. These are model" >&2
  echo "weights and vendored runtime files, so a mismatch means a bad or partial" >&2
  echo "download rather than an edit you made. Re-run:" >&2
  echo "    ./prepare_imtalker_personaplex.sh --hf-token <token>" >&2
  echo "If you replaced one of these on purpose, update its hash in $0." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
python - <<'PY'
import torch, torchaudio, bitsandbytes, aiohttp, av, sphn
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.__version__.startswith("2.8.0+cu128"), torch.__version__
print("CUDA:", torch.cuda.get_device_name(0)); print("Torch:", torch.__version__)
PY
python -m py_compile "$IM/imtalker_personaplex_try_vad2_8998.py" "$IM/seedvc_runtime.py" "$IM/liveTry.py" "$IM/liveTry_cached.py" \
  "$IM/search_helpers.py" "$IM/conversation_logger.py" "$IM/system_logger.py" "$IM/speech_text.py" \
  "$IM/experiments/original_pod_8998/FM.py" "$IM/experiments/original_pod_8998/FMT.py"
echo "Preflight OK: try_vad2, $VOICE_PROMPT, prompt cache=$PROMPT_CACHE, 2.0s/25-step chunks, 50 frames, CFG 1.24, NFE 3, renderer sub-batch $RENDER_SUB_BATCH, FP32, Opus."
echo "  logs=${LOG_DIR:-<console only>}  max_input_buffer=${MAX_INPUT_BUFFER_SEC}s  jpeg_q=$JPEG_QUALITY  prebuffer=$PREBUFFER_CHUNKS  incremental_publish=$INCREMENTAL_PUBLISH"
if [[ "$ENABLE_SEARCH" == "1" ]]; then
  echo "  search=ON ref_lora=$REF_LORA_DIR stt_pkg=$STT_PKG_DIR web_search=${WEB_SEARCH_ENABLED} provider=${WEB_SEARCH_PROVIDER:-tavily} router_threshold=${ROUTER_THRESHOLD:-0.40} thinking_sound=$THINKING_SOUND_PATH"
  echo "  merge_ref_lora=${MERGE_REF_LORA:-0} stt_inline=${STT_INLINE:-0} stt_queue_frames=${STT_QUEUE_FRAMES:-64} compressor_device=${COMPRESSOR_DEVICE:-cuda}"
  echo "  NOTE watch the rtf= value on the [liveTryStudio] lines. Below 1.00 the model"
  echo "       pipeline cannot keep up with real time and the avatar audio will starve."
else
  echo "  search=OFF (set ENABLE_SEARCH=1 with WEB_SEARCH_API_KEY to enable online search)"
fi
[[ "$CHECK_ONLY" -eq 1 ]] && exit 0
if ss -ltnp | grep -q ":${PORT}\\b"; then echo "Port $PORT is occupied:" >&2; ss -ltnp | grep ":${PORT}\\b" >&2; exit 1; fi
mkdir -p "$IM/logs" "$ROOT/pacing_compare/integrated"
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"
cd "$IM"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$IM:$BNB/moshi:$BNB:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export IMTALKER_CACHED_ENGINE="$PROMPT_CACHE" IMTALKER_PROMPT_STATE_CACHE="$PROMPT_CACHE" IMTALKER_TRANSITION_BLEND_FRAMES=0
exec python -u "$IM/imtalker_personaplex_try_vad2_8998.py" \
 --host 0.0.0.0 --port "$PORT" --html_path "$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html" \
 --generator_path "$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt" --renderer_path "$IM/checkpoints/renderer.ckpt" \
 --adapter_path "$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt" \
 --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 --adapter_window_mode lookahead --adapter_future_steps 0 \
 --ref_path "$IM/assets/3robert.jpeg" --wav2vec_model_path "$IM/checkpoints/wav2vec2-base-960h" \
 --moshi_root "$BNB" --mimi_hf_repo nvidia/personaplex-7b-v1 --moshi_weight "$BNB/model_bnb_4bit.pt" \
 --mimi_weight "$BNB/tokenizer-e351c8d8-checkpoint125.safetensors" --tokenizer "$BNB/tokenizer_spm_32k_3.model" \
 --text_prompt "$TEXT_PROMPT_VALUE" \
 --quantize_4bit --voice_prompt "$VOICE_PROMPT" --voice_prompt_dir "$BNB/voices" \
 --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 25 \
 --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 50 --helium_deque_size 25 \
 --prebuffer_chunks "$PREBUFFER_CHUNKS" --render_sub_batch "$RENDER_SUB_BATCH" --renderer_precision fp32 \
 --frame_q_backpressure 32 --buffer_ms 160 --skip_fm_audio_encoder \
 --max_input_buffer_sec "$MAX_INPUT_BUFFER_SEC" --conversation_log_dir "$LOG_DIR" \
 --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
 --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise --fp32 --tf32 \
 --silence_helium_path "$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt" \
 --jpeg_quality "$JPEG_QUALITY" --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
 --blink_motion_path "$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt" --enable_eye_blink_composite \
 "${EXTRA_ARGS[@]}" "${SEARCH_ARGS[@]}"
