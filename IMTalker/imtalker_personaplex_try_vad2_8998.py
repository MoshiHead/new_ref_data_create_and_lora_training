"""PersonaPlex + IMTalker AJ server with split audio/video WebSockets.

Raw PersonaPlex audio is never stored in or cleared with the video queue.
Assistant Opus audio stays on /ws/conversation, while JPEG video frames are
sent through /ws/video?session_id=... to avoid websocket head-of-line blocking.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import json
import os
import queue

import ws_av_binary_codec as _wsbin
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sphn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.original_pod_8998.FM import FMGenerator
from generator.train_lora import apply_lora_to_model
from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.unitalk_wav2vec_adapter import UniTalkLastLayerLiveAdapter
from generator.options.base_options import BaseOptions
from generator.wav2vec2 import Wav2VecModel
if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from liveTry_cached import MoshiOnlyEngine
else:
    from liveTry import MoshiOnlyEngine
from renderer.models import IMTRenderer
from seedvc_runtime import SeedVCStreamingConverter
from speech_text import normalize_for_speech
import system_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24_000          # Mimi sample rate (24kHz)
VIDEO_FPS = 25              # IMTalker frame rate
MIMI_FRAME_SIZE = 1_920     # samples per Mimi frame (80ms @ 24kHz)
MAIN_CODEBOOKS = 8          # codebooks used for Helium input embeddings
PREBUFFER_CHUNKS = 0        # produce this many chunks before sender starts pacing
WAV2VEC_SR = 16_000
TRANSITION_BLEND_FRAMES = max(
    0, int(os.environ.get("IMTALKER_TRANSITION_BLEND_FRAMES", "5"))
)
print(
    f"[TRANSITION-BLEND] configured frames={TRANSITION_BLEND_FRAMES}",
    flush=True,
)


class PlasticityProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(4096)
        self.net = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_ln(x))


class PlasticityUpsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(768, 768, kernel_size=4, stride=4)

    def forward(self, low: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.up(low.transpose(1, 2).contiguous())
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()


class PlasticityCausalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.15,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(768)
        self.ff = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
        )
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(attn)
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x


class PlasticityCausalTransformer(nn.Module):
    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PlasticityCausalBlock() for _ in range(8)])
        self.norm = nn.LayerNorm(768)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].to(device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class StudioNativeLiveAdapter(nn.Module):
    """Frontend fp32 adapter live wrapper.

    Training contract:
      raw 12.5Hz Helium -> Wav2Vec2 projected frontend [T50, 768]
      live contract:
      projected frontend -> frozen Wav2Vec2 encoder -> final hidden -> IMTalker audio_projection.
    """

    def __init__(self, wav2vec_model_path: str, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = HeliumToWav2VecFrontendAdapter(num_layers=int(num_layers), dropout=float(dropout))
        self.wav2vec = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).eval().float()
        for param in self.wav2vec.parameters():
            param.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward_single(self, source: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = source.unsqueeze(0).contiguous()
        target_len = int(target_len)
        frontend_len = max(1, target_len * 2)
        frontend50 = self.model(src.float(), target_len=frontend_len).float()
        final50 = self.wav2vec.encode_from_projected_frontend(frontend50).last_hidden_state.float()
        final25 = F.interpolate(
            final50.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)[0].float().contiguous()
        return frontend50[0].float().contiguous(), final50[0].float().contiguous(), final25


class MoshiOnlyEngineWithHidden(MoshiOnlyEngine):
    """Moshi reply engine that also returns the main LM hidden for each generated step.

    Layer[-2] is exposed as a native LMGen output so Moshi can keep CUDA graph
    replay enabled. We do not use Python forward hooks in this path.
    """

    def __init__(
        self,
        *args,
        capture_layer: int = -2,
        thinking_sound_path: str = "",
        search_max_filler_sec: float = 6.0,
        stt_inline: bool = False,
        stt_queue_frames: int = 64,
        wait_for_user_before_speaking: bool = True,
        **kwargs,
    ) -> None:
        self.tf_capture_layer = int(capture_layer)
        # Both must exist before super().__init__(), which runs reset_session()
        # and can reach _step() during warmup.
        self.wait_for_user_before_speaking = bool(wait_for_user_before_speaking)
        self._user_has_spoken = False
        self._loud_input_frames = 0
        # Must exist before super().__init__() -- it runs reset_session() during
        # warmup, and _step() can be reached from there.
        self.stt_inline = bool(stt_inline)
        self._stt_queue: "queue.Queue | None" = None
        self._stt_thread: threading.Thread | None = None
        self._stt_stop = threading.Event()
        # Serializes STT model access between the worker thread and the
        # session reset on the GPU thread. Uncontended in steady state:
        # only the worker ever takes it during normal operation.
        self._stt_lock = threading.RLock()
        self._stt_dropped_frames = 0
        self._stt_drop_last_log = 0.0
        # Real-time factor accounting. This is THE number that tells you whether
        # the pipeline can hold a live conversation: it is audio seconds
        # consumed divided by wall seconds elapsed. Below 1.0 the producer is
        # falling behind permanently, the fixed-cadence audio sender starves,
        # and the browser hears silence rather than slow speech -- so it needs
        # to be measured continuously, not inferred after the fact.
        self._rt_frames = 0
        self._rt_wall_start = 0.0
        super().__init__(*args, **kwargs)

        # How long a turn may wait for routing + search + compression before
        # giving up and injecting a generic fallback. The old pipeline shipped
        # a fixed ~2.0s cap and its forensic logs showed real searches taking
        # 2.5-3.7s end to end, so that cap fired first and discarded a
        # correctly-computed answer in every logged search turn. 6.0s leaves
        # comfortable margin above the worst latency actually measured.
        self._SEARCH_MAX_FILLER_FRAMES = max(
            1, round(float(search_max_filler_sec) * TARGET_SR / MIMI_FRAME_SIZE)
        )

        # "Thinking sound": played in place of the model's own audio ONLY while
        # an online search is actually running (see _start_thinking_sound and
        # its two call sites). It is deliberately never played on turns the
        # model answers from its own knowledge, nor while the router is still
        # deciding -- in both of those cases the model is about to speak within
        # a few hundred milliseconds, and a filler clip would delay and mask a
        # reply that needed no waiting at all.
        self.thinking_sound_pcm: np.ndarray | None = None
        self._thinking_sound_cursor = 0
        if thinking_sound_path:
            if Path(thinking_sound_path).is_file():
                try:
                    self.thinking_sound_pcm = load_audio_24k(thinking_sound_path)
                    print(
                        f"[liveTryPlasticity][search] thinking sound loaded: {thinking_sound_path} "
                        f"({self.thinking_sound_pcm.shape[0] / TARGET_SR:.2f}s)",
                        flush=True,
                    )
                    self.sys_logger.event(
                        "thinking_sound",
                        f"loaded {thinking_sound_path} "
                        f"({self.thinking_sound_pcm.shape[0] / TARGET_SR:.2f}s, looped while searching)",
                        path=str(thinking_sound_path),
                        duration_s=round(self.thinking_sound_pcm.shape[0] / TARGET_SR, 3),
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[liveTryPlasticity][search] failed to load thinking sound "
                        f"{thinking_sound_path!r}: {e!r}\n{tb}",
                        flush=True,
                    )
                    self.conv_logger.error("thinking_sound_load", e, tb)
                    self.sys_logger.component_failed("thinking_sound", e, tb)
                    self.thinking_sound_pcm = None
            else:
                print(
                    f"[liveTryPlasticity][search] thinking sound path not found: {thinking_sound_path} "
                    f"-- will stay silent during the search instead",
                    flush=True,
                )
                self.sys_logger.path("thinking_sound", thinking_sound_path, required=False)

        self._install_graph_hidden_capture()

        # The worker is NOT started here. See start_stt_worker().
        self._stt_queue_frames = max(8, int(stt_queue_frames))
        if self.stt_lm_gen is not None and self.stt_inline:
            print(
                "[liveTryPlasticity][STT] running INLINE on the GPU thread "
                "(--stt_inline): expect a 1B forward plus two device syncs inside "
                "every 80ms step. Use this only to compare against the threaded path.",
                flush=True,
            )
            self.sys_logger.event("stt_worker", "STT/VAD runs INLINE on the GPU thread", inline=True)

    def start_stt_worker(self) -> None:
        """Start the STT worker thread. Idempotent; safe to call per session.

        Deliberately NOT called from __init__. CUDA graph capture is exclusive
        process-wide -- capture_error_mode is "global", so work submitted by any
        other thread inside the capture window aborts it. The STT graphs are
        captured during single-threaded construction (see _warmup_stt), but the
        FM/renderer engine is constructed lazily on the first websocket
        connection and does its own warmup then. Starting the worker only after
        that point means no background thread is ever running while another
        component is capturing, which removes the whole class of race instead
        of the one instance that was observed."""
        if self.stt_lm_gen is None or self.stt_inline:
            return
        if self._stt_thread is not None and self._stt_thread.is_alive():
            return
        if self.search_hard_disabled:
            return
        self._stt_queue = queue.Queue(maxsize=self._stt_queue_frames)
        self._stt_stop.clear()
        self._stt_thread = threading.Thread(
            target=self._stt_worker_loop, name="stt-vad", daemon=True
        )
        self._stt_thread.start()
        self.sys_logger.event(
            "stt_worker",
            f"STT/VAD worker started on its own thread and CUDA stream "
            f"(queue={self._stt_queue_frames} frames = "
            f"{self._stt_queue_frames * MIMI_FRAME_SIZE / TARGET_SR:.1f}s of audio). "
            f"Keeps a 1B forward and two device syncs off the 80ms real-time budget.",
            queue_frames=self._stt_queue_frames, inline=False,
        )

    def realtime_factor(self) -> float:
        """Audio seconds consumed per wall second, since the session started.

        1.0 means the model pipeline exactly keeps up with the microphone.
        Below 1.0 it is falling behind permanently and cannot recover on its
        own, because the input arrives at exactly real time."""
        if self._rt_wall_start == 0.0:
            return 1.0
        wall = time.perf_counter() - self._rt_wall_start
        if wall <= 0.0:
            return 1.0
        return (self._rt_frames * MIMI_FRAME_SIZE / TARGET_SR) / wall

    def _next_thinking_sound_chunk(self) -> np.ndarray:
        """Next MIMI_FRAME_SIZE samples of the thinking sound, looping
        seamlessly. Advances self._thinking_sound_cursor and
        self._thinking_sound_play_count (incremented every time the clip wraps
        back to its start, i.e. every completed extra play-through)."""
        pcm = self.thinking_sound_pcm
        n = pcm.shape[0]
        out = np.empty(MIMI_FRAME_SIZE, dtype=np.float32)
        pos = 0
        cursor = self._thinking_sound_cursor % n
        while pos < MIMI_FRAME_SIZE:
            take = min(MIMI_FRAME_SIZE - pos, n - cursor)
            out[pos:pos + take] = pcm[cursor:cursor + take]
            pos += take
            new_cursor = (cursor + take) % n
            if new_cursor < cursor or (new_cursor == 0 and take > 0):
                self._thinking_sound_play_count += 1
            cursor = new_cursor
        self._thinking_sound_cursor = cursor
        return out

    def _will_actually_search(self) -> bool:
        """Whether a 'search' decision would really reach the network.

        The thinking sound is gated on this, not merely on the router saying
        'search': with web search disabled or no API key configured, the
        pipeline decides to search, discovers it cannot, and falls straight
        back to the model's own knowledge. Playing a 'searching...' cue for a
        search that never happens would tell the user something untrue."""
        return bool(self.web_search_enabled and self.web_search_api_key)

    def _start_thinking_sound(self, turn_id, transcript: str = "") -> None:
        """Begin the thinking sound. MUST run on the GPU thread -- it mutates
        the playback state that _step() reads every 80ms.

        Idempotent: the rules path starts the sound synchronously the instant
        it commits to searching, while the model-router path signals from the
        background thread, and both can land for the same turn."""
        # Withhold the model's words for the whole search, whether or not the
        # filler clip is available -- these are independent concerns (one is
        # what the user hears, the other is what the model commits to).
        if self.suppress_text_during_search:
            self.suppress_text_until_ref = True
        if self.thinking_sound_pcm is None or self.search_thinking_active:
            return
        self.search_thinking_active = True
        self._thinking_sound_started_at = time.perf_counter()
        self._thinking_sound_play_count = 1
        self.conv_logger.event("thinking_sound_start", transcript=transcript)
        self.conv_logger.narrate_thinking_start(turn_id)

    def _stop_thinking_sound(self, turn_id, reason: str, reason_text: str) -> None:
        """Shared stop logic for both the real-ref-ready and filler-timeout
        paths in _consume_pending -- keeps duration/loop-count bookkeeping in
        one place."""
        if not self.search_thinking_active:
            return
        self.search_thinking_active = False
        duration_s = max(0.0, time.perf_counter() - self._thinking_sound_started_at)
        clip_duration_s = (
            self.thinking_sound_pcm.shape[0] / TARGET_SR
            if self.thinking_sound_pcm is not None else 0.0
        )
        self.conv_logger.event(
            "thinking_sound_stop", reason=reason, duration_s=round(duration_s, 2),
            play_count=self._thinking_sound_play_count,
        )
        self.conv_logger.narrate_thinking_stop(
            turn_id, reason_text, duration_s, self._thinking_sound_play_count, clip_duration_s
        )
        self.conv_logger.record(
            turn_id, thinking_sound_s=round(duration_s, 3),
            thinking_sound_plays=self._thinking_sound_play_count,
        )

    def _install_graph_hidden_capture(self) -> None:
        lm_model = self.lm
        lm_gen = self.lm_gen
        if hasattr(lm_gen, "prepare_step_input") and hasattr(lm_gen, "process_transformer_output"):
            @torch.no_grad()
            def personaplex_step_with_hidden(
                self_gen,
                input_tokens: torch.Tensor = None,
                moshi_tokens: torch.Tensor = None,
                text_token: torch.Tensor = None,
                depformer_replace_tokens: torch.Tensor | None = None,
            ):
                prepared = self_gen.prepare_step_input(input_tokens, moshi_tokens, text_token)
                if prepared is None:
                    return None
                input_, provided_, target_, model_input_position, target_position = prepared
                state = self_gen._streaming_state
                transformer_out, text_logits = state.graphed_main(input_)
                output = self_gen.process_transformer_output(
                    transformer_out,
                    text_logits,
                    provided_,
                    target_,
                    model_input_position,
                    target_position,
                )
                return output, transformer_out, transformer_out

            lm_gen._step = types.MethodType(personaplex_step_with_hidden, lm_gen)
            # This override forwards text_token to prepare_step_input, and
            # process_transformer_output honors a provided token instead of
            # sampling one. That is what lets _step() force the model silent
            # while a search is in flight (see _step / suppress_text_until_ref).
            self._step_supports_text_token = True
            lm_gen.streaming_forever(1)
            self._warmup_runtime()
            print("[liveTryPlasticity] installed PersonaPlex graphed hidden capture", flush=True)
            return

        from moshi.models.lm import scatter_with_mask_
        from moshi.modules.transformer import create_sin_embedding
        from moshi.utils.sampling import sample_token

        capture_layer = int(self.tf_capture_layer) % len(lm_model.transformer.layers)

        old_state = getattr(lm_gen, "_streaming_state", None)
        if old_state is not None:
            with contextlib.suppress(Exception):
                old_state.__exit__(None, None, None)
            with contextlib.suppress(Exception):
                lm_gen._stop_streaming()

        def forward_text_with_layer(self_lm, sequence, sum_condition=None, cross_attention_src=None):
            B, K, S = sequence.shape
            assert K == self_lm.num_codebooks, (K, self_lm.num_codebooks)
            input_sequence = sequence
            input_ = None
            for cb_index in range(self_lm.num_audio_codebooks):
                audio_emb = self_lm.emb[cb_index](input_sequence[:, cb_index + self_lm.audio_offset])
                input_ = audio_emb if input_ is None else input_ + audio_emb
            text_emb = self_lm.text_emb(input_sequence[:, 0])
            input_ = text_emb if input_ is None else input_ + text_emb
            if sum_condition is not None:
                input_ = input_ + sum_condition.to(input_)
            if cross_attention_src is not None:
                cross_attention_src = cross_attention_src.to(input_)

            transformer = self_lm.transformer
            _, T, C = input_.shape
            dtype_input = input_.dtype
            state = transformer._streaming_state
            if state is None:
                offsets = torch.zeros(1, dtype=torch.long, device=input_.device)
            else:
                offsets = state.offsets

            x = input_
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                positions = positions + offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=x.dtype)
                x = x + transformer.positional_scale * pos_emb

            captured = x
            for idx, layer in enumerate(transformer.layers):
                x = layer(x, cross_attention_src=cross_attention_src)
                if idx == capture_layer:
                    captured = x

            if state is not None:
                state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)

            transformer_out = x.to(dtype_input)
            layer_hidden = captured.to(dtype_input)
            if self_lm.out_norm:
                transformer_out = self_lm.out_norm(transformer_out)
            text_logits = self_lm.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            return transformer_out, text_logits, layer_hidden

        @torch.no_grad()
        def step_with_layer(self_gen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None):
            state = self_gen._streaming_state
            if state is None:
                raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
            lm_model_local = self_gen.lm_model

            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
            assert S == 1, "Only support being given steps one by one."
            needed_tokens = lm_model_local.num_codebooks - lm_model_local.dep_q - 1
            assert Ki >= needed_tokens, f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
            if Ki > needed_tokens:
                input_tokens = input_tokens[:, :needed_tokens, :]

            CT = state.cache.shape[2]
            delays = self_gen.delays_cuda[lm_model_local.dep_q + 1:]
            write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
            scatter_with_mask_(state.cache[:, lm_model_local.dep_q + 1:], -1, write_positions, input_tokens, state.exec_mask[:, None, None])

            is_init = state.offsets[:, None, None] <= self_gen.delays_cuda[:, None]
            is_init |= ~state.exec_mask[:, None, None]
            positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
            input_ = state.cache.gather(dim=2, index=positions)
            input_ = torch.where(is_init, state.initial, input_)

            if self_gen.check:
                assert not (input_ == lm_model_local.ungenerated_token_id).any(), (state.offsets, input_)
                assert (input_[:, lm_model_local.audio_offset:] <= lm_model_local.card).all(), input_
                assert (input_[:, :1] <= lm_model_local.text_card).all()

            zero = torch.full((1,), lm_model_local.zero_token_id, dtype=torch.long, device=input_.device)
            if self_gen.cfg_coef != 1.:
                if state.cfg_is_masked_until is not None:
                    limit = self_gen.delays_cuda[:, None] + state.cfg_is_masked_until.view(-1, 1, 1)
                    is_zeroed = state.offsets[:, None, None] <= limit
                    masked = torch.where(is_zeroed & ~is_init, zero, input_)
                    input_ = torch.cat([input_, masked], dim=0)
                else:
                    input_ = input_.repeat(2, 1, 1)
                if self_gen.cfg_is_no_text:
                    input_[B:, :1] = torch.where(~is_init[:, :1], zero, input_[B:, :1])

            transformer_out, text_logits, layer_hidden = state.graphed_main(input_, state.condition_sum, state.condition_cross)
            if self_gen.cfg_coef != 1.:
                logits, logits_null = text_logits.chunk(2)
                if self_gen.cfg_is_no_text:
                    text_logits = logits
                    layer_hidden = layer_hidden[:B]
                else:
                    text_logits = logits_null + (logits - logits_null) * self_gen.cfg_coef
                    layer_hidden = layer_hidden[:B]

            if self_gen.on_text_logits_hook:
                self_gen.on_text_logits_hook(text_logits)
            text_token = sample_token(text_logits.float(), self_gen.use_sampling, self_gen.temp_text, self_gen.top_k_text)
            assert text_token.dim() == 3, text_token.shape
            assert text_token.shape[2] == 1
            assert text_token.shape[1] == 1, "Only one text stream supported."
            text_token = text_token[:, 0, 0]
            if self_gen.on_text_hook is not None:
                self_gen.on_text_hook(text_token)

            if state.graphed_depth is None:
                audio_tokens = None
            else:
                if depformer_replace_tokens is None:
                    audio_tokens = state.graphed_depth(text_token, transformer_out)
                else:
                    assert depformer_replace_tokens.dim() == 3
                    audio_tokens = depformer_replace_tokens.squeeze(-1)
                if self_gen.on_audio_hook is not None:
                    self_gen.on_audio_hook(audio_tokens)

            state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
            state.offset_cpu += 1
            positions = (state.offsets % CT)[:, None, None]
            scatter_with_mask_(state.cache[:, :1], -1, positions, text_token[:, None, None], state.exec_mask[:, None, None])
            if audio_tokens is not None:
                audio_tokens = audio_tokens[:, :, None]
                scatter_with_mask_(state.cache[:, 1: lm_model_local.dep_q + 1, :], -1, positions.expand_as(audio_tokens), audio_tokens, state.exec_mask[:, None, None])

            if not self_gen.support_out_of_sync and state.offset_cpu <= self_gen.max_delay:
                return None
            gen_delays_cuda = self_gen.delays_cuda[: lm_model_local.dep_q + 1]
            index = (state.offsets[:, None, None] - self_gen.max_delay + gen_delays_cuda[:, None]) % CT
            out = state.cache.gather(dim=2, index=index)
            mask = (state.offsets <= self_gen.max_delay) | ~state.exec_mask
            out[mask, :, :] = lm_model_local.ungenerated_token_id
            return out, transformer_out, layer_hidden

        lm_model.forward_text = types.MethodType(forward_text_with_layer, lm_model)
        lm_gen._step = types.MethodType(step_with_layer, lm_gen)
        # step_with_layer takes only input_tokens, so text cannot be forced on
        # this fallback path; search-window text suppression degrades to a
        # no-op here rather than raising a TypeError mid-conversation.
        self._step_supports_text_token = False
        lm_gen.streaming_forever(1)
        self._warmup_runtime()
        print(f"[liveTryPlasticity] installed graphed layer capture layer={self.tf_capture_layer}", flush=True)


    # -- Turn-detection constants (frame = one MIMI_FRAME_SIZE / 80ms chunk,
    # the same granularity as _step() itself) --
    _VAD_SILENCE_FRAMES_REQUIRED = 12   # ~960ms of silence ends an utterance
    # Class-level fallback only -- __init__ always overrides this with an
    # instance attribute derived from --search_max_filler_sec. Kept so the
    # attribute exists with a sane value on any path that reads it before
    # __init__ finishes.
    _SEARCH_MAX_FILLER_FRAMES = 25
    # RMS at which the model's own output counts as "it has started speaking".
    # Mirrors the default --assistant_speech_rms_threshold used by the avatar
    # gate; used only for the time-to-first-word metric.
    _SPEECH_RMS_THRESHOLD = 0.006
    # Conservative default; _install_graph_hidden_capture() sets the real value
    # for whichever step override it installed.
    _step_supports_text_token = False

    def reset_session(self) -> None:
        super().reset_session()
        # Per-conversation STT/turn-detection state. All of this no-ops
        # harmlessly when STT is not configured (self.stt_lm_gen stays None).
        self.stt_token_buffer: list = []
        self.stt_in_utterance = False
        self.stt_silence_frame_count = 0
        self.stt_last_vad_end = False
        # Set by the STT worker thread when it detects end-of-utterance; drained
        # by the GPU thread in _step(). Same cross-thread handoff pattern as
        # pending_ref_tokens: the worker never touches turn state or the LM, it
        # only publishes a finished transcript.
        self.pending_transcript: tuple[str, list] | None = None
        self.search_turn_epoch = 0
        self.search_ref_committed_this_turn = False
        self.search_awaiting_ref = False
        # Set True if an uncaught exception ever escapes the per-chunk STT /
        # search hook in _step(). Without that guard such an exception
        # propagates out of _step() and kills the whole GPU producer thread:
        # audio and video stop forever, and because the thread that would have
        # logged it is the one that died, the logs simply end mid-turn with no
        # error line. Once set, search is skipped for the rest of THIS session
        # while the underlying avatar conversation keeps working.
        self.search_hard_disabled = False
        # Per-turn stage-timing accumulator, see _start_turn / _log_timing_summary.
        self._turn_timing_start: float | None = None
        self._turn_timing_stages: dict[str, float] = {}
        # Cross-thread handoff slots. The background thread WRITES these; the
        # GPU thread POLLS them once per 80ms chunk in _consume_pending() and
        # is the only thread allowed to touch self.lm_gen. `pending_lookup` is
        # separate from `pending_ref` because the <lookup> filler is injected
        # only AFTER the router has committed to a search -- telling the model
        # to "please wait" and then not searching leaves it stalling for an
        # answer that never arrives.
        self.pending_lookup_tokens: list | None = None
        self.pending_ref_tokens: list | None = None
        self.pending_search_cancelled = False
        # Set by the background thread the moment it is about to hit the search
        # API; the GPU thread turns it into an actual _start_thinking_sound()
        # call on its next chunk. Needed because the model-router path only
        # learns that a search is happening after its forward pass, off-thread.
        self.pending_start_thinking = False
        # While True, _step() forces the model's text stream to its own
        # zero_text_code so it composes NOTHING during a search. Muting the
        # outgoing audio alone was never enough: the model keeps generating
        # text behind the filler, so by the time the <ref> arrives seconds
        # later it is already mid-sentence with an invented figure and simply
        # finishes it.
        self.suppress_text_until_ref = False
        # Cleared for every new session: a fresh conversation has to earn the
        # model's first word again.
        self._user_has_spoken = False
        self._loud_input_frames = 0
        self._pending_ref_token_counts = (0, 0)
        self.search_filler_frame_count = 0
        self.search_session_history: list[tuple[str, str]] = []
        self.search_current_transcript = ""
        # Snapshot of len(self.audio_text) at turn start (when the user STOPPED
        # speaking), marking where this turn's assistant response begins.
        self._turn_start_audio_text_len = 0
        # Snapshot of len(self.audio_text) at the moment the user STARTS
        # speaking again, marking where this turn's response ends. Both bounds
        # are needed: the model is full-duplex and keeps emitting text while
        # the user talks, so slicing to the end of audio_text pulls the NEXT
        # turn's opening words into the previous turn's logged response.
        self._utterance_start_audio_text_len = 0
        # True time-to-first-spoken-word tracking. This is the ONLY honest
        # latency metric for a full-duplex model: the SAID line cannot serve as
        # one, because it fires when the NEXT utterance ends, so its timestamp
        # measures how long the user waited before speaking again rather than
        # how long the assistant took.
        self._turn_awaiting_first_speech = False
        self._turn_first_speech_epoch = 0
        self.search_thinking_active = False
        self._thinking_sound_cursor = 0
        self._thinking_sound_started_at = 0.0
        self._thinking_sound_play_count = 0
        self._rt_frames = 0
        self._rt_wall_start = 0.0
        self._rt_last_warn = 0.0
        # Drop any frames the worker had not consumed yet and re-reset the STT
        # streams under the lock. super().reset_session() already reset them,
        # but it did so without the lock and possibly while the worker was
        # mid-forward, so this is the reset that actually sticks.
        stt_queue = getattr(self, "_stt_queue", None)
        if stt_queue is not None:
            drained = 0
            while True:
                try:
                    stt_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained:
                print(
                    f"[liveTryPlasticity][STT] dropped {drained} queued frame(s) on session reset",
                    flush=True,
                )
        if getattr(self, "stt_lm_gen", None) is not None:
            with self._stt_lock:
                with contextlib.suppress(Exception):
                    self.stt_lm_gen.reset_streaming()
                with contextlib.suppress(Exception):
                    self.stt_mimi.reset_streaming()
        # Guards the time-to-first-word metric: the model is full-duplex and is
        # usually still finishing the PREVIOUS answer when a turn boundary
        # lands, so the first loud frame after _start_turn is not this turn's
        # reply. Requiring a silent gap first is what makes the number mean
        # "the assistant started answering" instead of "the assistant was
        # already talking".
        self._turn_saw_silence_after_start = False
        self._turn_first_speech_deadline = 0

    def _inject_tokens(self, tokens: list[int]) -> None:
        """Force-feed text tokens via the PUBLIC lm_gen.step() (not the
        hidden-capturing `_step` installed above). reset_streaming() is never
        called: the shared KV cache -- and therefore the conversation heard so
        far -- stays intact across the injection, which is the whole point.
        This is context injection, not a context reset."""
        for tok in tokens:
            self.lm_gen.step(
                moshi_tokens=self.lm_gen._encode_zero_frame(),
                text_token=tok,
                input_tokens=self.lm_gen._encode_sine_frame(),
            )

    def _stt_submit(self, chunk: torch.Tensor) -> None:
        """GPU thread: hand this 80ms frame to the STT worker. Never blocks.

        The frame is queued together with `len(self.audio_text)` AS OF NOW.
        That pairing is the whole reason this is correct: the worker reaches
        this frame some time later, and if it read audio_text then, it would
        read the GPU thread's position at that later moment instead of the
        position this frame belongs to. The boundary would land past the start
        of the model's reply, and the reply recorded for the turn would begin
        mid-sentence -- which is exactly what the transcripts showed.

        `chunk` is freshly allocated by _step() every call and is never mutated
        afterwards, so it can be handed over by reference -- no copy, no extra
        device traffic on the critical path.

        On overflow the OLDEST frame is dropped rather than the newest: if STT
        has fallen behind, the useful thing to keep is the most recent speech.
        Dropping here degrades transcription only; it must never be allowed to
        stall the thread that generates audio and video."""
        if self._stt_queue is None:
            return
        item = (chunk, len(self.audio_text))
        try:
            self._stt_queue.put_nowait(item)
            return
        except queue.Full:
            pass
        with contextlib.suppress(queue.Empty):
            self._stt_queue.get_nowait()
        try:
            self._stt_queue.put_nowait(item)
        except queue.Full:
            pass
        self._stt_dropped_frames += 1
        now = time.perf_counter()
        if now - self._stt_drop_last_log >= 5.0:
            self._stt_drop_last_log = now
            print(
                f"[liveTryPlasticity][STT] worker is behind: dropped "
                f"{self._stt_dropped_frames} frames "
                f"({self._stt_dropped_frames * MIMI_FRAME_SIZE / TARGET_SR:.1f}s of audio) "
                f"this session. Transcripts may be clipped; the avatar is unaffected.",
                flush=True,
            )
            self.conv_logger.event(
                "stt_frames_dropped",
                f"total={self._stt_dropped_frames}",
                dropped_frames=self._stt_dropped_frames,
            )

    def _stt_worker_loop(self) -> None:
        """Dedicated STT thread. Runs on its own CUDA stream so its per-frame
        device syncs (the VAD score and the token buffer both come back to the
        host) do not serialize the renderer against PersonaPlex."""
        stream = None
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                stream = torch.cuda.Stream()
        print(
            f"[liveTryPlasticity][STT] worker thread started "
            f"(own CUDA stream={stream is not None})",
            flush=True,
        )
        while not self._stt_stop.is_set():
            try:
                item = self._stt_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break
            chunk, audio_text_len = item
            try:
                if stream is not None:
                    with torch.cuda.stream(stream):
                        self._stt_forward(chunk, audio_text_len)
                    stream.synchronize()
                else:
                    self._stt_forward(chunk, audio_text_len)
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][STT] worker failed, disabling search for the "
                    f"rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("stt_worker", e, tb)
                self.sys_logger.component_failed("stt_worker (runtime)", e, tb)
                self.search_hard_disabled = True
                break
        print("[liveTryPlasticity][STT] worker thread stopped", flush=True)

    def _stt_forward(self, chunk: torch.Tensor, audio_text_len: int | None = None) -> None:
        """Run the STT/VAD submodel one 80ms frame forward and, on a detected
        end-of-utterance, publish the transcript for the GPU thread to act on.

        Runs on the STT worker thread. Touches only STT state plus two plain
        attributes shared with the GPU thread (`pending_transcript`, written
        here and read there; `search_awaiting_ref`, the reverse) -- never
        self.lm_gen, self.mimi, or any turn bookkeeping."""
        if audio_text_len is None:
            # Inline mode: this frame IS the current one, so now is the right
            # moment to read the boundary.
            audio_text_len = len(self.audio_text)
        with self._stt_lock:
            self._stt_forward_locked(chunk, audio_text_len)

    def _stt_forward_locked(self, chunk: torch.Tensor, audio_text_len: int) -> None:
        stt_codes = self.stt_mimi.encode(chunk)
        stt_result = self.stt_lm_gen.step_with_extra_heads(stt_codes)
        if stt_result is None:
            return
        stt_tokens, vad_heads = stt_result
        vad_score = 0.0
        if vad_heads and len(vad_heads) > 2:
            vad_score = float(vad_heads[2][0, 0, 0].cpu().item())
        if stt_tokens is not None:
            self.stt_token_buffer.append(stt_tokens[:, :1, :].cpu())
            text_token = stt_tokens[0, 0, 0].item()
            if text_token not in (0, self.stt_padding_token_id):
                if not self._user_has_spoken:
                    self._user_has_spoken = True
                    print(
                        "[liveTryPlasticity][search] first user speech detected -- "
                        "the model may speak from here on",
                        flush=True,
                    )
                if not self.stt_in_utterance:
                    # First real word of a new user utterance: everything the
                    # model emitted before this point belongs to the PREVIOUS
                    # turn's response, everything after is it reacting to what
                    # it is hearing now. Use the position captured when this
                    # frame was SUBMITTED, not the current one -- see
                    # _stt_submit.
                    self._utterance_start_audio_text_len = audio_text_len
                self.stt_in_utterance = True
                self.stt_last_vad_end = False
        if vad_score > self.vad_threshold:
            self.stt_silence_frame_count += 1
        else:
            self.stt_silence_frame_count = 0
        vad_fired = (
            self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
            and not self.stt_last_vad_end
        )
        self.stt_last_vad_end = self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
        if not (vad_fired and self.stt_in_utterance and self.stt_token_buffer and not self.search_awaiting_ref):
            return

        import search_helpers

        transcript, transcript_token_ids = search_helpers.decode_stt_tokens_with_ids(
            self.stt_token_buffer, self.stt_tokenizer, self.stt_padding_token_id
        )
        self.stt_token_buffer = []
        self.stt_in_utterance = False
        self.stt_silence_frame_count = 0
        if not transcript.strip():
            return
        # Reset the STT streams here, on the worker's own thread, so the next
        # utterance starts clean without the GPU thread ever touching them.
        with contextlib.suppress(Exception):
            self.stt_lm_gen.reset_streaming()
            self.stt_mimi.reset_streaming()
        # Hand off. Everything below this line runs on the GPU thread, because
        # it reads audio_text boundaries and starts a turn.
        self.pending_transcript = (transcript, transcript_token_ids)

    def _consume_pending_transcript(self) -> None:
        """GPU thread: act on a transcript the STT worker finished.

        This is the second half of the old inline _stt_step -- the half that
        must stay on the GPU thread because it reads the audio_text boundaries
        that delimit the previous turn's reply and then starts a new turn."""
        pending = self.pending_transcript
        if pending is None:
            return
        self.pending_transcript = None
        transcript, transcript_token_ids = pending

        import search_helpers

        # Close out the PREVIOUS turn's response now that we know it finished
        # (the user has started speaking again). Slice between BOTH turn
        # boundaries: from where that turn's answer began, to where this new
        # utterance's first word arrived.
        resp_end = self._utterance_start_audio_text_len
        if resp_end <= self._turn_start_audio_text_len:
            resp_end = len(self.audio_text)
        prev_response = search_helpers.strip_injected_tags(
            self.audio_text[self._turn_start_audio_text_len:resp_end]
        )
        # Only a turn that actually began has a response to close out. Without
        # this, the very first utterance reports whatever the model composed
        # before the conversation started as "turn 0", attributed to an empty
        # question -- which is how prompt fragments ended up in the log as if
        # they were an answer.
        if prev_response and self.search_current_transcript:
            self.conv_logger.turn_replied(self.search_turn_epoch, prev_response)
            self.conv_logger.assistant_response(self.search_current_transcript, prev_response)
            self.conv_logger.narrate_response(
                self.search_turn_epoch, self.search_current_transcript, prev_response
            )
            # The response is the last fact a turn is missing, so the complete
            # per-turn report can only be written here.
            self.conv_logger.record(self.search_turn_epoch, response=prev_response)
            self.conv_logger.turn_report(
                self.search_turn_epoch,
                total_s=(
                    time.perf_counter() - self._turn_timing_start
                    if self._turn_timing_start else 0.0
                ),
                stages=dict(self._turn_timing_stages),
            )
            if self.search_current_transcript:
                self.search_session_history.append((self.search_current_transcript, prev_response))
                self.search_session_history = self.search_session_history[-6:]
        elif self.search_current_transcript:
            # The model produced zero spoken text for the previous turn before
            # the user spoke again. Flagged explicitly: silently missing
            # responses are indistinguishable from a turn that never happened.
            turn_start_perf = self._turn_timing_start
            gap_s = max(0.0, time.perf_counter() - turn_start_perf) if turn_start_perf else 0.0
            self.conv_logger.no_response_warning(
                self.search_turn_epoch, self.search_current_transcript, gap_s
            )
            self.conv_logger.narrate_no_response_warning(
                self.search_turn_epoch, self.search_current_transcript, gap_s
            )

        # -- Transcript sanity gate -----------------------------------------
        # Two independent checks, both cheap enough for _step()'s 80ms budget:
        #   1. writing system -- the STT submodel cannot emit Devanagari or
        #      Cyrillic, so those are decode garbage;
        #   2. language -- the model IS bilingual (en/fr) and on unclear audio
        #      hallucinates fluent Spanish/French in ordinary Latin letters,
        #      which check (1) cannot see.
        # Routing on either would answer a question nobody asked and can send
        # nonsense to a paid search API, so the turn is dropped here -- before
        # the router, before any injection. The avatar is unaffected:
        # PersonaPlex heard the real audio directly and keeps replying.
        usable, script_stats = search_helpers.check_transcript_usable(
            transcript, self.stt_max_non_latin_ratio, require_english=self.stt_require_english,
        )
        if not usable and self.stt_reject_foreign_script:
            self.conv_logger.turn_transcript_rejected(
                self.search_turn_epoch + 1, transcript, script_stats, transcript_token_ids
            )
            hint = ""
            if script_stats.get("kind") == "script":
                hint = (
                    f" first ids={transcript_token_ids[:24]} -- if these look like ordinary "
                    f"ids, the STT tokenizer is likely mismatched; compare the "
                    f"'stt tokenizer=' line printed at startup."
                )
            print(
                f"[liveTryPlasticity][STT] rejected transcript "
                f"({script_stats.get('reason', 'unusable')}, "
                f"{len(transcript_token_ids)} tok): {transcript[:120]!r}{hint}",
                flush=True,
            )
            return

        self.conv_logger.turn_heard(
            self.search_turn_epoch + 1, transcript, transcript_token_ids, script_stats
        )
        self.conv_logger.narrate_user_message(self.search_turn_epoch + 1, transcript)
        self._start_turn(transcript)

    def _begin_casual_turn(self, transcript: str, reason: str) -> None:
        """Mark a turn that will be answered from the model's own knowledge:
        no search, no injection, nothing added to the context. Only turn
        bookkeeping.

        This is the LOW-LATENCY path, and keeping it free of work is the whole
        point: the model is already generating, so the reply starts within a
        few hundred milliseconds. No thinking sound plays here by design --
        nothing is being waited on, and a filler clip would only delay and mask
        a reply that never needed covering."""
        self.search_turn_epoch += 1
        self.search_current_transcript = transcript
        self._turn_start_audio_text_len = len(self.audio_text)
        self._turn_awaiting_first_speech = True
        self._turn_first_speech_epoch = self.search_turn_epoch
        self._turn_saw_silence_after_start = False
        # ~4s of frames: long enough for the previous answer to finish, short
        # enough that the metric is still recorded if it never does.
        self._turn_first_speech_deadline = self.step + 50
        print(
            f"[liveTryPlasticity][search] no search ({reason}) -- "
            f"answering from the model's own knowledge",
            flush=True,
        )

    def _start_turn(self, transcript: str) -> None:
        """Decide how to answer this utterance.

        Tier 0 (here, on the GPU thread): pure-regex routing, microseconds. A
        confident rule verdict either starts the search immediately or ends the
        turn as a casual one, with no model call at all -- which is what keeps
        the no-search path as fast as a plain conversational server.

        Tier 1 (background thread): anything the rules could not resolve is
        scored by the router in `_route_and_search`. That call is tens of
        milliseconds -- far too slow for _step()'s 80ms budget, which is why it
        never runs here.

        <lookup> is deliberately NOT injected up front: telling the model to
        "please wait" and then deciding not to search would leave it stalling
        for an answer that never arrives."""
        import search_helpers

        # Per-turn stage-timing accumulator. Safe as a plain instance
        # attribute: search_awaiting_ref gates re-entry, so only one turn's
        # timing is ever in flight at once.
        self._turn_timing_start = time.perf_counter()
        self._turn_timing_stages = {}

        if not getattr(self, "search_enabled", False):
            self._begin_casual_turn(transcript, "search not configured")
            self.conv_logger.record(
                self.search_turn_epoch, transcript=transcript, needs_search=False,
                router_source="disabled", router_score=0.0,
                router_reason="STT/router/search not configured for this run",
            )
            return

        t_rules0 = time.perf_counter()
        ruled, rule_reason = search_helpers.rule_route_explain(transcript)
        rules_elapsed = time.perf_counter() - t_rules0
        self._turn_timing_stages["rule_route"] = rules_elapsed
        self.conv_logger.quick_gate_timing(rules_elapsed)

        if ruled is False:
            self.conv_logger.turn_decision(
                self.search_turn_epoch + 1, transcript, needs_search=False, source="rules",
                score=0.0, elapsed_s=rules_elapsed, reason=rule_reason,
            )
            self.conv_logger.narrate_router_decision(
                self.search_turn_epoch + 1, transcript, False, "rules", 0.0, rule_reason,
            )
            self._begin_casual_turn(transcript, "rule: static phrase")
            self.conv_logger.record(
                self.search_turn_epoch, transcript=transcript, needs_search=False,
                router_source="rules", router_score=0.0, router_reason=rule_reason,
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "answered from own knowledge",
                time.perf_counter() - self._turn_timing_start,
            )
            return

        # Either the rules demanded a search (ruled is True) or they could not
        # decide (None) and the router will settle it on the background thread.
        self.search_turn_epoch += 1
        my_epoch = self.search_turn_epoch
        self.search_ref_committed_this_turn = False
        self.search_awaiting_ref = True
        self.search_filler_frame_count = 0
        self.search_current_transcript = transcript
        self.pending_lookup_tokens = None
        self.pending_ref_tokens = None
        self.pending_search_cancelled = False
        self._turn_start_audio_text_len = len(self.audio_text)
        self._turn_awaiting_first_speech = True
        self._turn_first_speech_epoch = my_epoch
        self._turn_saw_silence_after_start = False
        # ~4s of frames: long enough for the previous answer to finish, short
        # enough that the metric is still recorded if it never does.
        self._turn_first_speech_deadline = self.step + 50
        self.conv_logger.record(my_epoch, transcript=transcript)

        # The thinking sound is NOT started here for the undecided case. Only
        # the rules path knows synchronously that a search is happening; when
        # the model router has to decide, that verdict arrives on the
        # background thread and may well be "no search", in which case no sound
        # should ever have played.
        if ruled is True:
            if self._will_actually_search():
                self._start_thinking_sound(my_epoch, transcript)
            self.conv_logger.turn_decision(
                my_epoch, transcript, needs_search=True, source="rules", score=1.0,
                elapsed_s=rules_elapsed, reason=rule_reason,
            )
            self.conv_logger.narrate_router_decision(
                my_epoch, transcript, True, "rules", 1.0, rule_reason,
            )
            self.conv_logger.record(
                my_epoch, needs_search=True, router_source="rules", router_score=1.0,
                router_reason=rule_reason,
            )

        threading.Thread(
            target=self._route_and_search,
            args=(transcript, my_epoch, ruled is True),
            daemon=True,
            name="query-search",
        ).start()

    def _route_and_search(self, transcript: str, my_epoch: int, rules_said_search: bool) -> None:
        """Background thread: (optionally) run the model router, then -- only
        if a search is warranted -- web search, clean, and compress into one
        short grounding statement. Only touches the router/compressor objects
        and plain attributes; never self.lm_gen / self.mimi / self.stt_lm_gen.
        Results are handed to the GPU thread through self.pending_* slots,
        which _consume_pending() polls once per chunk."""
        import search_helpers

        try:
            # --- Tier 1: the model-based decision, unless the rules already
            # committed to searching (in which case skip the forward pass). ---
            if not rules_said_search:
                verdict = self.query_router.decide(transcript)
                self._turn_timing_stages["router"] = float(verdict.get("elapsed_s", 0.0))
                self.conv_logger.turn_decision(
                    my_epoch, transcript,
                    needs_search=bool(verdict["needs_search"]),
                    source=str(verdict["source"]),
                    score=float(verdict["score"]),
                    elapsed_s=float(verdict["elapsed_s"]),
                    reason=str(verdict["reason"]),
                )
                self.conv_logger.narrate_router_decision(
                    my_epoch, transcript, bool(verdict["needs_search"]),
                    str(verdict["source"]), float(verdict["score"]), str(verdict["reason"]),
                )
                self.conv_logger.record(
                    my_epoch, needs_search=bool(verdict["needs_search"]),
                    router_source=str(verdict["source"]),
                    router_score=float(verdict["score"]),
                    router_reason=str(verdict["reason"]),
                )
                print(
                    f"[liveTryPlasticity][router] needs_search={verdict['needs_search']} "
                    f"src={verdict['source']} score={verdict['score']:.3f} "
                    f"in {1000.0 * float(verdict['elapsed_s']):.0f}ms :: {transcript!r}",
                    flush=True,
                )
                if not verdict["needs_search"]:
                    # Nothing will be injected. Tell the GPU thread to drop the
                    # thinking sound and let the model answer on its own. The
                    # <lookup> filler was deliberately never injected, so the
                    # context is untouched and the reply is a normal one.
                    self.conv_logger.turn_done(
                        my_epoch, "answered from own knowledge",
                        time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
                    )
                    self._log_timing_summary()
                    if my_epoch == self.search_turn_epoch:
                        self.pending_search_cancelled = True
                    return

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            # --- A search is happening: only now does the model get told to
            # wait. Injection itself must happen on the GPU thread. ---
            lookup_text = search_helpers.wrap_with_lookup_tags()
            self.pending_lookup_tokens = self.tokenizer.encode(lookup_text)

            hits: list[dict] = []
            if not self.web_search_enabled:
                print(
                    "[liveTryPlasticity][search] a search was needed but web search is "
                    "disabled -- falling back to the model's own knowledge",
                    flush=True,
                )
            elif not self.web_search_api_key:
                print(
                    "[liveTryPlasticity][search] a search was needed but no API key is "
                    "configured -- falling back to the model's own knowledge",
                    flush=True,
                )
            else:
                # An online search is genuinely about to happen -- this is the
                # ONLY place (besides the synchronous rules path in
                # _start_turn) that the thinking sound may begin. Raised here
                # rather than at the routing decision so that a "search"
                # verdict which cannot actually reach the network never makes
                # a sound.
                self.pending_start_thinking = True
                self.conv_logger.event(
                    "web_search_start", query=transcript, provider=self.web_search_provider,
                    triggered_reason="the router decided this needs current information",
                )
                self.conv_logger.narrate_web_search_start(
                    my_epoch, transcript, self.web_search_provider,
                    "the question needs current information",
                )
                t_web0 = time.perf_counter()
                web_hits = search_helpers.web_search_query_sync(
                    transcript, self.web_search_api_key, self.web_search_provider,
                    self.web_search_max_results, self.web_search_timeout,
                )
                web_elapsed = time.perf_counter() - t_web0
                self._turn_timing_stages["web_search"] = web_elapsed
                self.conv_logger.web_search(
                    transcript, self.web_search_provider, len(web_hits), web_elapsed,
                    triggered_reason="router decided live data was required",
                )
                # Relevance floor. Search engines always return something, so
                # this is the only thing standing between an unrelated page and
                # the assistant's spoken answer -- the old pipeline had no
                # floor at all and summarized results scoring as low as 0.04.
                relevant = [h for h in web_hits if h.get("similarity_score", 0.0) >= self.web_search_min_score]
                sorted_hits = sorted(relevant, key=lambda c: c["similarity_score"], reverse=True)
                hits = sorted_hits[: self.web_search_max_results]
                self.conv_logger.turn_search(
                    my_epoch, self.web_search_provider, len(web_hits), len(hits), web_elapsed
                )
                self.conv_logger.retrieval(transcript, "web", hits, web_elapsed)
                self.conv_logger.narrate_web_search_results(
                    my_epoch, sorted_hits, hits, self.web_search_max_results
                )
                self.conv_logger.record(
                    my_epoch, search_query=transcript, search_provider=self.web_search_provider,
                    search_found=len(web_hits), search_hits=[
                        {"source": h.get("source"), "similarity_score": h.get("similarity_score"),
                         "text": str(h.get("text", ""))[:600]}
                        for h in hits
                    ],
                    search_elapsed_s=round(web_elapsed, 3),
                )
                if web_hits and not relevant:
                    print(
                        f"[liveTryPlasticity][search] all {len(web_hits)} web result(s) scored below "
                        f"web_search_min_score={self.web_search_min_score} -- discarding them and "
                        f"falling back to the model's own knowledge",
                        flush=True,
                    )

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                # Superseded by a newer turn, or the filler timeout already
                # committed a fallback while we were searching -- skip
                # compression rather than spending GPU time on a result that
                # can never be used.
                return

            grounding = ""
            grounding_raw = ""
            used_fallback = False
            t_compress0 = time.perf_counter()
            if hits and self.context_compressor is not None:
                grounding = self.context_compressor.compress(question=transcript, chunks=hits)
                grounding_raw = getattr(self.context_compressor, "last_raw_result", "") or grounding
                primary_elapsed = time.perf_counter() - t_compress0
                self._turn_timing_stages["compression"] = (
                    self._turn_timing_stages.get("compression", 0.0) + primary_elapsed
                )
                self.conv_logger.compressor_call(
                    transcript, [h.get("text", "") for h in hits[:2]], grounding,
                    primary_elapsed, used_fallback=False,
                )
                if grounding:
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=False)
            if not grounding and hits:
                used_fallback = True
                t_fallback0 = time.perf_counter()
                grounding = search_helpers.summarize_web_fallback(
                    transcript, hits, max_sentences=2, max_chars=200
                )
                grounding_raw = grounding
                fallback_elapsed = time.perf_counter() - t_fallback0
                self._turn_timing_stages["compression"] = (
                    self._turn_timing_stages.get("compression", 0.0) + fallback_elapsed
                )
                self.conv_logger.compressor_call(
                    transcript, [h.get("text", "") for h in hits[:2]], grounding,
                    fallback_elapsed, used_fallback=True,
                )
                if grounding:
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=True)
            if not grounding:
                self.conv_logger.narrate_no_information(my_epoch)
            self.conv_logger.turn_ground(my_epoch, grounding, used_fallback=used_fallback)
            self.conv_logger.record(
                my_epoch, summary=grounding, summary_raw=grounding_raw,
                used_fallback=bool(used_fallback),
            )

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            ref_content = grounding.strip() if grounding else (
                "There's no specific information available on this, so answer from general knowledge."
            )
            # Last line of defence before the text reaches a speech model that
            # has no text normalizer of its own. compress() already returns
            # spoken form; the extractive fallback and this hard-coded string
            # go through the same door so nothing unnormalized can slip past.
            ref_content = normalize_for_speech(ref_content) or ref_content
            ids_before_trim = self.tokenizer.encode(ref_content)
            ids = ids_before_trim
            if len(ids) > self.max_ref_tokens:
                ids = ids[: self.max_ref_tokens]
                ref_content = self.tokenizer.decode(ids)
            new_ref_tokens = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(ref_content))
            # Set the metadata BEFORE the token list itself: the GPU thread
            # polls `pending_ref_tokens is not None` as its readiness signal, so
            # the counts must already be in place by the time that becomes true
            # (otherwise there is a narrow race where it reads stale counts).
            self._pending_ref_token_counts = (len(ids_before_trim), self.max_ref_tokens)
            self.pending_ref_tokens = new_ref_tokens
            print(
                f"[liveTryPlasticity][search] prepared <ref> block "
                f"({len(self.pending_ref_tokens)} tok): {ref_content[:150]!r}",
                flush=True,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTryPlasticity][search] route/search/compress failed: {e!r}\n{tb}", flush=True)
            self.conv_logger.error("route_and_search", e, tb)
            self.conv_logger.narrate_no_information(my_epoch)
            if my_epoch == self.search_turn_epoch:
                fallback_tokens = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(
                    "There's no specific information available on this, so answer from general knowledge."
                ))
                self._pending_ref_token_counts = (len(fallback_tokens), self.max_ref_tokens)
                self.pending_ref_tokens = fallback_tokens

    def _log_timing_summary(self) -> None:
        """Print the consolidated big-to-small timing breakdown for the turn
        that was just committed (real ref, fallback, or router cancel -- all
        call this). Never raises: a missing or stale timing dict must not break
        injection."""
        try:
            start = getattr(self, "_turn_timing_start", None)
            stages = getattr(self, "_turn_timing_stages", None)
            if start is None or not stages:
                return
            total = time.perf_counter() - start
            self.conv_logger.turn_timing_summary(self.search_turn_epoch, total, dict(stages))
            self.conv_logger.narrate_timing_summary(self.search_turn_epoch, total, dict(stages))
        except Exception:
            pass

    def _consume_pending(self) -> None:
        """Called once per chunk while a routing/search decision is in flight.
        This is the ONLY place the background thread's work reaches the LM,
        because token injection must happen on the GPU thread.

        Three outcomes, checked in priority order:
          1. cancelled  -- the router decided no search is needed; stop the
                           thinking sound and let the model answer normally.
                           Nothing is injected, so the context is untouched.
          2. ref ready  -- inject the grounded <ref> block.
          3. timed out  -- after self._SEARCH_MAX_FILLER_FRAMES chunks
                           (--search_max_filler_sec) inject a generic fallback
                           so the model never hangs on a slow or failed search.
        The <lookup> filler is injected on whatever chunk the background thread
        made it available, which is only ever after a search was committed."""
        import search_helpers

        self.search_filler_frame_count += 1

        if self.pending_search_cancelled:
            self.pending_search_cancelled = False
            self.pending_lookup_tokens = None
            self.pending_start_thinking = False
            self.suppress_text_until_ref = False
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            # Normally a no-op: the router path never starts the sound before
            # deciding, so on a "no search" verdict there is nothing playing.
            # Kept because the rules path CAN have started it already, and a
            # later cancel must still silence it.
            self._stop_thinking_sound(
                self.search_turn_epoch, "no_search_needed",
                "the assistant already knew this and did not need to search",
            )
            print(
                "[liveTryPlasticity][search] router said no search -- answering from "
                "the model's own knowledge (nothing injected)",
                flush=True,
            )
            return

        # A search is really in flight -> start the cue (idempotent, so the
        # rules path having already started it is fine). Skipped when the
        # answer is ALREADY waiting: a search that completed inside this same
        # 80ms tick would otherwise start the clip and stop it again a few
        # lines below, emitting a single-chunk blip of noise for no reason.
        if self.pending_start_thinking:
            self.pending_start_thinking = False
            if self.pending_ref_tokens is None:
                self._start_thinking_sound(self.search_turn_epoch, self.search_current_transcript)

        if self.pending_lookup_tokens is not None:
            lookup_tokens = self.pending_lookup_tokens
            self.pending_lookup_tokens = None
            t_lookup0 = time.perf_counter()
            self._inject_tokens(lookup_tokens)
            lookup_elapsed = time.perf_counter() - t_lookup0
            self._turn_timing_stages["lookup_inject"] = lookup_elapsed
            self.conv_logger.ref_injected(
                self.tokenizer.decode(lookup_tokens), len(lookup_tokens), lookup_elapsed, kind="lookup"
            )
            print(
                f"[liveTryPlasticity][search] <lookup> injected ({len(lookup_tokens)} tok) "
                f"-- searching in background",
                flush=True,
            )

        if self.pending_ref_tokens is not None:
            ref_tokens = self.pending_ref_tokens
            self.pending_ref_tokens = None
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            # Release the text hold BEFORE injecting: the very next normal
            # generation step must be free to speak the grounded answer.
            self.suppress_text_until_ref = False
            # Stop the thinking sound BEFORE injecting -- the injection burst
            # itself produces no audio (forced-token steps bypass decode), so
            # the very next normal generation step in this same _step() call is
            # the model's real, grounded reply: no silent gap, no overlap.
            self._stop_thinking_sound(self.search_turn_epoch, "ref_ready", "the answer was ready")
            t0 = time.perf_counter()
            self._inject_tokens(ref_tokens)
            elapsed = time.perf_counter() - t0
            self._turn_timing_stages["ref_inject"] = elapsed
            ref_text = self.tokenizer.decode(ref_tokens)
            self.conv_logger.ref_injected(ref_text, len(ref_tokens), elapsed, kind="ref")
            n_before, max_tok = self._pending_ref_token_counts
            self.conv_logger.narrate_injection(
                self.search_turn_epoch, ref_text, len(ref_tokens), n_before, max_tok, kind="ref",
            )
            self.conv_logger.record(
                self.search_turn_epoch, injected_text=ref_text, injected_tokens=len(ref_tokens),
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "grounded from web search",
                time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
            )
            self._log_timing_summary()
            print(
                f"[liveTryPlasticity][search] <ref> injected ({len(ref_tokens)} tok) "
                f"after {self.search_filler_frame_count} filler chunks",
                flush=True,
            )
        elif self.search_filler_frame_count >= self._SEARCH_MAX_FILLER_FRAMES:
            fallback_text = "There's no specific information available on this, so answer from general knowledge."
            fallback = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(fallback_text))
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            self.suppress_text_until_ref = False
            self._stop_thinking_sound(
                self.search_turn_epoch, "filler_timeout",
                "the search was taking too long, so the assistant moved on with what it had",
            )
            t0 = time.perf_counter()
            self._inject_tokens(fallback)
            fallback_inject_elapsed = time.perf_counter() - t0
            self._turn_timing_stages["ref_inject"] = fallback_inject_elapsed
            self.conv_logger.ref_injected(fallback_text, len(fallback), fallback_inject_elapsed, kind="ref_fallback")
            self.conv_logger.narrate_injection(
                self.search_turn_epoch, fallback_text, len(fallback), len(fallback),
                self.max_ref_tokens, kind="ref_fallback",
            )
            self.conv_logger.record(
                self.search_turn_epoch, injected_text=fallback_text, injected_tokens=len(fallback),
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "search timed out, answered from own knowledge",
                time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
            )
            self._log_timing_summary()
            print("[liveTryPlasticity][search] <ref> fallback injected after filler timeout", flush=True)

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        if self._rt_wall_start == 0.0:
            self._rt_wall_start = t0
        self._rt_frames += 1
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            self.mimi.reset_streaming()
            self.skip_first = False

        # -- STT/VAD forward pass + turn-boundary detection, then consume any
        # in-flight routing/search result. Both are no-ops unless
        # --stt_hf_repo/--stt_pkg_dir were configured at launch, so the plain
        # conversational path is untouched and pays nothing.
        #
        # Both calls run inside try/except: this hook runs on every single
        # chunk of the live conversation, so an uncaught exception here would
        # propagate out of _step() and kill the entire GPU producer thread --
        # audio and video generation stop forever, and the thread that would
        # have logged the failure is the one that died, so the logs just end
        # mid-turn. On failure, search is disabled for the rest of this session
        # but the avatar keeps talking. --
        if self.stt_lm_gen is not None and not self.search_hard_disabled:
            t_stt0 = time.perf_counter()
            try:
                if self.stt_inline:
                    # Opt-in A/B path: run the STT model right here, the way it
                    # used to. Kept only so the threaded path can be compared
                    # against it; it costs a 1B forward plus two device syncs
                    # inside this 80ms budget.
                    self._stt_forward(chunk)
                else:
                    self._stt_submit(chunk)
                self._consume_pending_transcript()
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][search] STT handling failed, disabling search "
                    f"for the rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("stt_step", e, tb)
                self.sys_logger.component_failed("stt_step (runtime)", e, tb)
                self.search_hard_disabled = True
                self.search_awaiting_ref = False
                self.search_thinking_active = False
                self.suppress_text_until_ref = False
            stt_ms = 1000.0 * (time.perf_counter() - t_stt0)
        else:
            stt_ms = 0.0
        if self.search_awaiting_ref and not self.search_hard_disabled:
            try:
                self._consume_pending()
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][search] _consume_pending failed, disabling "
                    f"search for the rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("consume_pending", e, tb)
                self.sys_logger.component_failed("consume_pending (runtime)", e, tb)
                self.search_hard_disabled = True
                self.search_awaiting_ref = False
                self.search_thinking_active = False
                self.suppress_text_until_ref = False

        # While a search is in flight, hand the model its own "say nothing"
        # token instead of letting it sample text. process_transformer_output
        # honors a provided token rather than sampling, so this is the model's
        # native silence mechanism, not a hack bolted on top. Audio and hidden
        # states keep flowing, so the avatar pipeline and chunk cadence are
        # untouched; only the words are withheld, and precisely while the model
        # would otherwise be inventing an answer it is about to be handed.
        t_lm0 = time.perf_counter()
        # Hold the model's words until the user has actually said something.
        # Same native mechanism as the search suppression below -- a provided
        # text token rather than a sampled one -- so audio, hidden states and
        # chunk cadence are unaffected and only the words wait.
        wait_for_user = (
            self.wait_for_user_before_speaking
            and not self._user_has_spoken
            and self.stt_lm_gen is not None
            and not self.search_hard_disabled
        )
        if (self.suppress_text_until_ref or wait_for_user) and self._step_supports_text_token:
            lm_out = self.lm_gen._step(
                codes[:, :, :1], text_token=getattr(self.lm_gen, "zero_text_code", 3)
            )
        else:
            lm_out = self.lm_gen._step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        tokens = None
        helium_hidden = None
        if lm_out is not None:
            if not (isinstance(lm_out, tuple) and len(lm_out) == 3):
                raise RuntimeError(f"Moshi graph layer[-2] contract failure: got {type(lm_out)} len={len(lm_out) if isinstance(lm_out, tuple) else 'n/a'}")
            tokens, _transformer_out, layer_hidden = lm_out
            helium_hidden = layer_hidden[:1, -1:].detach().float().cpu()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        reply_codes = None
        if tokens is None:
            reply_pcm = np.zeros(MIMI_FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < MIMI_FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, MIMI_FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > MIMI_FRAME_SIZE:
                reply_pcm = reply_pcm[:MIMI_FRAME_SIZE]

        # The model's OWN output level, measured before the thinking sound can
        # replace it below: after the swap, `reply_pcm` may be the filler clip,
        # whose level says nothing about whether the model is speaking.
        model_own_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))

        # First real audio of this turn -> the honest end-to-end latency. This
        # is the ONE number that matches what the user experiences; the SAID
        # line in the conversation log is emitted when the NEXT utterance ends
        # and therefore measures the user's own pause, not the assistant's.
        if self._turn_awaiting_first_speech:
            quiet = model_own_rms <= self._SPEECH_RMS_THRESHOLD
            if quiet:
                self._turn_saw_silence_after_start = True
            # Accept the first loud frame only once the model has actually gone
            # quiet since the turn started -- otherwise this measures the tail of
            # the PREVIOUS answer and always reports ~0.03s. The deadline is the
            # escape hatch for a model that never pauses: past it, the number is
            # still recorded but flagged approximate rather than lost.
            past_deadline = (
                self._turn_first_speech_deadline
                and self.step >= self._turn_first_speech_deadline
            )
            if not quiet and (self._turn_saw_silence_after_start or past_deadline):
                self._turn_awaiting_first_speech = False
                started = self._turn_timing_start
                if started is not None:
                    first_word_s = time.perf_counter() - started
                    self.conv_logger.turn_spoke(
                        self._turn_first_speech_epoch, first_word_s, self.input_backlog_sec(),
                    )
                    self.conv_logger.record(
                        self._turn_first_speech_epoch,
                        first_word_s=round(first_word_s, 3),
                        first_word_approximate=bool(past_deadline
                                                    and not self._turn_saw_silence_after_start),
                        input_backlog_s=round(self.input_backlog_sec(), 3),
                        realtime_factor=round(self.realtime_factor(), 3),
                    )

        # "Thinking sound": while an online search is in flight, replace what
        # the model would otherwise output with the looped clip. The model
        # keeps stepping normally above (KV cache and timing untouched); only
        # the audio actually sent out is swapped. force_idle tells the
        # avatar-motion gate downstream to stay visually idle rather than
        # lip-syncing to this non-speech sound -- its real RMS is well above
        # the speech threshold, so without it the avatar would appear to talk.
        # There is no "model started speaking" stop condition: the clip only
        # ever covers a real search, which ends in exactly two ways -- the
        # <ref> landing or the filler timeout, both in _consume_pending.
        force_idle = False
        if self.search_thinking_active and self.thinking_sound_pcm is not None:
            reply_pcm = self._next_thinking_sound_chunk()
            force_idle = True

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))

        # Independent release for the wait-for-user gate. The STT text stream is
        # the precise signal, but making it the ONLY one would mean a
        # tokenization failure mutes the avatar for the whole session -- a much
        # worse outcome than the startup babble the gate exists to prevent. Loud
        # microphone input for a quarter of a second is enough to conclude
        # somebody is talking, whatever STT makes of it.
        if self.wait_for_user_before_speaking and not self._user_has_spoken:
            if input_rms > self._SPEECH_RMS_THRESHOLD:
                self._loud_input_frames += 1
                if self._loud_input_frames >= 3:
                    self._user_has_spoken = True
                    print(
                        f"[liveTryPlasticity] user audio detected "
                        f"(in_rms={input_rms:.5f}) -- releasing the model to speak",
                        flush=True,
                    )
            else:
                self._loud_input_frames = 0
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")

        print(
            "[liveTryStudio] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"hidden={helium_hidden is not None} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms stt={stt_ms:.1f}ms "
            f"decode={decode_ms:.1f}ms total={total_ms:.1f}ms rtf={self.realtime_factor():.2f}",
            flush=True,
        )

        # Real-time watchdog. A sustained rtf below 1.0 is the one condition
        # under which this pipeline goes SILENT rather than merely slow: the
        # audio sender paces on a fixed 80ms grid, so a producer that cannot
        # supply 80ms of audio per 80ms of wall clock leaves the browser's
        # decoder permanently under-run. Say so explicitly, because the symptom
        # ("no sound at all") looks nothing like the cause ("GPU 20% too slow").
        if self.step % 125 == 0:
            rtf = self.realtime_factor()
            now = time.perf_counter()
            if rtf < 0.98 and now - self._rt_last_warn >= 10.0:
                self._rt_last_warn = now
                print(
                    f"[liveTryPlasticity] WARNING real-time factor {rtf:.2f} -- the model "
                    f"pipeline is consuming only {rtf:.2f}s of audio per second of wall clock. "
                    f"Below 1.0 the audio sender starves and the avatar will sound broken or "
                    f"silent. Reduce per-step GPU cost: merge the reference LoRA "
                    f"(--merge_ref_lora), move the router/compressor off this GPU "
                    f"(--compressor_device cpu), or launch with ENABLE_SEARCH=0.",
                    flush=True,
                )
                self.conv_logger.event(
                    "realtime_factor_low", f"rtf={rtf:.3f}",
                    realtime_factor=round(rtf, 3),
                    stt_dropped_frames=self._stt_dropped_frames,
                )
                self.sys_logger.event(
                    "realtime_factor_low",
                    f"rtf={rtf:.2f} -- producer is below real time; audio will starve",
                    realtime_factor=round(rtf, 3),
                )

        return {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": audio_b64,
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "force_idle": force_idle,
            "input_rms": input_rms,
            "token": token,
            "piece": token_piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "stt_ms": stt_ms,
            "realtime_factor": self.realtime_factor(),
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
        }

    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process a bounded number of Mimi frames.

        The base MoshiOnlyEngine drains the entire input buffer before returning.
        In live mode that is dangerous: mic audio can accumulate while Moshi is
        loading, then avatar frames do not reach the sender until the backlog is
        fully processed. Bounded draining keeps the producer/sender interleaved.
        """
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                break
            pcm = self.input_buffer[:MIMI_FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[MIMI_FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _supervise(name: str, fn, on_error=None, max_restarts: int = 0,
               restart_delay_s: float = 0.5):
    """Wrap a long-lived thread body so a crash is LOUD and lands in both logs.

    Every media thread here is a daemon started with threading.Thread(target=...).
    Python's default behaviour on an uncaught exception in such a thread is to
    print a traceback to stderr and let the thread die silently. For the GPU
    producer that means audio and video stop forever while the persona, STT and
    search threads keep running -- so the conversation log continues to look
    completely healthy and the only evidence is a traceback in a file nobody
    thinks to check. Both structured logs must record a thread death, because
    "the avatar went quiet" and "a thread died" have to be connectable.

    With `max_restarts` the body is re-entered after a crash. That is only safe
    for a body that owns all of its mutable state as locals -- the GPU producer
    does -- so re-entering gives a clean stream with the loaded models still
    resident. The budget is finite so a deterministic fault cannot spin."""

    def _runner():
        attempt = 0
        while True:
            try:
                fn()
            except BaseException as e:  # noqa: BLE001 - a dying thread must be reported
                tb = traceback.format_exc()
                print("[THREAD-DIED] %s: %r\n%s" % (name, e, tb), flush=True)
                with contextlib.suppress(Exception):
                    system_logger.get().component_failed("thread:" + name, e, tb)
                if on_error is not None:
                    with contextlib.suppress(Exception):
                        on_error(e, tb)
                if attempt >= max_restarts:
                    if max_restarts:
                        print(
                            f"[THREAD-DIED] {name}: restart budget exhausted "
                            f"({max_restarts}); staying down.",
                            flush=True,
                        )
                        with contextlib.suppress(Exception):
                            system_logger.get().event(
                                "thread_restart_exhausted",
                                f"{name} crashed {attempt + 1} times; not restarting again",
                                thread=name, restarts=attempt,
                            )
                    return
                attempt += 1
                # Reclaim whatever the dead attempt was holding before retrying;
                # if the crash was an OOM, retrying without this just repeats it.
                with contextlib.suppress(Exception):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(
                    f"[THREAD-RESTART] {name}: attempt {attempt}/{max_restarts} "
                    f"in {restart_delay_s:.1f}s",
                    flush=True,
                )
                with contextlib.suppress(Exception):
                    system_logger.get().event(
                        "thread_restart",
                        f"{name} crashed and is being restarted "
                        f"(attempt {attempt} of {max_restarts})",
                        thread=name, attempt=attempt, max_restarts=max_restarts,
                    )
                time.sleep(restart_delay_s)
            else:
                print(f"[THREAD-EXIT] {name} returned normally", flush=True)
                with contextlib.suppress(Exception):
                    system_logger.get().event(
                        "thread_exit", f"{name} returned normally", thread=name
                    )
                return

    return _runner


def _log_vram_headroom(fm_engine, args) -> None:
    """Write down how much VRAM is left once everything is resident.

    Every model is loaded by this point -- PersonaPlex, Mimi, the FM generator,
    the FP32 renderer, and with search on the STT model, the router/compressor
    and the reference LoRA. What is left over is what the renderer has to do its
    work in, and when that number is small the failure is an OOM on the render
    thread, which reads to the user as an avatar that never speaks.

    The estimate is the renderer's dominant transient: the resolution-64
    attention, batch x heads x N x N x 4 bytes. It is the allocation that
    actually failed, so it is the right thing to compare headroom against."""
    if not torch.cuda.is_available():
        return
    with contextlib.suppress(Exception):
        free_b, total_b = torch.cuda.mem_get_info()
        free_gb = free_b / (1 << 30)
        sub_batch = max(1, int(getattr(args, "render_sub_batch", 8)))
        # 64x64 tokens, 8 heads, fp32 -- one head at a time plus the running
        # mean, which is what the renderer now holds rather than all heads.
        est_peak_gb = 2 * sub_batch * (64 * 64) ** 2 * 4 / (1 << 30)
        fits = free_gb > est_peak_gb * 1.5
        msg = (
            f"free={free_gb:.2f}GB of {total_b / (1 << 30):.2f}GB after all models; "
            f"renderer needs about {est_peak_gb:.2f}GB per sub-batch of {sub_batch} frames"
        )
        if not fits:
            msg += (
                " -- this is tight. If the avatar goes silent, lower "
                "--render_sub_batch or run with ENABLE_SEARCH=0."
            )
        print(f"[VRAM] {msg}", flush=True)
        system_logger.get().gpu_memory("after FM engine and renderer")
        system_logger.get().event(
            "vram_headroom", msg,
            free_gb=round(free_gb, 2),
            total_gb=round(total_b / (1 << 30), 2),
            render_sub_batch=sub_batch,
            est_render_peak_gb=round(est_peak_gb, 2),
            comfortable=bool(fits),
        )


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def encode_jpeg_b64(frame_rgb: np.ndarray, quality: int) -> str:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def encode_jpeg_bytes(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def _pcm_f32_to_i16_b64(pcm: np.ndarray) -> str:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode((arr * 32767.0).astype(np.int16).tobytes()).decode("ascii")


def _pcm_f32_to_i16_bytes(pcm: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def split_audio_into_frame_slices(pcm: np.ndarray, fps: float) -> list[np.ndarray]:
    frame_samples = int(round(TARGET_SR / float(fps)))
    arr = np.asarray(pcm, dtype=np.float32)
    n_frames = max(0, int(round(arr.shape[0] / frame_samples)))
    if n_frames == 0:
        return []
    total = n_frames * frame_samples
    if arr.shape[0] < total:
        arr = np.pad(arr, (0, total - arr.shape[0]))
    elif arr.shape[0] > total:
        arr = arr[:total]
    return [arr[i * frame_samples:(i + 1) * frame_samples].copy() for i in range(n_frames)]


def load_audio_24k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).float().numpy()


def load_ref_image(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# FM + Renderer weight loading (identical to liveTryFM.py)
# ---------------------------------------------------------------------------

def _clean_generator_state(ckpt: dict) -> dict:
    raw = ckpt.get("ema_state_dict") or ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}


def _load_fm(args: argparse.Namespace, device: torch.device) -> FMGenerator:
    t_total = time.perf_counter()
    fm = FMGenerator(args).to(device).eval()
    ckpt = torch.load(args.generator_path, map_location="cpu")
    cleaned = _clean_generator_state(ckpt)
    missing, unexpected = fm.load_state_dict(cleaned, strict=False)
    print(
        f"[liveTryHeliumFM][FM] base loaded missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    lora_path = str(getattr(args, "lora_generator_path", "") or "")
    if lora_path:
        apply_lora_to_model(
            fm,
            rank=int(getattr(args, "lora_rank", 64) or 64),
            alpha=float(getattr(args, "lora_alpha", 128) or 128),
            dropout=float(getattr(args, "lora_dropout", 0.05)),
            include_pose_lora=not bool(getattr(args, "no_lora_pose_projection", False)),
            include_audio_lora=not bool(getattr(args, "no_lora_audio_projection", False)),
            only_pose_lora=bool(getattr(args, "only_lora_pose_projection", False)),
        )
        lora_ckpt = torch.load(lora_path, map_location="cpu")
        lora_cleaned = _clean_generator_state(lora_ckpt)
        missing_lora, unexpected_lora = fm.load_state_dict(lora_cleaned, strict=False)
        lora_keys = sum(1 for key in lora_cleaned if "lora_" in key)
        print(
            f"[liveTryHeliumFM][FM] lora loaded path={lora_path} "
            f"lora_keys={lora_keys} missing={len(missing_lora)} unexpected={len(unexpected_lora)}",
            flush=True,
        )
    fm.to(device).eval()
    _sync_cuda()
    print(f"[liveTryHeliumFM][FM] loaded in {_ms(t_total):.0f}ms", flush=True)
    return fm


def _load_renderer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> IMTRenderer:
    t_total = time.perf_counter()
    renderer = IMTRenderer(args).to(device).eval()
    ckpt = torch.load(args.renderer_path, map_location="cpu")
    raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {k.replace("gen.", "", 1).replace("model.", "", 1): v for k, v in raw.items()}
    missing, unexpected = renderer.load_state_dict(cleaned, strict=False)
    renderer = renderer.to(dtype=dtype)
    _sync_cuda()
    if getattr(args, "compile_renderer", False):
        @torch.no_grad()
        def _fused_render(motion_latent, g_r, m_r, f_r):
            ta_c = renderer.adapt(motion_latent, g_r)
            m_c = renderer.latent_token_decoder(ta_c)
            frames = renderer.decode(m_c, m_r, f_r)
            return frames
        renderer._fused_render = torch.compile(_fused_render)
    print(
        f"[liveTryHeliumFM][renderer] loaded in {_ms(t_total):.0f}ms "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return renderer


# ---------------------------------------------------------------------------
# Helium extractor (chunk-local, batch LM)
# ---------------------------------------------------------------------------

class HeliumExtractor:
    """Prefix-growing raw Helium + global interpolation.

    This matches the best Stage 3 diagnostic:
      audio prefix -> raw Helium -> append only new raw steps
      -> one global interpolation over accumulated raw Helium
      -> emit last target_frames
    """

    def __init__(
        self,
        helium_mimi: "MimiModel",
        helium_lm: "LMModel",
        device: torch.device,
    ) -> None:
        self.helium_mimi = helium_mimi
        self.helium_lm = helium_lm
        self.device = device
        self._lock = threading.Lock()
        self._prefix_pcm = np.empty(0, dtype=np.float32)
        self._raw_parts: list[torch.Tensor] = []
        self._prev_raw_len = 0
        self._emitted_frames = 0

    def reset(self) -> None:
        with self._lock:
            self._prefix_pcm = np.empty(0, dtype=np.float32)
            self._raw_parts = []
            self._prev_raw_len = 0
            self._emitted_frames = 0

    def _extract_raw(self, pcm_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.asarray(pcm_np, dtype=np.float32)).to(self.device, dtype=torch.float32)[None, None]

        codes = self.helium_mimi.encode(wav)
        codes = codes[:, :MAIN_CODEBOOKS, :].detach()
        batch_size, n_q, total_steps = codes.shape

        dtype = next(self.helium_lm.parameters()).dtype
        input_emb = torch.zeros(
            batch_size, total_steps, self.helium_lm.dim, device=self.device, dtype=dtype
        )
        for q in range(n_q):
            input_emb = input_emb + self.helium_lm.emb[q](codes[:, q].long())

        padding_ids = torch.full(
            (batch_size, total_steps),
            self.helium_lm.existing_text_padding_id,
            dtype=torch.long,
            device=self.device,
        )
        input_emb = input_emb + self.helium_lm.text_emb(padding_ids)

        if getattr(self.helium_lm.transformer, "_streaming_state", None) is not None:
            raise RuntimeError("helium_lm must stay in batch mode (non-streaming)")

        captured: list[torch.Tensor] = []

        def _hook(_mod, _inp, out):
            captured.append(out.detach())

        handle = self.helium_lm.transformer.layers[-2].register_forward_hook(_hook)
        try:
            self.helium_lm.transformer(input_emb)
        finally:
            if reply_engine is not None:
                pipeline_stop_event.set()
                session_started.clear()
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(f"Helium hook captured {len(captured)} tensors; expected 1")
        return captured[0].squeeze(0).float().contiguous()  # [T_raw, 4096]

    @torch.no_grad()
    def extract_raw_chunk(self, pcm_np: np.ndarray) -> torch.Tensor:
        """Return only the new raw 12.5Hz Helium steps for one new audio chunk."""
        pcm = np.asarray(pcm_np, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_raw_chunk expects non-empty 1D PCM")

        with self._lock:
            self._prefix_pcm = np.concatenate([self._prefix_pcm, pcm], axis=0)
            raw_prefix = self._extract_raw(self._prefix_pcm)
            new_raw = raw_prefix[self._prev_raw_len:]
            if int(new_raw.shape[0]) == 0 and int(raw_prefix.shape[0]) > 0:
                new_raw = raw_prefix[-1:]
            self._raw_parts.append(new_raw.cpu())
            self._prev_raw_len = int(raw_prefix.shape[0])
            return new_raw.contiguous()

    @torch.no_grad()
    def extract_exact_chunk_from_prefix(
        self,
        pcm_prefix: np.ndarray,
        chunk_start_frame: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Return the raw 12.5Hz Helium slice for a video-frame window.

        This path is used by file-mode lookahead. It must return raw Helium
        steps so the studio adapter remains the only temporal upsampler.
        """
        pcm = np.asarray(pcm_prefix, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_exact_chunk_from_prefix expects non-empty 1D PCM")
        with self._lock:
            raw_prefix = self._extract_raw(pcm)
        start_frame = int(chunk_start_frame)
        end_frame = start_frame + int(target_frames)
        start_raw = int(round(start_frame * 0.5))
        end_raw = int(round(end_frame * 0.5))
        start_raw = max(0, min(start_raw, int(raw_prefix.shape[0])))
        end_raw = max(start_raw + 1, min(end_raw, int(raw_prefix.shape[0])))
        if end_raw > int(raw_prefix.shape[0]):
            raise RuntimeError(
                f"Requested raw slice [{start_raw}, {end_raw}) exceeds prefix Helium length {raw_prefix.shape[0]}"
            )
        return raw_prefix[start_raw:end_raw].contiguous()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveHeliumFMEngine:
    """Helium extraction + FM + renderer, session-stateful."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.args = args
        self.device = torch.device(args.device)
        renderer_precision = str(
            getattr(args, "renderer_precision", "fp32")
        ).lower()
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[renderer_precision]
        self.fps = float(args.fps)
        self.audio_chunk_sec = float(getattr(args, "audio_chunk_sec", 0.96))
        self.audio_chunk_samples = int(round(self.audio_chunk_sec * TARGET_SR))
        self.fm_chunk_frames = max(1, int(getattr(args, "fm_chunk_frames", 24)))
        self.live_sliding_window = bool(getattr(args, "enable_live_sliding_window", False))
        self.slide_past_frames = max(0, int(getattr(args, "slide_past_frames", 10)))
        self.slide_future_frames = max(0, int(getattr(args, "slide_future_frames", 3)))
        self.render_sub_batch = max(1, int(args.render_sub_batch))
        self.jpeg_quality = int(args.jpeg_quality)
        trained_window = int(round(float(args.wav2vec_sec) * self.fps))
        if self.fm_chunk_frames != trained_window:
            print(
                f"[liveTryHeliumFM] WARNING fm_chunk_frames={self.fm_chunk_frames} "
                f"but wav2vec_sec*fps={trained_window}",
                flush=True,
            )
        if self.live_sliding_window:
            print(
                f"[liveTryHeliumFM][typeAC] IMTalker lookahead enabled "
                f"past={self.slide_past_frames}f current={self.fm_chunk_frames}f "
                f"future={self.slide_future_frames}f",
                flush=True,
            )

        t_total = time.perf_counter()

        # FM + renderer
        self.fm = _load_fm(args, self.device)
        self.renderer = _load_renderer(args, self.device, self.dtype)

        # Adapter: either the six-layer projected-frontend model or UniTalk's
        # 12-layer model with only its final layer passed to IMTalker.
        t_adapter = time.perf_counter()
        if args.adapter_type == "unitalk_last_layer":
            self.studio_adapter = UniTalkLastLayerLiveAdapter(
                args.wav2vec_model_path,
                args.adapter_dropout,
            ).to(self.device).float().eval()
        else:
            self.studio_adapter = StudioNativeLiveAdapter(
                args.wav2vec_model_path,
                args.adapter_num_layers,
                args.adapter_dropout,
            ).to(self.device).float().eval()
        payload = torch.load(args.adapter_path, map_location="cpu")
        if isinstance(payload, dict) and args.adapter_type == "frontend":
            saved_args = payload.get("args", {})
            if saved_args and int(saved_args.get("num_layers", args.adapter_num_layers)) != int(args.adapter_num_layers):
                print(
                    f"[liveTryHeliumFrontendFM] WARNING checkpoint num_layers={saved_args.get('num_layers')} "
                    f"but CLI adapter_num_layers={args.adapter_num_layers}",
                    flush=True,
                )
            state = payload.get("adapter", payload.get("model", payload))
        else:
            state = payload
        self.studio_adapter.load_state_dict(state, strict=True)
        _sync_cuda()
        print(
            f"[liveTryHeliumFrontendFM][adapter] type={args.adapter_type} loaded in {_ms(t_adapter):.0f}ms "
            f"path={args.adapter_path}",
            flush=True,
        )

        # Raw HF Wav2Vec2 target path used during Helium adapter training.
        t_w2v = time.perf_counter()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_model_path,
            local_files_only=True,
        )
        self.wav2vec_model = self.studio_adapter.wav2vec
        _sync_cuda()
        print(
            f"[liveTryHeliumStudioFM][wav2vec] loaded in {_ms(t_w2v):.0f}ms "
            f"path={args.wav2vec_model_path}",
            flush=True,
        )

        # Reference image: pre-compute identity + motion-ref features once
        ref_tensor = load_ref_image(args.ref_path, self.device, self.dtype)
        with torch.no_grad():
            self.f_r, self.g_r = self.renderer.dense_feature_encoder(ref_tensor)
            self.ref_x = self.renderer.latent_token_encoder(ref_tensor).to(dtype=torch.float32)
            ta_r = self.renderer.adapt(self.ref_x.to(dtype=self.dtype), self.g_r)
            self.m_r = self.renderer.latent_token_decoder(ta_r)
        _sync_cuda()
        self.eye_blink_enabled = bool(getattr(args, "enable_eye_blink_composite", False))
        self._blink_maps: tuple[torch.Tensor, ...] | None = None
        self._eye_masks: tuple[torch.Tensor, ...] | None = None
        self._render_frame_cursor: int = 0
        if self.eye_blink_enabled:
            self._init_eye_blink_composite()

        # Moshi models for Helium extraction
        self._init_moshi(args)

        # Optional local audio file (simulate-live mode)
        self.audio_pcm: np.ndarray | None = None
        if getattr(args, "audio_path", "") and Path(args.audio_path).is_file():
            self.audio_pcm = load_audio_24k(args.audio_path)
            print(
                f"[liveTryHeliumFM] audio_path loaded: {self.audio_pcm.shape[0]/TARGET_SR:.2f}s "
                f"chunk={self.audio_chunk_sec:.3f}s/{self.audio_chunk_samples} samples "
                f"fm_chunk={self.fm_chunk_frames}f",
                flush=True,
            )

        # Shared noise tensor (pre-generated, indexed by absolute frame position)
        self.noise_buf: torch.Tensor | None = None
        if getattr(args, "shared_noise", False):
            max_frames = int(getattr(args, "noise_max_frames", 5000))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(getattr(args, "noise_seed", 1234)))
            self.noise_buf = torch.randn(
                1, max_frames, int(args.dim_w), device=self.device, generator=gen
            )
            print(f"[liveTryHeliumFM] shared noise buf: {tuple(self.noise_buf.shape)}", flush=True)

        # Per-session state (reset on each new client)
        self.stream_state: dict | None = None
        self.abs_frame: int = 0
        self.helium_context_tail: torch.Tensor | None = None
        self.helium_deque_size = max(
            1, int(getattr(args, "helium_deque_size", 100))
        )
        self.helium_deque: torch.Tensor | None = None
        self.helium_deque_filled: int = 0
        self.silence_helium_seed: torch.Tensor | None = None
        silence_helium_path = str(
            getattr(args, "silence_helium_path", "") or ""
        )
        if silence_helium_path:
            payload = torch.load(silence_helium_path, map_location="cpu")
            seed = (
                payload.get("silence_helium_mean")
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(seed, torch.Tensor) or seed.numel() != 4096:
                raise RuntimeError(
                    f"Invalid silence Helium seed: {silence_helium_path}"
                )
            self.silence_helium_seed = seed.reshape(1, 4096).to(
                device=self.device, dtype=torch.float32
            )
            print(
                f"[liveTryHeliumFM] silence Helium seed loaded: "
                f"{silence_helium_path}",
                flush=True,
            )
        self._pcm_accum: np.ndarray = np.empty(0, dtype=np.float32)
        self.dump_motion = bool(getattr(args, "dump_motion", False))
        self.dump_dir = Path(getattr(args, "dump_dir", ROOT / "live_try_dumps"))
        self._session_motion_parts: list[torch.Tensor] = []
        self._session_helium_parts: list[torch.Tensor] = []
        self._session_adapter_50_parts: list[torch.Tensor] = []
        self._session_adapter_25_parts: list[torch.Tensor] = []
        self._session_projected_audio_parts: list[torch.Tensor] = []
        self._session_audio_parts: list[np.ndarray] = []
        self._session_live_token_parts: list[torch.Tensor] = []
        self._session_chunk_rows: list[dict] = []
        self._session_reply_events: list[dict] = []
        self._session_started_wall: float = time.time()

        # JPEG encoding thread pool (CPU-only work, parallelizable)
        self._jpeg_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="jpeg"
        )

        # Warmup
        self._warmup()

        print(
            f"[liveTryHeliumFM] ready — total startup {_ms(t_total):.0f}ms "
            f"fm_chunk={self.fm_chunk_frames} render_sub={self.render_sub_batch} "
            f"dtype={self.dtype}",
            flush=True,
        )

    def _init_moshi(self, args: argparse.Namespace) -> None:
        if bool(getattr(args, "direct_reply_hidden", False)) and bool(getattr(args, "enable_moshi_reply", False)):
            self.extractor = None
            print("[liveTryHeliumStudioFM] using direct Moshi reply hidden; batch Helium extractor skipped", flush=True)
            return

        from generate_helium import load_mimi_and_lm

        t0 = time.perf_counter()
        helium_mimi, helium_lm, _ = load_mimi_and_lm(args)
        helium_mimi.eval()
        helium_lm.eval()
        print(
            f"[liveTryHeliumFM] Moshi loaded in {_ms(t0):.0f}ms "
            f"dtype={next(helium_lm.parameters()).dtype}",
            flush=True,
        )

        self.extractor = HeliumExtractor(helium_mimi, helium_lm, self.device)

    def reset_session(self) -> None:
        """Call when a new WebSocket client connects or sends 'start'."""
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0
        self._pcm_accum = np.empty(0, dtype=np.float32)
        if self.extractor is not None:
            self.extractor.reset()
        self._session_motion_parts = []
        self._session_helium_parts = []
        self._session_adapter_50_parts = []
        self._session_adapter_25_parts = []
        self._session_projected_audio_parts = []
        self._session_audio_parts = []
        self._session_live_token_parts = []
        self._session_chunk_rows = []
        self._session_reply_events = []
        self._session_started_wall = time.time()

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_pcm = np.zeros(self.audio_chunk_samples, dtype=np.float32)

        t0 = time.perf_counter()
        if self.extractor is None:
            raw_steps = max(1, int(round(self.fm_chunk_frames * 12.5 / float(self.fps))))
            if self.silence_helium_seed is not None:
                dummy_helium = self.silence_helium_seed.expand(
                    raw_steps, -1
                ).contiguous()
            else:
                dummy_helium = torch.zeros(
                    raw_steps, 4096, device=self.device, dtype=torch.float32
                )
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium=skipped direct_hidden raw_steps={raw_steps}", flush=True)
        else:
            dummy_helium = self.extractor.extract_raw_chunk(dummy_pcm)
            _sync_cuda()
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium={_ms(t0):.0f}ms", flush=True)
            self.extractor.reset()

        t0 = time.perf_counter()
        motion, _info = self._sample_motion_from_helium(dummy_helium, self.fm_chunk_frames)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] fm={_ms(t0):.0f}ms motion={tuple(motion.shape)}", flush=True)
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0

        t0 = time.perf_counter()
        dummy_motion = torch.zeros(self.render_sub_batch, 32, device=self.device, dtype=self.dtype)
        _frames, _timings = self._render_motion(dummy_motion)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] renderer={_ms(t0):.0f}ms", flush=True)
        self._render_frame_cursor = 0

        # Warmup JPEG pool
        t0 = time.perf_counter()
        dummy_np = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = encode_jpeg_b64(dummy_np, self.jpeg_quality)
        print(f"[liveTryHeliumFM][warmup] jpeg={_ms(t0):.0f}ms", flush=True)

        self.stream_state = None

    def feed_pcm(self, pcm_s16le_bytes: bytes) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.feed_pcm_f32(pcm)

    def feed_pcm_f32(self, pcm_f32: np.ndarray) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.asarray(pcm_f32, dtype=np.float32)
        self._pcm_accum = np.concatenate([self._pcm_accum, pcm])
        if self._pcm_accum.shape[0] < self.audio_chunk_samples:
            return None
        chunk = self._pcm_accum[:self.audio_chunk_samples].copy()
        self._pcm_accum = self._pcm_accum[self.audio_chunk_samples:]
        motion, info = self._process_pcm_chunk(chunk, self.fm_chunk_frames)
        self._record_session_chunk(chunk, motion, info)
        return motion, info, chunk

    @torch.no_grad()
    def _process_pcm_chunk(self, pcm_chunk: np.ndarray, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        target_frames = max(1, min(int(target_frames), self.fm_chunk_frames))
        if self.extractor is None:
            raise RuntimeError("direct_reply_hidden mode cannot process raw browser audio directly")

        t0 = time.perf_counter()
        helium = self.extractor.extract_raw_chunk(pcm_chunk)
        timings["helium_ms"] = _ms(t0)
        motion, fm_info = self._sample_motion_from_helium(helium, target_frames)
        timings.update(fm_info)
        return motion, timings

    @torch.no_grad()
    def _sample_motion_from_helium(self, helium: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        t_adapter = time.perf_counter()
        helium = helium.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        current_steps = int(helium.shape[0])
        deque_size = int(getattr(self, "helium_deque_size", 100))
        if self.helium_deque is None:
            if self.silence_helium_seed is not None:
                self.helium_deque = self.silence_helium_seed.expand(
                    deque_size, -1
                ).clone()
            else:
                self.helium_deque = torch.zeros(
                    deque_size,
                    helium.shape[1],
                    device=self.device,
                    dtype=torch.float32,
                )
            self.helium_deque_filled = 0
        if current_steps >= deque_size:
            self.helium_deque = helium[-deque_size:].detach().clone()
            self.helium_deque_filled = deque_size
        else:
            self.helium_deque = torch.cat([self.helium_deque[current_steps:], helium], dim=0).contiguous()
            self.helium_deque_filled = min(deque_size, int(self.helium_deque_filled) + current_steps)

        adapter_window_mode = str(
            getattr(self.args, "adapter_window_mode", "tail")
        )
        if adapter_window_mode == "lookahead":
            # Match training: process the full 8-second window at 50 Hz,
            # emit .96 seconds, and retain .48 seconds as future context.
            future_steps = int(getattr(self.args, "adapter_future_steps", 6))
            target_len_50_full = deque_size * 4
            _baseline, _cnn, feat_50_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_50_full
            )
            if self.live_sliding_window:
                # Type AC: keep the Type A Helium deque + adapter path, but
                # give IMTalker/FM a small [past + current + future] feature
                # window and emit only current frames. This places lookahead in
                # IMTalker instead of discarding future context inside adapter.
                past_25 = int(self.slide_past_frames)
                future_25 = int(self.slide_future_frames)
                current_25 = int(target_frames)
                past_50 = past_25 * 2
                future_50 = future_25 * 2
                current_50 = current_25 * 2
                full_len_50 = int(feat_50_full.shape[0])
                current_end_50 = full_len_50 - future_50
                current_start_50 = current_end_50 - current_50
                window_start_50 = current_start_50 - past_50
                window_end_50 = current_end_50 + future_50
                if window_start_50 < 0 or current_start_50 < 0 or window_end_50 > full_len_50:
                    raise RuntimeError(
                        "Sliding adapter window is out of range: "
                        f"full={full_len_50} start={window_start_50} "
                        f"current_start={current_start_50} end={window_end_50}"
                    )
                feat_50 = feat_50_full[window_start_50:window_end_50].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=past_25 + current_25 + future_25,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
            else:
                emitted_frames_50 = max(1, current_steps * 4)
                future_frames_50 = max(0, future_steps * 4)
                segment_end = int(feat_50_full.shape[0]) - future_frames_50
                segment_start = segment_end - emitted_frames_50
                if segment_start < 0:
                    raise RuntimeError(
                        "Look-ahead adapter output is shorter than its emit/future region"
                    )
                feat_50 = feat_50_full[segment_start:segment_end].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
        else:
            # Legacy behavior: predict at 25 Hz and emit the newest tail.
            target_len_25_full = deque_size * 2
            _baseline, _cnn, feat_25_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_25_full
            )
            fresh_frames = max(1, current_steps * 2)
            if int(feat_25_full.shape[0]) < fresh_frames:
                raise RuntimeError(
                    f"Deque adapter output too short: got "
                    f"{feat_25_full.shape[0]}, need {fresh_frames}"
                )
            feat_25 = feat_25_full[-fresh_frames:].contiguous()
            feat_50 = feat_25
        if (not self.live_sliding_window) and int(feat_25.shape[0]) != target_frames:
            feat_25 = F.interpolate(
                feat_25.T.unsqueeze(0),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0).T.contiguous()
        projected_a = self.fm._project_audio(feat_25.unsqueeze(0).float())
        timings["adapter_ms"] = _ms(t_adapter)
        timings["helium_ms"] = timings["adapter_ms"]
        timings["helium_deque_filled"] = int(self.helium_deque_filled)

        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}
        if self.noise_buf is not None:
            if self.live_sliding_window:
                noise_start = max(0, self.abs_frame - int(self.slide_past_frames))
                noise_end = noise_start + int(feat_25.shape[0])
            else:
                noise_start = self.abs_frame
                noise_end = self.abs_frame + target_frames
            data["noise_init"] = self.noise_buf[:, noise_start:noise_end]
        t_fm = time.perf_counter()
        if self.live_sliding_window:
            motion = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
            )
        else:
            motion, self.stream_state = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
                stream_state=self.stream_state,
                return_state=True,
            )
        timings["fm_ms"] = _ms(t_fm)

        motion = motion.squeeze(0).detach()
        if self.live_sliding_window:
            start = int(self.slide_past_frames)
            motion = motion[start:start + target_frames]
        else:
            motion = motion[:target_frames]
        ref_blend = float(getattr(self.args, "motion_ref_blend", 0.0) or 0.0)
        if ref_blend > 0.0:
            ref_motion = self.ref_x.detach().float()
            if ref_motion.ndim == 3:
                ref_motion = ref_motion[0]
            if ref_motion.ndim == 1:
                ref_motion = ref_motion.unsqueeze(0)
            if int(ref_motion.shape[0]) == 1:
                ref_motion = ref_motion.expand(int(motion.shape[0]), -1)
            elif int(ref_motion.shape[0]) != int(motion.shape[0]):
                ref_motion = F.interpolate(
                    ref_motion.T.unsqueeze(0),
                    size=int(motion.shape[0]),
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T
            ref_motion = ref_motion.to(device=motion.device, dtype=motion.dtype)
            blend = max(0.0, min(1.0, ref_blend))
            motion = motion.mul(1.0 - blend).add(ref_motion, alpha=blend)
        timings["helium_feat"] = helium.detach().cpu()
        timings["adapter_feat_50"] = feat_50.detach().cpu()
        timings["adapter_feat_25"] = feat_25.detach().cpu()
        timings["projected_audio"] = projected_a.squeeze(0).detach().cpu()
        timings["frames"] = int(motion.shape[0])
        timings["abs_start"] = self.abs_frame
        self.abs_frame += timings["frames"]
        return motion, timings

    def _record_session_chunk(self, pcm_chunk: np.ndarray, motion: torch.Tensor, info: dict) -> None:
        self._session_audio_parts.append(np.asarray(pcm_chunk, dtype=np.float32).copy())
        self._session_motion_parts.append(motion.detach().float().cpu().clone())
        helium_feat = info.get("helium_feat")
        if isinstance(helium_feat, torch.Tensor):
            self._session_helium_parts.append(helium_feat.float().cpu().clone())
        adapter_feat_50 = info.get("adapter_feat_50")
        if isinstance(adapter_feat_50, torch.Tensor):
            self._session_adapter_50_parts.append(adapter_feat_50.float().cpu().clone())
        adapter_feat_25 = info.get("adapter_feat_25")
        if isinstance(adapter_feat_25, torch.Tensor):
            self._session_adapter_25_parts.append(adapter_feat_25.float().cpu().clone())
        projected_audio = info.get("projected_audio")
        if isinstance(projected_audio, torch.Tensor):
            self._session_projected_audio_parts.append(projected_audio.float().cpu().clone())
        self._session_chunk_rows.append({
            "chunk": len(self._session_chunk_rows) + 1,
            "abs_start": int(info.get("abs_start", 0)),
            "frames": int(info.get("frames", int(motion.shape[0]))),
            "samples": int(len(pcm_chunk)),
            "helium_ms": float(info.get("helium_ms", 0.0)),
            "fm_ms": float(info.get("fm_ms", 0.0)),
        })

    @torch.no_grad()
    def _extract_wav2vec_raw_50hz(self, audio_24k: np.ndarray) -> torch.Tensor:
        arr = np.asarray(audio_24k, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return torch.empty((0, 768), dtype=torch.float32)
        wav = torch.from_numpy(arr).view(1, -1)
        wav16 = torchaudio.functional.resample(wav, TARGET_SR, WAV2VEC_SR).squeeze(0).contiguous().numpy()
        inputs = self.wav2vec_feature_extractor(
            wav16,
            sampling_rate=WAV2VEC_SR,
            return_tensors="pt",
            padding=True,
        )
        kwargs = {
            "input_values": inputs.input_values.to(self.device),
        }
        if getattr(inputs, "attention_mask", None) is not None:
            kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        frontend = self.wav2vec_model.extract_projected_frontend(**kwargs)
        feat = self.wav2vec_model.encode_from_projected_frontend(
            frontend
        ).last_hidden_state.detach().float().cpu()[0].contiguous()
        return feat

    def dump_last_session(self, *, source: str = "") -> Optional[Path]:
        if not self.dump_motion or not self._session_motion_parts:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.dump_dir / "last_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        motion = torch.cat(self._session_motion_parts, dim=0).contiguous()
        audio = np.concatenate(self._session_audio_parts, axis=0) if self._session_audio_parts else np.empty(0, dtype=np.float32)

        motion_path = session_dir / "full_motion.pt"
        helium_path = session_dir / "full_helium_raw.pt"
        adapter_50_path = session_dir / "full_adapter_w2v_50hz.pt"
        adapter_25_path = session_dir / "full_adapter_w2v_25fps.pt"
        wav2vec_50_path = session_dir / "full_wav2vec_50hz.pt"
        projected_audio_path = session_dir / "full_projected_audio_32.pt"
        audio_path = session_dir / "full_moshi_reply_24k.wav"
        live_tokens_path = session_dir / "live_mimi_tokens.pt"
        reply_events_path = session_dir / "reply_events.jsonl"
        reply_text_path = session_dir / "reply_text.txt"
        meta_path = session_dir / "meta.json"
        helium = None
        adapter_50 = None
        adapter_25 = None
        wav2vec_50 = None
        projected_audio = None
        live_tokens = None
        if self._session_helium_parts:
            helium = torch.cat(self._session_helium_parts, dim=0).contiguous()
        if self._session_adapter_50_parts:
            adapter_50 = torch.cat(self._session_adapter_50_parts, dim=0).contiguous()
        if self._session_adapter_25_parts:
            adapter_25 = torch.cat(self._session_adapter_25_parts, dim=0).contiguous()
        if self._session_projected_audio_parts:
            projected_audio = torch.cat(self._session_projected_audio_parts, dim=0).contiguous()
        if self._session_live_token_parts:
            live_tokens = torch.cat(self._session_live_token_parts, dim=2).contiguous()
        if audio.size > 0:
            wav2vec_50 = self._extract_wav2vec_raw_50hz(audio)

        torch.save({
            "motion": motion,
            "chunks": self._session_chunk_rows,
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 768)),
            "audio_adapter_dim": int(getattr(self.args, "audio_adapter_dim", 512)),
            "wav2vec_sec": float(self.args.wav2vec_sec),
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
            "source": source,
        }, motion_path)
        if helium is not None:
            torch.save({
                "helium": helium,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "audio_chunk_sec": float(self.audio_chunk_sec),
                "fm_chunk_frames": int(self.fm_chunk_frames),
                "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 4096)),
                "source": source,
            }, helium_path)
        if adapter_50 is not None:
            torch.save({
                "adapter_feat_50": adapter_50,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, adapter_50_path)
        if adapter_25 is not None:
            torch.save({
                "adapter_feat_25": adapter_25,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, adapter_25_path)
        if wav2vec_50 is not None:
            torch.save({
                "wav2vec_50hz": wav2vec_50,
                "chunks": self._session_chunk_rows,
                "sample_rate": int(WAV2VEC_SR),
                "source": source,
            }, wav2vec_50_path)
        if projected_audio is not None:
            torch.save({
                "projected_audio": projected_audio,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, projected_audio_path)
        if live_tokens is not None:
            torch.save({
                "live_mimi_tokens": live_tokens,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, live_tokens_path)
        if audio.size > 0:
            torchaudio.save(str(audio_path), torch.from_numpy(audio).view(1, -1), TARGET_SR)
        if self._session_reply_events:
            with reply_events_path.open("w", encoding="utf-8") as f:
                for row in self._session_reply_events:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            reply_text = "".join(str(row.get("piece", "")) for row in self._session_reply_events)
            reply_text_path.write_text(reply_text, encoding="utf-8")
        meta_path.write_text(json.dumps({
            "source": source,
            "session_started_wall": float(self._session_started_wall),
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "motion_frames": int(motion.shape[0]),
            "helium_frames": int(helium.shape[0]) if helium is not None else 0,
            "adapter_50_frames": int(adapter_50.shape[0]) if adapter_50 is not None else 0,
            "adapter_25_frames": int(adapter_25.shape[0]) if adapter_25 is not None else 0,
            "wav2vec_50_frames": int(wav2vec_50.shape[0]) if wav2vec_50 is not None else 0,
            "projected_audio_frames": int(projected_audio.shape[0]) if projected_audio is not None else 0,
            "audio_samples": int(audio.shape[0]),
            "audio_seconds": float(audio.shape[0] / TARGET_SR) if audio.size > 0 else 0.0,
            "reply_text_chars": int(sum(len(str(row.get("piece", ""))) for row in self._session_reply_events)),
            "reply_events": int(len(self._session_reply_events)),
            "chunks": self._session_chunk_rows,
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
        }, indent=2), encoding="utf-8")
        print(f"[liveTryHeliumFM] dumped last session -> {session_dir}", flush=True)
        return session_dir

    def _extract_motion_tensor_from_payload(self, payload, path: str) -> torch.Tensor:
        if isinstance(payload, torch.Tensor):
            motion = payload
        elif isinstance(payload, dict):
            candidates = ["motion", "motion_latents", "latents", "full_motion", "pred_motion", "x"]
            motion = None
            for key in candidates:
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    motion = value
                    break
            if motion is None:
                tensor_items = [
                    value
                    for value in payload.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == int(self.args.dim_w)
                ]
                if tensor_items:
                    motion = tensor_items[0]
            if motion is None:
                raise ValueError(
                    f"No motion tensor found in blink_motion_path={path}. "
                    f"Available keys={list(payload.keys())[:20]}"
                )
        else:
            raise TypeError(f"Unsupported blink motion payload type {type(payload)} from {path}")

        motion = motion.detach().float().cpu()
        if motion.ndim == 3 and motion.shape[0] == 1:
            motion = motion[0]
        if motion.ndim != 2 or motion.shape[-1] != int(self.args.dim_w):
            raise ValueError(
                f"Blink motion must have shape (T,{int(self.args.dim_w)}) or (1,T,{int(self.args.dim_w)}); "
                f"got {tuple(motion.shape)} from {path}"
            )
        return motion.contiguous()

    def _make_eye_mask(self, height: int, width: int) -> torch.Tensor:
        yy = torch.linspace(0.0, 1.0, int(height), device=self.device, dtype=self.dtype).view(1, 1, height, 1)
        xx = torch.linspace(0.0, 1.0, int(width), device=self.device, dtype=self.dtype).view(1, 1, 1, width)
        center_y = float(getattr(self.args, "eye_center_y", 0.405))
        radius_x = max(float(getattr(self.args, "eye_radius_x", 0.145)), 1e-6)
        radius_y = max(float(getattr(self.args, "eye_radius_y", 0.070)), 1e-6)
        feather = max(float(getattr(self.args, "eye_feather", 0.10)), 1e-6)

        mask = torch.zeros(1, 1, height, width, device=self.device, dtype=self.dtype)
        for center_x in (
            float(getattr(self.args, "eye_left_x", 0.36)),
            float(getattr(self.args, "eye_right_x", 0.64)),
        ):
            dist = ((xx - center_x) / radius_x).square() + ((yy - center_y) / radius_y).square()
            ellipse = ((1.0 + feather) - dist).clamp(0.0, feather) / feather
            mask = torch.maximum(mask, ellipse)
        return mask.clamp(0.0, 1.0).contiguous()

    @torch.no_grad()
    def _init_eye_blink_composite(self) -> None:
        blink_path = str(getattr(self.args, "blink_motion_path", "") or "")
        if not blink_path:
            raise ValueError("--enable_eye_blink_composite requires --blink_motion_path")
        if not Path(blink_path).is_file():
            raise FileNotFoundError(f"blink_motion_path does not exist: {blink_path}")

        payload = torch.load(blink_path, map_location="cpu")
        blink_motion = self._extract_motion_tensor_from_payload(payload, blink_path)
        blink_maps_parts: list[list[torch.Tensor]] = []
        chunk = max(1, int(getattr(self.args, "render_sub_batch", 8)))
        for start in range(0, int(blink_motion.shape[0]), chunk):
            sub = blink_motion[start:start + chunk].to(self.device, dtype=self.dtype)
            g_sub = self.g_r.expand(int(sub.shape[0]), -1)
            ta_b = self.renderer.adapt(sub, g_sub)
            maps_b = self.renderer.latent_token_decoder(ta_b)
            if not blink_maps_parts:
                blink_maps_parts = [[] for _ in range(len(maps_b))]
            for idx, map_b in enumerate(maps_b):
                blink_maps_parts[idx].append(map_b.detach())

        self._blink_maps = tuple(torch.cat(parts, dim=0).contiguous() for parts in blink_maps_parts)
        self._eye_masks = tuple(self._make_eye_mask(m.shape[-2], m.shape[-1]) for m in self._blink_maps)
        _sync_cuda()
        shapes = [tuple(m.shape) for m in self._blink_maps]
        print(
            f"[liveTryHeliumFM][blink] cached blink maps from {blink_path}: "
            f"frames={int(blink_motion.shape[0])} shapes={shapes}",
            flush=True,
        )

    def _composite_eye_blink_maps(
        self,
        current_maps: tuple[torch.Tensor, ...] | list[torch.Tensor],
        start_frame: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, ...]:
        if self._blink_maps is None or self._eye_masks is None:
            return tuple(current_maps)
        blink_len = int(self._blink_maps[0].shape[0])
        if blink_len <= 0:
            return tuple(current_maps)
        indices = (torch.arange(int(num_frames), device=self.device) + int(start_frame)) % blink_len
        composited: list[torch.Tensor] = []
        for cur, blink_all, mask in zip(current_maps, self._blink_maps, self._eye_masks):
            blink = blink_all.index_select(0, indices).to(device=cur.device, dtype=cur.dtype)
            mask = mask.to(device=cur.device, dtype=cur.dtype)
            composited.append(blink * mask + cur * (1.0 - mask))
        return tuple(composited)

    @torch.no_grad()
    def _render_motion(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        timings: dict = {}
        t_total = time.perf_counter()
        motion = motion.to(self.device, dtype=self.dtype)
        n = int(motion.shape[0])

        render_start = self._render_frame_cursor
        fused = getattr(self.renderer, '_fused_render', None)
        if self.eye_blink_enabled:
            fused = None

        # torch.compile specializes the renderer to its warmup batch shape.
        # Pad the final short sub-batch so a 50-frame chunk does not compile a
        # second graph for the 2-frame tail during the first live response.
        render_motion = motion
        render_n = n
        if fused is not None and n < self.render_sub_batch:
            pad_n = self.render_sub_batch - n
            render_motion = torch.cat(
                [motion, motion[-1:].expand(pad_n, -1)],
                dim=0,
            ).contiguous()
            render_n = self.render_sub_batch

        g_r_sub = self.g_r.expand(render_n, -1)
        m_r_sub = tuple(m.expand(render_n, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(render_n, -1, -1, -1) for f in self.f_r]
        if fused is not None:
            frames = fused(render_motion, g_r_sub, m_r_sub, f_r_sub)[:n]
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            m_c = self.renderer.latent_token_decoder(ta_c)
            if self.eye_blink_enabled:
                m_c = self._composite_eye_blink_maps(m_c, render_start, n)
            frames = self.renderer.decode(m_c, m_r_sub, f_r_sub)
        self._render_frame_cursor += n

        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        timings["total_ms"] = _ms(t_total)
        return frames_np, timings

    @torch.no_grad()
    def _render_motion_oom_safe(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        """_render_motion, but an OOM halves the batch and retries instead of
        killing the caller's thread.

        Rendering is the largest transient allocation in the process: the
        renderer's attention at resolution 64 is O(batch), so halving the batch
        halves the peak. Recursing down to a single frame is worth trying before
        giving up, because a single frame is roughly a tenth of the peak of a
        full sub-batch and will fit in almost any surviving headroom.

        If even one frame will not fit, the chunk is dropped and the caller
        continues. A dropped chunk is a visible glitch; a dead producer thread is
        a dead avatar for the rest of the session."""
        try:
            return self._render_motion(motion)
        except torch.cuda.OutOfMemoryError:
            n = int(motion.shape[0])
            torch.cuda.empty_cache()
            if n <= 1:
                free_b, total_b = torch.cuda.mem_get_info()
                msg = (
                    f"renderer OOM on a single frame "
                    f"(free={free_b / 2**30:.2f}GB of {total_b / 2**30:.2f}GB); "
                    f"dropping this chunk and continuing"
                )
                print(f"[RENDER-OOM] {msg}", flush=True)
                with contextlib.suppress(Exception):
                    system_logger.get().event(
                        "render_oom", msg,
                        frames=n, free_gb=round(free_b / 2**30, 2),
                        recovered=False,
                    )
                raise
            half = n // 2
            print(
                f"[RENDER-OOM] sub-batch of {n} did not fit; retrying as "
                f"{half}+{n - half}. Lower --render_sub_batch to avoid this.",
                flush=True,
            )
            with contextlib.suppress(Exception):
                system_logger.get().event(
                    "render_oom",
                    f"sub-batch of {n} frames did not fit; split into "
                    f"{half}+{n - half} and continued",
                    frames=n, split_into=[half, n - half], recovered=True,
                )
            first, t1 = self._render_motion_oom_safe(motion[:half])
            second, t2 = self._render_motion_oom_safe(motion[half:])
            timings = dict(t1)
            timings["total_ms"] = float(t1.get("total_ms", 0.0)) + float(t2.get("total_ms", 0.0))
            timings["oom_split"] = True
            return np.concatenate([first, second], axis=0), timings

    def render_and_encode_subbatch(
        self,
        motion_sub: torch.Tensor,
        audio_slices: list[np.ndarray],
        abs_start: int,
        text_payload: str,
        avatar_chunk_id: int,
        total_gen_ms: float,
    ) -> list[dict]:
        """Render a sub-batch of frames, JPEG-encode in parallel, return packet dicts."""
        frames_np, _render_info = self._render_motion_oom_safe(motion_sub)

        jpeg_futures = []
        for frame_rgb in frames_np:
            jpeg_futures.append(
                self._jpeg_pool.submit(encode_jpeg_bytes, frame_rgb, self.jpeg_quality)
            )

        packets = []
        gen_ms_i = int(round(float(total_gen_ms)))
        sr_i = int(round(float(TARGET_SR)))
        for j, fut in enumerate(jpeg_futures):
            idx = abs_start + j
            audio_slice = audio_slices[j] if j < len(audio_slices) else np.zeros(
                int(round(TARGET_SR / self.fps)), dtype=np.float32
            )
            jpeg_bytes = fut.result()
            output_audio_codec = str(
                getattr(self.args, "output_audio_codec", "pcm")
            ).lower()
            pcm_b = (
                b""
                if output_audio_codec == "opus"
                else _pcm_f32_to_i16_bytes(audio_slice)
            )
            blob = _wsbin.pack_av_frame(
                idx,
                idx + 1,
                gen_ms_i,
                sr_i,
                jpeg_bytes,
                pcm_b,
                text_payload,
                int(avatar_chunk_id),
            )
            packets.append(
                {
                    "frame_number": idx,
                    "ws_kind": "bytes",
                    "data": blob,
                    "audio_pcm": np.asarray(audio_slice, dtype=np.float32).copy(),
                    "t_ready": time.perf_counter(),
                }
            )
        return packets

    def audio_slice(self, frame_idx: int) -> np.ndarray:
        if self.audio_pcm is None:
            frame_samples = int(round(TARGET_SR / self.fps))
            return np.zeros(frame_samples, dtype=np.float32)
        frame_samples = int(round(TARGET_SR / self.fps))
        start = frame_idx * frame_samples
        chunk = self.audio_pcm[start:start + frame_samples]
        if chunk.shape[0] < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.shape[0]))
        return chunk


# ---------------------------------------------------------------------------
# WebSocket streaming coroutine (file-driven mode, unchanged)
# ---------------------------------------------------------------------------

async def stream_from_file(ws: WebSocket, engine: LiveHeliumFMEngine) -> None:
    """Simulate live streaming using --audio_path, sending frames back over WS."""
    if engine.audio_pcm is None:
        await ws.send_json({"type": "error", "msg": "No --audio_path given for file-streaming mode"})
        return

    audio = engine.audio_pcm
    lookahead_chunks = max(0, int(getattr(engine.args, "file_chunk_lookahead", 0)))
    total_chunks = int(np.ceil(len(audio) / engine.audio_chunk_samples))
    start_wall = time.perf_counter()
    emitted = 0

    print(
        f"[liveTryHeliumFM] stream_from_file: {total_chunks} chunks "
        f"lookahead={lookahead_chunks}",
        flush=True,
    )

    async def _emit_motion_chunk(
        motion: torch.Tensor,
        fm_info: dict,
        chunk_label: int,
        emitted_so_far: int,
    ) -> int:
        helium_ms = float(fm_info["helium_ms"])
        fm_ms = float(fm_info["fm_ms"])
        n_frames = int(motion.shape[0])
        all_frames_np: list[np.ndarray] = []
        render_ms = 0.0
        for sb_start in range(0, n_frames, engine.render_sub_batch):
            sub = motion[sb_start:sb_start + engine.render_sub_batch].to(
                engine.device, dtype=engine.dtype
            )
            frames_np, render_info = engine._render_motion_oom_safe(sub)
            render_ms += float(render_info["total_ms"])
            all_frames_np.extend(frames_np)

        print(
            f"[liveTryHeliumFM][chunk#{chunk_label}] "
            f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
            f"render={render_ms:.0f}ms frames={n_frames} "
            f"abs_start={fm_info['abs_start']}",
            flush=True,
        )

        for j, frame_rgb in enumerate(all_frames_np):
            idx = emitted_so_far + j
            chunk_id = idx + 1
            audio_b64 = _pcm_f32_to_i16_b64(engine.audio_slice(idx))
            jpeg_b64 = encode_jpeg_b64(frame_rgb, engine.jpeg_quality)

            await ws.send_json({
                "type": "chunk_audio",
                "chunk_id": chunk_id,
                "sample_rate": TARGET_SR,
                "pcm_s16le_b64": audio_b64,
            })
            await ws.send_json({
                "type": "chunk_frame",
                "chunk_id": chunk_id,
                "frame_idx": 0,
                "jpeg_b64": jpeg_b64,
                "moshi_text": (
                    f"Helium+FM | chunk#{chunk_label} "
                    f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
                    f"render={render_ms:.0f}ms"
                ),
                "server_fps": round(float(engine.fps), 1),
                "chunks_done": chunk_label,
            })
            target_t = start_wall + (idx + 1) / engine.fps
            await asyncio.sleep(max(0.0, target_t - time.perf_counter()))
        return emitted_so_far + len(all_frames_np)

    if lookahead_chunks <= 0:
        for chunk_idx in range(total_chunks):
            pcm_chunk = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ]
            pcm_real = pcm_chunk.copy()
            target_frames = int(round(pcm_chunk.shape[0] * engine.fps / TARGET_SR))
            if pcm_chunk.shape[0] < engine.audio_chunk_samples:
                pcm_chunk = np.pad(pcm_chunk, (0, engine.audio_chunk_samples - pcm_chunk.shape[0]))

            motion, fm_info = engine._process_pcm_chunk(pcm_chunk, target_frames)
            engine._record_session_chunk(pcm_real, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_idx + 1, emitted)
    else:
        pending_real: list[np.ndarray] = []
        prefix_audio = np.empty(0, dtype=np.float32)
        chunk_counter = 0

        def _process_exact_pending(pcm_real: np.ndarray) -> tuple[torch.Tensor, dict]:
            nonlocal prefix_audio
            target_frames = int(round(pcm_real.shape[0] * engine.fps / TARGET_SR))
            t0 = time.perf_counter()
            helium = engine.extractor.extract_exact_chunk_from_prefix(
                prefix_audio,
                engine.abs_frame,
                target_frames,
            )
            _sync_cuda()
            fm_info: dict = {"helium_ms": _ms(t0)}
            motion, sample_info = engine._sample_motion_from_helium(helium, target_frames)
            fm_info.update(sample_info)
            return motion, fm_info

        for chunk_idx in range(total_chunks):
            pcm_real = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ].copy()
            prefix_audio = np.concatenate([prefix_audio, pcm_real], axis=0)
            pending_real.append(pcm_real)

            while len(pending_real) > lookahead_chunks:
                chunk_counter += 1
                oldest = pending_real.pop(0)
                motion, fm_info = _process_exact_pending(oldest)
                engine._record_session_chunk(oldest, motion, fm_info)
                emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

        while pending_real:
            chunk_counter += 1
            oldest = pending_real.pop(0)
            motion, fm_info = _process_exact_pending(oldest)
            engine._record_session_chunk(oldest, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

    await ws.send_json({"type": "stream_end", "total_frames": emitted})
    engine.dump_last_session(source=str(engine.args.audio_path))
    print(f"[liveTryHeliumFM] stream done: {emitted} frames", flush=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LiveHeliumFMOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.set_defaults(wav2vec_sec=0.96)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8998)
        parser.add_argument("--html_path", default=str(ROOT / "static" / "index_v3_binary_fullscreen.html"))
        parser.add_argument("--generator_path", required=True)
        parser.add_argument("--lora_generator_path", default="", help="Optional LoRA generator checkpoint to apply on top of --generator_path")
        parser.add_argument("--lora_rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=float, default=128.0)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        parser.add_argument("--no_lora_pose_projection", action="store_true")
        parser.add_argument("--no_lora_audio_projection", action="store_true")
        parser.add_argument("--only_lora_pose_projection", action="store_true")
        parser.add_argument("--renderer_path", required=True)
        parser.add_argument("--adapter_path", required=True, help="Frontend fp32 Helium->Wav2Vec2 projected-frontend adapter checkpoint")
        parser.add_argument(
            "--adapter_type",
            choices=("frontend", "unitalk_last_layer"),
            default="frontend",
            help="Adapter architecture. UniTalk passes only layer 12 to IMTalker.",
        )
        parser.add_argument(
            "--adapter_window_mode",
            choices=("tail", "lookahead"),
            default="tail",
        )
        parser.add_argument(
            "--adapter_future_steps",
            type=int,
            default=6,
            help="12.5 Hz future-context steps retained in lookahead mode.",
        )
        parser.add_argument("--adapter_num_layers", type=int, default=6, help="Transformer layers in the frontend adapter checkpoint")
        parser.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout value used when constructing the frontend adapter")
        parser.add_argument("--stats_path", default="", help="Unused for frontend adapter mode; accepted for compatibility")
        parser.add_argument(
            "--silence_helium_path",
            default="",
            help="Optional real-silence Helium mean used to initialize the deque.",
        )
        parser.add_argument("--ref_path", required=True)
        parser.add_argument("--audio_path", default="", help="WAV to stream in fixed chunks (simulate-live mode)")
        # Moshi
        parser.add_argument("--moshi_root", default="/workspace/moshi")
        parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
        parser.add_argument("--moshi_weight", default="", help="Optional local PersonaPlex/Moshi LM checkpoint")
        parser.add_argument("--mimi_weight", default="", help="Optional local Mimi checkpoint")
        parser.add_argument("--tokenizer", default="", help="Optional local sentencepiece tokenizer")
        parser.add_argument("--quantize_4bit", action="store_true", help="Load PersonaPlex/Moshi LM with bnb 4-bit quantization")
        parser.add_argument("--num_codebooks", type=int, default=8, help="PersonaPlex/Moshi audio codebooks")
        parser.add_argument("--moshi_context", type=int, default=0, help="Optional PersonaPlex/Moshi KV context length")
        parser.add_argument("--voice_prompt", default="", help="PersonaPlex voice prompt filename, e.g. NATM0.pt")
        parser.add_argument("--voice_prompt_dir", default="", help="Optional PersonaPlex voice prompt directory")
        parser.add_argument("--text_prompt", default="", help="Optional PersonaPlex system text prompt")
        parser.add_argument("--moshi_reply_device", default=None, help="Optional separate CUDA device for Moshi reply generation")
        parser.add_argument("--enable_moshi_reply", action="store_true", help="Mic -> Moshi reply audio -> Helium/FM avatar")
        parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)
        parser.add_argument("--direct_reply_hidden", action="store_true", default=True, help="Use Moshi generation hidden directly instead of re-encoding reply audio")
        parser.add_argument("--no_direct_reply_hidden", dest="direct_reply_hidden", action="store_false")
        parser.add_argument(
            "--disable_assistant_output_gate",
            action="store_true",
            help="Disable assistant-output RMS gating and always drive IMTalker from live PersonaPlex hidden states",
        )
        parser.add_argument(
            "--assistant_speech_rms_threshold",
            type=float,
            default=0.006,
            help="RMS threshold on PersonaPlex assistant reply audio before hidden states are allowed to drive avatar motion",
        )
        parser.add_argument(
            "--assistant_speech_hold_chunks",
            type=int,
            default=1,
            help="Keep assistant gate open for this many avatar chunks after reply audio drops below threshold",
        )
        # -- STT + query routing + web search ------------------------------
        # All optional and all additive: omitting every flag below reproduces
        # this script's plain conversational behaviour exactly, with no extra
        # model loaded and no per-chunk cost.
        parser.add_argument("--ref_lora_dir", default="", help="Dir containing lora/ with the <lookup>/<ref> context-injection LoRA adapter")
        parser.add_argument(
            "--merge_ref_lora", action=argparse.BooleanOptionalAction, default=False,
            help="Fold the reference LoRA into the base weights at startup. DEFAULT OFF, which "
                 "matches the old pipeline: it runs this adapter unmerged (QLoRA-style, computed "
                 "at forward time on top of the 4-bit base) and holds real time comfortably on the "
                 "same GPU, so unmerged is a proven configuration and not a performance problem. "
                 "Merging is also not currently possible against a bnb-4bit base -- peft attempts "
                 "`base_layer.weight.data += delta_weight` against the PACKED quantized blob and "
                 "raises a shape mismatch -- so enabling this just takes the loud fallback path.",
        )
        parser.add_argument("--max_ref_tokens", type=int, default=250, help="Cap on injected <ref> block length, in tokens")
        parser.add_argument(
            "--wait_for_user", action=argparse.BooleanOptionalAction, default=True,
            help="Withhold the model's words until the user has actually spoken. "
                 "On by default: at session start the model otherwise free-runs "
                 "from the system prompt and reads fragments of it aloud, which "
                 "also commits it to a topic before the first question arrives. "
                 "Requires STT (that is what detects the first word). Use "
                 "--no-wait_for_user if you want a greeting before the user speaks.",
        )
        parser.add_argument(
            "--spoken_form_numbers", action="store_true",
            help="Spell numbers out in words in the injected summary "
                 "(\"three hundred nine dollars\" instead of \"$309.32\"). OFF by "
                 "default, and the default is deliberate: PersonaPlex already reads "
                 "digits aloud correctly, whereas spelled-out numbers force it to "
                 "re-encode the value and it drops magnitudes -- $309.32 was spoken "
                 "back as $39.32, and a euro price came back in dollars. Markdown, "
                 "brackets and citation markers are stripped either way.",
        )
        parser.add_argument(
            "--router_threshold", type=float, default=0.40,
            help="P(needs live data) at or above which the router sends a question to web search. "
                 "Deliberately below 0.5: searching unnecessarily costs a couple of seconds of "
                 "thinking sound and is recoverable, whereas failing to search produces a "
                 "confidently wrong answer spoken aloud. Raise it if the assistant searches too "
                 "eagerly for your users' phrasing.",
        )
        parser.add_argument(
            "--router_rules", type=int, default=1,
            help="1 (default) runs the instant regex pre-pass before the model router, so obvious "
                 "cases are decided in microseconds with no forward pass at all. This is what keeps "
                 "the no-search path as fast as a plain conversational server. Set 0 to route every "
                 "turn through the model.",
        )
        parser.add_argument(
            "--search_max_filler_sec", type=float, default=6.0,
            help="Max seconds to wait for routing + search + compression before giving up and "
                 "injecting a generic 'no information' fallback. Real search+compression regularly "
                 "takes 2.5-3.7s end to end, so a ~2s cap discards a correctly-computed answer; "
                 "6.0s leaves margin above the worst latency actually measured.",
        )
        parser.add_argument(
            "--web_search_min_score", type=float, default=0.15,
            help="Discard web-search results scoring below this relevance floor before summarizing "
                 "or injecting them. Search engines always return something, so this floor is the "
                 "only thing between an unrelated page and the assistant's spoken answer.",
        )
        parser.add_argument(
            "--stt_inline", action="store_true",
            help="Run the STT/VAD forward pass inline on the GPU thread instead of on its own "
                 "thread. Diagnostic only. Inline costs a 1B model forward plus two device syncs "
                 "inside every 80ms step, which is what pushed this pipeline below real time and "
                 "silenced the avatar; the threaded default keeps that off the critical path.",
        )
        parser.add_argument(
            "--stt_queue_frames", type=int, default=64,
            help="Frames buffered for the STT worker (64 = ~5s of audio). On overflow the OLDEST "
                 "frame is dropped and the drop is logged: STT falling behind must degrade "
                 "transcription, never stall audio and video generation.",
        )
        parser.add_argument("--stt_hf_repo", default="", help="HF repo for the STT/VAD submodel, e.g. kyutai/stt-1b-en_fr-candle. Omit to disable routing/search entirely (no transcript, no turn-detection signal).")
        parser.add_argument("--stt_pkg_dir", default="", help="Dir containing an isolated `pip install --no-deps --target <dir> moshi` install of the upstream Kyutai moshi package")
        parser.add_argument(
            "--max_input_buffer_sec", type=float, default=2.0,
            help="Maximum seconds of unprocessed microphone audio to keep. THE most important "
                 "latency control in this server: the GPU producer is rate-limited to exactly real "
                 "time by frame_q backpressure, so it can never drain a backlog. Without this cap "
                 "any transient stall (model warmup, a slow render, a client hiccup) becomes a "
                 "PERMANENT reply delay for the rest of the session, which is how a 2-4s pipeline "
                 "ends up answering 10-12s late. When the cap is exceeded the OLDEST audio is "
                 "dropped and the drop is logged. Set 0 for the old unbounded behaviour.",
        )
        parser.add_argument("--vad_threshold", type=float, default=0.5)
        parser.add_argument(
            "--suppress_text_during_search", type=int, default=1,
            help="1 (default) forces the model's text stream to its own zero_text_code for the "
                 "whole time a web search is in flight, so it composes nothing until the <ref> "
                 "arrives. Muting the outgoing audio alone is not enough: the model keeps "
                 "generating behind the filler and is already mid-sentence with an invented number "
                 "by the time the real one lands. Set 0 to restore the old behaviour.",
        )
        parser.add_argument(
            "--prompt_settle_sec", type=float, default=0.0,
            help="Seconds of forced silence appended after the system prompt. The prompt is "
                 "force-fed as the assistant's OWN speech, so generation otherwise resumes from a "
                 "context ending mid-self-description and the model carries on describing itself "
                 "instead of answering. Default 0 keeps the opening greeting; raise in small steps.",
        )
        parser.add_argument(
            "--stt_reject_foreign_script", type=int, default=1,
            help="1 (default) discards any STT transcript that is mostly non-Latin script. The "
                 "bundled STT model (kyutai/stt-1b-en_fr) produces only English and French, so such "
                 "a transcript is decode garbage, not a language surprise -- routing on it answers "
                 "a question nobody asked and can send nonsense to a paid search API. The avatar is "
                 "unaffected either way: PersonaPlex hears the real audio directly.",
        )
        parser.add_argument(
            "--stt_require_english", type=int, default=1,
            help="1 (default) also discards transcripts that are another LANGUAGE in Latin script. "
                 "The STT model is bilingual (en/fr) and on unclear audio hallucinates fluent "
                 "Spanish/French, which the script check above cannot see. Detection is lexical "
                 "(function words plus accent rate) so it costs microseconds and needs no model.",
        )
        parser.add_argument(
            "--stt_max_non_latin_ratio", type=float, default=0.15,
            help="Fraction of non-Latin LETTERS (digits/punctuation ignored) a transcript may "
                 "contain before --stt_reject_foreign_script discards it. Not 0: one stray decoded "
                 "symbol in an otherwise clean English sentence should not throw the turn away.",
        )
        parser.add_argument("--compressor_model", default="", help="HF model id for the small instruct model that BOTH routes queries and compresses web results, e.g. Qwen/Qwen2.5-1.5B-Instruct. One model does both jobs, so routing costs no extra VRAM and no extra load time. Omit to disable routing and search entirely.")
        parser.add_argument("--compressor_device", default="cuda")
        parser.add_argument("--compressor_4bit", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--compressor_max_passages", type=int, default=2)
        parser.add_argument("--web_search_enabled", action="store_true")
        parser.add_argument("--web_search_api_key", default=None, help="Never hardcode this; pass via env var / secure prompt")
        parser.add_argument("--web_search_provider", default="tavily", choices=("tavily", "serper", "bing"))
        parser.add_argument("--web_search_max_results", type=int, default=3)
        parser.add_argument("--web_search_timeout", type=float, default=3.0)
        parser.add_argument(
            "--thinking_sound_path",
            default=str(ROOT.parent / "assets" / "ai-thinking-sound.wav"),
            help="WAV played over the assistant audio channel ONLY while an online search is "
                 "actually running (looped if shorter than the wait, stopped the instant the "
                 "grounded <ref> is injected or the search times out). Turns answered from the "
                 "model's own knowledge stay silent: the model starts speaking within a few "
                 "hundred ms there, so a filler would only delay and mask the reply. "
                 "Set to '' to disable.",
        )
        parser.add_argument(
            "--conversation_log_dir", default="",
            help="Directory for the two structured log files. system_<session>.log records what "
                 "the process IS (models, sources, LoRA paths, video config, GPU, startup timing); "
                 "conversation_<session>.log / detailed_<session>.log record what was SAID (every "
                 "transcript, router decision, search, summary, injection, reply and per-stage "
                 "duration). Both carry millisecond timestamps. Empty disables file logging; "
                 "console logging always happens.",
        )
        # FM
        parser.add_argument("--audio_chunk_sec", type=float, default=0.96)
        parser.add_argument("--fm_chunk_frames", type=int, default=24, help="Must match wav2vec_sec×fps")
        parser.add_argument(
            "--helium_deque_size",
            type=int,
            default=100,
            help="Number of 12.5 Hz PersonaPlex hidden steps exposed to the adapter.",
        )
        parser.add_argument(
            "--enable_live_sliding_window",
            action="store_true",
            help="Run IMTalker/FM on a past/current/future feature window and emit only the current slice.",
        )
        parser.add_argument(
            "--slide_past_frames",
            type=int,
            default=10,
            help="Past 25Hz frames used by --enable_live_sliding_window.",
        )
        parser.add_argument(
            "--slide_future_frames",
            type=int,
            default=3,
            help="Future 25Hz frames used by --enable_live_sliding_window; 3 frames = 120ms.",
        )
        parser.add_argument("--skip_fm_audio_encoder", action="store_true", help="Skip FMGenerator's raw-audio Wav2Vec encoder; live PersonaPlex passes precomputed adapter features")
        parser.add_argument("--reply_hidden_steps_per_chunk", type=int, default=0, help="Raw Moshi 12.5Hz hidden steps per avatar chunk; 0 derives from fm_chunk_frames/fps")
        parser.add_argument("--prebuffer_chunks", type=int, default=3, help="Avatar chunks queued before sender starts pacing")
        parser.add_argument("--frame_q_backpressure", type=int, default=160)
        parser.add_argument(
            "--file_chunk_lookahead",
            type=int,
            default=0,
            help="For --audio_path mode, wait this many future chunks before emitting the oldest chunk",
        )
        parser.add_argument(
            "--render_sub_batch", type=int, default=8,
            help="Frames rendered per renderer call. Larger batches mean fewer Python round-trips "
                 "and better GPU occupancy, which shortens the render stage that sits directly on "
                 "the reply-latency path. Prefer a value that divides --fm_chunk_frames exactly "
                 "(e.g. 10 for 50-frame chunks) so the final short sub-batch never needs padding.",
        )
        parser.add_argument("--jpeg_quality", type=int, default=86)
        parser.add_argument(
            "--incremental_chunk_publish", action="store_true",
            help="Publish each render sub-batch (frames plus the audio steps those frames cover) "
                 "as soon as it is encoded, instead of holding the whole chunk until every frame "
                 "is ready. Removes the render stage from the reply-latency path -- typically the "
                 "single largest remaining win after --max_input_buffer_sec -- at the cost of the "
                 "atomic A/V publication guarantee: if a later sub-batch stalls, the browser's "
                 "playback clock has already advanced past the earlier ones. Default OFF; enable "
                 "only after confirming render time is stable on your GPU.",
        )
        parser.add_argument("--reply_audio_gain", type=float, default=1.0, help="Accepted for launch-script compatibility")
        parser.add_argument("--enable_seedvc", action="store_true")
        parser.add_argument("--seedvc_repo", default="/home/ubuntu/project/seed-vc")
        parser.add_argument("--seedvc_reference_audio", default="")
        parser.add_argument("--seedvc_steps", type=int, default=6)
        parser.add_argument("--seedvc_cfg", type=float, default=0.7)
        parser.add_argument("--seedvc_prompt_seconds", type=float, default=5.0)
        parser.add_argument(
            "--motion_ref_blend",
            type=float,
            default=0.0,
            help="Blend generated motion latent toward the reference image latent to reduce head tilt.",
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--buffer_ms", type=int, default=80)
        parser.add_argument(
            "--renderer_precision",
            choices=("fp32", "fp16", "bf16"),
            default="fp32",
        )
        parser.add_argument(
            "--output_audio_codec",
            choices=("pcm", "opus"),
            default="pcm",
            help="Assistant audio transport; Opus is one persistent stream per session.",
        )
        parser.add_argument("--dump_motion", action="store_true", help="Dump last session motion/audio to disk")
        parser.add_argument("--dump_dir", default=str(ROOT / "live_try_dumps"))
        # Shared noise
        parser.add_argument("--shared_noise", action="store_true")
        parser.add_argument("--noise_seed", type=int, default=1234)
        parser.add_argument("--noise_max_frames", type=int, default=5000)
        parser.add_argument(
            "--noise_temporal_corr",
            type=float,
            default=0.0,
            help="AR(1)-style temporal correlation applied to FM initial noise.",
        )
        parser.add_argument(
            "--motion_prior_noise_blend",
            type=float,
            default=0.0,
            help="Small blend from previous/generated motion into FM initial noise.",
        )
        # Precision
        parser.add_argument("--fp32", action="store_true")
        parser.add_argument("--tf32", action="store_true")
        parser.add_argument("--compile_renderer", action="store_true")
        # Eye blink motion-map compositing
        parser.add_argument("--enable_eye_blink_composite", action="store_true")
        parser.add_argument("--blink_motion_path", default="", help="Cached blink motion latent .pt file")
        parser.add_argument("--eye_left_x", type=float, default=0.36)
        parser.add_argument("--eye_right_x", type=float, default=0.64)
        parser.add_argument("--eye_center_y", type=float, default=0.405)
        parser.add_argument("--eye_radius_x", type=float, default=0.145)
        parser.add_argument("--eye_radius_y", type=float, default=0.070)
        parser.add_argument("--eye_feather", type=float, default=0.10)
        return parser


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Helium MotionField Deque FM liveTry")

    # The system log is opened here, before any model is constructed, so it
    # captures how the models got loaded rather than only that they exist.
    sys_log = system_logger.configure(log_dir=getattr(args, "conversation_log_dir", ""))
    sys_log.process_start()
    sys_log.torch_info()
    sys_log.config(args, keys=[
        "host", "port", "device", "html_path",
        "generator_path", "lora_generator_path", "lora_rank", "lora_alpha", "lora_dropout",
        "renderer_path", "renderer_precision", "compile_renderer",
        "adapter_path", "adapter_type", "adapter_num_layers", "adapter_window_mode",
        "adapter_future_steps", "silence_helium_path", "ref_path", "wav2vec_model_path",
        "moshi_root", "mimi_hf_repo", "moshi_weight", "mimi_weight", "tokenizer",
        "quantize_4bit", "num_codebooks", "voice_prompt", "voice_prompt_dir",
        "enable_moshi_reply", "direct_reply_hidden", "reply_hidden_steps_per_chunk",
        "audio_chunk_sec", "wav2vec_sec", "fm_chunk_frames", "helium_deque_size",
        "prebuffer_chunks", "render_sub_batch", "jpeg_quality", "frame_q_backpressure",
        "buffer_ms", "incremental_chunk_publish", "max_input_buffer_sec",
        "a_cfg_scale", "nfe", "fps", "seed", "noise_seed", "fp32", "tf32",
        "output_audio_codec", "enable_eye_blink_composite", "blink_motion_path",
        "ref_lora_dir", "merge_ref_lora", "max_ref_tokens", "stt_hf_repo", "stt_pkg_dir",
        "compressor_model", "compressor_device", "compressor_4bit", "router_threshold",
        "stt_inline", "stt_queue_frames",
        "router_rules", "web_search_enabled", "web_search_api_key", "web_search_provider",
        "web_search_max_results", "web_search_timeout", "web_search_min_score",
        "search_max_filler_sec", "thinking_sound_path", "conversation_log_dir",
        "suppress_text_during_search", "vad_threshold", "spoken_form_numbers",
        "wait_for_user",
    ])
    sys_log.section("checkpoints and assets")
    for role, value, required in (
        ("generator (IMTalker FM)", getattr(args, "generator_path", ""), True),
        ("generator LoRA", getattr(args, "lora_generator_path", ""), False),
        ("renderer", getattr(args, "renderer_path", ""), True),
        ("helium->wav2vec adapter", getattr(args, "adapter_path", ""), True),
        ("silence Helium seed", getattr(args, "silence_helium_path", ""), False),
        ("wav2vec2 frontend", getattr(args, "wav2vec_model_path", ""), True),
        ("reference image", getattr(args, "ref_path", ""), True),
        ("blink motion", getattr(args, "blink_motion_path", ""), False),
        ("reference LoRA dir", getattr(args, "ref_lora_dir", ""), False),
        ("STT package dir", getattr(args, "stt_pkg_dir", ""), False),
        ("thinking sound", getattr(args, "thinking_sound_path", ""), False),
        ("html page", getattr(args, "html_path", ""), True),
    ):
        sys_log.path(role, value, required=required)
    sys_log.video_config(
        fps=float(getattr(args, "fps", 25)),
        frames_per_chunk=int(getattr(args, "fm_chunk_frames", 24)),
        chunk_seconds=round(
            float(getattr(args, "fm_chunk_frames", 24)) / float(getattr(args, "fps", 25)), 3
        ),
        render_sub_batch=int(getattr(args, "render_sub_batch", 8)),
        renderer_precision=str(getattr(args, "renderer_precision", "fp32")),
        jpeg_quality=int(getattr(args, "jpeg_quality", 86)),
        output_audio_codec=str(getattr(args, "output_audio_codec", "pcm")),
        buffer_ms=int(getattr(args, "buffer_ms", 80)),
        prebuffer_chunks=int(getattr(args, "prebuffer_chunks", 3)),
        frame_q_backpressure=int(getattr(args, "frame_q_backpressure", 160)),
        incremental_chunk_publish=bool(getattr(args, "incremental_chunk_publish", False)),
        eye_blink_composite=bool(getattr(args, "enable_eye_blink_composite", False)),
        nfe=int(getattr(args, "nfe", 5)),
        a_cfg_scale=float(getattr(args, "a_cfg_scale", 1.0)),
    )
    assets_dir = ROOT / "static" / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    engine: LiveHeliumFMEngine | None = LiveHeliumFMEngine(args)
    moshi_engine: MoshiOnlyEngine | None = None
    seedvc: SeedVCStreamingConverter | None = None
    if bool(getattr(args, "enable_seedvc", False)):
        if not args.seedvc_reference_audio:
            raise ValueError("--seedvc_reference_audio is required with --enable_seedvc")
        seedvc = SeedVCStreamingConverter(
            repo=args.seedvc_repo,
            reference_audio=args.seedvc_reference_audio,
            input_sample_rate=TARGET_SR,
            block_seconds=float(args.audio_chunk_sec),
            diffusion_steps=int(args.seedvc_steps),
            cfg_rate=float(args.seedvc_cfg),
            prompt_seconds=float(args.seedvc_prompt_seconds),
            device=args.device,
        )
        print(
            f"[SeedVC] ready load={seedvc.load_seconds:.1f}s "
            f"reference={args.seedvc_reference_audio} steps={args.seedvc_steps} "
            f"cfg={args.seedvc_cfg}",
            flush=True,
        )

    def get_engine() -> LiveHeliumFMEngine:
        nonlocal engine
        if engine is None:
            engine = LiveHeliumFMEngine(args)
        return engine

    def get_moshi_engine() -> MoshiOnlyEngine:
        nonlocal moshi_engine
        if moshi_engine is None:
            # Set before the engine exists: this decides both the compressor's
            # prompt and how its output is normalised, and the two must agree
            # from the very first turn.
            with contextlib.suppress(Exception):
                import search_helpers
                search_helpers.set_spell_numbers(
                    bool(getattr(args, "spoken_form_numbers", False))
                )
                system_logger.get().config(
                    "spoken_form_numbers",
                    bool(getattr(args, "spoken_form_numbers", False)),
                )
            moshi_engine = MoshiOnlyEngineWithHidden(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=getattr(args, "moshi_reply_device", None) or args.device,
                cfg_coef=float(args.moshi_cfg_coef),
                placeholder_jpeg_b64="",
                moshi_weight=getattr(args, "moshi_weight", ""),
                mimi_weight=getattr(args, "mimi_weight", ""),
                tokenizer=getattr(args, "tokenizer", ""),
                quantize_4bit=bool(getattr(args, "quantize_4bit", False)),
                num_codebooks=int(getattr(args, "num_codebooks", 8)),
                context=(int(args.moshi_context) if int(getattr(args, "moshi_context", 0)) > 0 else None),
                voice_prompt=getattr(args, "voice_prompt", ""),
                voice_prompt_dir=getattr(args, "voice_prompt_dir", ""),
                text_prompt=getattr(args, "text_prompt", ""),
                ref_lora_dir=getattr(args, "ref_lora_dir", ""),
                merge_ref_lora=bool(getattr(args, "merge_ref_lora", False)),
                max_ref_tokens=int(getattr(args, "max_ref_tokens", 250)),
                router_threshold=float(getattr(args, "router_threshold", 0.40)),
                router_use_rules=bool(int(getattr(args, "router_rules", 1))),
                stt_hf_repo=getattr(args, "stt_hf_repo", ""),
                stt_pkg_dir=getattr(args, "stt_pkg_dir", ""),
                vad_threshold=float(getattr(args, "vad_threshold", 0.5)),
                suppress_text_during_search=bool(int(getattr(args, "suppress_text_during_search", 1))),
                wait_for_user_before_speaking=bool(getattr(args, "wait_for_user", True)),
                prompt_settle_sec=float(getattr(args, "prompt_settle_sec", 0.0)),
                stt_reject_foreign_script=bool(int(getattr(args, "stt_reject_foreign_script", 1))),
                stt_max_non_latin_ratio=float(getattr(args, "stt_max_non_latin_ratio", 0.15)),
                stt_require_english=bool(int(getattr(args, "stt_require_english", 1))),
                max_input_buffer_sec=float(getattr(args, "max_input_buffer_sec", 2.0)),
                compressor_model=getattr(args, "compressor_model", ""),
                compressor_device=getattr(args, "compressor_device", "cuda"),
                compressor_4bit=bool(getattr(args, "compressor_4bit", True)),
                compressor_max_passages=int(getattr(args, "compressor_max_passages", 2)),
                web_search_enabled=bool(getattr(args, "web_search_enabled", False)),
                web_search_api_key=getattr(args, "web_search_api_key", None),
                web_search_provider=getattr(args, "web_search_provider", "tavily"),
                web_search_max_results=int(getattr(args, "web_search_max_results", 3)),
                web_search_timeout=float(getattr(args, "web_search_timeout", 3.0)),
                web_search_min_score=float(getattr(args, "web_search_min_score", 0.15)),
                conversation_log_dir=getattr(args, "conversation_log_dir", ""),
                thinking_sound_path=getattr(args, "thinking_sound_path", ""),
                search_max_filler_sec=float(getattr(args, "search_max_filler_sec", 6.0)),
                stt_inline=bool(getattr(args, "stt_inline", False)),
                stt_queue_frames=int(getattr(args, "stt_queue_frames", 64)),
            )
        return moshi_engine

    split_sessions: dict[str, dict] = {}
    # PersonaPlex/Mimi streaming state is process-global. Never let a new
    # browser session reset it while the previous session is still tearing down.
    conversation_lock = asyncio.Lock()

    async def _get_split_media_epoch(session: dict) -> float:
        while not session["prebuffer_ready"].is_set() and not session.get("closed", False):
            await asyncio.sleep(0.01)
        async with session["media_epoch_lock"]:
            if session.get("media_epoch") is None:
                session["media_epoch"] = time.perf_counter() + 0.08
            return float(session["media_epoch"])

    def _media_generation(session: dict) -> int:
        with session["media_generation_lock"]:
            return int(session["media_generation"])

    if bool(getattr(args, "enable_moshi_reply", False)):
        print("[liveTryHeliumFM_ws_binary] eager-loading Moshi/PersonaPlex reply engine", flush=True)
        get_moshi_engine()

    @app.get("/")
    async def index():
        if html_path.is_file():
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>", status_code=500)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "live_moshi_reply_helium_fm" if args.enable_moshi_reply else "live_helium_fm",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "loaded": engine is not None,
        })

    @app.websocket("/ws/video")
    async def video_stream(ws: WebSocket):
        await ws.accept()
        session_id = str(ws.query_params.get("session_id", ""))
        session = split_sessions.get(session_id)
        if session is None:
            await ws.send_json({
                "type": "error",
                "message": "unknown or expired video session",
            })
            await ws.close()
            return

        frame_q: asyncio.Queue = session["frame_q"]
        send_start_wall = await _get_split_media_epoch(session)
        active_generation = _media_generation(session)
        base_frame_idx: int | None = None
        frames_sent = 0
        starvation_events = 0
        starve_start: float | None = None
        print(
            f"[AJ][VIDEO] connected session={session_id[:8]} "
            f"queued={frame_q.qsize()}",
            flush=True,
        )
        try:
            while True:
                try:
                    packet = frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    if starve_start is None:
                        starve_start = time.perf_counter()
                        starvation_events += 1
                    await asyncio.sleep(0.004)
                    if session.get("closed", False) and frame_q.empty():
                        break
                    continue

                if packet is None:
                    break

                packet_generation = int(packet.get("media_generation", 0))
                current_generation = _media_generation(session)
                if packet_generation != current_generation:
                    continue
                if packet_generation != active_generation or base_frame_idx is None:
                    active_generation = packet_generation
                    base_frame_idx = int(packet["frame_number"])
                    send_start_wall = await _get_split_media_epoch(session)
                    await ws.send_json({
                        "type": "video_epoch",
                        "generation": active_generation,
                    })
                    print(f"[MEDIA-EPOCH][VIDEO] generation={active_generation} base_frame={base_frame_idx}", flush=True)

                if starve_start is not None:
                    gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                    if gap_ms > 100:
                        print(
                            f"[AJ][VIDEO] STARVED {gap_ms:.0f}ms "
                            f"(event #{starvation_events}) "
                            f"frame_q={frame_q.qsize()} sent={frames_sent}",
                            flush=True,
                        )
                    starve_start = None

                idx = int(packet["frame_number"])
                target_t = send_start_wall + (idx - base_frame_idx) / float(args.fps)
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

                if packet.get("ws_kind") == "bytes":
                    await ws.send_bytes(packet["data"])
                else:
                    await ws.send_json(packet["msg"])
                frames_sent += 1

                late_s = time.perf_counter() - target_t
                if late_s > 0.5:
                    print(
                        f"[AJ][VIDEO] frame {idx} is {late_s*1000:.0f}ms late",
                        flush=True,
                    )
                if frames_sent % 50 == 0:
                    elapsed = time.perf_counter() - send_start_wall
                    print(
                        f"[AJ][VIDEO] sent={frames_sent} frame={idx} "
                        f"frame_q={frame_q.qsize()} elapsed={elapsed:.1f}s "
                        f"starve_events={starvation_events}",
                        flush=True,
                    )
        except (WebSocketDisconnect, RuntimeError, Exception) as exc:
            print(f"[AJ][VIDEO] closed session={session_id[:8]} {exc!r}", flush=True)
        finally:
            print(f"[AJ][VIDEO] done session={session_id[:8]} sent={frames_sent}", flush=True)

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        if conversation_lock.locked():
            print("[SESSION-STRICT] waiting for previous teardown", flush=True)
        await conversation_lock.acquire()
        print("[SESSION-STRICT] acquired PersonaPlex session lock", flush=True)
        fm_engine = get_engine()
        fm_engine.reset_session()
        _log_vram_headroom(fm_engine, args)
        if seedvc is not None:
            seedvc.reset()
        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None
        if reply_engine is not None:
            # Safe to start now: get_engine() above has finished the FM/renderer
            # warmup, so nothing else in this process is capturing CUDA graphs.
            with contextlib.suppress(Exception):
                reply_engine.start_stt_worker()
        # PersonaPlex is reset exactly once at the Start boundary below. Doing
        # it here as well replays uncached prompts twice for every connection.
        browser_input_sr = 48000
        opus_reader = sphn.OpusStreamReader(TARGET_SR)
        audio_packets_seen = 0

        # -- Queues for the reply pipeline --
        # mic_q: (raw_bytes, input_sr) or None to stop
        # frame_q: per-frame packet dicts or None to stop
        mic_q: queue.Queue[tuple[bytes, int, int, float] | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        audio_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None
        mic_ingest_thread: threading.Thread | None = None
        persona_thread: threading.Thread | None = None
        sender_task: asyncio.Task | None = None
        audio_sender_task: asyncio.Task | None = None
        event_loop: asyncio.AbstractEventLoop | None = None
        ws_send_lock = asyncio.Lock()
        media_epoch_lock = asyncio.Lock()
        media_epoch: float | None = None
        session_id = uuid.uuid4().hex
        prebuffer_ready = threading.Event()
        media_generation_lock = threading.Lock()
        media_generation = 0
        playback_interrupt_event = threading.Event()
        playback_state = {"active": False, "hold": 0}
        mic_overlap_packets = 0
        session_started = threading.Event()
        last_mic_level_log_wall = 0.0

        stream_task: asyncio.Task | None = None

        if reply_engine is not None:
            # Match native PersonaPlex: microphone ingress must never be dropped
            # because video delivery is temporarily slower than generation.
            mic_q = queue.Queue()
            frame_q = asyncio.Queue(maxsize=512)
            audio_q = asyncio.Queue(maxsize=512)
            event_loop = asyncio.get_running_loop()
            reply_input_lock = threading.Lock()
            persona_event_q: queue.Queue[dict] = queue.Queue()
            pipeline_stop_event = threading.Event()
            split_sessions[session_id] = {
                "frame_q": frame_q,
                "prebuffer_ready": prebuffer_ready,
                "media_epoch_lock": media_epoch_lock,
                "media_epoch": media_epoch,
                "media_generation_lock": media_generation_lock,
                "media_generation": media_generation,
                "closed": False,
                "created_at": time.perf_counter(),
            }

        await ws.send_json({
            "type": "server_ready",
            "variant": (
                "AJ-NETWORK-ISO-CACHED-FP32"
                if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower()
                in {"1", "true", "yes", "on"}
                else "AJ-NETWORK-ISO-FP32"
            ),
            "session_id": session_id,
            "video_ws_path": f"/ws/video?session_id={session_id}",
            "sample_rate": TARGET_SR,
            "output_audio_codec": str(args.output_audio_codec),
            "model_type": "moshi_reply+helium_fm+renderer" if args.enable_moshi_reply else "helium_fm+renderer",
            "tokens_per_chunk": int(args.fm_chunk_frames),
            "has_audio_file": fm_engine.audio_pcm is not None,
            "buffer_ms": int(args.buffer_ms),
            "av_transport": "binary",
            "target_fps": round(float(args.fps), 2),
        })
        print("[liveTryHeliumFM] websocket connected; sent server_ready", flush=True)

        if reply_engine is not None:
            def _mic_ingest_worker() -> None:
                """Drain browser audio independently from video/render progress."""
                assert mic_q is not None
                packets = 0
                while True:
                    item = mic_q.get()
                    if item is None:
                        break
                    raw_bytes, input_sr, packet_seq, received_at = item
                    if not raw_bytes:
                        continue
                    with reply_input_lock:
                        reply_engine.append_browser_pcm(
                            np.frombuffer(raw_bytes, dtype=np.int16), input_sr
                        )
                    packets += 1
                    queue_age_ms = 1000.0 * (time.perf_counter() - received_at)
                    if queue_age_ms > 100.0:
                        print(
                            f"[OPUS-TRANSPORT] seq={packet_seq} "
                            f"ingest_age={queue_age_ms:.1f}ms pending_q={mic_q.qsize()}",
                            flush=True,
                        )
                    if packets % 250 == 0:
                        print(
                            f"[MIC-ISOLATED] ingested={packets} pending_q={mic_q.qsize()}",
                            flush=True,
                        )
                print(f"[MIC-ISOLATED] stopped ingested={packets}", flush=True)

            def _persona_buffered_samples() -> int:
                with reply_input_lock:
                    return int(reply_engine.input_buffer.shape[0])

            def _persona_process_ready(max_steps: int) -> list[dict]:
                with reply_input_lock:
                    return reply_engine.process_ready_steps_limited(max_steps)

            def _persona_priority_worker() -> None:
                """Run one PersonaPlex step whenever one complete Mimi frame is ready.

                Keeping the critical section to one 80 ms step lets microphone
                ingestion append new PCM between steps instead of waiting behind a
                multi-step catch-up burst.
                """
                priority_stream = None
                if torch.cuda.is_available():
                    try:
                        # PyTorch accepts -1 as a high-priority stream even in builds
                        # that do not expose get_stream_priority_range().
                        priority_stream = torch.cuda.Stream(priority=-1)
                        print("[PERSONA-CONTINUOUS] high-priority CUDA stream enabled", flush=True)
                    except Exception as exc:
                        print(f"[PERSONA-CONTINUOUS] stream fallback: {exc!r}", flush=True)
                total_events = 0
                native_audio_seq = 0
                suppress_media = False
                suppression_silence_confirmed = False
                suppression_silence_steps = 0

                async def _cancel_stale_media(generation: int) -> None:
                    nonlocal media_epoch
                    dropped_audio = 0
                    dropped_frames = 0
                    while True:
                        try: item = audio_q.get_nowait()
                        except asyncio.QueueEmpty: break
                        if item is not None: dropped_audio += 1
                    while True:
                        try: item = frame_q.get_nowait()
                        except asyncio.QueueEmpty: break
                        if item is not None: dropped_frames += 1
                    async with media_epoch_lock:
                        media_epoch = None
                        split_sessions[session_id]["media_epoch"] = None
                    playback_state["active"] = False
                    playback_state["hold"] = 0
                    async with ws_send_lock:
                        await ws.send_json({"type":"media_reset","generation":generation,"reason":"native_personaplex_barge_in"})
                    print(f"[MEDIA-RESET] generation={generation} dropped_audio={dropped_audio} dropped_frames={dropped_frames}", flush=True)
                    with contextlib.suppress(Exception):
                        sys_log.event(
                            "media_reset",
                            f"generation={generation} dropped_audio={dropped_audio} "
                            f"dropped_frames={dropped_frames} -- the GPU producer's partial "
                            f"chunk buffer is cleared by this; repeated resets can prevent a "
                            f"chunk ever completing",
                            generation=generation, dropped_audio=dropped_audio,
                            dropped_frames=dropped_frames,
                        )

                def _trigger_media_reset() -> int:
                    with media_generation_lock:
                        split_sessions[session_id]["media_generation"] += 1
                        generation = int(split_sessions[session_id]["media_generation"])
                    prebuffer_ready.clear()
                    asyncio.run_coroutine_threadsafe(_cancel_stale_media(generation), event_loop)
                    return generation

                while not pipeline_stop_event.is_set():
                    if not session_started.is_set() or _persona_buffered_samples() < MIMI_FRAME_SIZE:
                        time.sleep(0.002)
                        continue
                    buffered_before = _persona_buffered_samples()
                    ready_before = buffered_before // MIMI_FRAME_SIZE
                    if priority_stream is None:
                        events = _persona_process_ready(1)
                    else:
                        with torch.cuda.stream(priority_stream):
                            events = _persona_process_ready(1)
                        # Hidden/audio copies produced by _step synchronize the
                        # generated event before it is published to IMTalker.
                        priority_stream.synchronize()
                    for event in events:
                        input_rms = float(event.get("input_rms", 0.0) or 0.0)
                        reply_rms = float(event.get("reply_rms", 0.0) or 0.0)
                        if playback_interrupt_event.is_set():
                            playback_interrupt_event.clear()
                            generation = _trigger_media_reset()
                            suppress_media = True
                            suppression_silence_confirmed = False
                            suppression_silence_steps = 0
                            print(
                                f"[PLAYBACK-BARGE] generation={generation} "
                                f"step={event.get('step',-1)} input_rms={input_rms:.5f} "
                                f"reply_rms={reply_rms:.5f}",
                                flush=True,
                            )
                            with contextlib.suppress(Exception):
                                sys_log.event(
                                    "playback_barge_in",
                                    f"generation={generation} input_rms={input_rms:.5f} "
                                    f"reply_rms={reply_rms:.5f} -- media suppressed until the "
                                    f"model goes silent and starts a fresh reply",
                                    generation=generation,
                                    input_rms=round(input_rms, 5),
                                    reply_rms=round(reply_rms, 5),
                                )

                        if suppress_media:
                            if not suppression_silence_confirmed:
                                if reply_rms <= 0.003:
                                    suppression_silence_steps += 1
                                else:
                                    suppression_silence_steps = 0
                                if suppression_silence_steps >= 3:
                                    suppression_silence_confirmed = True
                                    print(
                                        f"[PLAYBACK-BARGE] native silence confirmed "
                                        f"step={event.get('step',-1)}",
                                        flush=True,
                                    )
                                continue
                            if reply_rms < 0.012:
                                continue
                            suppress_media = False
                            suppression_silence_confirmed = False
                            suppression_silence_steps = 0
                            print(
                                f"[PLAYBACK-BARGE] fresh reply starts "
                                f"step={event.get('step',-1)} reply_rms={reply_rms:.5f}",
                                flush=True,
                            )

                        with media_generation_lock:
                            event["media_generation"] = int(split_sessions[session_id]["media_generation"])
                        # Keep native audio private until its complete 2-second
                        # motion/video chunk has rendered. The GPU worker then
                        # publishes matching audio and video as one media epoch.
                        persona_event_q.put(event)
                    total_events += len(events)
                    if ready_before > 2:
                        print(
                            f"[PERSONA-CONTINUOUS] backlog={ready_before} "
                            f"drained={len(events)} event_q={persona_event_q.qsize()}",
                            flush=True,
                        )
                print(f"[PERSONA-CONTINUOUS] stopped events={total_events}", flush=True)

            def _gpu_producer_thread() -> None:
                """GPU thread: Moshi -> Helium -> FM -> render -> JPEG -> frame_q.

                Runs Moshi at maximum GPU speed. When real mic audio is not
                yet available, pads Moshi input with silence so it can keep
                generating reply audio without waiting for real-time mic
                arrival. This dramatically cuts first-reply latency.
                """
                assert (
                    mic_q is not None
                    and frame_q is not None
                    and audio_q is not None
                    and event_loop is not None
                )
                pending_reply_steps: list[dict] = []
                pending_reply_hidden: list[torch.Tensor] = []
                pending_reply_audio: list[np.ndarray] = []
                reply_step_history: list[dict] = []
                reply_audio_history: list[np.ndarray] = []
                reply_avatar_chunk_idx = 0
                chunk_produce_count = 0
                active_generation = 0
                staged_audio_seq = 0
                # Pause producer when queue is this deep (qsize() is heuristic across threads).
                FRAME_Q_BACKPRESS = max(1, int(getattr(args, "frame_q_backpressure", 96)))
                FRAME_Q_PUT_TIMEOUT_S = 120.0
                prebuffer_chunks = max(0, int(getattr(args, "prebuffer_chunks", PREBUFFER_CHUNKS)))
                hidden_steps_per_chunk = int(getattr(args, "reply_hidden_steps_per_chunk", 0))
                if hidden_steps_per_chunk <= 0:
                    hidden_steps_per_chunk = int(round(float(args.fm_chunk_frames) * 12.5 / float(args.fps)))
                hidden_steps_per_chunk = max(1, hidden_steps_per_chunk)
                # Native PersonaPlex drains every complete Mimi frame currently
                # buffered. A generous cap prevents an unbounded historical
                # backlog from monopolizing the worker after a client stall.
                max_moshi_steps_per_loop = 64
                incremental_publish = bool(getattr(args, "incremental_chunk_publish", False))
                if incremental_publish:
                    print(
                        "[GPU] incremental chunk publication ENABLED -- render time is off the "
                        "reply-latency path, atomic A/V publication is not guaranteed",
                        flush=True,
                    )
                was_silent = True
                assistant_gate_hold = 0
                last_motion_frame: torch.Tensor | None = None
                seedvc_was_active = False

                def _enqueue_frame(pkt: dict) -> None:
                    """Block until frame_q accepts pkt (real backpressure). Must run from GPU thread."""
                    pkt["media_generation"] = active_generation
                    fut = asyncio.run_coroutine_threadsafe(frame_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING frame_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING frame_q.put failed: {e!r}", flush=True)

                def _enqueue_audio(pkt: dict) -> None:
                    pkt["media_generation"] = active_generation
                    fut = asyncio.run_coroutine_threadsafe(audio_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING audio_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING audio_q.put failed: {e!r}", flush=True)

                sys_log.event(
                    "gpu_producer_started",
                    f"steps/chunk={hidden_steps_per_chunk} frames/chunk={int(args.fm_chunk_frames)} "
                    f"sub_batch={fm_engine.render_sub_batch} backpressure={FRAME_Q_BACKPRESS}",
                    hidden_steps_per_chunk=hidden_steps_per_chunk,
                    incremental_publish=incremental_publish,
                )

                if prebuffer_chunks <= 0 and not prebuffer_ready.is_set():
                    prebuffer_ready.set()
                    print("[GPU] prebuffer=0, sender starts immediately", flush=True)

                while not pipeline_stop_event.is_set():
                    # Mic ingestion runs in its own worker and remains independent
                    # from all video backpressure below.
                    while frame_q.qsize() >= FRAME_Q_BACKPRESS:
                        time.sleep(0.004)

                    if not session_started.is_set():
                        time.sleep(0.003)
                        continue

                    try:
                        first_event = persona_event_q.get(timeout=0.003)
                    except queue.Empty:
                        time.sleep(0.003)
                        continue

                    # PersonaPlex runs independently; this worker only batches
                    # completed native events for adapter/FM/rendering.
                    t_recv = time.perf_counter()
                    events = [first_event]
                    while len(events) < max_moshi_steps_per_loop:
                        try:
                            events.append(persona_event_q.get_nowait())
                        except queue.Empty:
                            break
                    for ev in events:
                        event_generation = int(ev.get("media_generation", 0))
                        if event_generation != active_generation:
                            active_generation = event_generation
                            pending_reply_steps.clear(); pending_reply_hidden.clear(); pending_reply_audio.clear()
                            reply_step_history.clear(); reply_audio_history.clear()
                            chunk_produce_count = 0; was_silent = True; assistant_gate_hold = 0
                            fm_engine.helium_deque = None
                            fm_engine.helium_deque_filled = 0
                            print(
                                f"[MEDIA-EPOCH][GPU] generation={active_generation} "
                                "cleared adapter batching and Helium window",
                                flush=True,
                            )
                        pending_reply_steps.append(ev)
                        fm_engine._session_reply_events.append({
                            "step": int(ev.get("step", -1)),
                            "token": int(ev.get("token", -1)),
                            "piece": str(ev.get("piece", "")),
                            "audio_text": str(ev.get("audio_text", "")),
                            "reply_rms": float(ev.get("reply_rms", 0.0)),
                            "reply_peak": float(ev.get("reply_peak", 0.0)),
                            "input_rms": float(ev.get("input_rms", 0.0)),
                            "hidden": bool(isinstance(ev.get("helium_hidden"), torch.Tensor)),
                            "total_ms": float(ev.get("total_ms", 0.0)),
                        })
                        reply_pcm = (
                            np.frombuffer(base64.b64decode(ev["reply_i16_b64"]), dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        if bool(getattr(args, "direct_reply_hidden", False)):
                            hidden = ev.get("helium_hidden")
                            if not isinstance(hidden, torch.Tensor):
                                continue
                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())
                            pending_reply_audio.append(reply_pcm)
                            if len(pending_reply_hidden) < hidden_steps_per_chunk:
                                continue

                            used_hidden = pending_reply_hidden[:hidden_steps_per_chunk]
                            used_audio = pending_reply_audio[:hidden_steps_per_chunk]
                            used_steps = pending_reply_steps[:hidden_steps_per_chunk]
                            pending_reply_hidden = pending_reply_hidden[hidden_steps_per_chunk:]
                            pending_reply_audio = pending_reply_audio[hidden_steps_per_chunk:]
                            pending_reply_steps = pending_reply_steps[hidden_steps_per_chunk:]

                            if str(getattr(args, "adapter_window_mode", "tail")) == "lookahead":
                                history_size = int(fm_engine.helium_deque_size)
                                future_steps = int(
                                    getattr(args, "adapter_future_steps", 6)
                                )
                                reply_audio_history.extend(used_audio)
                                reply_step_history.extend(used_steps)
                                reply_audio_history = reply_audio_history[-history_size:]
                                reply_step_history = reply_step_history[-history_size:]

                                zero_audio = np.zeros_like(used_audio[0])
                                audio_window = (
                                    [zero_audio]
                                    * (history_size - len(reply_audio_history))
                                    + reply_audio_history
                                )
                                silent_step = {
                                    "token": 0,
                                    "reply_rms": 0.0,
                                    "total_ms": 0.0,
                                    "audio_text": "",
                                }
                                step_window = (
                                    [silent_step]
                                    * (history_size - len(reply_step_history))
                                    + reply_step_history
                                )
                                output_end = history_size - future_steps
                                output_start = (
                                    output_end - hidden_steps_per_chunk
                                )
                                used_audio = audio_window[output_start:output_end]
                                used_steps = step_window[output_start:output_end]

                            pcm_chunk = np.concatenate(used_audio, axis=0).astype(np.float32, copy=False)
                            reply_rms = float(np.sqrt(np.mean(np.square(pcm_chunk)))) if pcm_chunk.size else 0.0
                            step_rms = max(
                                (
                                    float(s.get("reply_rms", 0.0) or 0.0)
                                    for s in used_steps
                                ),
                                default=0.0,
                            )
                            speech_threshold = float(
                                getattr(args, "assistant_speech_rms_threshold", 0.006)
                            )
                            # The thinking sound is real (non-speech) audio played over
                            # the assistant channel, and its RMS is well above
                            # speech_threshold -- so without this override the gate below
                            # reads it as the assistant talking and lip-syncs the avatar
                            # to a filler clip. force_idle keeps the avatar visually idle
                            # (silence_helium_seed) for exactly the frames the clip covers.
                            force_idle_window = any(bool(st.get("force_idle")) for st in used_steps)
                            if force_idle_window:
                                assistant_active_now = False
                                assistant_gate_hold = 0
                            else:
                                assistant_active_now = max(reply_rms, step_rms) > speech_threshold
                                if assistant_active_now:
                                    assistant_gate_hold = max(
                                        0,
                                        int(getattr(args, "assistant_speech_hold_chunks", 1)),
                                    )
                                elif assistant_gate_hold > 0:
                                    assistant_gate_hold -= 1
                            assistant_active = assistant_active_now or assistant_gate_hold > 0
                            if bool(getattr(args, "disable_assistant_output_gate", False)):
                                assistant_active = True

                            previous_active = not was_silent
                            transition = assistant_active != previous_active
                            if transition:
                                print(
                                    f"[GPU] Avatar gate transition "
                                    f"{'speech' if assistant_active else 'idle'} "
                                    f"reply_rms={reply_rms:.5f} step_rms={step_rms:.5f}",
                                    flush=True,
                                )
                            was_silent = not assistant_active

                            if assistant_active:
                                helium_chunk = torch.cat(used_hidden, dim=0)
                            else:
                                if fm_engine.silence_helium_seed is not None:
                                    helium_chunk = fm_engine.silence_helium_seed.expand(
                                        hidden_steps_per_chunk, -1
                                    ).contiguous()
                                else:
                                    hidden_dim = int(used_hidden[0].shape[-1])
                                    helium_chunk = torch.zeros(
                                        hidden_steps_per_chunk,
                                        hidden_dim,
                                        device=fm_engine.device,
                                        dtype=torch.float32,
                                    )
                            target_frames = max(1, int(round(len(pcm_chunk) * float(args.fps) / TARGET_SR)))
                            motion, fm_info = fm_engine._sample_motion_from_helium(helium_chunk, target_frames)
                            if (
                                transition
                                and last_motion_frame is not None
                                and TRANSITION_BLEND_FRAMES > 0
                            ):
                                blend_frames = min(
                                    TRANSITION_BLEND_FRAMES,
                                    int(motion.shape[0]),
                                )
                                alpha = torch.linspace(
                                    0.0,
                                    1.0,
                                    blend_frames + 2,
                                    device=motion.device,
                                    dtype=motion.dtype,
                                )[1:-1].unsqueeze(-1)
                                previous = last_motion_frame.to(
                                    device=motion.device,
                                    dtype=motion.dtype,
                                ).unsqueeze(0)
                                motion[:blend_frames] = (
                                    previous * (1.0 - alpha)
                                    + motion[:blend_frames] * alpha
                                )
                            last_motion_frame = motion[-1].detach().clone()
                            fm_info["assistant_active"] = bool(assistant_active)
                            fm_info["assistant_reply_rms"] = float(reply_rms)
                            fm_info["assistant_step_rms"] = float(step_rms)
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)
                            fm_engine._record_session_chunk(pcm_chunk, motion, fm_info)
                        else:
                            result = fm_engine.feed_pcm_f32(reply_pcm)
                            if result is None:
                                continue

                            motion, fm_info, pcm_chunk = result
                            steps_per_avatar_chunk = max(1, int(round(len(pcm_chunk) / MIMI_FRAME_SIZE)))
                            used_steps = pending_reply_steps[:steps_per_avatar_chunk]
                            pending_reply_steps = pending_reply_steps[steps_per_avatar_chunk:]
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)

                        reply_avatar_chunk_idx += 1
                        chunk_produce_count += 1

                        moshi_total_ms = sum(float(s.get("total_ms", 0.0)) for s in used_steps)

                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        if seedvc is not None:
                            seedvc_active = bool(fm_info.get("assistant_active", True))
                            if seedvc_active:
                                if not seedvc_was_active:
                                    try:
                                        seedvc.reset()
                                    except Exception as exc:
                                        print(f"[SeedVC] reset failed; continuing: {exc!r}", flush=True)
                                try:
                                    pcm_chunk, seedvc_info = seedvc.convert(pcm_chunk)
                                    fm_info.update(seedvc_info)
                                    print(
                                        f"[SeedVC] chunk={reply_avatar_chunk_idx} "
                                        f"ms={seedvc_info['seedvc_ms']:.1f} "
                                        f"rtf={seedvc_info['seedvc_rtf']:.3f} "
                                        f"sola={int(seedvc_info['sola_offset'])}",
                                        flush=True,
                                    )
                                except Exception as exc:
                                    print(f"[SeedVC] conversion failed, using raw PCM: {exc!r}", flush=True)
                            else:
                                pcm_chunk = np.zeros_like(pcm_chunk)
                            seedvc_was_active = seedvc_active
                        frame_audio = split_audio_into_frame_slices(pcm_chunk, args.fps)
                        n_frames = int(motion.shape[0])
                        emitted = int(fm_info["abs_start"])
                        output_step = used_steps[-1] if used_steps else ev
                        text_payload = (
                            output_step.get("audio_text")
                            or output_step.get("sampled_text")
                            or ""
                        )
                        total_gen_ms = (
                            moshi_total_ms + float(fm_info["helium_ms"]) + float(fm_info["fm_ms"])
                        )


                        t_chunk_start = time.perf_counter()
                        staged_frames: list[dict] = []
                        # Audio steps are 80ms, video frames are 1/fps, so each
                        # audio step covers a fixed number of frames. Incremental
                        # publication needs that ratio to release the audio that
                        # belongs to the frames it just published, and nothing else.
                        frames_per_audio_step = max(
                            1, int(round(MIMI_FRAME_SIZE / (TARGET_SR / float(args.fps))))
                        )
                        published_audio = 0

                        def _publish(packets: list[dict], upto_frame: int) -> None:
                            """Send `packets` and the audio steps their frames cover.
                            Frames go first so the browser has its visual reserve
                            before audio advances the playback clock."""
                            nonlocal published_audio, staged_audio_seq
                            for pkt in packets:
                                _enqueue_frame(pkt)
                            want_audio = min(
                                len(used_audio),
                                (upto_frame + frames_per_audio_step - 1) // frames_per_audio_step,
                            )
                            while published_audio < want_audio:
                                _enqueue_audio({
                                    "audio_packet_index": staged_audio_seq,
                                    "audio_pcm": np.asarray(
                                        used_audio[published_audio], dtype=np.float32
                                    ).copy(),
                                    "created_at": time.perf_counter(),
                                })
                                published_audio += 1
                                staged_audio_seq += 1

                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sb_end = min(sb_start + fm_engine.render_sub_batch, n_frames)
                            sub_motion = motion[sb_start:sb_end]
                            sub_audio = frame_audio[sb_start:sb_end]

                            packets = fm_engine.render_and_encode_subbatch(
                                sub_motion,
                                sub_audio,
                                abs_start=emitted + sb_start,
                                text_payload=text_payload,
                                avatar_chunk_id=avatar_chunk_id,
                                total_gen_ms=total_gen_ms,
                            )

                            if incremental_publish:
                                # Release this sub-batch immediately: the render
                                # stage stops sitting on the reply-latency path.
                                # A/V order is still exact -- the audio released
                                # here is only what these frames cover -- but the
                                # atomic guarantee is gone: if a later sub-batch
                                # stalls, playback has already advanced past this
                                # one. That trade is why it is opt-in.
                                _publish(packets, sb_end)
                            staged_frames.extend(packets)

                        if not incremental_publish:
                            # Atomic publication (default). Deliberately kept as
                            # the original straight-line code rather than routed
                            # through _publish: this is the proven path, and the
                            # default must not depend on the correctness of the
                            # opt-in one. Do not let speech escape before all
                            # matching frames are ready -- this intentionally
                            # holds one whole generation chunk to preserve A/V
                            # sync. Frames go first so the browser has the
                            # complete visual reserve before audio advances its
                            # playback clock.
                            for pkt in staged_frames:
                                _enqueue_frame(pkt)
                            for audio_step in used_audio:
                                _enqueue_audio({
                                    "audio_packet_index": staged_audio_seq,
                                    "audio_pcm": np.asarray(audio_step, dtype=np.float32).copy(),
                                    "created_at": time.perf_counter(),
                                })
                                staged_audio_seq += 1

                        if not prebuffer_ready.is_set():
                            prebuffer_ready.set()
                            print(
                                f"[GPU][ATOMIC-2S] released audio={len(used_audio)} "
                                f"frames={len(staged_frames)} generation={active_generation}",
                                flush=True,
                            )

                        chunk_wall_ms = _ms(t_chunk_start)
                        produce_latency_ms = _ms(t_recv)
                        q_depth = frame_q.qsize() if frame_q is not None else -1

                        backlog_s = reply_engine.input_backlog_sec()
                        print(
                            f"[GPU][chunk#{reply_avatar_chunk_idx}] "
                            f"moshi={moshi_total_ms:.0f}ms "
                            f"helium={float(fm_info['helium_ms']):.0f}ms "
                            f"fm={float(fm_info['fm_ms']):.0f}ms "
                            f"render+jpeg={chunk_wall_ms:.0f}ms "
                            f"frames={n_frames} "
                            f"produce_latency={produce_latency_ms:.0f}ms "
                            f"frame_q={q_depth} "
                            f"mic_backlog={backlog_s:.2f}s "
                            f"abs={emitted}",
                            flush=True,
                        )

                        if reply_avatar_chunk_idx <= 3 or reply_avatar_chunk_idx % 25 == 0:
                            # Microphone backlog is the single best predictor of
                            # perceived reply delay: the producer is pinned to real
                            # time by frame_q backpressure, so a backlog here is a
                            # delay that never shrinks on its own. Everything else
                            # in this line is per-chunk cost; this one is cumulative.
                            if backlog_s >= 1.0:
                                print(
                                    f"[GPU] WARNING microphone backlog {backlog_s:.1f}s -- replies "
                                    f"are running this far behind and the producer cannot catch up "
                                    f"(it is rate-limited to real time). Lower the per-chunk cost "
                                    f"(--render_sub_batch / --jpeg_quality / --nfe) or lower "
                                    f"--max_input_buffer_sec to cap the delay.",
                                    flush=True,
                                )
                            status_kwargs = {
                                "chunk": reply_avatar_chunk_idx,
                                "frame_q_depth": q_depth,
                                "input_backlog_s": round(backlog_s, 2),
                                "realtime_factor": round(reply_engine.realtime_factor(), 3),
                                "stt_dropped_frames": int(
                                    getattr(reply_engine, "_stt_dropped_frames", 0)
                                ),
                                "input_dropped_s": round(
                                    reply_engine._input_dropped_samples / TARGET_SR, 2
                                ),
                                "render_ms": round(chunk_wall_ms, 1),
                                "fm_ms": round(float(fm_info["fm_ms"]), 1),
                                "helium_ms": round(float(fm_info["helium_ms"]), 1),
                                "search_enabled": getattr(reply_engine, "search_enabled", False),
                                "stt_active": reply_engine.stt_lm_gen is not None,
                                "compressor_active": reply_engine.context_compressor is not None,
                                "router_active": getattr(reply_engine, "query_router", None) is not None,
                            }
                            if torch.cuda.is_available():
                                status_kwargs["gpu_mem_allocated_gb"] = round(
                                    torch.cuda.memory_allocated() / (1 << 30), 2
                                )
                                status_kwargs["gpu_mem_reserved_gb"] = round(
                                    torch.cuda.memory_reserved() / (1 << 30), 2
                                )
                            reply_engine.conv_logger.component_status(**status_kwargs)
                            sys_log.active_work(
                                f"streaming chunk #{reply_avatar_chunk_idx}: "
                                f"render={chunk_wall_ms:.0f}ms fm={float(fm_info['fm_ms']):.0f}ms "
                                f"mic_backlog={backlog_s:.2f}s frame_q={q_depth}",
                                **status_kwargs,
                            )

                fut_done = asyncio.run_coroutine_threadsafe(frame_q.put(None), event_loop)
                try:
                    fut_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING sentinel put: {e!r}", flush=True)
                fut_audio_done = asyncio.run_coroutine_threadsafe(
                    audio_q.put(None),
                    event_loop,
                )
                try:
                    fut_audio_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING audio sentinel put: {e!r}", flush=True)

            async def _get_media_epoch() -> float:
                nonlocal media_epoch
                while not prebuffer_ready.is_set():
                    await asyncio.sleep(0.01)
                async with media_epoch_lock:
                    session = split_sessions.get(session_id)
                    if media_epoch is None and session is not None and session.get("media_epoch") is not None:
                        media_epoch = float(session["media_epoch"])
                    if media_epoch is None:
                        media_epoch = time.perf_counter() + 0.08
                        if session is not None:
                            session["media_epoch"] = media_epoch
                    return media_epoch

            async def _reply_sender() -> None:
                """Wait for prebuffer, then drain frame_q at 25fps."""
                assert frame_q is not None
                send_start_wall = await _get_media_epoch()
                frames_sent = 0
                starvation_events = 0
                starve_start: float | None = None
                ws_closed = False

                q_depth_at_start = frame_q.qsize()
                print(
                    f"[SENDER] prebuffer filled, starting pacing with "
                    f"{q_depth_at_start} frames queued",
                    flush=True,
                )

                while True:
                    if ws_closed:
                        break

                    try:
                        packet = frame_q.get_nowait()
                    except asyncio.QueueEmpty:
                        if send_start_wall is not None and starve_start is None:
                            starve_start = time.perf_counter()
                            starvation_events += 1
                        await asyncio.sleep(0.004)
                        continue

                    if packet is None:
                        break

                    if starve_start is not None:
                        gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                        if gap_ms > 100:
                            print(
                                f"[SENDER] STARVED {gap_ms:.0f}ms "
                                f"(event #{starvation_events}) "
                                f"frame_q={frame_q.qsize()} sent={frames_sent}",
                                flush=True,
                            )
                        starve_start = None

                    idx = int(packet["frame_number"])

                    target_t = send_start_wall + idx / float(args.fps)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                    try:
                        async with ws_send_lock:
                            if packet.get("ws_kind") == "bytes":
                                await ws.send_bytes(packet["data"])
                            else:
                                await ws.send_json(packet["msg"])
                    except (WebSocketDisconnect, RuntimeError, Exception):
                        ws_closed = True
                        break
                    frames_sent += 1

                    late_s = time.perf_counter() - target_t
                    if late_s > 0.5:
                        print(
                            f"[SENDER] video frame {idx} is {late_s*1000:.0f}ms late",
                            flush=True,
                        )

                    if frames_sent % 50 == 0:
                        q_depth = frame_q.qsize()
                        elapsed = time.perf_counter() - send_start_wall
                        print(
                            f"[SENDER] sent={frames_sent} frame={idx} "
                            f"frame_q={q_depth} "
                            f"elapsed={elapsed:.1f}s "
                            f"starve_events={starvation_events}",
                            flush=True,
                        )

            async def _audio_sender() -> None:
                assert audio_q is not None
                output_audio_codec = str(args.output_audio_codec).lower()
                if output_audio_codec != "opus":
                    raise RuntimeError("AJ requires --output_audio_codec opus")
                opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                send_start_wall = await _get_media_epoch()
                packets_sent = 0
                bytes_sent = 0
                max_queue_depth = 0
                max_lateness_ms = 0.0
                last_send_wall: float | None = None
                # AG: prevent catch-up bursts after GPU/render stalls.
                # Even if target_t is already late, keep websocket audio writes
                # spaced near the media frame cadence instead of flushing a
                # backlog back-to-back into the browser decoder/worklet.
                # PersonaPlex emits one native Mimi audio step every 80 ms.
                native_packet_sec = float(MIMI_FRAME_SIZE) / float(TARGET_SR)
                min_send_interval_s = max(0.001, native_packet_sec - 0.005)
                next_send_wall = send_start_wall
                active_generation = 0
                base_audio_idx: int | None = None

                while True:
                    packet = await audio_q.get()
                    if packet is None:
                        break
                    session = split_sessions.get(session_id)
                    if session is None: break
                    packet_generation = int(packet.get("media_generation", 0))
                    current_generation = _media_generation(session)
                    if packet_generation != current_generation: continue
                    idx = int(packet["audio_packet_index"])
                    if packet_generation != active_generation or base_audio_idx is None:
                        active_generation = packet_generation
                        base_audio_idx = idx
                        send_start_wall = await _get_media_epoch()
                        next_send_wall = send_start_wall
                        last_send_wall = None
                        opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                        print(f"[MEDIA-EPOCH][AUDIO] generation={active_generation} base_audio={base_audio_idx}", flush=True)
                    target_t = send_start_wall + (idx - base_audio_idx) * native_packet_sec
                    target_t = max(target_t, next_send_wall)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    session = split_sessions.get(session_id)
                    if session is None or packet_generation != _media_generation(session):
                        continue
                    send_wall = time.perf_counter()
                    lateness_ms = max(0.0, (send_wall - target_t) * 1000.0)
                    max_lateness_ms = max(max_lateness_ms, lateness_ms)
                    interval_ms = (
                        0.0 if last_send_wall is None
                        else (send_wall - last_send_wall) * 1000.0
                    )
                    last_send_wall = send_wall
                    next_send_wall = max(target_t, send_wall) + min_send_interval_s
                    max_queue_depth = max(max_queue_depth, audio_q.qsize())

                    audio_pcm = np.asarray(
                        packet["audio_pcm"],
                        dtype=np.float32,
                    ).reshape(-1)
                    packet_rms = (
                        float(np.sqrt(np.mean(np.square(audio_pcm, dtype=np.float32))))
                        if audio_pcm.size else 0.0
                    )
                    if packet_rms >= 0.006:
                        playback_state["active"] = True
                        playback_state["hold"] = 4
                    elif int(playback_state["hold"]) > 0:
                        playback_state["hold"] = int(playback_state["hold"]) - 1
                        playback_state["active"] = True
                    else:
                        playback_state["active"] = False
                    opus_payload = opus_writer.append_pcm(audio_pcm)
                    if opus_payload is None and hasattr(opus_writer, "read_bytes"):
                        opus_payload = opus_writer.read_bytes()
                    if opus_payload:
                        try:
                            async with ws_send_lock:
                                await ws.send_bytes(b"\x01" + opus_payload)
                        except (WebSocketDisconnect, RuntimeError, Exception):
                            break
                        packets_sent += 1
                        bytes_sent += len(opus_payload)

                    if packets_sent and packets_sent % 50 == 0:
                        print(
                            f"[NATIVE-AUDIO] sent={packets_sent} seq={idx} "
                            f"audio_q={audio_q.qsize()} max_q={max_queue_depth} "
                            f"interval={interval_ms:.1f}ms "
                            f"late={lateness_ms:.1f}ms max_late={max_lateness_ms:.1f}ms "
                            f"bytes={bytes_sent}",
                            flush=True,
                        )

            def _report_thread_death(kind: str):
                def _cb(exc, tb):
                    with contextlib.suppress(Exception):
                        reply_engine.conv_logger.error(f"thread_{kind}", exc, tb)
                return _cb

            mic_ingest_thread = threading.Thread(
                target=_supervise(
                    "mic-ingest", _mic_ingest_worker, _report_thread_death("mic_ingest")
                ),
                daemon=True,
                name="mic-ingest",
            )
            mic_ingest_thread.start()
            persona_thread = threading.Thread(
                target=_supervise(
                    "persona-priority", _persona_priority_worker,
                    _report_thread_death("persona_priority"),
                ),
                daemon=True,
                name="persona-priority",
            )
            persona_thread.start()
            gpu_thread = threading.Thread(
                target=_supervise(
                    "gpu-producer", _gpu_producer_thread, _report_thread_death("gpu_producer"),
                    max_restarts=3,
                ),
                daemon=True, name="gpu-producer",
            )
            gpu_thread.start()
            print(
                f"[AJ][AUDIO] session={session_id[:8]} "
                f"video_path=/ws/video?session_id={session_id}",
                flush=True,
            )
            audio_sender_task = asyncio.create_task(_audio_sender())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if reply_engine is None:
                        continue
                    assert mic_q is not None
                    packet_sr = int(browser_input_sr)
                    if len(data) > 0 and data[0] == 1:
                        decoded_pcm = opus_reader.append_bytes(bytes(data[1:]))
                        if decoded_pcm.shape[-1] == 0:
                            continue
                        decoded_pcm = np.asarray(decoded_pcm, dtype=np.float32).reshape(-1)
                        decoded_i16 = (
                            np.clip(decoded_pcm, -1.0, 1.0) * 32767.0
                        ).astype(np.int16)
                        data = decoded_i16.tobytes()
                        packet_sr = TARGET_SR
                    audio_packets_seen += 1
                    pcm_i16 = np.frombuffer(data, dtype=np.int16)
                    mic_rms = float(np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))) if pcm_i16.size else 0.0
                    mic_peak = float(np.max(np.abs(pcm_i16.astype(np.float32) / 32768.0))) if pcm_i16.size else 0.0
                    if bool(playback_state["active"]) and mic_rms >= 0.020:
                        mic_overlap_packets += 1
                    else:
                        mic_overlap_packets = 0
                    if mic_overlap_packets >= 2 and not playback_interrupt_event.is_set():
                        playback_interrupt_event.set()
                        mic_overlap_packets = 0
                        print(
                            f"[PLAYBACK-OVERLAP] packet={audio_packets_seen} "
                            f"mic_rms={mic_rms:.5f} assistant_playback=active",
                            flush=True,
                        )
                    now_wall = time.perf_counter()
                    if audio_packets_seen <= 3 or now_wall - last_mic_level_log_wall >= 1.0:
                        voice = "VOICE" if mic_rms >= 0.02 else "quiet"
                        print(
                            f"[MIC] packet={audio_packets_seen} rms={mic_rms:.5f} "
                            f"peak={mic_peak:.3f} sr={packet_sr} {voice}",
                            flush=True,
                        )
                        last_mic_level_log_wall = now_wall
                    if audio_packets_seen <= 3:
                        print(
                            f"[liveTryHeliumFM] rx binary mic packet#{audio_packets_seen} "
                            f"bytes={len(data)} sr={packet_sr}",
                            flush=True,
                        )
                    if not session_started.is_set():
                        session_started.set()
                        print(
                            "[liveTryHeliumFM] auto-started session from first binary mic packet",
                            flush=True,
                        )
                    try:
                        mic_q.put_nowait(
                            (bytes(data), int(packet_sr), audio_packets_seen, time.perf_counter())
                        )
                    except queue.Full:
                        # The queue is intentionally unbounded. If this ever
                        # triggers, fail loudly rather than losing speech invisibly.
                        print(
                            f"[MIC-NODROP] ERROR unexpected queue.Full "
                            f"packet={audio_packets_seen}",
                            flush=True,
                        )
                    continue
                text = msg.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type", "")).lower()
                print(f"[liveTryHeliumFM] rx text type={msg_type or '<empty>'}", flush=True)

                if msg_type == "start":
                    session_started.clear()
                    playback_interrupt_event.clear()
                    playback_state["active"] = False
                    playback_state["hold"] = 0
                    mic_overlap_packets = 0
                    browser_input_sr = int(payload.get("sample_rate", payload.get("sampleRate", browser_input_sr)))
                    # A Start message is a hard session boundary. Remove any
                    # transport data that arrived before Start and recreate the
                    # stateful Opus decoder before resetting model/avatar state.
                    cleared_mic_packets = 0
                    if mic_q is not None:
                        while True:
                            try:
                                stale_item = mic_q.get_nowait()
                            except queue.Empty:
                                break
                            if stale_item is not None:
                                cleared_mic_packets += 1
                    cleared_persona_events = 0
                    while True:
                        try:
                            persona_event_q.get_nowait()
                            cleared_persona_events += 1
                        except queue.Empty:
                            break
                    opus_reader = sphn.OpusStreamReader(TARGET_SR)
                    fm_engine.reset_session()
                    if seedvc is not None:
                        seedvc.reset()
                    if fm_engine.audio_pcm is not None and (stream_task is None or stream_task.done()):
                        stream_task = asyncio.create_task(stream_from_file(ws, fm_engine))
                    if reply_engine is not None:
                        with reply_input_lock:
                            reply_engine.reset_session()
                        session_started.set()
                    print(
                        f"[SESSION-FULL-RESET] session={session_id[:8]} "
                        f"cleared_mic_packets={cleared_mic_packets} "
                        f"cleared_persona_events={cleared_persona_events} "
                        "opus=reset persona=reset avatar=reset",
                        flush=True,
                    )
                    print(
                        "[liveTryHeliumFM] start → "
                        + ("streaming from file" if fm_engine.audio_pcm is not None else "live Moshi reply mode"),
                        flush=True,
                    )

                elif msg_type == "chunk_audio":
                    pcm_b64 = payload.get("pcm_s16le_b64", "")
                    if not pcm_b64:
                        continue
                    pcm_bytes = base64.b64decode(pcm_b64)
                    result = fm_engine.feed_pcm(pcm_bytes)
                    if result is not None:
                        motion, fm_info, _pcm_chunk = result
                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        n_frames = int(motion.shape[0])
                        emitted = fm_info["abs_start"]
                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sub = motion[sb_start:sb_start + fm_engine.render_sub_batch].to(
                                fm_engine.device, dtype=fm_engine.dtype
                            )
                            frames_np, _ = fm_engine._render_motion_oom_safe(sub)
                            for j, frame_rgb in enumerate(frames_np):
                                idx = emitted + sb_start + j
                                await ws.send_json({
                                    "type": "chunk_frame",
                                    "chunk_id": idx + 1,
                                    "frame_idx": 0,
                                    "jpeg_b64": encode_jpeg_b64(frame_rgb, fm_engine.jpeg_quality),
                                    "moshi_text": (
                                        f"live Helium+FM "
                                        f"helium={fm_info['helium_ms']:.0f}ms "
                                        f"fm={fm_info['fm_ms']:.0f}ms"
                                    ),
                                    "server_fps": round(float(args.fps), 1),
                                    "chunks_done": avatar_chunk_id,
                                })

                elif msg_type == "stop":
                    print("[liveTryHeliumFM] stop requested", flush=True)
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session = split_sessions.get(session_id)
            if session is not None:
                session["closed"] = True
            # Signal every worker before joining it. Previously teardown joined
            # workers whose loops were still active, so a rapid second Start
            # waited forever and stale PersonaPlex output survived the boundary.
            if reply_engine is not None:
                pipeline_stop_event.set()
                session_started.clear()
            if stream_task is not None:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            if mic_q is not None:
                with contextlib.suppress(queue.Full):
                    mic_q.put_nowait(None)
            if mic_ingest_thread is not None:
                await asyncio.to_thread(mic_ingest_thread.join, 5.0)
            if persona_thread is not None:
                await asyncio.to_thread(persona_thread.join, 10.0)
            if gpu_thread is not None:
                await asyncio.to_thread(gpu_thread.join, 30.0)
            if sender_task is not None:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await sender_task
            if audio_sender_task is not None:
                audio_sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await audio_sender_task
            fm_engine.dump_last_session(source="websocket_live")
            split_sessions.pop(session_id, None)
            if conversation_lock.locked():
                conversation_lock.release()
                print("[SESSION-STRICT] released PersonaPlex session lock", flush=True)
            print("[liveTryHeliumFM] websocket closed", flush=True)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = LiveHeliumFMOptions()
    args = parser.parse()
    args.rank = args.device
    parser.print_options()

    app = build_app(args)

    import uvicorn

    print(f"[liveTryHeliumFM_ws_binary] serving {args.html_path} (binary av_transport)")
    print(f"[liveTryHeliumFM] open http://{args.host}:{args.port}/")
    system_logger.get().ready(url=f"http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
