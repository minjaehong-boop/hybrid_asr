from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from config.settings import SenseVoiceConfig
from asr.transducer_stage import _import_sherpa_onnx


def _resolve_provider(provider: str) -> str:
    """Resolve runtime provider value from explicit choice or auto mode."""
    if provider in ("cpu", "cuda", "coreml"):
        return provider
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_model_file(model_dir: Path) -> Path:
    """Resolve the preferred ONNX file inside the SenseVoice model directory."""
    for filename in ("model.int8.onnx", "model.onnx"):
        candidate = model_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"model.int8.onnx or model.onnx not found in {model_dir}")


def _normalize_refine_text(text: str) -> str:
    """Normalize whitespace from offline SenseVoice output."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def _dedup_overlap(prev_text: str, new_text: str) -> str:
    """Remove longest suffix-prefix overlap between prev and new text."""
    if not prev_text or not new_text:
        return new_text
    prev_stripped = "".join(prev_text.split())
    seg_stripped = "".join(new_text.split())
    best_overlap = 0
    for k in range(min(len(prev_stripped), len(seg_stripped)), 0, -1):
        if prev_stripped.endswith(seg_stripped[:k]):
            best_overlap = k
            break
    if best_overlap > 0:
        skipped = 0
        cut_pos = 0
        for ci, ch in enumerate(new_text):
            if ch.isspace():
                continue
            skipped += 1
            if skipped >= best_overlap:
                cut_pos = ci + 1
                break
        new_text = new_text[cut_pos:].strip()
    return new_text


def _extract_center_tokens(
    tokens: list,
    timestamps: list,
    chunk_dur: float,
    left_trim: float,
    right_trim: float,
) -> str:
    """Extract tokens within the center commit region based on per-token timestamps."""
    committed = []
    for ti, tok in enumerate(tokens):
        if ti >= len(timestamps):
            break
        t = float(timestamps[ti])
        if t >= left_trim and t <= chunk_dur - right_trim:
            committed.append(tok)
    return re.sub(r"\s+", " ", "".join(committed)).strip()


def _extract_center_by_ratio(
    raw_text: str,
    chunk_dur: float,
    left_trim: float,
    right_trim: float,
) -> str:
    """Fallback: estimate center text by character ratio when timestamps unavailable."""
    if not raw_text or chunk_dur <= 0:
        return raw_text
    left_ratio = left_trim / chunk_dur
    right_ratio = right_trim / chunk_dur
    left_chars = int(len(raw_text) * left_ratio)
    right_chars = int(len(raw_text) * right_ratio)
    end_idx = len(raw_text) - right_chars if right_chars > 0 else len(raw_text)
    return raw_text[left_chars:end_idx].strip()


@dataclass(slots=True)
class RefineUpdate:
    start_sec: float
    end_sec: float
    text: str
    is_final: bool = False


class SenseVoiceRefineStage:
    """Chunk-based refine stage built on sherpa-onnx SenseVoiceSmall.

    Three modes (selected by trim_sec and context_mode):
    - Simple (trim_sec == 0): decode each chunk independently
    - CC-Accumulate (trim_sec > 0, context_mode="accumulate"): growing multi-chunk context buffer
    - CC-Overlap    (trim_sec > 0, context_mode="overlap"):    fixed 2*trim_sec overlap buffer
    """

    sample_rate = 16000
    min_retry_samples = sample_rate

    def __init__(self, cfg: SenseVoiceConfig) -> None:
        self.cfg = cfg
        self.trim_sec = getattr(cfg, "trim_sec", 0.0)
        self.context_mode = getattr(cfg, "context_mode", "accumulate")
        self.chunk_offset_sec = 0.0
        self._chunk_index = 0
        self._prev_committed_text = ""

        # accumulate mode
        self._max_context_chunks = 4
        self._chunk_buffers: list[np.ndarray] = []
        self._context_buffer: np.ndarray | None = None

        # overlap mode
        self._overlap_buffer: np.ndarray | None = None

        # shared finalize state (both CC modes)
        self._last_result_tokens: list = []
        self._last_result_timestamps: list = []
        self._last_result_text: str = ""
        self._last_chunk_dur: float = 0.0
        self._last_right_trim: float = 0.0

        self._sherpa_onnx = _import_sherpa_onnx()
        provider = _resolve_provider(cfg.provider)
        model_dir = cfg.model_dir.resolve()
        model_file = _resolve_model_file(model_dir)
        tokens_file = model_dir / "tokens.txt"
        if not tokens_file.is_file():
            raise FileNotFoundError(f"tokens.txt not found in {model_dir}")
        try:
            self.recognizer = self._sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_file),
                tokens=str(tokens_file),
                language=cfg.language,
                use_itn=cfg.use_itn,
                provider=provider,
                num_threads=cfg.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize SenseVoice refine stage. "
                "Pass --sensevoice-model-dir to a directory containing model.int8.onnx and tokens.txt."
            ) from exc

    def reset(self) -> None:
        """Reset per-file state so the same recognizer can be reused across files."""
        self.chunk_offset_sec = 0.0
        self._chunk_index = 0
        self._prev_committed_text = ""
        self._chunk_buffers = []
        self._context_buffer = None
        self._overlap_buffer = None
        self._last_result_tokens = []
        self._last_result_timestamps = []
        self._last_result_text = ""
        self._last_chunk_dur = 0.0
        self._last_right_trim = 0.0

    def _decode_chunk(self, samples: np.ndarray):
        """Decode one chunk, retrying once if the tail chunk is too short."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        stream = self.recognizer.create_stream()
        try:
            stream.accept_waveform(self.sample_rate, samples)
            self.recognizer.decode_stream(stream)
            return stream.result
        except RuntimeError:
            if len(samples) >= self.min_retry_samples:
                raise
            padded = np.pad(samples, (0, self.min_retry_samples - len(samples))).astype(np.float32, copy=False)
            stream = self.recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, padded)
            self.recognizer.decode_stream(stream)
            return stream.result

    def _commit(self, text: str, commit_start_sec: float, commit_end_sec: float) -> RefineUpdate | None:
        """Dedup against previous committed text and return RefineUpdate, or None if empty."""
        text = _dedup_overlap(self._prev_committed_text, text)
        if not text:
            return None
        self._prev_committed_text += text
        return RefineUpdate(
            start_sec=commit_start_sec,
            end_sec=commit_end_sec,
            text=_normalize_refine_text(text),
            is_final=False,
        )
    #청크 간 겹침 없이 간단히 디코딩하는 모드. 디코딩 결과에 timestamps가 있으면 해당 구간만 추출, 없으면 비율로 추정.
    def _feed_simple(self, samples: np.ndarray) -> RefineUpdate | None:
        """Simple chunking: decode each chunk independently."""
        chunk = np.asarray(samples, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return None

        chunk_start_sec = self.chunk_offset_sec
        chunk_end_sec = chunk_start_sec + len(chunk) / self.sample_rate
        self.chunk_offset_sec = chunk_end_sec

        result = self._decode_chunk(chunk)
        text = _normalize_refine_text(result.text)
        if not text:
            return None

        start_sec, end_sec = chunk_start_sec, chunk_end_sec
        timestamps = list(getattr(result, "timestamps", []) or [])
        if timestamps:
            start_sec = max(chunk_start_sec, min(chunk_end_sec, chunk_start_sec + float(timestamps[0])))
            end_sec = max(start_sec, min(chunk_end_sec, chunk_start_sec + float(timestamps[-1])))
            if end_sec <= start_sec:
                end_sec = chunk_end_sec

        return RefineUpdate(start_sec=start_sec, end_sec=end_sec, text=text, is_final=False)
    
    #청크 간 겹침이 있는 CC 모드. trim_sec 만큼 겹치도록 청크를 구성하여 디코딩, 겹치는 구간은 timestamps 또는 비율로 추정하여 결과에서 추출. 커밋 시점은 각 청크의 left_trim 끝으로 고정하여 실시간성 최대화, 단 tail이 짤리는 문제 존재 (finalize에서 보정). overlap 모드는 고정된 겹침 버퍼 사용, accumulate 모드는 최대 4청크까지 누적된 컨텍스트 버퍼 사용. 두 모드 모두 커밋 텍스트는 이전 커밋과 겹치는 부분을 제거하여 반환 (중복 제거는 단순 longest overlap이지만, finalize에서 tail 보정 시에는 좀 더 보수적으로 _dedup_overlap 적용).
    def _feed_center_commit(self, samples: np.ndarray) -> RefineUpdate | None:
        """CC 공통 흐름 (overlap / accumulate).

        overlap:    prefix = _overlap_buffer (2*trim_sec 고정),  left_trim = 0 or trim_sec
        accumulate: prefix = _context_buffer (최대 4청크 LRU),   left_trim = 0 or context_dur - trim_sec
        stride는 두 모드 동일: 청크0 = new_dur - right_trim, 이후 = new_dur
        """
        new_samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if new_samples.size == 0:
            return None

        # ── prefix 선택 & full_chunk 구성 ──
        prefix = self._overlap_buffer if self.context_mode == "overlap" else self._context_buffer
        full_chunk = (
            np.concatenate([prefix, new_samples])
            if prefix is not None and self._chunk_index > 0
            else new_samples
        )
        chunk_dur = len(full_chunk) / self.sample_rate

        # ── left_trim 계산 ──
        if self._chunk_index == 0:
            left_trim = 0.0
        elif self.context_mode == "overlap":
            left_trim = self.trim_sec
        else:
            left_trim = len(prefix) / self.sample_rate - self.trim_sec
        right_trim = self.trim_sec

        if left_trim + right_trim >= chunk_dur:
            if left_trim > 0 and chunk_dur > 0.5:
                left_trim = max(0.0, chunk_dur - 0.5) * (left_trim / (left_trim + right_trim))
            right_trim = 0.0

        new_dur = len(new_samples) / self.sample_rate
        stride_sec = new_dur - right_trim if self._chunk_index == 0 else new_dur
        commit_start_sec = self.chunk_offset_sec
        self.chunk_offset_sec += stride_sec

        # ── 디코딩 ──
        result = self._decode_chunk(full_chunk)
        tokens = list(getattr(result, "tokens", []) or [])
        timestamps = list(getattr(result, "timestamps", []) or [])

        self._last_result_tokens = tokens
        self._last_result_timestamps = timestamps
        self._last_result_text = result.text
        self._last_chunk_dur = chunk_dur
        self._last_right_trim = right_trim

        if tokens and timestamps:
            text = _extract_center_tokens(tokens, timestamps, chunk_dur, left_trim, right_trim)
        else:
            text = _extract_center_by_ratio(_normalize_refine_text(result.text), chunk_dur, left_trim, right_trim)

        # ── prefix buffer 갱신 ──
        if self.context_mode == "overlap":
            overlap_samples = int(2 * self.trim_sec * self.sample_rate)
            self._overlap_buffer = full_chunk[-overlap_samples:] if len(full_chunk) > overlap_samples else full_chunk.copy()
        else:
            self._chunk_buffers.append(new_samples)
            if len(self._chunk_buffers) > self._max_context_chunks:
                self._chunk_buffers.pop(0)
            self._context_buffer = np.concatenate(self._chunk_buffers)

        self._chunk_index += 1
        return self._commit(text, commit_start_sec, self.chunk_offset_sec)

    def feed(self, samples: np.ndarray) -> RefineUpdate | None:
        """Decode one chunk and return a refine update."""
        if self.trim_sec > 0:
            return self._feed_center_commit(samples)
        return self._feed_simple(samples)

    def finalize(self) -> RefineUpdate:
        """Recover tail tokens cut by right_trim and signal pipeline completion.

        Both CC modes: chunk_offset_sec == 실제 커밋 끝, end_sec = chunk_offset_sec + last_right_trim.
        Always returns is_final=True, even when tail text is empty.
        """
        tail_text = ""
        if self._last_right_trim > 0:
            tail_left = self._last_chunk_dur - self._last_right_trim
            if self._last_result_tokens and self._last_result_timestamps:
                tail_text = _extract_center_tokens(
                    self._last_result_tokens, self._last_result_timestamps,
                    self._last_chunk_dur, tail_left, 0.0,
                )
            elif self._last_result_text:
                tail_text = _extract_center_by_ratio(
                    _normalize_refine_text(self._last_result_text),
                    self._last_chunk_dur, tail_left, 0.0,
                )
            if tail_text:
                tail_text = _dedup_overlap(self._prev_committed_text, tail_text)
            if tail_text:
                tail_text = _normalize_refine_text(tail_text)
        return RefineUpdate(
            start_sec=self.chunk_offset_sec,
            end_sec=self.chunk_offset_sec + self._last_right_trim,
            text=tail_text,
            is_final=True,
        )
