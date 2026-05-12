from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config.settings import TransducerConfig


def _link_onnxruntime_for_sherpa() -> None:
    """Create runtime symlink expected by sherpa-onnx when missing."""
    spec_ort = importlib.util.find_spec("onnxruntime")
    spec_sherpa = importlib.util.find_spec("sherpa_onnx")
    if spec_ort is None or spec_ort.origin is None:
        return
    if spec_sherpa is None or spec_sherpa.origin is None:
        return

    ort_capi_dir = Path(spec_ort.origin).parent / "capi"
    sherpa_lib_dir = Path(spec_sherpa.origin).parent / "lib"
    candidates = sorted(ort_capi_dir.glob("libonnxruntime.so.*"))
    if not candidates:
        return

    target_lib = candidates[-1]
    ort_link = ort_capi_dir / "libonnxruntime.so"
    if ort_link.is_symlink() or ort_link.exists():
        try:
            if ort_link.resolve() != target_lib.resolve():
                ort_link.unlink()
                ort_link.symlink_to(target_lib.name)
        except OSError:
            pass
    else:
        ort_link.symlink_to(target_lib.name)

    sherpa_link = sherpa_lib_dir / "libonnxruntime.so"
    rel_target = os.path.relpath(ort_link, sherpa_lib_dir)
    if sherpa_link.is_symlink() or sherpa_link.exists():
        try:
            if sherpa_link.resolve() != ort_link.resolve():
                sherpa_link.unlink()
                sherpa_link.symlink_to(rel_target)
        except OSError:
            pass
    else:
        sherpa_link.symlink_to(rel_target)


def _import_sherpa_onnx():
    """Import sherpa-onnx with fallback runtime link fix."""
    try:
        return importlib.import_module("sherpa_onnx")
    except ImportError as exc:
        if "libonnxruntime.so" not in str(exc):
            raise
        _link_onnxruntime_for_sherpa()
        return importlib.import_module("sherpa_onnx")


@dataclass(slots=True)
class TransducerUpdate:
    start_sec: float
    end_sec: float
    text: str
    is_final: bool = False


def _lcp_len(a: list[str], b: list[str]) -> int:
    """Compute longest common prefix length for token arrays."""
    n = min(len(a), len(b))
    idx = 0
    while idx < n and a[idx] == b[idx]:
        idx += 1
    return idx


class TransducerStage:
    def __init__(self, cfg: TransducerConfig) -> None:
        """Initialize sherpa-onnx online transducer recognizer."""
        self.cfg = cfg
        self._sherpa_onnx = _import_sherpa_onnx()
        self.recognizer = self._sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(cfg.tokens),
            encoder=str(cfg.encoder),
            decoder=str(cfg.decoder),
            joiner=str(cfg.joiner),
            sample_rate=cfg.sample_rate,
            feature_dim=cfg.feature_dim,
            num_threads=cfg.num_threads,
            provider=cfg.provider,
            decoding_method="greedy_search",
        )
        self.stream = self.recognizer.create_stream()
        self.prev_text = ""
        self.prev_tokens: list[str] = []

    def _decode_ready(self) -> None:
        """Drain decoder while stream is ready for inference."""
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def _build_update(self, end_sec: float, is_final: bool) -> TransducerUpdate | None:
        """Build an incremental subtitle update from recognizer state."""
        text = self.recognizer.get_result(self.stream).strip()
        if not text or text == self.prev_text:
            return None

        tokens = list(self.recognizer.tokens(self.stream))
        timestamps = list(self.recognizer.timestamps(self.stream))
        #print(f"Recognizer: {self.recognizer}")
        #print(f"Tokens: {tokens}")
        #print(f"Timestamps: {timestamps}")
        lcp = _lcp_len(self.prev_tokens, tokens)
        #print("tokens이 뭐지?", tokens)

        #print("lcp가 뭐지?", lcp)
        #print("timestamps가 뭐지?", timestamps)
        
        if lcp < len(timestamps):
            
            start_sec = float(timestamps[lcp])
        elif timestamps:
            start_sec = float(timestamps[-1])
        else:
            start_sec = max(0.0, end_sec - 0.2)

        self.prev_text = text
        self.prev_tokens = tokens
        return TransducerUpdate(
            start_sec=start_sec,
            end_sec=end_sec,
            text=text,
            is_final=is_final,
        )

    def feed(self, samples: np.ndarray, end_sec: float) -> TransducerUpdate | None:
        """Push one audio frame and return incremental live update."""
        self.stream.accept_waveform(self.cfg.sample_rate, samples)
        self._decode_ready()
        return self._build_update(end_sec=end_sec, is_final=False)

    def finalize(self, end_sec: float) -> TransducerUpdate | None:
        print("실시간 자막 완료")
        """Finalize stream and return last live update if available."""
        self.stream.input_finished()
        self._decode_ready()
        return self._build_update(end_sec=end_sec, is_final=True)
