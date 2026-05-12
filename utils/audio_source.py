from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(slots=True)
class AudioFrame:
    start_sec: float
    end_sec: float
    samples: np.ndarray
    sample_rate: int


class WavChunkSource:
    def __init__(
        self,
        path: Path,
        frame_sec: float,
        target_sr: int = 16000,
        realtime: bool = True,
        max_seconds: float | None = None,
    ) -> None:
        """Load wav and prepare frame iterator for streaming simulation."""
        self.path = path
        self.frame_sec = frame_sec
        self.target_sr = target_sr
        self.realtime = realtime
        self.max_seconds = max_seconds
        self.audio = self._load_audio()
        self.duration_sec = len(self.audio) / self.target_sr

    def _load_audio(self) -> np.ndarray:
        """Load, mono-mix, resample, and trim audio input."""
        if not self.path.exists():
            raise FileNotFoundError(f"Audio file not found: {self.path}")
        audio, sr = sf.read(str(self.path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        if self.max_seconds is not None:
            n = int(self.max_seconds * self.target_sr)
            audio = audio[:n]
        return np.asarray(audio, dtype=np.float32)

    def iter_frames(self):
        """Yield fixed-size audio frames with optional wall-clock pacing."""
        chunk_size = max(1, int(self.frame_sec * self.target_sr))
        wall_start = time.monotonic()
        for offset in range(0, len(self.audio), chunk_size):
            chunk = self.audio[offset : offset + chunk_size]
            if chunk.size == 0:
                continue
            start_sec = offset / self.target_sr
            end_sec = (offset + len(chunk)) / self.target_sr
            if self.realtime:
                elapsed_audio = end_sec
                elapsed_wall = time.monotonic() - wall_start
                if elapsed_audio > elapsed_wall:
                    time.sleep(elapsed_audio - elapsed_wall)
            yield AudioFrame(
                start_sec=start_sec,
                end_sec=end_sec,
                samples=chunk,
                sample_rate=self.target_sr,
            )

class MicChunkSource:
    """Real-time microphone input with sherpa-onnx Silero VAD.

    Yields AudioFrame objects from the microphone. Automatically stops
    after ``silence_timeout`` seconds of continuous silence (VAD off).
    Also stops on KeyboardInterrupt or if ``max_seconds`` is reached.
    """

    def __init__(
        self,
        frame_sec: float = 0.1,
        target_sr: int = 16000,
        max_seconds: float | None = None,
        vad_model: str = "models/silero_vad.onnx",
        vad_threshold: float = 0.5,
        silence_timeout: float = 2.0,
    ) -> None:
        """Set up microphone capture and VAD."""
        import sounddevice as sd
        from asr.transducer_stage import _import_sherpa_onnx

        self.frame_sec = frame_sec
        self.target_sr = target_sr
        self.max_seconds = max_seconds
        self.silence_timeout = silence_timeout
        self.duration_sec = 0.0
        self._sd = sd
        self._stopped = False

        # Init sherpa-onnx Silero VAD
        sherpa_onnx = _import_sherpa_onnx()
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
                model=vad_model,
                threshold=vad_threshold,
                min_silence_duration=silence_timeout,
                min_speech_duration=0.25,
                window_size=512,
            ),
            sample_rate=target_sr,
            num_threads=1,
            provider="cpu",
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

    def iter_frames(self):
        """Yield AudioFrame from microphone. Stops on silence timeout or Ctrl+C."""
        chunk_size = max(1, int(self.frame_sec * self.target_sr))
        vad_window = 512
        sample_offset = 0
        silence_start = None
        speech_ever = False

        print("Listening... (speak now, silence will auto-stop)", flush=True)

        try:
            with self._sd.InputStream(
                samplerate=self.target_sr,
                channels=1,
                dtype="float32",
                blocksize=chunk_size,
            ) as stream:
                while not self._stopped:
                    data, overflowed = stream.read(chunk_size)
                    if overflowed:
                        print("[WARN] audio buffer overflow", flush=True)

                    samples = data[:, 0].copy()
                    start_sec = sample_offset / self.target_sr
                    end_sec = (sample_offset + len(samples)) / self.target_sr
                    sample_offset += len(samples)
                    self.duration_sec = end_sec

                    # Feed VAD in 512-sample windows
                    for wi in range(0, len(samples), vad_window):
                        window = samples[wi : wi + vad_window]
                        if len(window) == vad_window:
                            self.vad.accept_waveform(window)

                    # Drain completed segments (keeps VAD internal state clean)
                    while not self.vad.empty():
                        self.vad.pop()

                    # Track silence for auto-stop
                    if self.vad.is_speech_detected():
                        speech_ever = True
                        silence_start = None
                    else:
                        if speech_ever and silence_start is None:
                            silence_start = time.monotonic()

                    if (
                        speech_ever
                        and silence_start is not None
                        and time.monotonic() - silence_start >= self.silence_timeout
                    ):
                        print("Silence detected, stopping.", flush=True)
                        break

                    # max_seconds limit
                    if self.max_seconds is not None and end_sec >= self.max_seconds:
                        break

                    yield AudioFrame(
                        start_sec=start_sec,
                        end_sec=end_sec,
                        samples=samples,
                        sample_rate=self.target_sr,
                    )

        except KeyboardInterrupt:
            print("\nStopped by user.", flush=True)

