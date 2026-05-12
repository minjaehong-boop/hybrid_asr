from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class TransducerConfig:
    tokens: Path
    encoder: Path
    decoder: Path
    joiner: Path
    sample_rate: int = 16000
    feature_dim: int = 80
    num_threads: int = 2
    provider: str = "cpu"


@dataclass(slots=True)
class SenseVoiceConfig:
    language: str
    model_dir: Path
    provider: str = "cpu"
    num_threads: int = 2
    use_itn: bool = False
    chunk_sec: float = 4.0
    trim_sec: float = 0.5
    context_mode: str = "accumulate"  # "accumulate" | "overlap"


@dataclass(slots=True)
class PipelineConfig:
    audio_path: Path | None
    frame_sec: float
    refine_chunk_sec: float
    realtime: bool
    max_seconds: float | None
    transducer: TransducerConfig
    refine: SenseVoiceConfig
    use_mic: bool = False
    vad_model: str = "models/silero_vad.onnx"
    silence_timeout: float = 3.0


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "default.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Parse minimal CLI arguments. Model/pipeline params live in YAML."""
    parser = argparse.ArgumentParser(
        description="Two-stage real-time subtitle pipeline (transducer + SenseVoice refine)."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Input wav file (16kHz mono recommended).",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Use microphone input with VAD (instead of wav file).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config file (default: config/default.yaml).",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Process as fast as possible without simulated playback.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional audio duration limit for quick debugging.",
    )
    parser.add_argument(
        "--refine-mode",
        choices=["accumulate", "overlap"],
        default=None,
        help="Refine context mode: 'accumulate' (up to 4 chunks) or 'overlap' (fixed 2*trim_sec buffer).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build PipelineConfig from YAML + CLI overrides."""
    config_path = args.config or _default_config_path()
    cfg = _load_yaml(config_path)

    t = cfg.get("transducer", {})
    r = cfg.get("refine", {})
    p = cfg.get("pipeline", {})

    model_dir = Path(t.get("model_dir", "models/epoch30-avg9"))
    use_int8 = t.get("use_int8", True)
    suffix = ".int8" if use_int8 else ""

    transducer = TransducerConfig(
        tokens=model_dir / "tokens.txt",
        encoder=model_dir / f"encoder-epoch-30-avg-9-with-averaged-model{suffix}.onnx",
        decoder=model_dir / f"decoder-epoch-30-avg-9-with-averaged-model{suffix}.onnx",
        joiner=model_dir / f"joiner-epoch-30-avg-9-with-averaged-model{suffix}.onnx",
        sample_rate=t.get("sample_rate", 16000),
        feature_dim=t.get("feature_dim", 80),
        num_threads=t.get("num_threads", 2),
        provider=t.get("provider", "cpu"),
    )

    refine_model_dir = Path(r.get(
        "model_dir",
        "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
    ))
    if not refine_model_dir.is_dir():
        raise FileNotFoundError(
            f"SenseVoice model directory not found: {refine_model_dir}\n"
            "Set refine.model_dir in your YAML config."
        )

    refine_chunk_sec = r.get("chunk_sec", 4.0)
    context_mode = getattr(args, "refine_mode", None) or r.get("context_mode", "accumulate")
    refine = SenseVoiceConfig(
        language=r.get("language", "ko"),
        model_dir=refine_model_dir.resolve(),
        provider=r.get("provider", "cpu"),
        num_threads=max(1, r.get("num_threads", 2)),
        use_itn=r.get("use_itn", False),
        chunk_sec=refine_chunk_sec,
        trim_sec=r.get("trim_sec", 0.5),
        context_mode=context_mode,
    )

    realtime = p.get("realtime", True)
    if args.no_realtime:
        realtime = False

    max_seconds = args.max_seconds or p.get("max_seconds")

    use_mic = args.mic
    audio_path = args.audio
    if not use_mic and audio_path is None:
        audio_path = Path("data/utterances/utt_0001.wav")

    return PipelineConfig(
        audio_path=audio_path,
        frame_sec=p.get("frame_sec", 0.1),
        refine_chunk_sec=refine_chunk_sec,
        realtime=realtime,
        max_seconds=max_seconds,
        transducer=transducer,
        refine=refine,
        use_mic=use_mic,
        vad_model=p.get("vad_model", "models/vad_model/silero_vad.onnx"),
        silence_timeout=p.get("silence_timeout", 2.0),
    )
