#!/usr/bin/env python3
"""aval.py — Compare accumulate vs overlap CC modes on a WAV directory or manifest.

Models are loaded once and reused across all files.

Usage:
    python aval.py                                          # default: data/utterances/, both modes
    python aval.py --wav data/utterances/
    python aval.py --wav data/utterances/ --mode overlap
    python aval.py --manifest data/label.txt
    python aval.py --manifest data/label.txt --limit 10
    python aval.py --wav data/utterances/ --chunk-sec 4.0 --trim-sec 0.5
    python aval.py --wav data/utterances/ --no-verbose     # summary only
"""
from __future__ import annotations

import argparse
import csv
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf

from config.settings import SenseVoiceConfig
from asr.refine_stage import SenseVoiceRefineStage

SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = Path("models/refine_model")
DEFAULT_WAV_DIR = Path("data_003/utterances")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_wav(path: Path) -> np.ndarray:
    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        samples = librosa.resample(samples, orig_sr=sr, target_sr=SAMPLE_RATE)
    return samples.astype(np.float32)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compact(text: str) -> str:
    return "".join(text.split())


def cer_no_space(hyp: str, ref: str) -> float:
    h, r = compact(hyp), compact(ref)
    if not r:
        return 0.0
    return max(0.0, 1.0 - SequenceMatcher(None, h, r, autojunk=False).ratio())


# ---------------------------------------------------------------------------
# Stage factory — load once
# ---------------------------------------------------------------------------

def make_stages(
    modes: list[str],
    chunk_sec: float,
    trim_sec: float,
    model_dir: Path,
) -> dict[str, SenseVoiceRefineStage]:
    """Create one SenseVoiceRefineStage per mode (model loaded once each)."""
    stages: dict[str, SenseVoiceRefineStage] = {}
    for mode in modes:
        print(f"Loading model for mode={mode} ...", flush=True)
        cfg = SenseVoiceConfig(
            language="ko",
            model_dir=model_dir,
            provider="cpu",
            num_threads=2,
            use_itn=False,
            chunk_sec=chunk_sec,
            trim_sec=trim_sec,
            context_mode=mode,
        )
        stages[mode] = SenseVoiceRefineStage(cfg)
    return stages


# ---------------------------------------------------------------------------
# Core runner (reuses an existing stage)
# ---------------------------------------------------------------------------

def run_mode(
    samples: np.ndarray,
    stage: SenseVoiceRefineStage,
    mode: str,
    verbose: bool,
) -> tuple[str, float]:
    """Feed samples through stage (after reset). Returns (hyp, elapsed_sec)."""
    stage.reset()
    chunk_samples = int(stage.cfg.chunk_sec * SAMPLE_RATE)
    parts: list[str] = []

    t0 = time.perf_counter()
    pos = 0
    while pos < len(samples):
        chunk = samples[pos : pos + chunk_samples]
        pos += chunk_samples
        update = stage.feed(chunk)
        if update and update.text:
            if verbose:
                print(f"  [{mode}] {update.start_sec:.1f}~{update.end_sec:.1f}s  {update.text}")
            parts.append(update.text)

    final = stage.finalize()
    if final and final.text:
        if verbose:
            print(f"  [{mode}] {final.start_sec:.1f}~{final.end_sec:.1f}s  {final.text}  [FINAL]")
        parts.append(final.text)

    elapsed = time.perf_counter() - t0
    return " ".join(parts), elapsed


# ---------------------------------------------------------------------------
# Per-file evaluation
# ---------------------------------------------------------------------------

def eval_wav(
    wav: Path,
    stages: dict[str, SenseVoiceRefineStage],
    ref: str = "",
    verbose: bool = True,
) -> dict[str, dict]:
    modes = list(stages.keys())
    samples = load_wav(wav)
    dur = len(samples) / SAMPLE_RATE
    first_stage = next(iter(stages.values()))
    chunk_sec = first_stage.cfg.chunk_sec
    trim_sec = first_stage.trim_sec
    stride = chunk_sec - trim_sec
    print(f"\n=== {wav.name}  dur={dur:.1f}s  chunk={chunk_sec}s  trim={trim_sec}s  stride={stride:.1f}s ===")

    results: dict[str, dict] = {}
    for mode, stage in stages.items():
        print(f"\n-- {mode} --")
        hyp, elapsed = run_mode(samples, stage, mode, verbose=verbose)
        cer = cer_no_space(hyp, ref) if ref else None
        rtf = elapsed / dur if dur > 0 else 0.0
        results[mode] = {"hyp": hyp, "cer": cer, "rtf": rtf}

    print(f"\n{'─' * 72}")
    for mode in modes:
        r = results[mode]
        cer_str = f"CER={r['cer']:.4f}  " if r["cer"] is not None else ""
        print(f"  [{mode:10s}]  RTF={r['rtf']:.3f}  {cer_str}{r['hyp']}")
    if ref:
        print(f"  [{'ref':10s}]  {ref}")

    return results


# ---------------------------------------------------------------------------
# Label lookup
# ---------------------------------------------------------------------------

def load_labels(label_file: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not label_file.exists():
        return labels
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if "::" not in line:
            continue
        p, t = line.split("::", 1)
        labels[Path(p.strip()).stem] = t.strip()
    return labels


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def csv_path(mode: str, chunk_sec: float, trim_sec: float) -> Path:
    return Path(f"results_{chunk_sec:.1f}s_{trim_sec:.1f}s_{mode}.csv")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "ref", "hyp", "cer", "rtf"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved: {path}")


def _run_batch(
    items: list[tuple[Path, str]],
    stages: dict[str, SenseVoiceRefineStage],
    verbose: bool,
) -> tuple[dict[str, list[float]], dict[str, list[dict]]]:
    modes = list(stages.keys())
    all_cer: dict[str, list[float]] = {m: [] for m in modes}
    all_rows: dict[str, list[dict]] = {m: [] for m in modes}
    for wav_path, ref in items:
        r = eval_wav(wav_path, stages, ref=ref, verbose=verbose)
        for m in modes:
            all_rows[m].append({
                "filename": wav_path.name,
                "ref": ref,
                "hyp": r[m]["hyp"],
                "cer": f"{r[m]['cer']:.4f}" if r[m]["cer"] is not None else "",
                "rtf": f"{r[m]['rtf']:.4f}",
            })
            if r[m]["cer"] is not None:
                all_cer[m].append(r[m]["cer"])
    return all_cer, all_rows


def eval_dir(
    wav_dir: Path,
    stages: dict[str, SenseVoiceRefineStage],
    limit: int | None,
    verbose: bool,
) -> None:
    wav_files = sorted(wav_dir.glob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]
    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    labels = load_labels(Path("data/label.txt"))
    items = [(w, labels.get(w.stem, "")) for w in wav_files]
    first_stage = next(iter(stages.values()))
    chunk_sec = first_stage.cfg.chunk_sec
    trim_sec = first_stage.trim_sec
    all_cer, all_rows = _run_batch(items, stages, verbose)

    for mode, rows in all_rows.items():
        write_csv(rows, csv_path(mode, chunk_sec, trim_sec))

    if len(wav_files) > 1:
        modes = list(stages.keys())
        print(f"\n{'=' * 72}")
        print(f"DIRECTORY SUMMARY  ({len(wav_files)} files, {wav_dir})")
        for m in modes:
            cers = all_cer[m]
            if cers:
                print(f"  {m:12s}  avg_CER={sum(cers)/len(cers):.4f}  n={len(cers)}")


def load_manifest(path: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    base = path.parent
    for line in path.read_text(encoding="utf-8").splitlines():
        if "::" not in line:
            continue
        wav_str, text = line.split("::", 1)
        wav_path = Path(wav_str.strip())
        if not wav_path.is_absolute():
            wav_path = base / wav_path
        items.append((wav_path, text.strip()))
    return items


def eval_manifest(
    manifest: Path,
    stages: dict[str, SenseVoiceRefineStage],
    limit: int | None,
    verbose: bool = True,
) -> None:
    items = load_manifest(manifest)
    if limit:
        items = items[:limit]
    modes = list(stages.keys())
    first_stage = next(iter(stages.values()))
    chunk_sec = first_stage.cfg.chunk_sec
    trim_sec = first_stage.trim_sec
    all_cer, all_rows = _run_batch(items, stages, verbose=verbose)

    for mode, rows in all_rows.items():
        write_csv(rows, csv_path(mode, chunk_sec, trim_sec))

    print(f"\n{'=' * 72}")
    print(f"MANIFEST SUMMARY  ({len(items)} files)")
    for m in modes:
        cers = all_cer[m]
        if cers:
            print(f"  {m:12s}  avg_CER={sum(cers)/len(cers):.4f}  n={len(cers)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare accumulate vs overlap CC modes (model loaded once)."
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="WAV directory (all *.wav) or single WAV file.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="label.txt manifest (path :: ref).")
    parser.add_argument(
        "--mode",
        choices=["accumulate", "overlap", "both"],
        default="both",
        help="Which mode(s) to run (default: both).",
    )
    parser.add_argument("--chunk-sec", type=float, default=4.0)
    parser.add_argument("--trim-sec", type=float, default=0.5)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Max files to process.")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--no-verbose", dest="verbose", action="store_false",
                        help="Summary only, no chunk-by-chunk output.")
    args = parser.parse_args()

    modes = ["accumulate", "overlap"] if args.mode == "both" else [args.mode]
    stages = make_stages(modes, args.chunk_sec, args.trim_sec, args.model_dir)

    if args.manifest:
        eval_manifest(args.manifest, stages, args.limit, verbose=args.verbose)
    elif args.wav is not None and args.wav.is_dir():
        eval_dir(args.wav, stages, args.limit, verbose=args.verbose)
    elif args.wav is not None and args.wav.is_file():
        labels = load_labels(Path("data/label.txt"))
        ref = labels.get(args.wav.stem, "")
        r = eval_wav(args.wav, stages, ref=ref, verbose=args.verbose)
        first_stage = next(iter(stages.values()))
        chunk_sec = first_stage.cfg.chunk_sec
        trim_sec = first_stage.trim_sec
        for mode, data in r.items():
            rows = [{
                "filename": args.wav.name,
                "ref": ref,
                "hyp": data["hyp"],
                "cer": f"{data['cer']:.4f}" if data["cer"] is not None else "",
                "rtf": f"{data['rtf']:.4f}",
            }]
            write_csv(rows, csv_path(mode, chunk_sec, trim_sec))
    else:
        eval_dir(DEFAULT_WAV_DIR, stages, args.limit, verbose=args.verbose)


if __name__ == "__main__":
    main()
