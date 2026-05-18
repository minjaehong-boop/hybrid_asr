"""
Convert KsponSpeech .trn to FunASR jsonl.

Input line format:  KsponSpeech_XX/.../KsponSpeech_NNNNNN.pcm :: text
Output jsonl line:  {"key": ..., "source": absolute_flac_path, "target": text, "source_len": frames_hint, "target_len": chars}

Uses .flac files (same stem as .pcm). Punctuation/ITN kept as-is.
"""
import argparse
import json
import soundfile as sf
from pathlib import Path

DEFAULT_ROOT = Path("/deepet/jh/sensevoice/dataset/ksponspeech")


def build(trn_path: Path, out_path: Path, root: Path, skip_missing: bool = True, probe_duration: bool = False):
    n_out = 0
    n_miss = 0
    with trn_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n").rstrip("\r")
            if " :: " not in line:
                continue
            rel, text = line.split(" :: ", 1)
            flac = (root / rel).with_suffix(".flac")
            if not flac.exists():
                n_miss += 1
                if skip_missing:
                    continue
            key = flac.stem
            if probe_duration and flac.exists():
                info = sf.info(str(flac))
                source_len = int(info.frames / info.samplerate * 100)  # 10ms frames
            else:
                source_len = 0
            rec = {
                "key": key,
                "source": str(flac),
                "source_len": source_len,
                "target": text,
                "target_len": len(text),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"[{trn_path.name}] wrote {n_out}, missing {n_miss} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="ksponspeech root (contains train.trn, dev.trn, KsponSpeech_*)")
    ap.add_argument("--out-dir", default="/deepet/jh/sensevoice/finetune/jsonl")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip flac header probing (fast but source_len=0; length-batch filter must be relaxed)")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = not args.no_probe
    build(root / "train.trn", out_dir / "train.jsonl", root, probe_duration=probe)
    build(root / "dev.trn", out_dir / "dev.jsonl", root, probe_duration=probe)


if __name__ == "__main__":
    main()
