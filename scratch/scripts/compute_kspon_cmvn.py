"""
KsponSpeech 기반 CMVN(.mvn, SenseVoice 포맷) 재계산.

80-dim log-mel fbank에 대해 mean/std를 누적 후, SenseVoice가 쓰는 Kaldi-style
Nnet 포맷으로 덤프한다. LFR(m=7) stack을 가정해 80-dim 통계를 7번 반복해 560-dim.

Usage:
  python scripts/compute_kspon_cmvn.py                        # default 5000 uttr
  python scripts/compute_kspon_cmvn.py --n 20000
  python scripts/compute_kspon_cmvn.py --jsonl finetune/jsonl/train.jsonl \
      --out resources/kspon.mvn
"""
import argparse
import json
import random
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def fbank80(wav_np, sr=16000):
    x = torch.from_numpy(wav_np).float().unsqueeze(0) * (1 << 15)
    return kaldi.fbank(
        x,
        num_mel_bins=80,
        sample_frequency=sr,
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        frame_length=25.0,
        frame_shift=10.0,
        snip_edges=True,
    )


def write_mvn(out_path: Path, mean80: torch.Tensor, istd80: torch.Tensor, lfr_m: int = 7):
    # SenseVoice 포맷: 80-dim 통계를 lfr_m번 복제해 560-dim
    shift_560 = (-mean80).repeat(lfr_m).tolist()
    scale_560 = istd80.repeat(lfr_m).tolist()
    dim = 80 * lfr_m

    def fmt(vals):
        return " ".join(f"{v:.7g}" for v in vals)

    lines = [
        "<Nnet>",
        f"<Splice> {dim} {dim}",
        "[ 0 ]",
        f"<AddShift> {dim} {dim} ",
        f"<LearnRateCoef> 0 [ {fmt(shift_560)} ]",
        f"<Rescale> {dim} {dim}",
        f"<LearnRateCoef> 0 [ {fmt(scale_560)} ]",
        "</Nnet>",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", default="/deepet/jh/sensevoice/finetune/jsonl/train.jsonl")
    p.add_argument("--out", default="resources/kspon.mvn")
    p.add_argument("--n", type=int, default=5000, help="무작위 샘플 utterance 수 (-1=전체)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lfr-m", type=int, default=7)
    args = p.parse_args()

    jsonl_path = Path(args.jsonl).resolve()
    items = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    random.Random(args.seed).shuffle(items)
    if args.n > 0:
        items = items[: args.n]

    sum_x = torch.zeros(80, dtype=torch.float64)
    sum_x2 = torch.zeros(80, dtype=torch.float64)
    n_frames = 0
    skipped = 0

    for it in tqdm(items, desc="fbank"):
        src = it["source"]
        try:
            wav, sr = sf.read(src)
        except Exception:
            skipped += 1
            continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            skipped += 1
            continue
        if len(wav) < 400:
            skipped += 1
            continue
        feat = fbank80(wav, sr=sr).double()
        sum_x += feat.sum(0)
        sum_x2 += (feat * feat).sum(0)
        n_frames += feat.shape[0]

    if n_frames == 0:
        raise RuntimeError("No frames accumulated")

    mean = sum_x / n_frames
    var = (sum_x2 / n_frames - mean * mean).clamp_min(1e-8)
    istd = 1.0 / var.sqrt()

    out = Path(args.out).resolve()
    write_mvn(out, mean.float(), istd.float(), lfr_m=args.lfr_m)
    print(f"[ok] wrote {out}  (n_utt={len(items) - skipped}, n_frames={n_frames}, skipped={skipped})")


if __name__ == "__main__":
    main()
