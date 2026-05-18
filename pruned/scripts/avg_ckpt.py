"""
Average multiple FunASR checkpoints into a single model.pt.avgN.

Usage
  # 지정한 체크포인트들 평균
  python avg_ckpt.py --model-dir /deepet/jh/sensevoice/finetune/pruned/workdir/sensevoice_pruned_small \
                     --ckpts model.pt.ep28.10000 model.pt.ep29 model.pt.ep29.5000 \
                             model.pt.ep29.10000 model.pt.ep30

  # 디렉토리에서 최신 N개 자동 선택
  python avg_ckpt.py --model-dir .../sensevoice_pruned_small --last 5

  # 출력 파일명 직접 지정
  python avg_ckpt.py --model-dir ... --last 5 --out model.pt.avg5
"""
import argparse
import re
from pathlib import Path

import torch


def sort_key(name: str):
    # model.pt.ep29 / model.pt.ep29.5000 / model.pt.ep29.10000
    m = re.match(r"model\.pt\.ep(\d+)(?:\.(\d+))?$", name)
    if not m:
        return (-1, -1)
    ep = int(m.group(1))
    step = int(m.group(2)) if m.group(2) else 0
    return (ep, step)


def discover_last(model_dir: Path, n: int) -> list[str]:
    cands = [p.name for p in model_dir.glob("model.pt.ep*") if sort_key(p.name) != (-1, -1)]
    cands.sort(key=sort_key)
    picked = cands[-n:]
    if len(picked) < n:
        raise SystemExit(f"only {len(picked)} ep ckpts found, need {n}")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ckpts", nargs="*", default=None, help="explicit ckpt filenames")
    ap.add_argument("--last", type=int, default=0, help="auto-pick last N ep*")
    ap.add_argument("--out", default=None, help="output filename (default model.pt.avg<N>)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if args.ckpts:
        names = args.ckpts
    elif args.last > 0:
        names = discover_last(model_dir, args.last)
    else:
        raise SystemExit("need --ckpts or --last")

    paths = [model_dir / n for n in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")

    print(f"[avg] averaging {len(paths)} ckpts:")
    for p in paths:
        print(f"  - {p.name}")

    def unwrap(sd):
        # FunASR may save pure state_dict or {"model"/"state_dict": state_dict, ...}
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            return sd["model"]
        if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            return sd["state_dict"]
        return sd

    def tensor_items(sd):
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                yield k, v

    avg = None
    extras = None  # carry non-float tensor entries from the last ckpt (ints, buffers, etc.)
    for i, p in enumerate(paths, 1):
        raw = torch.load(p, map_location="cpu", weights_only=False)
        sd = unwrap(raw)
        if avg is None:
            avg = {k: v.detach().float().clone() for k, v in tensor_items(sd)}
            extras = {k: v for k, v in sd.items() if k not in avg}
        else:
            for k, v in tensor_items(sd):
                avg[k] += v.detach().float()
        print(f"  [{i}/{len(paths)}] loaded {p.name}")

    for k in avg:
        avg[k] /= len(paths)

    # Preserve dtype of each parameter (FunASR often saves fp32, but cast back just in case).
    final = {}
    # reference dtype from first ckpt
    ref_raw = torch.load(paths[0], map_location="cpu", weights_only=False)
    ref_sd = unwrap(ref_raw)
    for k, v in avg.items():
        final[k] = v.to(ref_sd[k].dtype)
    for k, v in extras.items():
        if k not in final:
            final[k] = v

    out_name = args.out or f"model.pt.avg{len(paths)}"
    out_path = model_dir / out_name
    torch.save(final, out_path)
    print(f"[avg] averaged {len(avg)} float tensors, kept {len(extras)} non-float entries")
    print(f"[avg] saved -> {out_path}")


if __name__ == "__main__":
    main()
