"""
SenseVoice-Small 구조적 프루닝: 레이어를 균등 간격으로 선택하여 축소된 체크포인트 생성.

원본: encoders0(1) + encoders(49) + tp_encoders(20) = 70 blocks, 234M params
프루닝: encoders0(1) + encoders(keep_main-1) + tp_encoders(keep_tp) blocks

Usage:
  python scripts/prune_sensevoice.py --keep-main 25 --keep-tp 10
  python scripts/prune_sensevoice.py --keep-main 12 --keep-tp 6 --tag tiny
"""
import argparse
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path(
    "/home/alswoghd/.cache/modelscope/hub/models/iic/SenseVoiceSmall/model.pt"
)


def select_indices(total: int, keep: int) -> list[int]:
    """균등 간격으로 keep개 인덱스 선택."""
    if keep >= total:
        return list(range(total))
    return [int(round(i * (total - 1) / (keep - 1))) for i in range(keep)]


def prune(src_path: Path, keep_main: int, keep_tp: int, tag: str):
    print(f"Loading {src_path} ...")
    ckpt = torch.load(str(src_path), map_location="cpu")

    # 원본 블록 수 파악
    main_indices = set()
    tp_indices = set()
    for k in ckpt:
        m = re.match(r"encoder\.encoders\.(\d+)\.", k)
        if m:
            main_indices.add(int(m.group(1)))
        m = re.match(r"encoder\.tp_encoders\.(\d+)\.", k)
        if m:
            tp_indices.add(int(m.group(1)))

    n_main = len(main_indices)  # 49
    n_tp = len(tp_indices)      # 20
    print(f"Original: encoders0(1) + encoders({n_main}) + tp_encoders({n_tp}) = {1+n_main+n_tp} blocks")

    # 선택할 인덱스
    sel_main = select_indices(n_main, keep_main - 1)  # keep_main includes encoders0
    sel_tp = select_indices(n_tp, keep_tp)
    print(f"Pruned:   encoders0(1) + encoders({len(sel_main)}) + tp_encoders({len(sel_tp)}) = {1+len(sel_main)+len(sel_tp)} blocks")
    print(f"  main selected: {sel_main}")
    print(f"  tp selected:   {sel_tp}")

    # 새 state_dict 생성
    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        # encoders (main blocks)
        m = re.match(r"encoder\.encoders\.(\d+)\.(.*)", k)
        if m:
            old_idx = int(m.group(1))
            if old_idx in sel_main:
                new_idx = sel_main.index(old_idx)
                new_key = f"encoder.encoders.{new_idx}.{m.group(2)}"
                new_ckpt[new_key] = v
            continue

        # tp_encoders
        m = re.match(r"encoder\.tp_encoders\.(\d+)\.(.*)", k)
        if m:
            old_idx = int(m.group(1))
            if old_idx in sel_tp:
                new_idx = sel_tp.index(old_idx)
                new_key = f"encoder.tp_encoders.{new_idx}.{m.group(2)}"
                new_ckpt[new_key] = v
            continue

        # 나머지 (encoders0, after_norm, tp_norm, ctc, embed 등)
        new_ckpt[k] = v

    # 파라미터 수 비교
    old_params = sum(v.numel() for v in ckpt.values())
    new_params = sum(v.numel() for v in new_ckpt.values())
    print(f"\nParams: {old_params:,} → {new_params:,} ({new_params/old_params*100:.1f}%)")

    # 저장
    out_dir = ROOT / "workdir" / f"sensevoice_pruned_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pt.pruned"
    torch.save(dict(new_ckpt), str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")

    # YAML에 넣을 설정값 출력
    print(f"\n--- YAML config ---")
    print(f"num_blocks: {keep_main}")
    print(f"tp_blocks: {keep_tp}")
    print(f"init_param: {out_path}")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="SenseVoice 구조적 프루닝")
    ap.add_argument("--src", type=str, default=str(DEFAULT_SRC), help="원본 체크포인트")
    ap.add_argument("--keep-main", type=int, default=25,
                    help="유지할 main 블록 수 (encoders0 포함, 원본 50)")
    ap.add_argument("--keep-tp", type=int, default=10,
                    help="유지할 tp 블록 수 (원본 20)")
    ap.add_argument("--tag", type=str, default="half",
                    help="출력 디렉토리 태그")
    args = ap.parse_args()
    prune(Path(args.src), args.keep_main, args.keep_tp, args.tag)


if __name__ == "__main__":
    main()
