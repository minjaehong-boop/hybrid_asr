"""
SenseVoice CTC-scratch 학습 (서버 실행용, pretrained weight 로드 없음).

기본 경로 (서버):
  프로젝트: /deepet/jh/sensevoice/finetune/scratch/
  jsonl:    /deepet/jh/sensevoice/finetune/jsonl/   (scratch 외부, 형제 디렉토리)

Usage:
  python train_sensevoice_scratch.py                                     # train/dev
  python train_sensevoice_scratch.py --train train_kspon --dev dev_kspon
  python train_sensevoice_scratch.py ++train_conf.max_epoch=20           # Hydra override
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sensevoice_scratch.py
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
CONF_DIR = PROJECT / "conf"
DEFAULT_JSONL_DIR = PROJECT.parent / "jsonl"


def main():
    import argparse

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--tag", default="ko5k")
    pre.add_argument("--config", default=None)
    pre.add_argument("--train", default="train", help="train jsonl 이름 (.jsonl 제외)")
    pre.add_argument("--dev", default="dev", help="dev jsonl 이름 (.jsonl 제외)")
    pre.add_argument("--jsonl-dir", default=str(DEFAULT_JSONL_DIR))
    known, extra = pre.parse_known_args()

    tag = known.tag
    config_name = known.config or f"sensevoice_scratch_{tag}"
    output_dir = PROJECT / "finetune" / f"sensevoice_scratch_{tag}"

    jsonl_dir = Path(known.jsonl_dir)
    train_jsonl = jsonl_dir / f"{known.train}.jsonl"
    dev_jsonl = jsonl_dir / f"{known.dev}.jsonl"

    cmvn = PROJECT / "assets" / "kspon.mvn"
    bpe = PROJECT / "assets" / "lang_bpe_5000" / "bpe.model"

    missing = [p for p in [train_jsonl, dev_jsonl, cmvn, bpe] if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: {p} not found.")
        if not cmvn.exists():
            print("\n[hint] CMVN 먼저 계산:")
            print(f"  python scripts/compute_kspon_cmvn.py --jsonl {train_jsonl} "
                  f"--out {cmvn} --n 20000")
        sys.exit(1)

    def q(p):
        return f"'{p}'"

    argv = [
        "funasr-train",
        f"--config-path={CONF_DIR}",
        f"--config-name={config_name}",
        f"++train_data_set_list={q(train_jsonl)}",
        f"++valid_data_set_list={q(dev_jsonl)}",
        f"++output_dir={q(output_dir)}",
    ] + extra

    sys.argv = argv

    # tables.register 데코레이터가 Hydra import 전에 실행되어야 함
    sys.path.insert(0, str(PROJECT))
    import sensevoice_scratch  # noqa: F401

    from funasr.bin.train import main_hydra

    main_hydra()


if __name__ == "__main__":
    os.chdir(PROJECT)
    main()
