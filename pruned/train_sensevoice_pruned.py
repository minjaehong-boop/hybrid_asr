"""
프루닝된 SenseVoice-Small 파인튜닝 (서버 실행용).

기본 경로 (서버):
  프로젝트: /deepet/jh/sensevoice/finetune/pruned/
  jsonl:    /deepet/jh/sensevoice/finetune/jsonl/   (finetuning 외부, 형제 디렉토리)

Usage:
  python train_sensevoice_pruned.py                                   # tag=tiny, jsonl=train/dev
  python train_sensevoice_pruned.py --tag tiny --train train_kspon --dev dev_kspon
  python train_sensevoice_pruned.py ++train_conf.max_epoch=10         # Hydra override
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
CONF_DIR = PROJECT / "conf"
# jsonl은 finetuning 폴더 외부 (sibling): /deepet/jh/sensevoice/finetune/jsonl/
DEFAULT_JSONL_DIR = PROJECT.parent / "jsonl"


def main():
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--tag", default="tiny")
    pre_parser.add_argument("--config", default=None)
    pre_parser.add_argument("--train", default="train_both", help="train jsonl 이름 (.jsonl 제외)")
    pre_parser.add_argument("--dev", default="dev_both", help="dev jsonl 이름 (.jsonl 제외)")
    pre_parser.add_argument("--jsonl-dir", default=str(DEFAULT_JSONL_DIR))
    known, extra = pre_parser.parse_known_args()

    tag = known.tag
    config_name = known.config or f"sensevoice_pruned_{tag}"
    output_dir = PROJECT / "workdir" / f"sensevoice_pruned_{tag}"
    init_param = output_dir / "model.pt.pruned"

    jsonl_dir = Path(known.jsonl_dir)
    train_jsonl = jsonl_dir / f"{known.train}.jsonl"
    dev_jsonl = jsonl_dir / f"{known.dev}.jsonl"

    for p in [init_param, train_jsonl, dev_jsonl]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
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
        f"++init_param={q(init_param)}",
        "++ignore_init_mismatch=true",
    ] + extra

    sys.argv = argv
    from funasr.bin.train import main_hydra

    main_hydra()


if __name__ == "__main__":
    os.chdir(PROJECT)
    main()
