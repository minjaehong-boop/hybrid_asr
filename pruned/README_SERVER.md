# SenseVoiceSmall 프루닝 파인튜닝

서버 경로:
- 프로젝트: `/deepet/jh/sensevoice/finetune/pruned/`
- jsonl:    `/deepet/jh/sensevoice/finetune/jsonl/` (sibling)

## 1. 폴더 구조

```
/deepet/jh/sensevoice/finetune/
├── jsonl/                               # 이미 서버에 존재
│   ├── train.jsonl  dev.jsonl
│   ├── train_kspon.jsonl  dev_kspon.jsonl
│   ├── train_zeroth.jsonl  dev_zeroth.jsonl
│   └── train_both.jsonl  dev_both.jsonl
└── pruned/
    ├── conf/sensevoice_pruned_tiny.yaml
    ├── scripts/
    │   ├── prune_sensevoice.py
    │   ├── export_pruned_onnx.py
    │   └── prepare_ksponspeech_jsonl.py
    ├── train_sensevoice_pruned.py
    ├── assets/
    │   ├── am.mvn
    │   └── chn_jpn_yue_eng_ko_spectok.bpe.model
    ├── workdir/sensevoice_pruned_tiny/
    │   └── model.pt.pruned              # 프루닝된 초기 가중치 (267MB)
    └── requirements.txt
```

## 2. 환경 활성화

```bash
cd /deepet/jh/sensevoice/
source .venv/bin/activate
```

## 3. 학습

```bash
# 기본: train.jsonl + dev.jsonl
python train_sensevoice_pruned.py --tag tiny

# 특정 jsonl 선택
python train_sensevoice_pruned.py --tag tiny --train train_kspon --dev dev_kspon
python train_sensevoice_pruned.py --tag tiny --train train_both  --dev dev_both

# Hydra override
python train_sensevoice_pruned.py --tag tiny ++train_conf.max_epoch=10

# 멀티 GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sensevoice_pruned.py --tag tiny
```

출력: `workdir/sensevoice_pruned_tiny/` (체크포인트, tensorboard 로그).

## 4. ONNX Export + 양자화

```bash
python scripts/export_pruned_onnx.py --tag tiny \
  --ckpt workdir/sensevoice_pruned_tiny/model.pt.avg5
```

결과물: `workdir/sensevoice_pruned_tiny/onnx/` 하위에 `model.onnx`, `model.int8.onnx`, `model.int4.onnx`.

## 5. 주의사항

- **text_language**: jsonl 항목에 `text_language` 필드가 없으면 데이터셋은 기본값 `<|zh|>`를 사용. 한국어 학습 시 `text_language: "<|ko|>"` 추가 확인 필요.
- **init_param**: `workdir/sensevoice_pruned_tiny/model.pt.pruned` 가 반드시 존재해야 함.
- **asset 경로**: [conf/sensevoice_pruned_tiny.yaml](conf/sensevoice_pruned_tiny.yaml) 의 `bpemodel` / `cmvn_file` 이 `/deepet/jh/sensevoice/finetune/pruned/assets/...` 로 하드코딩됨. 설치 위치가 다르면 수정 필요.
