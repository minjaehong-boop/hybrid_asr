# SenseVoiceSmall CTC 순수 재학습 (서버)

원본 SenseVoiceSmall 공식 아키텍처에서 task embedding / rich CE / task query concat을 제거하고, 한국어 전용 5000 BPE vocab으로 **scratch 재학습** (pretrained weight 로드 없음).

서버 경로:
- 프로젝트: `/deepet/jh/sensevoice/finetune/scratch/`
- jsonl:    `/deepet/jh/sensevoice/finetune/jsonl/` (sibling)

## 1. 폴더 구조

```
/deepet/jh/sensevoice/finetune/
├── jsonl/                               # 이미 서버에 존재
│   ├── train.jsonl  dev.jsonl
│   ├── train_kspon.jsonl  dev_kspon.jsonl
│   └── ...
└── scratch/
    ├── conf/sensevoice_scratch_ko5k.yaml
    ├── sensevoice_scratch/              # FunASR 등록 패키지
    │   ├── __init__.py
    │   ├── model.py                     # SenseVoiceSmallCTCScratch
    │   └── dataset.py                   # SenseVoiceCTCDatasetNoTask
    ├── scripts/
    │   └── compute_kspon_cmvn.py        # KsponSpeech 기반 CMVN 재계산
    ├── assets/
    │   ├── lang_bpe_5000/               # icefall 호환 5000 BPE
    │   │   ├── bpe.model
    │   │   └── tokens.txt
    │   └── kspon.mvn                    # (2번 단계에서 생성)
    ├── workdir/sensevoice_scratch_ko5k/ # 출력 (체크포인트/로그)
    ├── train_sensevoice_scratch.py
    ├── requirements.txt
    └── README_SERVER.md
```

## 2. 환경 설치

```bash
cd /deepet/jh/sensevoice/finetune/scratch
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 3. CMVN 재계산 (최초 1회)

SenseVoice 기본 `am.mvn`은 중국어/일본어/영어 섞인 통계. 한국어 scratch에선 KsponSpeech 통계로 교체 필요:

```bash
python scripts/compute_kspon_cmvn.py \
    --jsonl /deepet/jh/sensevoice/finetune/jsonl/train.jsonl \
    --out   assets/kspon.mvn \
    --n 20000
```

출력: `assets/kspon.mvn` (~5분). 5000 uttr이면 통계 충분 — `--n`을 더 키워봤자 값 거의 안 변함.

## 4. 학습

```bash
# 기본: train.jsonl + dev.jsonl
python train_sensevoice_scratch.py

# 특정 jsonl 선택
python train_sensevoice_scratch.py --train train_kspon --dev dev_kspon

# Hydra override
python train_sensevoice_scratch.py ++train_conf.max_epoch=30
python train_sensevoice_scratch.py ++dataset_conf.batch_size=8000  # OOM 시

# 멀티 GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sensevoice_scratch.py
```

출력: `workdir/sensevoice_scratch_ko5k/` (체크포인트, tensorboard 로그).

## 5. 주의사항

- **text_language 불필요**: `SenseVoiceCTCDatasetNoTask`는 task prefix를 넣지 않으므로 jsonl의 `text_language` 필드는 무시됨. 기존 `add_text_language.py`를 돌릴 필요 없음.
- **init_param 없음**: scratch이므로 pretrained 로드 없음. random init부터 학습.
- **vocab=5000**: icefall BPE. `<blk>=0, <sos/eos>=1, <unk>=2`. SenseVoice 기본 `blank_id=0`과 일치.
- **CMVN 경로**: [conf/sensevoice_scratch_ko5k.yaml](conf/sensevoice_scratch_ko5k.yaml)의 `cmvn_file`과 `bpemodel`이 `/deepet/jh/sensevoice/finetune/scratch/assets/...`로 하드코딩. 설치 위치가 다르면 수정.
- **아키텍처**: `num_blocks=24, tp_blocks=0` (원본 70 → 24로 축소). 조정하려면 YAML 편집 후 재시작.

## 6. 기존 pruned 방식과의 차이

| | pruned_tiny | scratch (여기) |
|---|---|---|
| 초기화 | 원본 ckpt에서 레이어 샘플링 | random init |
| vocab | 25055 (다국어) | 5000 (한국어) |
| task embed | 유지 (LID/emo/event/itn) | 제거 |
| rich CE loss | 있음 | 없음 (CTC only) |
| num_blocks | 12 + tp 6 | 24 + tp 0 |
| 학습 비용 | 낮음 (fine-tune) | 높음 (scratch) |
