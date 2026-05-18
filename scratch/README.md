# scratch — SenseVoice CTC 한국어 처음부터 학습

## 개요

pruned버전처럼 pretrained weight를 사용하지 않고, KsponSpeech + Zeroth 한국어 데이터만으로 SenseVoice 아키텍처 기반 CTC 모델을 처음부터 학습합니다.
한국어 5000 BPE vocab을 별도로 학습해 사용하며, task embedding(언어인식, 감정인식, 오디오 이벤트 인식) / rich cross-entropy 등
SenseVoice 고유의 멀티태스크 헤드를 제거한 순수 CTC 구조(`SenseVoiceSmallCTCScratch`)를 씁니다.

```
KsponSpeech + Zeroth jsonl
        ↓ (train_sensevoice_scratch.py — pretrained weight 없음)
workdir/sensevoice_scratch_<tag>/
        ↓ avg_ckpt.py
model.pt.avgN
        ↓ export_pruned_onnx.py
workdir/sensevoice_scratch_<tag>/{model.onnx, model.int8.onnx, model.int4.onnx}
```

---

## 사용 순서

### 0. 의존성

```bash
source ../.venv/bin/activate
```

### 1. jsonl 준비

학습 데이터(jsonl)는 scratch, pruned폴더와 같은 위치에 둬야 합니다. 
이름: jsonl/

```bash
# KsponSpeech
python ../pruned/scripts/prepare_ksponspeech_jsonl.py \
    --root /deepet/jh/sensevoice/dataset/ksponspeech \
    --out-dir ../jsonl
# Zeroth 등 추가 데이터는 별도 스크립트로 merge → train.jsonl / dev.jsonl
```

> `train_sensevoice_scratch.py`의 기본 jsonl 이름은 `train_both` / `dev_both`입니다.
(여기서 both는 ksponspeech + zeroth)


### 2. CMVN 계산 (최초 1회)

scratch는 SenseVoice 공식 CMVN(5개언어)이 아닌 KsponSpeech 기반 CMVN(한국어만 존재)을 사용합니다.
`assets/kspon.mvn`이 이미 있으면 이 단계를 건너뛰어도 됩니다.


```bash
python scripts/compute_kspon_cmvn.py \
    --jsonl ../jsonl/train_both.jsonl \
    --out assets/kspon.mvn \
    --n 20000
```

### 3. 학습

```bash
# 기본 (tag=ko5k, conf=sensevoice_scratch_ko5k.yaml)
python train_sensevoice_scratch.py

# mcu 변형 지정
python train_sensevoice_scratch.py --tag ko5k_mcu

# Hydra override
python train_sensevoice_scratch.py ++train_conf.max_epoch=60

# 멀티 GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_sensevoice_scratch.py --tag ko5k_mcu
```

체크포인트: `finetune/sensevoice_scratch_<tag>/model.pt.ep<N>[.<step>]`

### 4. 체크포인트 평균

```bash
python scripts/avg_ckpt.py \
    --model-dir workdir/sensevoice_scratch_ko5k_mcu \
    --last 5
# 또는 직접 지정
python scripts/avg_ckpt.py \
    --model-dir workdir/sensevoice_scratch_ko5k_mcu \
    --ckpts model.pt.ep40 model.pt.ep39.15000 model.pt.ep39.14000
```
pruned와 마찬가지로 가중치하나만으로는 정상적인 성능을 발휘하지 못합니다.
최소 2개 이상(저는 보통 3~5개로 평균)의 학습된 가중치를 평균내어 사용했습니다.


### 5. ONNX export + 양자화

scratch용 config를 `--conf`로 넘겨야 합니다. (pruned와 동일 스크립트 사용).

```bash
python scripts/export_pruned_onnx.py \
    --tag ko5k_mcu \
    --ckpt workdir/sensevoice_scratch_ko5k_mcu/model.pt.avg5
```

> 내부적으로 `conf/sensevoice_scratch_<tag>.yaml`을 읽어 모델을 빌드합니다.
(vocab_size=5002(BPE 5000 + blank/unk)가 맞는지 확인할 것)

출력: `workdir/sensevoice_scratch_<tag>/`
- `model.onnx`
- `model.int8.onnx`
- `model.int4.onnx`

---

## conf 비교

| 파일 | num_blocks | tp_blocks | linear_units | batch_size | 비고 |
|------|:----------:|:---------:|:------------:|:----------:|------|
| `sensevoice_scratch_ko5k_mcu.yaml` | 6 | 0 | **1024** | 80000 | 기본. accum_grad=2 |
| `sensevoice_scratch_ko5k_mcu_more.yaml` | 6 | 0 | **512** | 80000 | width 절반 축소 |
| `sensevoice_scratch_ko5k_mcu_much.yaml` | 3 | 0 | **512** | 80000 | depth도 절반 |

공통 설정:
- Model: `SenseVoiceSmallCTCScratch` (task embedding / rich CE 제거, 순수 CTC)
- Dataset: `SenseVoiceCTCDatasetNoTask` (task prefix 없음)
- Tokenizer: `assets/lang_bpe_5000/bpe.model` (한국어 전용 5000 BPE)
- CMVN: `assets/kspon.mvn` (KsponSpeech에서 재계산)
- Frontend: WavFrontend, 80-mel, LFR 7/6
- Optimizer: AdamW (lr=5e-4, betas=[0.9, 0.98], weight_decay=5e-4), warmuplr 25000 steps
- max_epoch=40, validate/save 주기: 1000 step

> pruned와의 차이: scratch는 `init_param` 없이 시작합니다.

---

## 실험 결과 요약

평가 명령: `python eval_asr.py --models scratch_int4 scratch_int8 scratch_fp32 --datasets custom zeroth`

| 모델 | custom CER | zeroth CER | RTF(int8) | 학습 데이터 |
|------|:----------:|:----------:|:---------:|------------|
| ko5k_mcu (fp32) | **0.247** | **0.101** | 0.0086 | KsponSpeech+Zeroth, ep39~40 avg |
| ko5k_mcu_more (fp32) | 0.298 | 0.131 | 0.0082 | KsponSpeech+Zeroth, ep39~40 avg |
| ko5k_mcu_much (fp32) | 0.476 | 0.295 | 0.0082 | KsponSpeech+Zeroth, ep39~40 avg |

당연히 ko5k_mcu가 가장 좋습니다. 
custom 데이터셋 기준으로는 pruned_tiny(CER 0.12)가 scratch_mcu(0.25)보다 상당히 우수합니다.
결국 pretrained weight의 효과가 크다는 것을 알 수 있습니다.
(이것 때문에 LG공모에 pruned 버전을 기재했습니다.)
---

## 파일 구조

```
scratch/
├── assets/
│   ├── kspon.mvn                  # KsponSpeech CMVN (compute_kspon_cmvn.py로 생성)
│   └── lang_bpe_5000/
│       ├── bpe.model              # 한국어 5000 BPE
│       └── tokens.txt
├── conf/
│   ├── sensevoice_scratch_ko5k_mcu.yaml
│   ├── sensevoice_scratch_ko5k_mcu_more.yaml
│   └── sensevoice_scratch_ko5k_mcu_much.yaml
├── scripts/
│   ├── compute_kspon_cmvn.py      # KsponSpeech CMVN 재계산
│   ├── avg_ckpt.py                # 체크포인트 평균
│   └── export_pruned_onnx.py      # ONNX + 양자화
├── sensevoice_scratch/
│   ├── __init__.py                # tables.register 실행 (Hydra 호출 전 import 필수)
│   ├── model.py                   # SenseVoiceSmallCTCScratch 정의
│   └── dataset.py                 # SenseVoiceCTCDatasetNoTask 정의
├── workdir/
│   └── sensevoice_scratch_<tag>/
│       ├── model.pt.ep<N>…        # 학습 체크포인트
│       ├── bpe.model              # export 시 복사된 vocab
│       ├── tokens.txt
│       ├── kspon.mvn
│       ├── config.yaml
│       ├── model.onnx
│       ├── model.int8.onnx
│       └── model.int4.onnx
├── train_sensevoice_scratch.py
└── requirements.txt
```

---

## 결론

- 더 많은 한국어 데이터: 현재 KsponSpeech(약 969h) + Zeroth(약 51h)를 사용했습니다. 하지만 아시다시피 ksponspeech 데이터셋은 완전히 믿고 사용하기엔 아쉬운 부분이 있습니다.
- 가전 명령어 특화 BPE: 현재 BPE는 KsponSpeech 기반. "타이머", "예약", "전원" 등 가전 도메인 어휘를 forced vocab에 추가한 BPE를 학습하면 단편 명령어 인식률 향상 기대.
- mcu depth에 따른 trade off: much에 경우 depth가 3이기 때문에 성능이 너무 낮고 depth 6이(more) 실용 최소치로 보입니다.
