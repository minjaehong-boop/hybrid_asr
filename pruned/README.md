# pruned — SenseVoice-Small 구조적 프루닝 + 파인튜닝

## 개요

SenseVoice-Small 원본(234M params, 70 blocks)에서 레이어를 균등 간격으로 선택해 블록 수를 줄인 뒤,프루닝된 가중치를 초기값으로 삼아 KsponSpeech(+Zeroth)로 파인튜닝합니다.



원본 vocab(SenseVoice spectok BPE, 25055개)과 task embedding 구조를 그대로 유지합니다.

```
원본: encoders0(1) + encoders(49) + tp_encoders(20) = 70 blocks, 234M
              ↓ prune_sensevoice.py
프루닝됨: encoders0(1) + encoders(N-1) + tp_encoders(M) = (N+M) blocks
              ↓ train_sensevoice_pruned.py
파인튜닝: workdir/sensevoice_pruned_<tag>/
              ↓ avg_ckpt.py + export_pruned_onnx.py
배포: workdir/sensevoice_pruned_<tag>/onnx/{model.onnx, model.int8.onnx, model.int4.onnx}
```

---

## 사용 순서

### 0. 의존성

```bash
# 프로젝트 루트의 .venv 사용
source ../.venv/bin/activate   # 또는 서버의 venv 경로
```

### 1. jsonl 준비

학습 데이터(jsonl)는 pruned/ 폴더 내부가 아닌 동등한 위치에 두어야 합니다.

```bash
# pruned/../jsonl/ 경로에 생성
python scripts/prepare_ksponspeech_jsonl.py \
    --root /deepet/jh/sensevoice/dataset/ksponspeech \
    --out-dir ../jsonl
# → train_kspon.jsonl, dev_kspon.jsonl 생성
# Zeroth도 사용할 경우 별도 스크립트로 train_both.jsonl / dev_both.jsonl 만들어야 함
```

> `train_sensevoice_scratch.py`의 기본 jsonl 이름은 `train_both` / `dev_both`입니다.
(여기서 both는 ksponspeech + zeroth)

### 2. 구조적 프루닝 (초기 가중치 생성)

원본 체크포인트에서 keep_main·keep_tp 블록을 균등 선택해 `.pruned` 파일을 생성합니다.

```bash
# tiny (18 blocks)
python scripts/prune_sensevoice.py --keep-main 12 --keep-tp 6 --tag tiny

# 다른 크기 예시
python scripts/prune_sensevoice.py --keep-main 14 --keep-tp 7 --tag small
python scripts/prune_sensevoice.py --keep-main 25 --keep-tp 10 --tag half
```
main_encoder와 tp_encoder는 이름만 다를 뿐 완전히 동일합니다(main인코더 뒤에 tp인코더가 붙습니다.)
sensevoice 모델이 어떤 모델로부터 파생되어 나온 영향이라고 알고 있습니다.



생성 위치: `workdir/sensevoice_pruned_<tag>/model.pt.pruned`

### 3. 파인튜닝

```bash
# 기본 (tag=tiny, jsonl=train_both/dev_both)
python train_sensevoice_pruned.py

# KsponSpeech만 사용
python train_sensevoice_pruned.py --tag tiny --train train_kspon --dev dev_kspon

# Hydra override (에폭, LR 등)
python train_sensevoice_pruned.py ++train_conf.max_epoch=50 ++optim_conf.lr=0.00005

# 특정 tag
python train_sensevoice_pruned.py --tag small
```

체크포인트: `workdir/sensevoice_pruned_<tag>/model.pt.ep<N>[.<step>]`

### 4. 체크포인트 평균 (선택)

```bash
# 최신 5개 자동 선택
python scripts/avg_ckpt.py \
    --model-dir workdir/sensevoice_pruned_tiny \
    --last 5

# 직접 지정
python scripts/avg_ckpt.py \
    --model-dir workdir/sensevoice_pruned_tiny \
    --ckpts model.pt.ep28.10000 model.pt.ep29 model.pt.ep30
```

출력: `workdir/sensevoice_pruned_<tag>/model.pt.avg5`

### 5. ONNX export + 양자화

```bash
# model.pt.pruned (초기) 또는 학습 후 체크포인트 사용
python scripts/export_pruned_onnx.py --tag tiny

# 학습 후 체크포인트 지정
python scripts/export_pruned_onnx.py --tag tiny \
    --ckpt workdir/sensevoice_pruned_tiny/model.pt.avg5
```

출력: `workdir/sensevoice_pruned_<tag>/onnx/`
- `model.onnx` — FP32
- `model.int8.onnx` — MatMul/Gemm INT8
- `model.int4.onnx` — MatMul NF4 (block_size=32)

---

## conf 비교

| 파일 | num_blocks (yaml) | tp_blocks | linear_units | 총 blocks | 비고 |
|------|:-----------------:|:---------:|:------------:|:---------:|------|
| `sensevoice_pruned_tiny.yaml` | 12 | 6 | 2048 | 18 | 기본 실험 대상. lr=5e-4 |
| `sensevoice_pruned_small.yaml` | 14 | 7 | 2048 | 21 | tiny보다 약간 크다 |
| `sensevoice_pruned_medium.yaml` | 15 | 8 | 2048 | 23 | — |
| `sensevoice_pruned_half.yaml` | 25 | 10 | 2048 | 35 | 원본 50% 수준. keep_nbest_models=5 |
| `sensevoice_pruned_tiny_more.yaml` | 6 | 0 | **512** | 6 | width도 축소. lr=5e-4|
| `sensevoice_pruned_tiny_much.yaml` | 3 | 0 | **512** | 3 | 최소 구성. lr=5e-4 |

> `tiny_more` / `tiny_much`는 `tp_blocks=0`이라 tp_encoders가 없습니다.(결과에 미치는 영향은 없습니다.)
> `tiny` ~ `half`는 원본 SenseVoiceSmall 구조 그대로(task embedding 포함)이므로`SenseVoiceSmall` 모델 클래스를 사용합니다.
> `tiny_more` / `tiny_much`도 동일 모델 클래스를 쓰지만 depth가 매우 작아 성능이 낮습니다.

공통 설정 (전체 conf 동일):
- Tokenizer: SenseVoice spectok BPE (`assets/chn_jpn_yue_eng_ko_spectok.bpe.model`)
- CMVN: `assets/am.mvn` (SenseVoice 공식 mvn)
- Frontend: WavFrontend, 80-mel, LFR 7/6
- Optimizer: AdamW, warmuplr

---

## 실험 결과 요약

평가 명령: `python eval_asr.py --models pruned_int4 pruned_int8 pruned_fp32 --pruned-dir workdir/sensevoice_pruned_<tag>/onnx/ --datasets custom zeroth`

| 모델 | custom CER | zeroth CER | RTF(int8) | 학습 데이터 |
|------|:----------:|:----------:|:---------:|------------|
| pruned_tiny (fp32) | **0.124** | — | 0.0191 | KsponSpeech, ep56~60 avg |
| pruned_tiny_more (fp32) | 0.246 | 0.156 | 0.0088 | KsponSpeech+Zeroth, ep39~40 avg |
| pruned_tiny_much (fp32) | 0.407 | 0.321 | 0.0085 | KsponSpeech+Zeroth, ep39~40 avg |

tiny가 성능 대비 크기가 가장 좋습니다. tiny_more/tiny_much는 RTF가 더 빠르지만 CER 열화가 큽니다(당연히 depth가 더 깊은 구조가 더 성능이 좋습니다.)

---

## 파일 구조

```
pruned/
├── assets/
│   ├── am.mvn                    # SenseVoice 공식 CMVN
│   └── chn_jpn_yue_eng_ko_spectok.bpe.model
├── conf/
│   ├── sensevoice_pruned_tiny.yaml
│   ├── sensevoice_pruned_small.yaml
│   ├── sensevoice_pruned_medium.yaml
│   ├── sensevoice_pruned_half.yaml
│   ├── sensevoice_pruned_tiny_more.yaml
│   └── sensevoice_pruned_tiny_much.yaml
├── scripts/
│   ├── prune_sensevoice.py       # 구조적 프루닝 (model.pt → model.pt.pruned)
│   ├── avg_ckpt.py               # 체크포인트 평균
│   ├── export_pruned_onnx.py     # ONNX + INT8/INT4 양자화
│   ├── prepare_ksponspeech_jsonl.py
│   └── add_text_language.py
├── workdir/
│   └── sensevoice_pruned_<tag>/
│       ├── model.pt.pruned       # 프루닝 초기 가중치
│       ├── model.pt.ep<N>…       # 학습 체크포인트
│       └── onnx/
│           ├── model.onnx
│           ├── model.int8.onnx
│           └── model.int4.onnx
├── train_sensevoice_pruned.py
└── requirements.txt
```

---
