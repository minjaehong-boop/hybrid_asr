# KsponSpeech ASR 레시피 (+ Zeroth-Korean)

## 출처 및 변경 사항

이 레시피는 [icefall PR #1664](https://github.com/k2-fsa/icefall/pull/1664) (작성자: [@johnSHLee96](https://github.com/johnSHLee96))의 KsponSpeech 레시피를 기반으로 합니다.

원본 레시피에서 **Zeroth-Korean 학습 데이터를 추가**한 버전입니다. 그 외 구조와 모델은 원본과 동일합니다.

| | 원본 (PR #1664) | 이 레시피 |
|---|---|---|
| KsponSpeech | ✅ | ✅ |
| Zeroth-Korean (train) | ❌ | ✅ |
| MUSAN (노이즈 증강) | ✅ | ✅ |

---

## 모델 구조

| 디렉터리 | 인코더 | 디코더 | 비고 |
|---|---|---|---|
| `pruned_transducer_stateless7_streaming` | Streaming Zipformer | Embedding + Conv1d | 실시간 스트리밍 추론 지원 |
| `zipformer` | Upgraded Zipformer | Embedding + Conv1d | 비스트리밍, 최신 레시피 |

---

## 1. 클론

```bash
git clone -b streaming_model https://github.com/minjaehong-boop/hybrid_asr.git
cd hybrid_asr
```
download폴더를 만들어, zeroth_korean, ksponspeech, musan(필요 시) 데이터셋을 위치시킵니다.

## 2. 가상환경 및 패키지 설치

```bash
python -m venv .venv
source .venv/bin/activate

pip install "numpy<2"
pip install -r requirements.txt
pip install -e .
```

## 3. k2 설치

k2는 PyPI에 없으며 PyTorch + CUDA 버전에 맞는 wheel을 직접 설치해야 합니다.

```bash
# 현재 버전 확인
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

확인한 버전에 맞게 설치합니다. 예시 (torch 2.4.0 + CUDA 12.1):

```bash
pip install k2==1.24.4.dev20241030+cuda12.1.torch2.4.0 \
  -f https://k2-fsa.github.io/k2/cuda.html
```

전체 버전 목록: https://k2-fsa.github.io/k2/cuda.html

설치 확인:
```bash
python -c "import k2; print(k2.__version__)"
```

## 4. 데이터 준비

`prepare.sh` 실행 전에 아래 데이터를 준비합니다:

| 데이터셋 | 위치 | 비고 |
|---|---|---|
| KsponSpeech | `download/KsponSpeech/` | [AIHub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)에서 수동 다운로드 (회원가입 필요) |
| Zeroth-Korean | `download/zeroth_korean/` | 선택 사항 — 없으면 KsponSpeech만 사용 |
| MUSAN | `download/musan/` | `prepare.sh`에서 자동 다운로드 |

```bash
./prepare.sh --stage 0 --stop-stage 5
```

각 스테이지 내용:

| Stage | 내용 |
|---|---|
| 0 | MUSAN 다운로드 |
| 1 | KsponSpeech + Zeroth-Korean lhotse 매니페스트 생성 |
| 2 | MUSAN 매니페스트 생성 |
| 3 | 80-dim fbank 특징 추출 → `data/fbank/` |
| 4 | MUSAN fbank 추출 |
| 5 | BPE 토크나이저 학습 (vocab=5000) → `data/lang_bpe_5000/` |

특정 스테이지만 재실행: `./prepare.sh --stage 3 --stop-stage 3`

## 5. 학습

```bash
export CUDA_VISIBLE_DEVICES="0"

./pruned_transducer_stateless7_streaming/train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --max-duration 750 \
  --enable-musan True
```

- CUDA OOM 발생 시 `--max-duration`을 낮춥니다 (500 → 300)
- 중간 재개: `--start-epoch N`
- 체크포인트 저장 위치: `pruned_transducer_stateless7_streaming/exp/epoch-N.pt`

## 6. 디코딩 평가

```bash
./pruned_transducer_stateless7_streaming/decode.py \
  --epoch 30 \
  --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --max-duration 600 \
  --decode-chunk-len 32 \
  --decoding-method greedy_search
```

- `--avg 9`: 해당 epoch 이전 9개 체크포인트를 평균하여 성능 향상
- 디코딩 방식: `greedy_search`, `modified_beam_search`, `fast_beam_search`

## 7. ONNX 변환 (배포용)

```bash
./pruned_transducer_stateless7_streaming/export-onnx.py \
  --tokens data/lang_bpe_5000/tokens.txt \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --epoch 30 \
  --avg 9 \
  --use-averaged-model true \
  --decode-chunk-len 32
```

`exp/` 아래에 `encoder/decoder/joiner-epoch-30-avg-9.onnx` 3개 파일이 생성됩니다.
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)로 바로 배포 가능합니다.
