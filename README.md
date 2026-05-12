# HYBRID — 2단계 한국어 실시간 자막 파이프라인

Raspberry Pi 5를 타겟으로 한 실시간 한국어 자막 파이프라인.
Transducer(sherpa-onnx)가 저지연 실시간 자막을 제공하고, SenseVoice가 청크 단위로 오프라인 정제를 수행한다.
quickspacer가 최종 출력에 한국어 띄어쓰기를 교정한다.

---

## 동작 원리

```
오디오 입력
   │
   ├─▶ [Stage 1] Transducer (온라인, 저지연)
   │      sherpa-onnx 스트리밍 인식 → LIVE 자막 출력
   │
   └─▶ [Stage 2] SenseVoice Refine (오프라인, 청크 단위)
          chunk_sec 단위로 정확도 높은 텍스트 생성
          center-commit 전략으로 경계 오류 제거
          → LIVE 자막 위에 오버레이(overlay)
```

**오버레이 전략**: `SubtitleStore`가 confirmed 텍스트(REFINE 누적)와 현재 LIVE 텍스트를 compact SequenceMatcher로 비교해, LIVE의 REFINE 커버리지 끝 이후 suffix만 붙여 최종 표시 텍스트를 만든다.

---

## 환경 요구사항

- Python 3.10+
- Raspberry Pi 5 (aarch64) 또는 x86_64
- sherpa-onnx (SenseVoice + Transducer ONNX 모델)

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

모델 파일 배치:

```
models/
  streaming_model/          # Transducer 모델 (tokens.txt, encoder/decoder/joiner .onnx)
  refine_model/             # SenseVoice 모델 (model.int8.onnx, tokens.txt)
  vad_model/
    silero_vad.onnx         # Silero VAD (마이크 입력 시 사용)
```

---

## 설정

모든 모델·파이프라인 파라미터는 `config/default.yaml`에서 관리한다.

```yaml
transducer:
  model_dir: models/streaming_model
  num_threads: 2
  provider: cpu          # cpu | cuda
  use_int8: true

refine:
  model_dir: models/refine_model
  language: ko           # auto | zh | en | ja | ko | yue
  chunk_sec: 3           # 청크 길이 (초)
  trim_sec: 0.5          # center-commit 양쪽 트림 (0 = CC 비활성)
  context_mode: accumulate  # accumulate | overlap

pipeline:
  frame_sec: 0.01        # 실시간 스테이지 프레임 크기
  realtime: true         # 실시간 재생 시뮬레이션 여부
```

커스텀 설정:

```bash
cp config/default.yaml config/my_config.yaml
python main.py --config config/my_config.yaml --audio test.wav
```

---

## 사용법

### CLI

```bash
# 기본 실행 (실시간 재생 시뮬레이션)
python main.py --audio data/utterances/utt_0001.wav

# 빠른 처리 (실시간 딜레이 없음)
python main.py --audio data/utterances/utt_0001.wav --no-realtime

# 마이크 입력 (Silero VAD 자동 감지, 침묵 시 자동 종료)
python main.py --mic

# 일부만 처리 (디버깅)
python main.py --audio test.wav --no-realtime --max-seconds 10

# refine 모드 지정
python main.py --audio test.wav --refine-mode overlap
```

CLI 옵션 요약:

| 옵션 | 설명 |
|------|------|
| `--audio PATH` | 입력 WAV 파일 (16kHz mono 권장) |
| `--mic` | 마이크 입력 + Silero VAD |
| `--config PATH` | YAML 설정 파일 (기본: `config/default.yaml`) |
| `--no-realtime` | 가능한 빠르게 처리 |
| `--max-seconds N` | 오디오 길이 제한 (디버깅용) |
| `--refine-mode` | `accumulate` 또는 `overlap` (YAML 설정 오버라이드) |

### 디버그 모드

오버레이 연산의 단계별 출력을 확인한다:

```bash
python main_debug.py --audio data/utterances/utt_0001.wav --no-realtime
```

REFINE이 도착할 때마다 SequenceMatcher 분석, compact 변환, suffix 추출, join 결합의 각 단계를 컬러 출력으로 보여준다.

### Web UI

```bash
source .venv/bin/activate
python server.py
# 브라우저에서 http://<host>:8080 접속
```

기능:
- 샘플 WAV 파일 선택 또는 파일 업로드
- 마이크 실시간 입력 (VAD)
- LIVE + REFINE 오버레이 자막 표시
- 진행 바, 이벤트 로그
- 서버 시작 시 모델 사전 로드 (워커 프로세스 유지)

---

## 평가

```bash
python eval.py
```

---

## 프로젝트 구조

```
main.py                  CLI 진입점
main_debug.py            단계별 오버레이 디버그 실행기
server.py                Web UI 서버 (FastAPI + WebSocket)
eval.py                  평가 스크립트
config/
  default.yaml           파이프라인 설정 (YAML)
  settings.py            설정 로더 + CLI 파싱
asr/
  pipeline.py            파이프라인 오케스트레이터 (오디오 스트리밍 + 자막 병합)
  transducer_stage.py    Stage 1: 실시간 스트리밍 ASR (온라인 Transducer)
  refine_stage.py        Stage 2: SenseVoice 오프라인 정제 (center-commit)
  protocol.py            워커 간 메시지 타입 정의
  workers.py             워커 프로세스 진입점
utils/
  audio_source.py        오디오 소스 (WAV 파일 / 마이크 + VAD)
  subtitle_store.py      자막 상태 관리 + 오버레이 연산
  time_utils.py          타임스탬프 포맷 유틸
data/                    테스트 오디오 파일 + 레이블
models/                  ONNX 모델 파일 (git 미추적)
docs/                    프로젝트 문서
```

---

## center-commit 전략

`trim_sec > 0` 일 때 활성화. 청크 경계 근처의 오인식을 줄이기 위해 청크 양쪽 `trim_sec`초를 커밋에서 제외하고, 다음 청크에서 해당 구간을 context로 재사용한다.

| 모드 | 설명 |
|------|------|
| `accumulate` | 최대 4개 청크를 누적 context로 사용 |
| `overlap` | 고정 `2*trim_sec` 길이의 overlap buffer 사용 |

`trim_sec: 0` 으로 설정하면 center-commit 없이 각 청크를 독립적으로 디코딩한다.
