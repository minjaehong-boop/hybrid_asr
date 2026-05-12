# hybrid1

## 프로젝트 목적
실시간 1차 스트리밍 ASR 자막에 덮어씌울 **2차 정제(offline re-decode) 엔진** 개발.
SenseVoice (sherpa-onnx) 모델을 청크 단위로 offline 디코딩하여 1차 자막을 교정한다.
quickspacer로 최종 표시 텍스트의 한국어 띄어쓰기를 보정한다.

## 핵심 모델
- **Transducer** (sherpa-onnx, 스트리밍): 1차 실시간 자막
- **SenseVoice** (sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8): 2차 정제
- **Silero VAD**: 마이크 입력 시 음성 구간 감지
- 타겟: 한국어(ko), CPU only (Raspberry Pi 5)

## 프로젝트 구조

```
main.py              CLI 진입점
server.py            웹 UI 서버 (FastAPI + WebSocket, 모델 사전로딩)
config/
  default.yaml       파이프라인 설정 (YAML)
  settings.py        설정 로더 + CLI 파싱
asr/                 핵심 ASR 파이프라인
  pipeline.py        오케스트레이터 (오디오 스트리밍 + 자막 병합)
  transducer_stage.py  Stage 1: 실시간 스트리밍 ASR
  refine_stage.py      Stage 2: SenseVoice 오프라인 정제
  protocol.py        워커 간 메시지 타입
  workers.py         워커 프로세스 진입점 (CLI용, finalize 후 종료)
utils/               공유 유틸리티
  audio_source.py    오디오 프레임 소스 (WAV 파일 + 마이크/VAD)
  subtitle_store.py  자막 상태 관리 + 오버레이 처리
  time_utils.py      타임스탬프 포맷
data/                테스트 오디오 + 라벨
models/              ONNX 모델 파일
  streaming_model/   Transducer 모델 (encoder/decoder/joiner)
  refine_model/      SenseVoice 모델
  vad_model/         Silero VAD 모델
docs/                프로젝트 문서
eval/                평가 스크립트
```

## 설정

모든 모델/파이프라인 파라미터는 `config/default.yaml`에서 관리.

## 실행

### CLI
```bash
python main.py --audio data/utterances/utt_0001.wav
python main.py --audio data/utterances/utt_0001.wav --no-realtime --max-seconds 10
python main.py --mic
python main.py --config my_config.yaml --audio test.wav
```

CLI 옵션: `--audio`, `--mic`, `--config`, `--no-realtime`, `--max-seconds`

### 웹 UI
```bash
python server.py
# http://<host>:8080
```

샘플 파일 선택, 파일 업로드, 마이크 입력 지원.
모델은 서버 시작 시 1회 로드, persistent worker로 유지.

## 2차 정제 전략: Center-Commit (CC)

현재 적용된 설정: **4.0s chunk / 0.5s trim**

### 청킹 전략 벤치마크 결과 (568초 테스트셋)

#### 기준선 (전체 인식)
- CER(raw): 0.0843 / CER(clean): 0.0454 / RTF: ~0.074

#### CER(clean, 부호제거) 비교

| Chunk | 단순청킹 | CC 0.5s | CC 1.0s | CC 1.5s |
|-------|---------|---------|---------|---------|
| 4.0s  | 0.0877  | 0.0548  | 0.0591  | 0.0631  |
| 4.5s  | 0.0735  | 0.0558  | 0.0579  | 0.0635  |
| 5.0s  | 0.0649  | 0.0603  | 0.0591  | 0.0620  |
| 5.5s  | 0.0593  | 0.0544  | 0.0549  | 0.0580  |
| 6.0s  | 0.0539  | 0.0495  | 0.0565  | 0.0555  |

#### 핵심 발견

1. **Center-commit >> 단순 청킹**: CC는 전체 인식 대비 CER +0.003~0.011
2. **0.5s overlap이 최선**: 경계 제거에 0.5s면 충분
3. **최종결정: 4.0s/0.5s** — stride 3.0s, CER 0.0548, RTF 0.099
