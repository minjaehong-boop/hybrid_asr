# 하이브리드 실시간 자막 시스템 원리

이 문서는 현재 프로젝트의 **1차 실시간 자막(LIVE)**, **2차 정제(REFINE)**, 그리고 **자막 병합(Overlay)** 전체를 코드 기준으로 자세히 설명한다.

---

## 1. 전체 구조 한눈에 보기

이 시스템은 같은 오디오를 **두 갈래**로 동시에 처리한다.

```text
오디오 입력 (WAV 파일 또는 마이크)
       |
       v
  WavChunkSource / MicChunkSource
  (0.1초 프레임으로 자름, 마이크는 Silero VAD 포함)
       |
  +---------+----------+
  |                    |
  v                    v
transducer-worker   refine buffer
(Sherpa-ONNX        (chunk_sec 누적 후
 온라인 트랜스듀서)   SenseVoice로 전송)
  |                    |
  v                    v
[LIVE] 이벤트       [REFINE] 이벤트
  |                    |
  +----------+---------+
             |
             v
        SubtitleStore
    (confirmed_text 축적 +
     SequenceMatcher 오버레이)
             |
             v
        quickspacer
    (최종 표시 텍스트 띄어쓰기 보정, level=3)
             |
             v
    터미널 출력 / 웹 UI (WebSocket)
```

- **1차 (LIVE)**: 빠르지만 거친 임시 자막. 0.1초마다 갱신.
- **2차 (REFINE)**: 느리지만 정확한 교정 자막. 기본값: 30초 단순 청킹 (`trim_sec=0`). Center-Commit 활성화 시 accumulate/overlap 두 모드 선택 가능.
- **SubtitleStore**: REFINE 결과를 `confirmed_text`로 축적하고, LIVE 텍스트 위에 오버레이.
- **quickspacer**: REFINE 텍스트는 SubtitleStore 저장 전에 보정, LIVE는 표시 직전 보정 (level=3).
- **MicChunkSource**: 마이크 입력 시 Silero VAD로 음성 구간 감지, 무음 자동 종료.

멀티프로세스 구조로, transducer-worker와 refine-worker가 **별도 프로세스**에서 돌아 서로 블로킹하지 않는다.
웹 UI(`server.py`)에서는 persistent worker로 모델을 1회 로드 후 유지한다.

---

## 2. 1차 실시간 자막 (LIVE) 원리

### 엔진

**Sherpa-ONNX Online Transducer** (스트리밍 ASR)

- `asr/transducer_stage.py` → `TransducerStage`
- 세 가지 ONNX 모델 파일로 구성: encoder, decoder, joiner

### 오디오 청크 방식

| 항목 | 값 |
|------|-----|
| 샘플레이트 | 16,000 Hz (16kHz 모노) |
| 프레임 크기 | `frame_sec=0.01`초 (기본값) |
| 프레임당 샘플 수 | 160개 |

`WavChunkSource`가 wav를 0.01초 프레임으로 잘라 하나씩 transducer-worker에 보낸다.

### 동작 원리: 스트리밍 누적 갱신

Transducer는 **스트리밍 누적 갱신** 방식이다. 이번 0.1초만 번역하는 게 아니라, 지금까지 들어온 오디오 전체 스트림을 반영해 **현재까지의 전체 문장**을 갱신한다.

```text
0.1초 도달 → "이천"
0.2초 도달 → "이천 십"
0.3초 도달 → "이천 십 오"
0.4초 도달 → "이천 십 오 년"
...
```

매번 앞부분은 유지한 채 뒤가 늘어나는 이유가 여기에 있다.

### 핵심 알고리즘: LCP (Longest Common Prefix)로 시작 시간 계산

`_build_update()` 메서드에서, 새 결과가 이전 결과와 달라진 지점을 **토큰 단위 LCP**로 찾는다.

```text
이전 토큰: [이천, 십, 오]
현재 토큰: [이천, 십, 오, 년, 현재]

LCP = 3 (공통 접두부: [이천, 십, 오])
새로 시작된 지점: tokens[3] = "년" → timestamps[3] = 0.42초 → start_sec
```

코드 흐름:

```python
# transducer_stage.py:_build_update()
tokens = list(self.recognizer.tokens(self.stream))
timestamps = list(self.recognizer.timestamps(self.stream))
lcp = _lcp_len(self.prev_tokens, tokens)  # 이전과 공통인 토큰 수

if lcp < len(timestamps):
    start_sec = float(timestamps[lcp])    # 새로 달라진 첫 토큰의 시간
elif timestamps:
    start_sec = float(timestamps[-1])     # 모든 토큰이 동일하면 마지막 시간
else:
    start_sec = max(0.0, end_sec - 0.2)   # 타임스탬프 없으면 fallback
```

### 핵심 변수

| 변수 | 위치 | 역할 |
|------|------|------|
| `self.prev_tokens` | TransducerStage | 이전 프레임의 토큰 배열. LCP 비교 기준 |
| `self.prev_text` | TransducerStage | 이전 프레임의 전체 텍스트. 변경 없으면 업데이트 skip |
| `self.stream` | TransducerStage | sherpa-onnx 온라인 스트림. 오디오가 계속 누적됨 |

### 출력 형태

```
[LIVE] 00:00:02.400 ~ 00:00:04.200 | 이천 십 오 년 현재 음성 인식
[LIVE-FINAL] 00:00:02.400 ~ 00:00:09.800 | 이천 십 오 년 현재 음성 인식 기술은
```

`is_final=True`는 `finalize()` 호출 시에만 발생하며, 전체 스트림이 끝났음을 의미.

---

## 3. 2차 정제 자막 (REFINE) 원리

### 엔진

**Sherpa-ONNX SenseVoiceSmall** (오프라인 배치 ASR)

- `asr/refine_stage.py` → `SenseVoiceRefineStage`
- 한국어, 영어, 중국어, 일본어, 광동어 지원
- `OfflineRecognizer.from_sense_voice()` 사용

### 청킹 모드 선택

`trim_sec` 값과 `context_mode`에 따라 세 가지 모드로 동작한다:

| 조건 | 모드 | 동작 |
|------|------|------|
| `trim_sec == 0` | **Simple** (기본값) | 각 청크를 독립적으로 디코딩 |
| `trim_sec > 0`, `context_mode="accumulate"` | **CC-Accumulate** | 이전 청크(최대 4개)를 컨텍스트로 붙여 디코딩, left_trim으로 기 커밋 구간 제외 |
| `trim_sec > 0`, `context_mode="overlap"` | **CC-Overlap** (구 방식) | 고정 2×trim_sec 버퍼만 유지하며 center-commit |

**현재 기본 설정** (`config/default.yaml`): `chunk_sec=3, trim_sec=0.5, context_mode=accumulate` → **CC-Accumulate 모드**

CLI로 모드 전환:
```bash
# overlap 모드
python main.py --audio test.wav --refine-mode overlap
# Simple 모드 (CC 비활성)는 YAML에서 trim_sec: 0 으로 설정
```

(trim_sec는 YAML에서 설정; `trim_sec: 0`이면 Simple 모드, `trim_sec > 0`이면 CC 활성)

---

### CC-Accumulate 모드 (`context_mode="accumulate"`)

이전 청크들을 컨텍스트로 누적하여 문맥 인식 품질을 높인다. 최대 4개 청크 보관.

#### 동작 흐름

```text
청크 0: [===3초===]
         left_trim=0, right_trim=0.5s
         full_chunk = 청크0
         context_buffer ← [청크0]

청크 1: [===3초===]
         full_chunk = [청크0 | 청크1] = 6초
         left_trim = context_dur - trim_sec = 3.0 - 0.5 = 2.5초
         right_trim = 0.5초
         커밋 영역 = [2.5초 ~ 5.5초] (청크1의 실제 내용)
         context_buffer ← [청크0 | 청크1]

청크 2: [===3초===]
         full_chunk = [청크0 | 청크1 | 청크2] = 9초
         left_trim = 6.0 - 0.5 = 5.5초
         right_trim = 0.5초
         커밋 영역 = [5.5초 ~ 8.5초]
         ...
```

stride는 항상 `new_samples` 길이(= `chunk_sec`)로 일정. offset = 0 → chunk_sec → 2×chunk_sec ...

#### 핵심 코드 흐름: `_feed_center_commit()`

```python
def _feed_center_commit(self, samples):
    # 1) context_buffer + 새 샘플 → full_chunk 조립
    if self._context_buffer is not None and self._chunk_index > 0:
        full_chunk = np.concatenate([self._context_buffer, new_samples])
    else:
        full_chunk = new_samples

    # 2) trim 경계 결정
    if self._chunk_index == 0:
        left_trim = 0.0
    else:
        context_dur = len(self._context_buffer) / self.sample_rate
        left_trim = context_dur - self.trim_sec   # 컨텍스트 구간 제외, trim_sec 여유만 남김
    right_trim = self.trim_sec

    # 3) stride = new_samples 길이 (chunk_sec 고정)
    stride_sec = len(new_samples) / self.sample_rate
    commit_start_sec = self.chunk_offset_sec
    commit_end_sec = commit_start_sec + stride_sec
    self.chunk_offset_sec = commit_end_sec

    # 4) 디코딩 + 중앙 영역 추출
    result = self._decode_chunk(full_chunk)
    text = _extract_center_tokens(tokens, timestamps, chunk_dur, left_trim, right_trim)

    # 5) 인접 청크 간 중복 제거
    text = _dedup_overlap(self._prev_committed_text, text)

    # 6) 청크 버퍼 갱신 (최대 4개)
    self._chunk_buffers.append(new_samples)
    if len(self._chunk_buffers) > self._max_context_chunks:
        self._chunk_buffers.pop(0)
    self._context_buffer = np.concatenate(self._chunk_buffers)
```

#### 핵심 변수 (Accumulate 모드)

| 변수 | 역할 |
|------|------|
| `self._chunk_buffers` | 개별 청크 samples 리스트 (최대 4개). context 구성용 |
| `self._context_buffer` | `_chunk_buffers`를 concat한 컨텍스트 버퍼 |
| `self._max_context_chunks` | 4 (고정). 오래된 청크 LRU 제거 |
| `self.chunk_offset_sec` | 지금까지 커밋된 절대 시간. 다음 청크 `commit_start_sec` |
| `self._chunk_index` | 청크 번호. 0이면 left_trim=0 |
| `self._prev_committed_text` | 이전 커밋 텍스트. 중복 제거용 |
| `self._last_result_tokens/timestamps` | 마지막 디코딩 결과. `_finalize_accumulate()`에서 tail 복구용 |
| `self._last_chunk_dur` / `self._last_right_trim` | finalize tail 복구 경계 계산용 |

---

### CC-Overlap 모드 (`context_mode="overlap"`)

기존(구) 방식. 고정 크기(2×trim_sec) 버퍼만 유지. stride = `chunk_dur - left_trim - right_trim`.

```text
청크 0: [===========4.0초===========]
         left_trim=0, right_trim=0.5s
         stride = new_dur - right_trim = 3.5초

청크 1:              [==overlap(1.0s)=|===4.0초===]
         full_chunk = 5.0초
         left_trim=0.5, right_trim=0.5s
         stride = new_dur = 4.0초

finalize: _last_right_trim > 0이면 저장된 결과에서 tail 추출 → 누락 구간 복원
```

핵심 변수:
| 변수 | 역할 |
|------|------|
| `self._overlap_buffer` | 이전 full_chunk 끝 2×trim_sec 샘플. 다음 청크 앞에 붙임 |
| `self._last_result_tokens/timestamps` | 마지막 디코딩 결과. finalize tail 복구용 |
| `self._last_result_text` | 마지막 디코딩 텍스트. timestamps 없을 때 ratio fallback용 |
| `self._last_chunk_dur` / `self._last_right_trim` | finalize tail 복구 경계 계산용 |

---

#### 중앙 영역 추출: `_extract_center_tokens()`

per-token timestamp를 사용해 `left_trim ≤ t ≤ chunk_dur - right_trim` 범위의 토큰만 선택:

```text
4.0초 청크, left_trim=0.5, right_trim=0.5인 경우:

토큰:      [이천, 이, 십, 오, 년, 현재, 음성, 인식, 기술은]
timestamp: [0.1,  0.3, 0.4, 0.5, 0.8, 1.2,  2.0,  2.8,  3.6]

left_trim=0.5 이상 & chunk_dur-right_trim=3.5 이하인 토큰:
→ [오, 년, 현재, 음성, 인식] (0.5 ≤ t ≤ 3.5)

timestamp 없으면 문자 비율로 fallback (_extract_center_by_ratio)
```

#### 인접 청크 간 중복 제거: `_dedup_overlap()`

CC의 overlap 때문에 인접 청크가 같은 내용을 반복할 수 있다. 공백 제거 후 suffix-prefix 최장 일치로 제거:

```text
prev_committed: "음성 인식 기술은"
new_text:       "기술은 단순히 서류를"

prev_stripped: "음성인식기술은"
new_stripped:  "기술은단순히서류를"

suffix-prefix 매칭: "기술은" (3글자 일치)
→ "기술은" 제거 → "단순히 서류를" 만 커밋
```

#### `finalize()`: 마지막 청크의 오른쪽 가장자리 복원

정상 처리 중에는 right_trim만큼 잘리므로, 마지막 청크의 오른쪽 끝이 누락된다.
`finalize()`는 두 CC 모드 공통으로 하나의 메서드에서 처리한다:

```python
def finalize(self) -> RefineUpdate:
    tail_text = ""
    if self._last_right_trim > 0:
        tail_left = self._last_chunk_dur - self._last_right_trim
        if self._last_result_tokens and self._last_result_timestamps:
            tail_text = _extract_center_tokens(..., tail_left, 0.0)
        elif self._last_result_text:
            tail_text = _extract_center_by_ratio(..., tail_left, 0.0)
        if tail_text:
            tail_text = _dedup_overlap(self._prev_committed_text, tail_text)
    return RefineUpdate(..., is_final=True)
```

`is_final=True`는 **텍스트가 비어있어도** 반드시 반환된다. 파이프라인이 REFINE 완료를 인식하기 위함.

### 1차와 결정적인 차이

| | 1차 (LIVE) | 2차 (REFINE, Simple) | 2차 (REFINE, CC) |
|--|--|--|--|
| 처리 단위 | 0.1초 프레임 | 30초 청크 | 설정된 chunk_sec |
| 누적 방식 | 전체 스트림 누적 갱신 | 청크 독립 디코딩 | accumulate/overlap |
| 모델 | Online Transducer | Offline SenseVoice | Offline SenseVoice |
| 지연 | 즉각 (<100ms) | ~30초 | ~chunk_sec |
| 정확도 | 중간 | 높음 | 높음 (CER 0.0548 @ 4s CC) |
| 예시 | "이천 십 오 년 현재" | "2026년 현재" | "2026년 현재" |

---

## 4. 파이프라인 버퍼링과 REFINE 플러시 타이밍

`pipeline.py`의 `StreamingSubtitlePipeline`이 오디오 프레임을 양쪽 워커에 분배한다.

### REFINE 버퍼링 핵심 변수

| 변수 | 역할 |
|------|------|
| `self.refine_buffer` | 아직 REFINE에 전송되지 않은 오디오 프레임 누적 리스트 |
| `self.refine_buffer_samples` | 누적된 총 샘플 수 |
| `self.refine_chunk_samples` | `int(chunk_sec × sample_rate)`. 이 값에 도달하면 flush (단일 threshold) |

flush threshold는 항상 동일하다 (두 단계 전환 없음). `chunk_sec=3`이면 항상 48,000 샘플(3초)마다 flush.

### 플러시 타이밍 예시

```text
frame_sec = 0.01초, sample_rate = 16000, chunk_sec = 3 (기본값)

flush threshold: 3초 × 16000 = 48,000 샘플 (항상 고정)

프레임 반복:
  frame 1~299 (0~2.99초): 각 160샘플 누적
  frame 300 (2.99~3.0초): total 48,000 → FLUSH! (첫 번째)
  frame 301~600: 다시 48,000 누적 → FLUSH! (두 번째)
  ...
```

CC 모드 예시 (`chunk_sec=4.0`): threshold=64,000 (4.0초×16000), 매 4초마다 flush.

### quickspacer 적용 시점

LIVE와 REFINE의 적용 시점이 다르다:

- **LIVE**: `store.update_live()` 이후 표시 직전에 overlay 결과 전체에 적용
- **REFINE**: `store.apply_refine()` **이전**에 refine_text에 먼저 적용 후 저장, 표시 시 다시 overlay에 적용

```python
def _emit_live(self, update: TransducerUpdate):
    display_line = self.store.update_live(...)      # REFINE으로 overlay
    display_text = display_line.text
    if display_text.strip():
        display_text = self.spacer.space([display_text.replace(" ", "")])[0]
    # display_text 출력

def _emit_refine(self, update: RefineUpdate):
    refine_text = update.text
    if refine_text.strip():
        refine_text = self.spacer.space([refine_text.replace(" ", "")])[0]  # 먼저 보정
    self.store.apply_refine(SubtitleLine(text=refine_text, ...))             # 보정된 텍스트 저장
    display_text = self.store.last_refine_display_text or update.text
    if display_text.strip():
        display_text = self.spacer.space([display_text.replace(" ", "")])[0]
    # display_text 출력
```

SubtitleStore 내부의 매칭은 공백 제거(compact) 후 수행되므로, 저장 텍스트의 띄어쓰기는 매칭 정확도에 영향 없다.

quickspacer level=3은 한국어 복합어를 적절히 유지하면서 띄어쓰기를 교정한다.

```text
overlay 결과:      "지하면서 조각 단위로 음성을 잘라 처리함으로써 실 시간성을 확보합니다"
공백 제거:          "지하면서조각단위로음성을잘라처리함으로써실시간성을확보합니다"
spacer level=3:    "지하면서 조각 단위로 음성을 잘라 처리함으로써 실시간성을 확보합니다"
```

---

## 5. 자막 병합 전략 (SubtitleStore)

`utils/subtitle_store.py` → `SubtitleStore`

### 핵심 개념: confirmed_text 축적 + SequenceMatcher 오버레이

REFINE 결과가 들어올 때마다 `confirmed_text`에 **누적 연결**한다.
LIVE 텍스트가 들어오면, confirmed_text가 LIVE의 어디까지를 커버하는지 찾아 **confirmed + 나머지 LIVE suffix**를 표시한다.

```text
시간 경과에 따른 자막 변화:

t=2초 (LIVE만):
  confirmed_text = ""
  LIVE: "이천 십 오 년 현재"
  표시: "이천 십 오 년 현재"

t=4초 (첫 REFINE 도착):
  confirmed_text = "2026년 현재 음성 인식 기술은"
  LIVE: "이천 십 오 년 현재 음성 인식 기술은 단순히"
  표시: "2026년 현재 음성 인식 기술은 단순히"   ← confirmed + LIVE suffix

t=7초 (두 번째 REFINE 도착):
  confirmed_text = "2026년 현재 음성 인식 기술은 단순히 서류를 텍스트로"
  LIVE: "이천 십 오 년 현재 음성 인식 기술은 단순히 서류를 텍스트로 바꾸는"
  표시: "2026년 현재 음성 인식 기술은 단순히 서류를 텍스트로 바꾸는"
```

### 핵심 변수

| 변수 | 역할 |
|------|------|
| `self.confirmed_text` | REFINE 결과가 누적된 확정 텍스트. 가장 중요한 상태 |
| `self._confirmed_compact` | `confirmed_text`에서 공백 제거한 버전. SequenceMatcher 비교용 |
| `self.live_line` | 가장 최근 LIVE 원본 라인 |
| `self.live_display_line` | 오버레이 적용 후 실제 표시되는 라인 |
| `self.last_refine_display_text` | REFINE 적용 직후의 표시 텍스트. `[REFINE]` 출력에 사용 |

### 5-1. LIVE 업데이트: `update_live()`

```python
def update_live(self, line: SubtitleLine) -> SubtitleLine:
    self.live_line = line

    if not self.confirmed_text:
        return line  # 아직 REFINE 없으면 LIVE 그대로

    display_text = self._overlay(line.text)  # confirmed 위에 LIVE suffix 붙이기
    return SubtitleLine(text=display_text, ...)
```

### 5-2. REFINE 업데이트: `apply_refine()`

중복 제거는 `refine_stage.py`의 `_dedup_overlap()`에서 이미 처리되므로, SubtitleStore에서는 단순 공백+concat만 수행한다.

```python
def apply_refine(self, line, now_sec, is_final=False) -> bool:
    new_text = line.text.strip()
    if not new_text:
        return False

    # 단순 연결 (중복 제거는 refine_stage에서 완료)
    if self.confirmed_text:
        merged = self.confirmed_text + " " + new_text
    else:
        merged = new_text

    self.confirmed_text = merged
    self._confirmed_compact = merged.replace(" ", "")
    return True
```

### 5-3. 오버레이 알고리즘: `_overlay()`

LIVE 텍스트에서 confirmed_text가 커버하는 범위를 SequenceMatcher로 찾고, 그 이후의 LIVE 텍스트만 suffix로 붙인다.

#### 단계별 흐름 예시

```text
confirmed_text = "2026년 현재 음성 인식 기술은"
live_text      = "이천 십 오 년 현재 음성 인식 기술은 단순히 서류를"
```

**Step 1**: 공백 제거 (compact 변환)
```text
confirmed_compact = "2026년현재음성인식기술은"
live_compact      = "이천십오년현재음성인식기술은단순히서류를"
```

**Step 2**: `_find_coverage_end()` — SequenceMatcher로 커버 범위 탐색

```python
sm = SequenceMatcher(None, confirmed_compact, live_compact, autojunk=False)
# ratio() ≥ 0.3이면 유효한 매칭

# matching_blocks 결과 예시:
# ("년현재음성인식기술은"이 양쪽에서 일치)
# → live_compact에서의 끝 위치 = live_end
```

```text
confirmed_compact: "2026년현재음성인식기술은"
                        ^^^^^^^^^^^^^^^^^ 매칭
live_compact:      "이천십오년현재음성인식기술은단순히서류를"
                        ^^^^^^^^^^^^^^^^^ 매칭 끝 = 위치 17

live_cut_compact = 17
```

핵심: "2026"과 "이천십오"는 다르지만, "년현재음성인식기술은"이 공통이므로 **부분 매칭**으로 커버 범위를 찾을 수 있다. 순수 시간 기반 매칭의 한계(SenseVoice와 Transducer의 타임스탬프가 다름)를 텍스트 매칭으로 해결.

**Step 3**: `_compact_index_to_original()` — compact 인덱스를 원래 텍스트 위치로 변환

compact에서 17글자가 매칭되면, 원래 live_text에서 공백을 세면서 17번째 비공백 문자 위치를 찾는다:

```text
live_text: "이천 십 오 년 현재 음성 인식 기술은 단순히 서류를"
            1  2 3 4 5 6  7  8  9 10 11 12 13 14 15 16 17
            ^                                          ^
            (공백은 건너뛰고 비공백만 카운트)          17번째 = "은" 뒤
→ live_cut = 27 (live_text[27:] = " 단순히 서류를")
```

**Step 4**: `_join()` — confirmed + suffix 결합

```python
suffix = live_text[27:]  # " 단순히 서류를"
return self._join(confirmed_text, suffix)
# → "2026년 현재 음성 인식 기술은 단순히 서류를"
```

`_join()`은 추가로:
- suffix 앞 공백 제거 (lstrip)
- confirmed와 suffix의 suffix-prefix 중복 제거
- 필요시 공백 삽입

### 5-4. REFINE 청크 간 중복 처리

REFINE 청크 간 중복 제거는 `refine_stage.py`의 `_dedup_overlap()`에서 수행된다. `SubtitleStore.apply_refine()`은 이미 dedup된 텍스트를 받으므로 단순 concat만 한다.

`_join()`은 여전히 suffix-prefix overlap 제거를 수행하여, overlay 결과의 경계 부분 중복을 방지한다.

---

## 6. 멀티프로세스 구조

### 프로세스 구성

```text
Main Process (pipeline.py)
├── transducer-worker (별도 프로세스, mp.Process)
│     └── TransducerStage
└── refine-worker (별도 프로세스, mp.Process)
      └── SenseVoiceRefineStage
```

### 큐 기반 통신

| 큐 | 방향 | 용도 | maxsize |
|----|------|------|---------|
| `transducer_task_q` | Main → Worker | 오디오 프레임 전송 | 32 |
| `transducer_event_q` | Worker → Main | LIVE 결과 반환 | 무제한 |
| `refine_task_q` | Main → Worker | 4초 청크 전송 | 1 |
| `refine_event_q` | Worker → Main | REFINE 결과 반환 | 무제한 |

`refine_task_q`의 maxsize=1인 이유: SenseVoice 디코딩이 느리므로, 미처리 청크가 쌓이지 않도록 **backpressure** 적용.

### 메시지 타입 (`asr/protocol.py`)

```python
# Main → Worker
TransducerTask(action="feed"|"finalize", end_sec, samples)
RefineTask(action="feed"|"finalize", samples)

# Worker → Main
TransducerEvent(kind="update"|"done"|"error", update, error)
RefineEvent(kind="update"|"done"|"error", update, error)
```

### Backpressure 처리: `_put_task()`

task 큐가 가득 차면, 결과 큐를 드레인하면서 재시도:

```python
def _put_task(self, task_queue, task):
    while True:
        try:
            task_queue.put(task, timeout=0.05)  # 50ms 대기
            return
        except queue.Full:
            self._drain_worker_events()   # 밀린 결과 소비
            self._check_worker_health()   # 워커 비정상 종료 감지
```

### 이벤트 드레인: `_drain_worker_events()`

양쪽 워커의 결과를 **공정하게** 번갈아 소비:

```python
def _drain_worker_events(self):
    self._drain_refine_events(max_items=8)      # REFINE 먼저 8개
    self._drain_transducer_events(max_items=16) # LIVE 16개
    self._drain_refine_events(max_items=8)      # REFINE 다시 8개
```

`max_items`로 한쪽이 독점하지 않도록 제한.

---

## 7. 전체 실행 흐름 요약

```text
pipeline.run()
│
├─ _start_workers()    # 두 워커 프로세스 생성 및 시작
│
├─ for frame in source.iter_frames():   # 0.1초 프레임 반복
│   │
│   ├─ transducer_task_q.put(feed)      # 프레임 → transducer worker
│   │
│   ├─ refine_buffer에 누적
│   │   └─ 임계치 도달 시 → _flush_refine_buffer()
│   │       └─ refine_task_q.put(feed)  # 4초 청크 → refine worker
│   │
│   └─ _drain_worker_events()           # 결과 소비
│       ├─ TransducerEvent(update) → _emit_live()
│       │   └─ store.update_live() → quickspacer → [LIVE] 출력
│       │
│       └─ RefineEvent(update) → _emit_refine()
│           └─ store.apply_refine() → quickspacer → [REFINE] 출력
│
├─ _flush_refine_buffer()               # 남은 버퍼 전송
│
├─ _finalize_workers()                  # 양쪽에 finalize 전송
│   ├─ transducer_task_q.put(finalize)
│   └─ refine_task_q.put(finalize)
│
├─ _drain_until_done()                  # done 이벤트까지 대기
│   └─ [LIVE-FINAL] + [REFINE-FINAL] 출력
│
└─ _stop_workers()                      # 프로세스 정리
```

---

## 8. 핵심 파일 위치

| 파일 | 역할 |
|------|------|
| `main.py` | CLI 진입점 |
| `server.py` | 웹 UI 서버 (FastAPI + WebSocket, persistent workers) |
| `config/default.yaml` | 파이프라인 설정 (YAML) |
| `config/settings.py` | 설정 로더 + CLI 파싱 |
| `asr/pipeline.py` | CLI 오케스트레이션, 두 worker 관리, spacer 적용 |
| `asr/transducer_stage.py` | 1차 Transducer 인식 (LCP 기반 갱신) |
| `asr/refine_stage.py` | 2차 SenseVoice 정제 (Center-commit) |
| `asr/protocol.py` | 워커 간 메시지 타입 |
| `asr/workers.py` | 워커 프로세스 진입점 (CLI용) |
| `utils/subtitle_store.py` | confirmed_text 축적 + SequenceMatcher 오버레이 |
| `utils/audio_source.py` | 오디오 프레임 소스 (WAV 파일 + 마이크/VAD) |
| `utils/time_utils.py` | 타임스탬프 포맷 |
| `eval.py` | 배치 평가 스크립트 |
