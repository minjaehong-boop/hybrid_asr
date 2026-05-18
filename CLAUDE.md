# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **KsponSpeech ASR recipe** within the [icefall](https://github.com/k2-fsa/icefall) framework — a k2/lhotse-based speech recognition system. KsponSpeech is a 969-hour Korean spontaneous speech corpus that must be downloaded manually from [AIHub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123).

Two model recipes are available:

| Directory | Encoder | Parameters | Notes |
|---|---|---|---|
| `pruned_transducer_stateless7_streaming/` | Streaming Zipformer | 79M | Supports real-time chunk-wise decoding |
| `zipformer/` | Upgraded Zipformer | 74M | Latest recipe; non-streaming by default |

Both use a **pruned stateless transducer** with Embedding + Conv1d decoder.

## Data Pipeline

Run `prepare.sh` in stages (all scripts run from `egs/ksponspeech/ASR/`):

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # Required to avoid segfault

# Stage 0: Download MUSAN noise data (auto)
# Stage 1: Prepare lhotse manifests for KsponSpeech and Zeroth-Korean
# Stage 2: Prepare MUSAN manifest
# Stage 3: Compute 80-dim fbank features → data/fbank/
# Stage 4: Compute MUSAN fbank
# Stage 5: Train BPE model → data/lang_bpe_5000/

./prepare.sh --stage 0 --stop-stage 5
# Or run a single stage:
./prepare.sh --stage 3 --stop-stage 3
```

**Prerequisites before running:**
- Place KsponSpeech at `download/KsponSpeech/` (or symlink: `ln -svf /path/to/KsponSpeech download/KsponSpeech`)
- Optional: place Zeroth-Korean data at `zeroth_korean/` for additional training data
- MUSAN is downloaded automatically to `download/musan/`

Intermediate outputs use `.done` sentinel files — delete them to force re-runs of specific steps.

## Training

All training scripts use PyTorch DDP and must be run from `egs/ksponspeech/ASR/`.

**Streaming Zipformer (pruned_transducer_stateless7_streaming):**
```bash
./pruned_transducer_stateless7_streaming/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --max-duration 750 \
  --enable-musan True
```

**Zipformer (non-streaming):**
```bash
./zipformer/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir zipformer/exp \
  --max-duration 750 \
  --enable-musan True \
  --base-lr 0.035   # Reduced from default 0.045 to avoid grad_scale errors
```

Checkpoints are saved to `exp/epoch-N.pt` and `exp/best-train-loss.pt`. Use `--start-epoch` to resume.

## Decoding

**Simulated streaming (full utterance fed at once, decode.py):**
```bash
./pruned_transducer_stateless7_streaming/decode.py \
  --epoch 30 --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --max-duration 600 \
  --decode-chunk-len 32 \     # 320ms at 10ms frame shift
  --decoding-method greedy_search   # or: modified_beam_search, fast_beam_search
```

**True chunk-wise streaming (streaming_decode.py):**
```bash
./pruned_transducer_stateless7_streaming/streaming_decode.py \
  --epoch 30 --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --decoding-method greedy_search \
  --decode-chunk-len 32 \
  --num-decode-streams 2000
```

**Zipformer decoding:**
```bash
./zipformer/decode.py \
  --epoch 30 --avg 9 \
  --exp-dir ./zipformer/exp \
  --decoding-method modified_beam_search
```

`--avg N` averages the last N checkpoints before the specified epoch.

## ONNX Export and Deployment

Export to ONNX for deployment with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx):

```bash
uv run ./pruned_transducer_stateless7_streaming/export-onnx.py \
  --tokens data/lang_bpe_5000/tokens.txt \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --epoch 30 --avg 9 \
  --use-averaged-model true \
  --decode-chunk-len 32
```

Produces `encoder-epoch-30-avg-9.onnx`, `decoder-epoch-30-avg-9.onnx`, `joiner-epoch-30-avg-9.onnx` in `exp/`.

sherpa-onnx inference:
```bash
./build/bin/sherpa-onnx \
  --tokens=/path/to/tokens.txt \
  --encoder=/path/to/encoder-epoch-30-avg-9.onnx \
  --decoder=/path/to/decoder-epoch-30-avg-9.onnx \
  --joiner=/path/to/joiner-epoch-30-avg-9.onnx \
  /path/to/test.wav
```

Hotword/context biasing only works with `--decoding-method modified_beam_search`.

## Architecture

- **Optimizer**: `ScaledAdam` with `Eden` LR scheduler (both in `icefall/optim.py`)
- **Data loading**: `KsponSpeechAsrDataModule` uses lhotse's `DynamicBucketingSampler` for dynamic batching by total duration (`--max-duration`)
- **Features**: Pre-computed 80-dim log-Mel filterbank stored as `.jsonl.gz` cut manifests
- **Tokenization**: SentencePiece BPE with vocab size 5000 (`data/lang_bpe_5000/bpe.model`)
- **Streaming**: Controlled by `--causal 1` flag and `--decode-chunk-len` (frames); 32 frames ≈ 320ms

## Package Manager

This project uses `uv` for dependency management. Use `uv run` to execute scripts with the managed environment.
