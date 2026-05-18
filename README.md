# KsponSpeech ASR Recipe

KsponSpeech is a 969-hour Korean spontaneous speech corpus, recorded from dialogues of about 2,000 native speakers on open-domain topics. This recipe trains a streaming transducer model (Streaming Zipformer + pruned stateless decoder) on KsponSpeech, optionally augmented with Zeroth-Korean and MUSAN noise.

- Dataset: [AIHub KsponSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)
- Paper: https://www.mdpi.com/2076-3417/10/19/6936
- Results: [RESULTS.md](./RESULTS.md)

---

## 1. Clone

```bash
git clone -b streaming_model https://github.com/minjaehong-boop/hybrid_asr.git
cd hybrid_asr
```

## 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate

pip install "numpy<2"
pip install -r requirements.txt
pip install -e .
```

## 3. Install k2

k2 is not on PyPI — install the wheel that matches your **PyTorch + CUDA** version.

```bash
# Check your versions first
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Then install the matching wheel. Example for torch 2.4.0 + CUDA 12.1:

```bash
pip install k2==1.24.4.dev20241030+cuda12.1.torch2.4.0 \
  -f https://k2-fsa.github.io/k2/cuda.html
```

Browse all available wheels: https://k2-fsa.github.io/k2/cuda.html

Verify:
```bash
python -c "import k2; print(k2.__version__)"
```

> **Note:** `requirements.txt` only includes packages needed for this recipe (lhotse, lilcom, sentencepiece, tensorboard). ONNX-related packages are optional and only needed for export.

## 4. Prepare datasets

Place the data before running `prepare.sh`:

| Dataset | Path | How to get |
|---|---|---|
| KsponSpeech | `download/KsponSpeech/` | Manual download from AIHub (requires account) |
| Zeroth-Korean | `download/zeroth_korean/` | Optional — skip if not using |
| MUSAN | `download/musan/` | Auto-downloaded by `prepare.sh` |

Then run:

```bash
./prepare.sh --stage 0 --stop-stage 5
```

This runs the following stages in order:

| Stage | What it does |
|---|---|
| 0 | Download MUSAN |
| 1 | Generate lhotse manifests for KsponSpeech and Zeroth-Korean |
| 2 | Generate MUSAN manifest |
| 3 | Extract 80-dim fbank features → `data/fbank/` |
| 4 | Extract MUSAN fbank features |
| 5 | Train BPE tokenizer (vocab=5000) → `data/lang_bpe_5000/` |

To re-run a specific stage: `./prepare.sh --stage 3 --stop-stage 3`

## 5. Train

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

- Reduce `--max-duration` if CUDA OOM (try 500 or 300)
- Resume from checkpoint: `--start-epoch N`
- Checkpoints saved to `pruned_transducer_stateless7_streaming/exp/epoch-N.pt`

## 6. Decode

```bash
./pruned_transducer_stateless7_streaming/decode.py \
  --epoch 30 \
  --avg 9 \
  --exp-dir ./pruned_transducer_stateless7_streaming/exp \
  --max-duration 600 \
  --decode-chunk-len 32 \
  --decoding-method greedy_search
```

`--avg 9` averages the last 9 checkpoints before the given epoch for better accuracy.
Decoding methods: `greedy_search`, `modified_beam_search`, `fast_beam_search`

## 7. Export to ONNX (for deployment)

```bash
./pruned_transducer_stateless7_streaming/export-onnx.py \
  --tokens data/lang_bpe_5000/tokens.txt \
  --exp-dir pruned_transducer_stateless7_streaming/exp \
  --epoch 30 \
  --avg 9 \
  --use-averaged-model true \
  --decode-chunk-len 32
```

Produces `encoder/decoder/joiner-epoch-30-avg-9.onnx` in `exp/`, ready for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
