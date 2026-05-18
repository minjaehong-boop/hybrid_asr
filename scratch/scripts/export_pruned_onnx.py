"""
Scratch SenseVoiceSmallCTCScratch 체크포인트 → ONNX export + INT8/INT4 양자화.

Pruned 버전과 달리 task embedding / rich CE / task query concat이 없으므로
입력은 (speech, speech_lengths) 2개뿐, 출력은 (ctc_log_probs, encoder_out_lens).

Usage:
  python scripts/export_scratch_onnx.py --tag ko5k
  python scripts/export_scratch_onnx.py --tag ko5k --ckpt workdir/sensevoice_scratch_ko5k_mcu/model.pt.avg5

서버 기본 경로: /deepet/jh/sensevoice/finetune/scratch/
  conf/sensevoice_scratch_<tag>.yaml
  workdir/sensevoice_scratch_<tag>/{model.pt.avgN, onnx/}
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# tables.register 데코레이터가 Hydra-free 환경에서 실행되도록 import
import sensevoice_scratch  # noqa: F401
from funasr.register import tables


ASSETS = ROOT / "assets"


class ExportWrapper(torch.nn.Module):
    """
    ONNX export용 래퍼. training forward에서 쓰는 CTC loss / specaug / normalize를
    모두 제거하고 (speech, speech_lengths) → (ctc_log_probs, encoder_out_lens)만 노출.
    """
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.ctc = model.ctc

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor):
        encoder_out, encoder_out_lens = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        return ctc_log_probs, encoder_out_lens


def build_model(conf_path: Path, ckpt_path: Path):
    with open(conf_path) as f:
        cfg = yaml.safe_load(f)

    model_cls = tables.model_classes.get(cfg["model"])
    if model_cls is None:
        raise RuntimeError(
            f"model_class '{cfg['model']}' not registered. "
            f"sensevoice_scratch 패키지가 import됐는지 확인"
        )

    # vocab_size는 tokens.txt 줄 수로 결정 (icefall BPE 5000)
    bpe_path = Path(cfg["tokenizer_conf"]["bpemodel"])
    tokens_txt = bpe_path.parent / "tokens.txt"
    if not tokens_txt.exists():
        raise FileNotFoundError(f"{tokens_txt} (vocab_size 결정에 필요)")
    with open(tokens_txt, encoding="utf-8") as f:
        vocab_size = sum(1 for line in f if line.strip())

    model_conf = cfg.get("model_conf", {}) or {}
    model = model_cls(
        specaug=None, specaug_conf=None,
        normalize=None, normalize_conf=None,
        encoder=cfg["encoder"],
        encoder_conf=cfg["encoder_conf"],
        ctc_conf=model_conf.get("ctc_conf"),
        input_size=560,
        vocab_size=vocab_size,
        ignore_id=model_conf.get("ignore_id", -1),
        blank_id=model_conf.get("blank_id", 0),
        length_normalized_loss=model_conf.get("length_normalized_loss", True),
    )

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        state = raw["model"]
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state = raw["state_dict"]
    else:
        state = raw

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    model.eval()
    return model, vocab_size


def export_onnx(model, out_path: Path):
    wrapper = ExportWrapper(model)
    wrapper.eval()

    # dummy: LFR 이후 560-dim 특징, 100 frames ≈ 6초 오디오
    T = 100
    speech = torch.randn(1, T, 560, dtype=torch.float32)
    speech_lengths = torch.tensor([T], dtype=torch.int32)

    # sanity check
    with torch.no_grad():
        logp, lens = wrapper(speech, speech_lengths)
        print(f"  dummy forward OK: logp={tuple(logp.shape)} lens={lens.tolist()}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (speech, speech_lengths),
            str(out_path),
            input_names=["speech", "speech_lengths"],
            output_names=["ctc_log_probs", "encoder_out_lens"],
            dynamic_axes={
                "speech": {0: "B", 1: "T"},
                "speech_lengths": {0: "B"},
                "ctc_log_probs": {0: "B", 1: "T_out"},
                "encoder_out_lens": {0: "B"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  FP32 ONNX: {out_path} ({size_mb:.1f} MB)")


def quantize_int8(fp32: Path, out: Path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(fp32), str(out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    print(f"  INT8 ONNX: {out} ({out.stat().st_size/1024/1024:.1f} MB)")


def quantize_int4(fp32: Path, out: Path):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    m = onnx.load(str(fp32))
    q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=False, accuracy_level=4)
    q.process()
    q.model.save_model_to_file(str(out), use_external_data_format=False)
    print(f"  INT4 ONNX: {out} ({out.stat().st_size/1024/1024:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ko5k")
    ap.add_argument("--ckpt", default=None,
                    help="체크포인트 경로 (기본: workdir/sensevoice_scratch_<tag>/model.pt.avg5)")
    ap.add_argument("--config", default=None,
                    help="YAML 경로 (기본: conf/sensevoice_scratch_<tag>.yaml)")
    ap.add_argument("--skip-int4", action="store_true",
                    help="INT4 스킵 (onnxruntime 1.16+ 필요)")
    args = ap.parse_args()

    conf_path = Path(args.config) if args.config else (
        ROOT / "conf" / f"sensevoice_scratch_{args.tag}.yaml"
    )
    work_dir = ROOT / "workdir" / f"sensevoice_scratch_{args.tag}"
    ckpt_path = Path(args.ckpt) if args.ckpt else (work_dir / "model.pt.avg5")
    out_dir = work_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not conf_path.exists():
        sys.exit(f"Config not found: {conf_path}")
    if not ckpt_path.exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}\n"
                 f"  (avg_ckpt.py --model-dir {work_dir} --last 5 먼저 실행)")

    # 추론 시 함께 필요한 파일들을 onnx 디렉토리에 복사
    tokens_src = ASSETS / "lang_bpe_5000" / "tokens.txt"
    bpe_src    = ASSETS / "lang_bpe_5000" / "bpe.model"
    cmvn_src   = ASSETS / "kspon.mvn"
    for src in (tokens_src, bpe_src, cmvn_src):
        if src.exists():
            shutil.copy2(str(src), str(out_dir / src.name))
            print(f"  copied {src.name}")
        else:
            print(f"  [warn] not found, skip: {src}")

    print(f"\n[1/4] Loading {conf_path.name} + {ckpt_path.name}...")
    model, vocab = build_model(conf_path, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  vocab_size: {vocab}  params: {n_params:,}")

    print("\n[2/4] Exporting ONNX (FP32)...")
    fp32 = out_dir / "model.onnx"
    export_onnx(model, fp32)

    print("\n[3/4] Quantizing INT8...")
    quantize_int8(fp32, out_dir / "model.int8.onnx")

    if args.skip_int4:
        print("\n[4/4] INT4 skipped (--skip-int4)")
    else:
        print("\n[4/4] Quantizing INT4...")
        try:
            quantize_int4(fp32, out_dir / "model.int4.onnx")
        except Exception as e:
            print(f"  [warn] INT4 실패: {e}")
            print("  onnxruntime 1.16+ 필요. --skip-int4 로 재실행")

    print(f"\nDone. Output dir: {out_dir}")
    print("  추론 입력: speech [B,T,560] float32, speech_lengths [B] int32")
    print("  추론 출력: ctc_log_probs [B,T',vocab] float32, encoder_out_lens [B] int32")
    print("  greedy CTC: yseq = ctc_log_probs[b,:lens[b]].argmax(-1), blank_id=0")


if __name__ == "__main__":
    main()

