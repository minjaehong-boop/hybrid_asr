"""
프루닝된 SenseVoiceSmall 체크포인트를 ONNX로 export 후 INT8/INT4 양자화.
FunASR 공식 export_meta.export_forward 시그니처 사용 (sherpa-onnx 호환).

Usage:
  python scripts/export_pruned_onnx.py --tag tiny
  python scripts/export_pruned_onnx.py --tag tiny --ckpt workdir/sensevoice_pruned_tiny/model.pt.avg5
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from funasr.models.sense_voice.model import SenseVoiceSmall
from funasr.models.sense_voice.export_meta import export_rebuild_model


SV_CACHE = Path("/home/alswoghd/.cache/modelscope/hub/models/iic/SenseVoiceSmall")


def build_model(conf_path: Path, ckpt_path: Path) -> SenseVoiceSmall:
    with open(conf_path) as f:
        cfg = yaml.safe_load(f)

    model_conf = cfg.get("model_conf", {}) or {}
    model = SenseVoiceSmall(
        specaug=None,
        specaug_conf=None,
        normalize=None,
        normalize_conf=None,
        encoder=cfg["encoder"],
        encoder_conf=cfg["encoder_conf"],
        ctc_conf=model_conf.get("ctc_conf"),
        input_size=560,          # n_mels(80) * lfr_m(7)
        vocab_size=25055,        # SenseVoice spectok vocab size
        ignore_id=model_conf.get("ignore_id", -1),
        blank_id=model_conf.get("blank_id", 0),
        sos=model_conf.get("sos", 1),
        eos=model_conf.get("eos", 2),
        length_normalized_loss=model_conf.get("length_normalized_loss", True),
    )

    state = torch.load(str(ckpt_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (first: {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})")
    model.eval()
    return model


def export_onnx(model: SenseVoiceSmall, out_path: Path):
    export_rebuild_model(model, device="cpu", max_seq_len=512)
    dummy = model.export_dummy_inputs()

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=model.export_input_names(),
        output_names=model.export_output_names(),
        dynamic_axes=model.export_dynamic_axes(),
        opset_version=14,
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  FP32 ONNX: {out_path} ({size_mb:.1f} MB)")


def quantize_int8(fp32_path: Path, out_path: Path):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        str(fp32_path),
        str(out_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  INT8 ONNX: {out_path} ({size_mb:.1f} MB)")


def quantize_int4(fp32_path: Path, out_path: Path):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    model = onnx.load(str(fp32_path))
    quant = MatMulNBitsQuantizer(model, block_size=32, is_symmetric=False, accuracy_level=4)
    quant.process()
    quant.model.save_model_to_file(str(out_path), use_external_data_format=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  INT4 ONNX: {out_path} ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="tiny")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="체크포인트 경로 (기본: workdir/sensevoice_pruned_<tag>/model.pt.pruned)")
    args = ap.parse_args()

    conf_path = ROOT / "conf" / f"sensevoice_pruned_{args.tag}.yaml"
    work_dir = ROOT / "workdir" / f"sensevoice_pruned_{args.tag}"
    ckpt_path = Path(args.ckpt) if args.ckpt else work_dir / "model.pt.pruned"
    out_dir = work_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tokens.txt 복사 (sherpa-onnx 로딩용)
    tokens_src = SV_CACHE / "tokens.txt"
    if tokens_src.exists():
        shutil.copy2(str(tokens_src), str(out_dir / "tokens.txt"))

    print(f"[1/4] Building SenseVoiceSmall from {conf_path.name} + {ckpt_path.name}...")
    model = build_model(conf_path, ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    print("[2/4] Exporting ONNX (FP32)...")
    fp32_path = out_dir / "model.onnx"
    export_onnx(model, fp32_path)

    print("[3/4] Quantizing INT8...")
    quantize_int8(fp32_path, out_dir / "model.int8.onnx")

    print("[4/4] Quantizing INT4...")
    quantize_int4(fp32_path, out_dir / "model.int4.onnx")

    print("\nDone!")


if __name__ == "__main__":
    main()
