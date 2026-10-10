"""realesr-general-x4v3（SRVGGNetCompact, 64ch・32 層, BSD 3-Clause）を固定形状の ONNX に出力する。

重み（SHA-256 8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292）:
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth

実行例（torch / onnx / onnxsim / onnxruntime の入った Python で）:
  python scripts/npu/export_generalv3.py --weights <realesr-general-x4v3.pth> --out <出力フォルダ>

出力:
  generalv3_nchw_256x256_fp32.onnx      GPU 用
  generalv3_nchw_512x512_fp32.onnx      GPU 用
  generalv3dp_nchw_512x512_fp32.onnx    NPU 用の元（PReLU を ReLU 2 本に分解。Anime Video v3 と同じ回避策）

NPU 用は、この後 bf16cast 変換と末尾の切り離しを行う（docs/model-conversion.md）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ARGV = sys.argv[1:]
sys.argv = [sys.argv[0]]  # 取り込み元のスクリプトが読み込み時に引数を見るので、空にしてから取り込む
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_animevideov3 as base  # noqa: E402


def build(weights: Path, decompose: bool) -> nn.Module:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    params = state.get("params", state)
    model = base.SRVGGNetCompact(num_feat=64, num_conv=32, upscale=4)
    model.load_state_dict(params, strict=True)
    if decompose:
        for i, layer in enumerate(model.body):
            if isinstance(layer, nn.PReLU):
                model.body[i] = base.DecomposedPReLU(layer.weight.data)
    return model.eval()


def export(weights: Path, out_dir: Path, size: int, name: str, decompose: bool) -> None:
    import onnx
    import onnxruntime as ort
    from onnxsim import simplify

    raw = out_dir / f"{name}_raw.onnx"
    out = out_dir / f"{name}.onnx"
    torch.onnx.export(build(weights, decompose), torch.rand(1, 3, size, size), str(raw),
                      input_names=["input"], output_names=["output"],
                      opset_version=17, do_constant_folding=True, dynamic_axes=None)
    simplified, ok = simplify(onnx.load(str(raw)))
    onnx.save(simplified if ok else onnx.load(str(raw)), str(out))
    raw.unlink()
    # 検証は、分解していない素のモデルの出力と比べる
    x = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)
    with torch.no_grad():
        ref = build(weights, False)(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"{out.name}: 出力形状 {got.shape}, 元の torch との最大差 {diff:.2e}", flush=True)
    if got.shape != (1, 3, size * 4, size * 4) or diff > 1e-4:
        raise SystemExit("検証NG")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(_ARGV)
    args.out.mkdir(parents=True, exist_ok=True)
    export(args.weights, args.out, 256, "generalv3_nchw_256x256_fp32", False)
    export(args.weights, args.out, 512, "generalv3_nchw_512x512_fp32", False)
    export(args.weights, args.out, 512, "generalv3dp_nchw_512x512_fp32", True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
