r"""Download the pinned Remaster Student ONNX and prepare it for DirectML.

Build-time dependency: pip install onnx numpy (not needed by the application).
Usage: .venv\Scripts\python.exe scripts/get_remaster.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = "018e7815aa8ef6e3eb6433d2572433d4f36e180e"
URL = f"https://huggingface.co/seantempesta/remaster-drunet/resolve/{REVISION}/drunet_student.onnx"
SHA256 = "b366ac68ed8e30a5d5a4db7776219ea3bd72929a81a53ef3b6f7e8fd0703206a"
FILENAME = "remaster_nchw_512x512_fp32.onnx"


def prepare(source: Path, destination: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, numpy_helper

    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Remaster download SHA-256 mismatch")
    model = onnx.load(source)
    # The published graph is entirely FP16 Conv/Add/ReLU/ConvTranspose.
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT16:
            tensor.CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(tensor).astype(np.float32), tensor.name))
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    for value in (*model.graph.input, *model.graph.output):
        for dim, size in zip(value.type.tensor_type.shape.dim, (1, 3, 512, 512)):
            dim.ClearField("dim_param")
            dim.dim_value = size
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".onnx.part")
    try:
        onnx.save(model, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models" / "ai")
    args = parser.parse_args()
    source = ROOT / "tmp" / "remaster" / "drunet_student.onnx"
    source.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        temporary = source.with_suffix(".download")
        try:
            request = urllib.request.Request(URL, headers={"User-Agent": "togu-scaler"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
                raise ValueError("Remaster download SHA-256 mismatch")
            os.replace(temporary, source)
        finally:
            temporary.unlink(missing_ok=True)
    destination = args.output_dir / FILENAME
    prepare(source, destination)
    for name in ("LICENSE-MIT-Remaster.txt", "NOTICE-Remaster.txt"):
        shutil.copyfile(ROOT / "scripts" / "release-licenses" / name, args.output_dir / name)
    print(f"Prepared: {destination}")


if __name__ == "__main__":
    main()
