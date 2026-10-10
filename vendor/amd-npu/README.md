# AMD Ryzen AI NPU assets

This folder holds NPU models from the first NPU experiments (August 2026, Ryzen AI Software 1.7.1).
The app no longer offers this route in its window or its command line; the current NPU route uses
`models/ai` and `vendor/amd-npu-1.8` (see `docs/technical.md`).

Both models here are converted from weights published by xinntao/Real-ESRGAN under BSD 3-Clause.

## animevideov3
- Source weights: xinntao/Real-ESRGAN `realesr-animevideov3.pth` (BSD-3-Clause)
- Export: `scripts/npu/export_animevideov3.py --decompose-prelu`, then
  `quark.onnx.tools.convert_fp32_to_bf16 --format with_cast`
- Model: `onnx-models/animevideov3dp_nchw_256x256_bf16cast.onnx`
- "dp" = decomposed PReLU (ReLU(x) - w*ReLU(-x)): the VAIML BF16 flow of Ryzen AI 1.7.1
  miscompiled trained PReLU weights (negative slopes); the equivalent decomposition avoids the PReLU kernel.
- Cache: `modelcachekey_animevideov3dp_nchw_256x256_bf16cast/`. It exceeds GitHub's 100 MB file limit and
  is not committed; the first NPU run after a fresh clone recompiles it (about 5 minutes).

## x4plus_anime
- Source weights: xinntao/Real-ESRGAN `RealESRGAN_x4plus_anime_6B.pth` (BSD-3-Clause),
  exported via `scripts/npu/export_x4plus_anime.py`
- Model: `onnx-models/x4plus_anime_nchw_256x256_bf16cast.onnx`
- Cache: `modelcachekey_x4plus_anime_nchw_256x256_bf16cast/`

The cache directory is kept next to the ONNX model. Ryzen AI's VitisAI Execution Provider uses this path
through explicit `cache_dir` and `cache_key` options.

A Real-ESRGAN model converted from AMD's `amd/realesrgan-256x256-tiles-amdnpu` (Research-only RAIL-MS)
was kept here until v0.13.0 and has been removed.
