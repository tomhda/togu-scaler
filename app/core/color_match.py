"""等倍処理の出力の色と明るさを入力に合わせ直す。

Remaster 本家の --color-transfer（pipeline_cpp/color_transfer.cu）と同じ処理。
1 コマ全体の RGB ごとの平均と標準偏差を入力に合わせる。タイル単位ではなく
1 コマ全体で揃えるので、継ぎ目は出ない。
"""
from __future__ import annotations

import numpy as np

# 本家は [0, 1] の分散の下限を 1e-8（標準偏差 1e-4）にしている。0〜255 に換算した値。
_MIN_STD = 1e-4 * 255.0
_LEVELS = np.arange(256, dtype=np.float64)


def _channel_stats(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各チャネルの平均と標準偏差を、256 階調の度数から求める。"""
    means = np.empty(3)
    stds = np.empty(3)
    for c in range(3):
        counts = np.bincount(image[..., c].ravel(), minlength=256).astype(np.float64)
        total = counts.sum()
        mean = float(counts @ _LEVELS) / total
        variance = float(counts @ (_LEVELS - mean) ** 2) / total
        means[c] = mean
        stds[c] = max(variance, 0.0) ** 0.5
    return means, stds


def match_channel_stats(source: np.ndarray, result: np.ndarray) -> np.ndarray:
    """result の RGB 各チャネルの平均と標準偏差を source に合わせた画像を返す。

    どちらも同じ寸法の HxWx3 uint8。形が違うときは result をそのまま返す。
    チャネルごとの一次変換なので、256 段の対応表で置き換える。
    """
    if (
        source.dtype != np.uint8 or result.dtype != np.uint8
        or source.shape != result.shape or source.ndim != 3 or source.shape[2] != 3
        or source.size == 0
    ):
        return result
    src_mean, src_std = _channel_stats(source)
    out_mean, out_std = _channel_stats(result)
    scale = src_std / np.maximum(out_std, _MIN_STD)
    matched = np.empty_like(result)
    for c in range(3):
        table = (_LEVELS - out_mean[c]) * scale[c] + src_mean[c]
        lut = np.clip(np.rint(table), 0, 255).astype(np.uint8)
        matched[..., c] = lut[result[..., c]]
    return matched
