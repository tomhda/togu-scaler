"""利用者が持ち込んだ ONNX モデルの読み込み（Qt 非依存）。

アプリのフォルダの ``models/custom/`` に置かれた ``<名前>.onnx`` を、
起動時に 1 回読み込む（置いたあとは再起動が必要）。同じ名前の
``<名前>.json`` があれば設定として読む（無くてよい）。

GUI と CLI はこのモジュールの ``scan()`` だけを使う。推論自体は
``helper_backend`` が通常の GPU 経路で行う。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..i18n import N_
from .settings import _app_root

# モデルのキーに付ける接頭辞。設定・CLI・一覧の内部値はこの形。
CUSTOM_PREFIX = "custom:"
# テスト用に場所を差し替える環境変数。
CUSTOM_MODELS_ENV = "TOGU_CUSTOM_MODELS_DIR"

# JSON が壊れているときの理由（詳細設定の「読み込めなかった行」に出す文言そのまま）。
# N_() を付けて訳文の洗い出し対象にする（表示時に t() で訳す）。
REASON_BAD_JSON = N_("設定ファイルを読めません")
# overlap が範囲外のときの理由（同上）。
REASON_BAD_OVERLAP = N_("overlap は 0〜64 の整数にしてください")


class CustomModelError(Exception):
    """追加モデルの失敗。Vulkan への退避はしない。"""


class CustomModelLoadFailed(CustomModelError):
    """モデルを開けなかった・4 倍でなかった。"""


class CustomModelNeedsGpu(CustomModelError):
    """GPU（DirectML）でだけ実行できるのに GPU が使えない。"""


class CustomModelImagesOnly(CustomModelError):
    """動画に使えないモデルを動画に指定した。"""


@dataclass(frozen=True)
class CustomModel:
    """読み込めた追加モデル 1 件。name は一覧に出す前の素の名前。"""

    key: str       # custom:<ファイル名（拡張子なし）>
    name: str      # JSON の name、無ければファイル名（拡張子なし）
    path: Path     # .onnx の絶対パス
    overlap: int   # serve の --overlap（片側マージン、入力 px）
    video: bool    # False なら動画には使えない


@dataclass(frozen=True)
class CustomModelFailure:
    """一覧に出さなかったモデル 1 件。reason は表示文言そのまま。"""

    file: str      # .onnx のファイル名
    reason: str    # REASON_BAD_JSON / REASON_BAD_OVERLAP のいずれか


def is_custom_key(value: object) -> bool:
    """追加モデルのキー（custom: で始まる文字列）か。"""
    return isinstance(value, str) and value.startswith(CUSTOM_PREFIX)


def custom_models_dir() -> Path:
    """追加モデルの置き場所。無ければ作る。

    決め方は ``settings._app_root()`` と同じ（配布版では exe のある
    フォルダの ``models\\custom\\``、開発時はリポジトリの ``models/custom/``）。
    環境変数 ``TOGU_CUSTOM_MODELS_DIR`` があればそちらを使う（テスト用）。
    """
    override = os.environ.get(CUSTOM_MODELS_ENV)
    folder = Path(override).expanduser() if override else _app_root() / "models" / "custom"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def scan() -> tuple[list[CustomModel], list[CustomModelFailure]]:
    """フォルダを読み、（読めた一覧、読めなかった一覧）を返す。

    1 モデル = ``<名前>.onnx``。同じ名前の ``<名前>.json`` があれば
    ``{ "name", "overlap", "video" }`` を読む（無くてよい・知らない項目は無視）。
    読めたものは名前の順に並べる。JSON が壊れている・overlap が
    0〜64 の整数でないモデルは一覧に出さず、理由を記録する。
    """
    from .settings import HELPER_DEFAULT_OVERLAP

    folder = custom_models_dir()
    models: list[CustomModel] = []
    failures: list[CustomModelFailure] = []
    onnx_files = sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() == ".onnx"),
        key=lambda path: path.name,
    )
    for onnx in onnx_files:
        stem = onnx.stem
        key = f"{CUSTOM_PREFIX}{stem}"
        name = stem
        overlap = HELPER_DEFAULT_OVERLAP
        video = True
        sidecar = onnx.with_suffix(".json")
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                failures.append(CustomModelFailure(onnx.name, REASON_BAD_JSON))
                continue
            if not isinstance(data, dict):
                failures.append(CustomModelFailure(onnx.name, REASON_BAD_JSON))
                continue
            raw_name = data.get("name", name)
            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
            raw_overlap = data.get("overlap", overlap)
            if (
                isinstance(raw_overlap, bool)
                or not isinstance(raw_overlap, int)
                or not 0 <= raw_overlap <= 64
            ):
                failures.append(CustomModelFailure(onnx.name, REASON_BAD_OVERLAP))
                continue
            overlap = raw_overlap
            # video は false のときだけ動画に使えない（それ以外は使える）。
            video = data.get("video", True) is not False
            video = bool(video)
        models.append(CustomModel(
            key=key, name=name, path=onnx.resolve(),
            overlap=overlap, video=video,
        ))
    models.sort(key=lambda model: (model.name.casefold(), model.key))
    return models, failures


def get(key: str) -> CustomModel | None:
    """キーに対応する追加モデル。無ければ None（起動後に消された場合など）。"""
    for model in scan()[0]:
        if model.key == key:
            return model
    return None
