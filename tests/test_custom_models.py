"""追加モデル（app/core/custom_models.py）のテスト。推論は回さない。

helper の起動・READY 応答は偽物に差し替える。GUI は offscreen で開き、
終わったら必ず MainWindow を close する。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# QApplication 生成前にオフスクリーンを強制
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtWidgets import QLabel

from app import cli
from app.core import custom_models as cm
from app.core import helper_backend
from app.core.settings import (
    UpscaleBackend,
    UpscaleSettings,
    canonical_helper_model,
)


@pytest.fixture
def custom_dir(monkeypatch, tmp_path):
    """TOGU_CUSTOM_MODELS_DIR を空の一時フォルダへ向ける。"""
    folder = tmp_path / "custom-models"
    folder.mkdir()
    monkeypatch.setenv(cm.CUSTOM_MODELS_ENV, str(folder))
    return folder


def _onnx(folder: Path, name: str = "mymodel") -> Path:
    path = folder / f"{name}.onnx"
    path.write_bytes(b"fake-onnx")
    return path


def _sidecar(folder: Path, name: str, payload: object) -> Path:
    path = folder / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _make_png(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 120, 200)).save(path, "PNG")
    return path


# ------------------------------------------------------------------ scan

def test_scan_onnx_only_uses_filename_and_defaults(custom_dir):
    _onnx(custom_dir)
    models, failures = cm.scan()
    assert failures == []
    assert len(models) == 1
    model = models[0]
    assert model.key == "custom:mymodel"
    assert model.name == "mymodel"
    assert model.path == (custom_dir / "mymodel.onnx").resolve()
    assert model.overlap == 16
    assert model.video is True


def test_scan_sidecar_name_overlap_video(custom_dir):
    _onnx(custom_dir, "mymodel")
    _sidecar(custom_dir, "mymodel", {
        "name": "My Model", "overlap": 32, "video": False, "unknown": 1,
    })
    models, failures = cm.scan()
    assert failures == []
    assert [(m.key, m.name, m.overlap, m.video) for m in models] == [
        ("custom:mymodel", "My Model", 32, False)]


def test_scan_broken_json_is_recorded(custom_dir):
    bad = _onnx(custom_dir, "broken")
    (custom_dir / "broken.json").write_text("{壊れた", encoding="utf-8")
    models, failures = cm.scan()
    assert models == []
    assert failures == [
        cm.CustomModelFailure(file=bad.name, reason="設定ファイルを読めません")]


def test_scan_non_dict_json_is_recorded(custom_dir):
    _onnx(custom_dir, "array")
    _sidecar(custom_dir, "array", [1, 2, 3])
    models, failures = cm.scan()
    assert models == []
    assert [f.reason for f in failures] == ["設定ファイルを読めません"]


@pytest.mark.parametrize("overlap", [65, -1, 1.5, "16", True, None])
def test_scan_overlap_out_of_range_is_recorded(custom_dir, overlap):
    _onnx(custom_dir, "odd")
    _sidecar(custom_dir, "odd", {"overlap": overlap})
    models, failures = cm.scan()
    assert models == []
    assert [f.reason for f in failures] == ["overlap は 0〜64 の整数にしてください"]


@pytest.mark.parametrize("overlap", [0, 16, 64])
def test_scan_overlap_edges_are_accepted(custom_dir, overlap):
    _onnx(custom_dir, "edge")
    _sidecar(custom_dir, "edge", {"overlap": overlap})
    models, failures = cm.scan()
    assert failures == []
    assert models[0].overlap == overlap


def test_scan_ignores_non_onnx(custom_dir):
    (custom_dir / "note.txt").write_text("x", encoding="utf-8")
    (custom_dir / "photo.png").write_bytes(b"x")
    models, failures = cm.scan()
    assert (models, failures) == ([], [])


def test_scan_sorted_by_name(custom_dir):
    _onnx(custom_dir, "c-model")
    _sidecar(custom_dir, "c-model", {"name": "B second"})
    _onnx(custom_dir, "a-model")
    _sidecar(custom_dir, "a-model", {"name": "A first"})
    models, failures = cm.scan()
    assert failures == []
    assert [m.key for m in models] == ["custom:a-model", "custom:c-model"]


def test_scan_empty_and_missing_folder(monkeypatch, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv(cm.CUSTOM_MODELS_ENV, str(empty))
    assert cm.scan() == ([], [])
    missing = tmp_path / "not-there-yet"
    monkeypatch.setenv(cm.CUSTOM_MODELS_ENV, str(missing))
    assert cm.scan() == ([], [])
    assert missing.is_dir()


def test_custom_models_dir_default_is_models_custom(monkeypatch, tmp_path):
    monkeypatch.delenv(cm.CUSTOM_MODELS_ENV, raising=False)
    # 既定の場所を触らないよう、ルートだけ一時場所へ向ける。
    monkeypatch.setattr("app.core.custom_models._app_root", lambda: tmp_path)
    folder = cm.custom_models_dir()
    assert folder == tmp_path / "models" / "custom"
    assert folder.is_dir()


# ------------------------------------------------- 解決・セッションの指定

def test_canonical_keeps_custom_key():
    assert canonical_helper_model("custom:mymodel") == "custom:mymodel"
    assert canonical_helper_model("custom:日本語モデル") == "custom:日本語モデル"


def test_session_spec_returns_custom_file(custom_dir):
    onnx = _onnx(custom_dir, "mymodel")
    _sidecar(custom_dir, "mymodel", {"overlap": 32})
    settings = UpscaleSettings(backend=UpscaleBackend.WINML_GPU, model="custom:mymodel")
    backend, _tile, path = helper_backend._session_spec(settings, 640, 360)
    assert backend == UpscaleBackend.WINML_GPU
    assert path == onnx.resolve()
    assert helper_backend.overlap_for_model("custom:mymodel") == 32
    assert helper_backend.overlap_for_model("custom:gone") == 16


def test_resolve_model_returns_custom_file(custom_dir):
    onnx = _onnx(custom_dir, "mymodel")
    assert (helper_backend._resolve_model(
        UpscaleBackend.WINML_GPU, "custom:mymodel", 512) == onnx.resolve())
    with pytest.raises(cm.CustomModelLoadFailed):
        helper_backend._resolve_model(
            UpscaleBackend.WINML_GPU, "custom:gone", 256)


class _FakeServeClient:
    """READY の scale/tile だけ返す偽物。ready=(scale, tile_w, tile_h)。"""

    ready: tuple[int, int, int] = (4, 256, 256)
    last_command: list = []
    fail_connect: Exception | None = None

    def __init__(self, command, workdir, env=None, log=None, creationflags=0) -> None:
        type(self).last_command = list(command)

    def connect(self, timeout=None, progress=None, cancel=None) -> None:
        if type(self).fail_connect is not None:
            raise type(self).fail_connect
        self.scale, self.tile_w, self.tile_h = type(self).ready

    def close(self, force=False) -> None:
        return None


@pytest.fixture
def _fake_gpu(monkeypatch):
    monkeypatch.setattr(
        helper_backend, "_winml_helper", lambda: Path("winml-sr.exe"))
    monkeypatch.setattr(helper_backend, "ServeClient", _FakeServeClient)
    _FakeServeClient.ready = (4, 256, 256)
    _FakeServeClient.fail_connect = None
    return _FakeServeClient


def test_npu_request_runs_on_gpu_with_message(custom_dir, _fake_gpu):
    _onnx(custom_dir, "mymodel")
    _sidecar(custom_dir, "mymodel", {"overlap": 24})
    settings = UpscaleSettings(
        backend=UpscaleBackend.NPU_NATIVE, model="custom:mymodel")
    messages: list[str] = []
    session = helper_backend.open_session(
        settings, 640, 360, progress=lambda _f, msg: messages.append(msg))
    assert session.backend == UpscaleBackend.WINML_GPU
    assert messages.count("追加したモデルは GPU で実行します") == 1
    command = _FakeServeClient.last_command
    assert command[command.index("--overlap") + 1] == "24"
    assert "--seam-template" not in command


def test_no_gpu_raises_needs_gpu_without_vulkan_fallback(
        custom_dir, monkeypatch, tmp_path):
    from app.core import upscaler
    from app.core.helper_backend import HelperBackendUnavailable

    _onnx(custom_dir, "mymodel")
    monkeypatch.setattr(
        helper_backend, "_winml_helper",
        lambda: (_ for _ in ()).throw(HelperBackendUnavailable("no gpu")))
    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU, model="custom:mymodel")
    with pytest.raises(cm.CustomModelNeedsGpu) as exc_info:
        helper_backend.open_session(settings, 64, 48)
    assert str(exc_info.value) == \
        "追加したモデルは GPU（DirectML）でだけ実行できます。"
    # upscaler 経由でも Vulkan へは切り替わらない（切り替われば別の例外になる）。
    src = _make_png(tmp_path / "a.png")
    with pytest.raises(cm.CustomModelNeedsGpu):
        upscaler.upscale_image(str(src), str(tmp_path / "o.png"), settings)


def test_vulkan_backend_raises_needs_gpu(custom_dir):
    _onnx(custom_dir, "mymodel")
    settings = UpscaleSettings(
        backend=UpscaleBackend.VULKAN, model="custom:mymodel")
    with pytest.raises(cm.CustomModelNeedsGpu):
        helper_backend._session_spec(settings, 64, 48)


def test_scale_2_ready_raises_load_failed(custom_dir, _fake_gpu):
    _onnx(custom_dir, "mymodel")
    _sidecar(custom_dir, "mymodel", {"name": "My Model"})
    _FakeServeClient.ready = (2, 256, 256)
    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU, model="custom:mymodel")
    with pytest.raises(cm.CustomModelLoadFailed) as exc_info:
        helper_backend.open_session(settings, 64, 48)
    assert str(exc_info.value).startswith(
        "このモデルを読み込めませんでした: My Model。")


def test_open_failure_raises_load_failed(custom_dir, _fake_gpu):
    from app.core.serve_client import ServeClientError

    _onnx(custom_dir, "mymodel")
    _FakeServeClient.fail_connect = ServeClientError("helper died")
    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU, model="custom:mymodel")
    with pytest.raises(cm.CustomModelLoadFailed) as exc_info:
        helper_backend.open_session(settings, 64, 48)
    assert "このモデルを読み込めませんでした: mymodel。" in str(exc_info.value)


def test_engine_rejects_video_only_model_for_videos(custom_dir, tmp_path):
    from app.core import engine
    from app.core.jobs import Job, JobKind

    _onnx(custom_dir, "stills")
    _sidecar(custom_dir, "stills", {"video": False})
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    job = Job(input_path=clip, kind=JobKind.VIDEO)
    settings = UpscaleSettings(
        backend=UpscaleBackend.WINML_GPU, model="custom:stills")
    with pytest.raises(cm.CustomModelImagesOnly) as exc_info:
        engine.process_job(job, settings)
    assert str(exc_info.value) == "このモデルは動画には使えません: stills"


# ------------------------------------------------------------------- GUI

@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
    application.processEvents()


@pytest.fixture
def gui_models(custom_dir):
    _onnx(custom_dir, "alpha")
    _onnx(custom_dir, "beta")
    _sidecar(custom_dir, "beta", {"name": "Beta Model", "video": False})
    _onnx(custom_dir, "broken")
    (custom_dir / "broken.json").write_text("{壊れた", encoding="utf-8")
    return custom_dir


def test_gui_lists_custom_models_with_separator(qt_app, gui_models):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    qt_app.processEvents()
    try:
        image_datas = [
            win.image_model_combo.itemData(i)
            for i in range(win.image_model_combo.count())]
        # 内蔵の後ろに区切り線を入れて追加モデルが並ぶ。
        # （区切り線は itemData を持たない Qt の separator。空文字で見分ける）
        assert "custom:alpha" in image_datas
        assert "custom:beta" in image_datas
        texts = [
            win.image_model_combo.itemText(i)
            for i in range(win.image_model_combo.count())]
        alpha_index = image_datas.index("custom:alpha")
        assert texts[alpha_index - 1] == ""
        assert image_datas[alpha_index - 1] is None
        alpha_text = texts[image_datas.index("custom:alpha")]
        assert alpha_text == "alpha（追加）"
        assert texts[image_datas.index("custom:beta")] == "Beta Model（追加）"
        # video: false は動画用の一覧に出ない。
        video_datas = [
            win.video_model_combo.itemData(i)
            for i in range(win.video_model_combo.count())]
        assert "custom:alpha" in video_datas
        assert "custom:beta" not in video_datas
        # 右列の説明行は追加モデルの文言になる。
        win.image_model_combo.setCurrentIndex(
            win.image_model_combo.findData("custom:alpha"))
        qt_app.processEvents()
        assert win.image_model_combo.toolTip() == "追加したモデル・GPU で実行"
    finally:
        win.close()
        qt_app.processEvents()


def test_gui_settings_drawer_shows_custom_section(qt_app, gui_models):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    qt_app.processEvents()
    try:
        drawer = win.drawer
        labels = [w.text() for w in drawer.findChildren(QLabel)]
        assert "追加モデル" in labels
        assert "2 個のモデルを読み込みました" in drawer.custom_count_label.text()
        assert str(gui_models) in drawer.custom_folder_label.text()
        assert drawer.custom_open_button.text() == "フォルダを開く"
        assert "読み込めませんでした: broken.onnx（設定ファイルを読めません）" in labels
    finally:
        win.close()
        qt_app.processEvents()


def test_gui_drawer_empty_folder_shows_none_yet(qt_app, custom_dir):
    from app.gui.main_window import MainWindow

    win = MainWindow()
    qt_app.processEvents()
    try:
        assert win.drawer.custom_count_label.text() == "まだありません"
        labels = [w.text() for w in win.drawer.findChildren(QLabel)]
        assert not [text for text in labels if text.startswith("読み込めませんでした")]
    finally:
        win.close()
        qt_app.processEvents()


# ------------------------------------------------------------------- CLI

@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """CLI テストは英語・GPU 解決・一時フォルダの追加モデルにする。"""
    import tempfile

    from app.core import binaries

    folder = tmp_path / "custom-models"
    folder.mkdir()
    monkeypatch.setenv(cm.CUSTOM_MODELS_ENV, str(folder))
    monkeypatch.setenv("TOGU_LANG", "en")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        helper_backend, "_winml_helper", lambda: Path("C:/fake/winml-sr.exe"))
    monkeypatch.setattr(binaries, "ffmpeg_exe", lambda: "C:/fake/ffmpeg.exe")
    monkeypatch.setattr(binaries, "ffprobe_exe", lambda: "C:/fake/ffprobe.exe")
    monkeypatch.setattr(
        binaries, "available_interpolation_models",
        lambda: ["rife-v4.6", binaries.FILM_MODEL])
    _onnx(folder, "alpha")
    _onnx(folder, "beta")
    _sidecar(folder, "beta", {"name": "Beta Model", "video": False})
    return folder


def _run(argv, capsys):
    code = cli.main(argv)
    out, err = capsys.readouterr()
    return code, out, err


def _json_out(out: str) -> dict:
    import json as _json

    assert out.strip().count("\n") == 0, f"stdout is not one JSON: {out!r}"
    return _json.loads(out)


def test_cli_models_lists_custom(cli_env, capsys):
    code, out, _err = _run(["models", "--backend", "gpu", "--json"], capsys)
    assert code == 0
    payload = _json_out(out)
    customs = [e for e in payload["upscale"] if e["key"].startswith("custom:")]
    assert [e["key"] for e in customs] == ["custom:alpha", "custom:beta"]
    # 内蔵の後ろに並ぶ。
    assert payload["upscale"].index(customs[0]) >= 5
    by_key = {e["key"]: e for e in customs}
    assert by_key["custom:alpha"]["name"] == "alpha (added)"
    assert by_key["custom:beta"]["name"] == "Beta Model (added)"
    assert by_key["custom:alpha"]["video"] is True
    assert by_key["custom:beta"]["video"] is False
    for entry in customs:
        assert entry["custom"] is True
        assert entry["available"] is True
        assert entry["scales"] == [4]
        assert entry["note"] == "Added model · Runs on the GPU"


def test_cli_models_hides_custom_on_vulkan(cli_env, capsys):
    code, out, _err = _run(["models", "--backend", "vulkan", "--json"], capsys)
    assert code == 0
    keys = [e["key"] for e in _json_out(out)["upscale"]]
    assert not [k for k in keys if k.startswith("custom:")]


def test_cli_unknown_custom_key_is_usage_error(cli_env, capsys, tmp_path):
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--model", "custom:gone", "--json"], capsys)
    assert code == 2
    payload = _json_out(out)
    assert payload["error"]["code"] == "model_unknown"


def test_cli_vulkan_with_custom_needs_gpu(cli_env, capsys, tmp_path):
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--model", "custom:alpha",
         "--backend", "vulkan", "--json"], capsys)
    assert code == 2
    payload = _json_out(out)
    assert payload["error"]["code"] == "custom_model_needs_gpu"
    assert payload["error"]["fix"]


def test_cli_run_image_with_custom(cli_env, monkeypatch, tmp_path, capsys):
    from app.core import upscaler

    def fake(in_path, out_path, settings, progress=None, cancel=None):
        assert settings.model == "custom:alpha"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(Path(in_path).read_bytes())

    monkeypatch.setattr(upscaler, "upscale_image", fake)
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--model", "custom:alpha", "--json"], capsys)
    assert code == 0
    result = _json_out(out)["results"][0]
    assert result["status"] == "done"
    assert result["settings"]["model"] == "custom:alpha"


def test_cli_run_video_with_images_only_custom(cli_env, tmp_path, capsys):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fake-video")
    code, out, _err = _run(
        ["run", str(clip), "--model", "custom:beta", "--json"], capsys)
    assert code == 2
    payload = _json_out(out)
    assert payload["error"]["code"] == "custom_model_images_only"
    assert payload["error"]["message"] == \
        "This model cannot be used for videos: Beta Model"
    assert payload["error"]["fix"]


def test_cli_run_load_failed_code(cli_env, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        helper_backend, "open_session",
        lambda *_a, **_k: (_ for _ in ()).throw(
            cm.CustomModelLoadFailed("Could not load this model: alpha.")))
    src = _make_png(tmp_path / "a.png")
    code, out, _err = _run(
        ["run", str(src), "--model", "custom:alpha", "--json"], capsys)
    assert code == 1
    result = _json_out(out)["results"][0]
    assert result["status"] == "failed"
    assert result["error"]["code"] == "custom_model_load_failed"
    assert "ONNX" in result["error"]["fix"]


def test_cli_run_needs_gpu_code(cli_env, monkeypatch, tmp_path, capsys):
    from app.core.helper_backend import HelperBackendUnavailable

    monkeypatch.setattr(
        helper_backend, "_winml_helper",
        lambda: (_ for _ in ()).throw(HelperBackendUnavailable("no gpu")))
    src = _make_png(tmp_path / "a.png")
    # --backend gpu を明示する（auto だと GPU 不在で vulkan 解決になり、
    # 実行前の使い方の誤りとして同じ code で止まる）。
    code, out, _err = _run(
        ["run", str(src), "--model", "custom:alpha",
         "--backend", "gpu", "--json"], capsys)
    assert code == 1
    result = _json_out(out)["results"][0]
    assert result["status"] == "failed"
    assert result["error"]["code"] == "custom_model_needs_gpu"
    assert result["error"]["fix"]


def test_cli_quick_check_with_custom(cli_env, monkeypatch, tmp_path, capsys):
    from app.core import upscaler

    def fake(in_path, out_path, _settings, progress=None, cancel=None):
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(Path(in_path).read_bytes())

    monkeypatch.setattr(upscaler, "upscale_image", fake)
    src = _make_png(tmp_path / "a.png", (100, 80))
    out = tmp_path / "check.png"
    code, out_s, _err = _run(
        ["quick-check", str(src), "--out", str(out),
         "--model", "custom:alpha", "--json"], capsys)
    assert code == 0
    payload = _json_out(out_s)
    assert payload["settings"]["model"] == "custom:alpha"
    assert (payload["width"], payload["height"]) == (100, 80)
