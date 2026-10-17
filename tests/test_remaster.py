"""Native-size restoration, backend routing, and failure guarantees."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core import engine, helper_backend, media, upscaler, video
from app.core.jobs import Job, JobKind
from app.core.settings import HELPER_MODEL_REMASTER, UpscaleBackend, UpscaleSettings


@pytest.mark.parametrize("backend", [UpscaleBackend.WINML_GPU, UpscaleBackend.NPU_NATIVE])
def test_remaster_uses_directml_at_native_resolution(monkeypatch, tmp_path, backend):
    model = tmp_path / "remaster.onnx"
    monkeypatch.setattr(helper_backend, "_resolve_model", lambda *args: model)
    settings = UpscaleSettings(model=HELPER_MODEL_REMASTER, backend=backend)
    assert settings.scale == 1
    assert settings.output_suffix() == "_Remaster_denoised"
    assert helper_backend._session_spec(settings, 731, 549) == (
        UpscaleBackend.WINML_GPU, 512, model)
    assert helper_backend.overlap_for_model(settings.model) == 64


@pytest.mark.parametrize("method", ["upscale_image", "upscale_folder"])
def test_remaster_failure_never_substitutes_upscaler(monkeypatch, tmp_path, method):
    def fail(*args, **kwargs):
        raise helper_backend.HelperBackendUnavailable("DML failed")

    def forbidden(*args, **kwargs):
        pytest.fail("Remaster must never fall back to a 4x upscaler")

    monkeypatch.setattr(helper_backend, method, fail)
    monkeypatch.setattr(upscaler, "_spawn", forbidden)
    with pytest.raises(helper_backend.HelperBackendUnavailable, match="DML failed"):
        getattr(upscaler, method)(str(tmp_path), str(tmp_path / "out"),
                                  UpscaleSettings(model=HELPER_MODEL_REMASTER))


def test_remaster_video_failure_preserves_existing_output(monkeypatch, tmp_path):
    src = tmp_path / "clip.mp4"
    src.touch()
    original = tmp_path / "upscaled" / "clip_Remaster_denoised.mp4"
    original.parent.mkdir()
    original.write_bytes(b"previous result")
    monkeypatch.setattr(media, "probe", lambda _: media.MediaInfo(
        kind="video", width=320, height=240, fps=24))

    def fail(_src, out, *args, **kwargs):
        Path(out).write_bytes(b"partial result")
        raise helper_backend.HelperBackendUnavailable("DML failed")

    monkeypatch.setattr(video, "upscale_video_piped", fail)
    with pytest.raises(helper_backend.HelperBackendUnavailable):
        engine.process_job(Job(src, JobKind.VIDEO), UpscaleSettings(
            model=HELPER_MODEL_REMASTER, overwrite=True))
    assert original.read_bytes() == b"previous result"
    assert list(original.parent.iterdir()) == [original]


@pytest.mark.parametrize("size,expected_format", [
    ((7680, 4320), "yuv420p"), ((321, 241), "yuv444p"),
])
def test_remaster_video_never_resizes_or_pads(size, expected_format):
    settings = UpscaleSettings(model=HELPER_MODEL_REMASTER, max_video_dim=(128, 128))
    filters, fitted, pixel_format = video._encoding_layout(settings, *size)
    assert "scale=" not in filters and "pad=" not in filters
    assert fitted is None
    assert pixel_format == expected_format


def test_remaster_preserves_alpha_and_odd_image_dimensions(monkeypatch, tmp_path):
    src = tmp_path / "alpha.png"
    image = Image.new("RGBA", (37, 29), (50, 100, 150, 71))
    image.putpixel((3, 4), (0, 0, 0, 0))
    image.save(src)

    class Session:
        scale = 1
        backend = UpscaleBackend.WINML_GPU

        def upscale(self, rgb, **kwargs):
            return np.full_like(rgb, 100)

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(helper_backend, "open_session", lambda *a, **kw: Session())
    out = engine.process_job(Job(src, JobKind.IMAGE), UpscaleSettings(model=HELPER_MODEL_REMASTER))
    assert out.name == "alpha_Remaster_denoised.png"
    with Image.open(out) as result:
        assert result.size == image.size
        assert result.getchannel("A").tobytes() == image.getchannel("A").tobytes()


def test_remaster_rejects_non_native_model(monkeypatch, tmp_path):
    monkeypatch.setattr(helper_backend, "_resolve_model", lambda *a: tmp_path / "wrong.onnx")
    monkeypatch.setattr(helper_backend, "_winml_helper", lambda: tmp_path / "helper.exe")
    closed = []

    class Client:
        scale = 4

        def __init__(self, *args, **kwargs):
            pass

        def connect(self, **kwargs):
            pass

        def close(self, **kwargs):
            closed.append(kwargs)

    monkeypatch.setattr(helper_backend, "ServeClient", Client)
    with pytest.raises(RuntimeError, match="等倍"):
        helper_backend.open_session(UpscaleSettings(model=HELPER_MODEL_REMASTER), 320, 240)
    assert closed == [{"force": True}]
