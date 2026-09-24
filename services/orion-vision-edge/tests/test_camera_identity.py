"""The capture URL (with its password) never becomes the camera's identity.

Regression for the leak in the walkway spec: SOURCE was published as
camera_id and ended up as stream_id in substrate_perception_embedding_baseline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import Settings
from app.stream_filter import is_own_frame

URL = "rtsp://admin:hunter2@192.168.1.50:554/Preview_01_sub"


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, SOURCE=kw.pop("SOURCE", URL), **kw)


def test_camera_id_is_stream_name_not_source() -> None:
    s = _settings(STREAM_ID="walkway")
    assert s.camera_id == "walkway"
    assert "://" not in s.camera_id


def test_source_redacted_drops_password() -> None:
    red = _settings().source_redacted
    assert "hunter2" not in red and "admin" not in red
    assert red == "rtsp://192.168.1.50:554/Preview_01_sub"


def test_empty_source_refused() -> None:
    with pytest.raises(Exception):
        _settings(SOURCE="")


def test_no_module_publishes_settings_source() -> None:
    app_dir = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        f"{p.name}:{i}"
        for p in app_dir.glob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "camera_id=settings.SOURCE" in line.replace(" ", "")
    ]
    assert offenders == []


def test_instance_detects_only_its_own_stream() -> None:
    assert is_own_frame("walkway", "walkway")
    assert not is_own_frame("cam0", "walkway")
    assert not is_own_frame(None, "walkway")
    assert is_own_frame(None, "cam0")
