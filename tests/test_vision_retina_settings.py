from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")

SERVICE_ROOT = Path(__file__).resolve().parents[1] / "services" / "orion-vision-retina"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))


def _reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    for key in list(os.environ):
        if key.startswith("RETINA_") or key in {
            "ORION_BUS_URL",
            "ORION_BUS_ENFORCE_CATALOG",
            "CHANNEL_RETINA_PUB",
            "FRAME_STORAGE_DIR",
            "JPEG_QUALITY",
            "HEALTH_INTERVAL_SECONDS",
            "SOURCE_RECONNECT_SECONDS",
        }:
            monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.settings as mod

    importlib.reload(mod)
    return mod.Settings()


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch)
    assert s.SERVICE_NAME == "vision-retina"
    assert s.SERVICE_VERSION == "0.2.0"
    assert s.CHANNEL_RETINA_PUB == "orion:vision:frames"
    assert s.RETINA_SOURCE_TYPE == "folder"
    assert s.RETINA_SOURCE == "/mnt/telemetry/vision/intake"
    assert s.RETINA_FPS == 1.0
    assert s.FRAME_STORAGE_DIR == "/mnt/telemetry/vision/frames"


def test_retina_source_path_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(
        monkeypatch,
        RETINA_SOURCE_PATH="/tmp/legacy-intake",
    )
    assert s.RETINA_SOURCE == "/tmp/legacy-intake"


def test_retina_source_explicit_overrides_path_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(
        monkeypatch,
        RETINA_SOURCE="/tmp/explicit",
        RETINA_SOURCE_PATH="/tmp/legacy-intake",
    )
    assert s.RETINA_SOURCE == "/tmp/explicit"


def test_jpeg_quality_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch, JPEG_QUALITY="999")
    assert s.JPEG_QUALITY == 100
    s2 = _reload_settings(monkeypatch, JPEG_QUALITY="-5")
    assert s2.JPEG_QUALITY == 1


@pytest.mark.parametrize(
    ("env_key", "env_value"),
    [
        ("RETINA_WIDTH", ""),
        ("RETINA_HEIGHT", ""),
        ("RETINA_WIDTH", "   "),
        ("RETINA_HEIGHT", "   "),
    ],
)
def test_blank_retina_dimensions_coerced_to_none(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_value: str,
) -> None:
    """Docker compose passes empty RETINA_WIDTH/HEIGHT; must not crash Settings()."""
    s = _reload_settings(monkeypatch, **{env_key: env_value})
    assert getattr(s, env_key) is None


def test_blank_retina_width_and_height_together(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch, RETINA_WIDTH="", RETINA_HEIGHT="")
    assert s.RETINA_WIDTH is None
    assert s.RETINA_HEIGHT is None


def test_retina_dimensions_explicit_integers(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch, RETINA_WIDTH="640", RETINA_HEIGHT="480")
    assert s.RETINA_WIDTH == 640
    assert s.RETINA_HEIGHT == 480
