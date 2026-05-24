from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT))


def _reload_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> object:
    for key in list(os.environ):
        if key.startswith(
            (
                "SERVICE_",
                "LOG_",
                "ORION_BUS_",
                "CHANNEL_",
                "ROUTER_",
                "DEFAULT_",
                "MAX_",
                "TASK_",
                "REQUIRE_",
                "DRY_",
            )
        ):
            monkeypatch.delenv(key, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.settings as mod

    importlib.reload(mod)
    return mod.Settings()


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _reload_settings(monkeypatch)
    assert s.SERVICE_NAME == "vision-frame-router"
    assert s.SERVICE_VERSION == "0.1.0"
    assert s.CHANNEL_FRAMES_IN == "orion:vision:frames"
    assert s.CHANNEL_HOST_INTAKE == "orion:exec:request:VisionHostService"
    assert s.CHANNEL_HOST_ARTIFACTS == "orion:vision:artifacts"
    assert s.CHANNEL_REPLY_PREFIX == "orion:vision:reply"
    assert s.CHANNEL_SYSTEM_HEALTH == "orion:system:health"
    assert s.ROUTER_ENABLED is True
    assert s.DEFAULT_TASK_TYPE == "retina_fast"
    assert s.DEFAULT_EVERY_N_FRAMES == 10
    assert s.MAX_INFLIGHT_TOTAL == 2
    assert s.DRY_RUN is False
