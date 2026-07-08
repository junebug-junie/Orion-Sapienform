from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_harness_settings():
    path = Path(__file__).resolve().parents[1] / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("harness_governor_settings", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.HarnessGovernorSettings


def test_finalize_cortex_rpc_defaults_to_background_lane(monkeypatch) -> None:
    monkeypatch.delenv("CHANNEL_CORTEX_EXEC_REQUEST", raising=False)
    monkeypatch.delenv("CORTEX_EXEC_REQUEST_CHANNEL", raising=False)
    settings_cls = _load_harness_settings()
    settings = settings_cls()
    assert settings.channel_cortex_exec_request == "orion:cortex:exec:request:background"
