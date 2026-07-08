"""Regression: town speech must not default to legacy cortex intake."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_embodiment_settings():
    path = Path(__file__).resolve().parents[1] / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("embodiment_settings", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.Settings


def test_speech_defaults_use_chat_lane_and_skip_unified() -> None:
    settings_cls = _load_embodiment_settings()
    fields = settings_cls.model_fields
    assert fields["cortex_request_channel"].default == "orion:cortex:exec:request:chat"
    assert fields["speech_unified_enabled"].default is False
    assert fields["speech_verb"].default == "chat_quick"
