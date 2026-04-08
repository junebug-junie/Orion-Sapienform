from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_sql_writer_settings():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "app" / "settings.py"
    spec = importlib.util.spec_from_file_location("sql_writer_settings_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.settings


def test_chat_response_feedback_kind_and_channel_are_mapped() -> None:
    settings = _load_sql_writer_settings()
    assert settings.route_map.get("chat.response.feedback.v1") == "ChatResponseFeedbackSQL"
    assert "orion:chat:response:feedback" in settings.effective_subscribe_channels
