from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_settings_module():
    module_path = Path("services/orion-sql-writer/app/settings.py")
    spec = importlib.util.spec_from_file_location("sql_writer_settings_phase21", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_feedback_wiring_exists_in_settings_env_and_compose() -> None:
    settings_module = _load_settings_module()
    assert "chat.response.feedback.v1" in settings_module.DEFAULT_ROUTE_MAP

    cfg = settings_module.Settings()
    assert "orion:chat:response:feedback" in cfg.sql_writer_subscribe_channels

    env_example = Path("services/orion-sql-writer/.env_example").read_text(encoding="utf-8")
    compose = Path("services/orion-sql-writer/docker-compose.yml").read_text(encoding="utf-8")

    assert "orion:chat:response:feedback" in env_example
    assert "chat.response.feedback.v1\":\"ChatResponseFeedbackSQL" in env_example
    assert "orion:chat:response:feedback" in compose


def test_feedback_wiring_exists_in_bus_and_schema_registry() -> None:
    channels = Path("orion/bus/channels.yaml").read_text(encoding="utf-8")
    registry = Path("orion/schemas/registry.py").read_text(encoding="utf-8")

    assert "orion:chat:response:feedback" in channels
    assert "chat.response.feedback.v1" in channels
    assert "ChatResponseFeedbackV1" in registry
