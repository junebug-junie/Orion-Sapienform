"""Governor wiring for the warm chat pool (spec L5): shipped ON, settings -> pool config, health surface."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

_SERVICE = Path(__file__).resolve().parents[1]
_KEYS = (
    "HARNESS_FCC_CHAT_WARM_POOL_ENABLED",
    "HARNESS_FCC_CHAT_WARM_POOL_SIZE",
    "HARNESS_FCC_CHAT_WARM_POOL_MAX_TURNS",
    "HARNESS_FCC_CHAT_WARM_POOL_MAX_AGE_SEC",
    "HARNESS_FCC_CHAT_WARM_POOL_MODEL_LABEL",
    "HARNESS_FCC_CHAT_WARM_POOL_RELAY_PORT",
    "HARNESS_FCC_CHAT_WARM_POOL_SPAWN_TIMEOUT_SEC",
    "HARNESS_FCC_CHAT_WARM_POOL_CLEAR_TIMEOUT_SEC",
)


def test_pool_ships_on_in_every_config_surface() -> None:
    from app.settings import HarnessGovernorSettings

    field = HarnessGovernorSettings.model_fields["harness_fcc_chat_warm_pool_enabled"]
    assert field.default is True
    example = (_SERVICE / ".env_example").read_text()
    compose = (_SERVICE / "docker-compose.yml").read_text()
    assert re.search(r"^HARNESS_FCC_CHAT_WARM_POOL_ENABLED=true$", example, re.M)
    assert "HARNESS_FCC_CHAT_WARM_POOL_ENABLED=${HARNESS_FCC_CHAT_WARM_POOL_ENABLED:-true}" in compose
    for key in _KEYS:
        assert re.search(rf"^{key}=", example, re.M), key
        assert f"{key}=${{{key}:-" in compose, key


def test_start_passes_settings_into_pool_config(monkeypatch: Any) -> None:
    import app.main as main

    captured: dict[str, Any] = {}

    async def fake_start(config):
        captured["config"] = config

    monkeypatch.setattr(main, "start_warm_pool", fake_start)
    monkeypatch.setattr(main.settings, "harness_fcc_chat_warm_pool_size", 2)
    monkeypatch.setattr(main.settings, "harness_fcc_chat_warm_pool_max_turns", 7)
    monkeypatch.setattr(main.settings, "harness_fcc_chat_warm_pool_relay_port", 7999)
    asyncio.run(main.start_fcc_warm_pool())
    cfg = captured["config"]
    assert (cfg.size, cfg.max_turns, cfg.relay_port) == (2, 7, 7999)


def test_start_failure_is_logged_not_raised(monkeypatch: Any, caplog: Any) -> None:
    import app.main as main

    async def boom(_config):
        raise OSError("address in use")

    monkeypatch.setattr(main, "start_warm_pool", boom)
    asyncio.run(main.start_fcc_warm_pool())
    assert any("fcc_warm_pool_start_failed" in r.getMessage() for r in caplog.records)


def test_health_reports_pool_absent_as_null() -> None:
    import app.main as main

    assert main._warm_pool_status() is None
