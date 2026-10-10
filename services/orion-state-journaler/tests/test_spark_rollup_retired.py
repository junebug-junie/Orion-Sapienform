"""Regression: the spark-state rollup stays retired.

orion-spark-introspector (the only real producer of orion:spark:state:snapshot)
was deleted 2026-07-28. This service kept rolling up that dead channel and
wrote all-zero rows into spark_state_rollups every 30s until 2026-10-10. These
tests fail if any part of that writer comes back: a bus subscription, a
Postgres write, the rollup settings, or a /rollups endpoint that serves the
frozen table as if it were current.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.main as journaler_main
from app.service import build_chassis
from app.settings import Settings
from orion.core.bus.bus_service_chassis import HeartbeatOnly

APP_DIR = Path(__file__).resolve().parents[1] / "app"
SERVICE_DIR = APP_DIR.parent


def test_chassis_is_heartbeat_only() -> None:
    chassis = build_chassis()
    assert type(chassis) is HeartbeatOnly


class _FakeBus:
    """Records every bus call; enough surface for BaseChassis.start_background()."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []
        self.subscribe = MagicMock(name="subscribe")

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def reconnect(self) -> None:
        return None

    async def publish(self, channel: str, env: object) -> None:
        self.published.append((channel, env))


def test_running_service_heartbeats_and_never_subscribes() -> None:
    """Drive the real start_background() path: the one job left is the
    SystemHealthV1 heartbeat; no subscription may come back."""
    with patch("app.service.settings.heartbeat_interval_sec", 0.01):
        chassis = build_chassis()
    assert chassis.cfg.heartbeat_interval_sec == 0.01
    bus = _FakeBus()
    chassis.bus = bus

    async def _drive() -> None:
        await chassis.start_background()
        await asyncio.sleep(0.1)
        await chassis.stop()

    asyncio.run(_drive())
    assert not bus.subscribe.called
    channels = {ch for ch, _ in bus.published}
    assert channels == {chassis.cfg.health_channel}, channels
    env = bus.published[0][1]
    assert env.kind == "system.health.v1"
    assert env.payload["service"] == "state-journaler"


def test_settings_carry_no_rollup_or_spark_keys() -> None:
    fields = set(Settings.model_fields)
    retired = {
        "channel_spark_state_snapshot",
        "channel_equilibrium_snapshot",
        "postgres_uri",
        "rollup_table",
        "windows_sec",
        "rollup_interval_sec",
        "retention_hours",
    }
    assert not (fields & retired), fields & retired


def test_no_source_reintroduces_the_rollup_writer() -> None:
    """Static backstop for code, env template and compose."""
    for path in APP_DIR.glob("*.py"):
        text = path.read_text()
        tree = ast.parse(text)
        imported = {
            name
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for name in ([n.module or ""] if isinstance(n, ast.ImportFrom) else []) + [a.name for a in n.names]
        }
        assert "asyncpg" not in imported, path
        assert "SparkStateSnapshotV1" not in imported, path
        assert "INSERT INTO" not in text.upper(), path
    for path in (SERVICE_DIR / ".env_example", SERVICE_DIR / "docker-compose.yml"):
        live = [ln for ln in path.read_text().splitlines() if not ln.lstrip().startswith("#")]
        for token in ("CHANNEL_SPARK_STATE_SNAPSHOT", "SPARK_ROLLUP_TABLE", "ROLLUP_", "POSTGRES_URI"):
            assert not any(token in ln for ln in live), (path, token)


def test_rollups_endpoint_is_marked_absent_not_served_from_frozen_table() -> None:
    with patch.object(journaler_main.chassis, "start_background"), patch.object(journaler_main.chassis, "stop"):
        with TestClient(journaler_main.app) as client:
            resp = client.get("/rollups", params={"window": 300, "hours": 24})
    assert resp.status_code == 410
    body = resp.json()
    assert body["retired"] is True
    assert "rows" not in body
