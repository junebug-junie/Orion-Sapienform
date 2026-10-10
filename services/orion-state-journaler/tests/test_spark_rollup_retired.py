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


def test_run_loop_never_subscribes_or_touches_postgres() -> None:
    chassis = build_chassis()
    chassis.bus = MagicMock()

    async def _drive() -> None:
        task = asyncio.create_task(chassis._run())
        await asyncio.sleep(0.05)
        chassis._stop.set()
        await asyncio.wait_for(task, timeout=1.0)

    with patch.dict("sys.modules", {"asyncpg": MagicMock()}) as mods:
        asyncio.run(_drive())
        assert not mods["asyncpg"].connect.called
    assert not chassis.bus.subscribe.called
    assert not chassis.bus.publish.called


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
