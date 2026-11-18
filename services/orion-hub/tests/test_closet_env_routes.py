"""Tests for Hub server-closet environment latest and history APIs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]

for _key, _val in (
    ("CHANNEL_VOICE_TRANSCRIPT", "orion:voice:transcript"),
    ("CHANNEL_VOICE_LLM", "orion:voice:llm"),
    ("CHANNEL_VOICE_TTS", "orion:voice:tts"),
    ("CHANNEL_COLLAPSE_INTAKE", "orion:collapse:intake"),
    ("CHANNEL_COLLAPSE_TRIAGE", "orion:collapse:triage"),
):
    os.environ.setdefault(_key, _val)


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
        if key == "app" or key.startswith("app."):
            del sys.modules[key]
    for path in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(path)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import closet_env_routes, office_env_routes  # noqa: E402
import asyncpg  # noqa: E402

NOW = datetime(2026, 10, 27, 20, 0, 30, tzinfo=timezone.utc)
SAMPLE_TS = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)


def _row(**overrides) -> dict:
    row = {
        "ts": SAMPLE_TS,
        "closet_co2_ppm": 812,
        "closet_temp_c": 27.3,
        "closet_humidity_pct": 40.0,
        "sensor_online": True,
        "stale": False,
        "sample_age_sec": 1.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(closet_env_routes.settings, "OFFICE_ENV_NODE", "pan-1")
    monkeypatch.setattr(closet_env_routes.settings, "OFFICE_ENV_STALE_AFTER_SEC", 90.0)
    monkeypatch.setattr(office_env_routes, "_now_utc", lambda: NOW)

    async def no_rows(*, node: str, hours: int):
        assert node == "pan-1"
        return []

    async def no_latest(*, node: str):
        assert node == "pan-1"
        return None

    monkeypatch.setattr(closet_env_routes, "_history_query", no_rows)
    monkeypatch.setattr(closet_env_routes, "_latest_query", no_latest)
    app = FastAPI()
    app.include_router(closet_env_routes.router)
    return TestClient(app)


def test_router_prefix():
    assert closet_env_routes.router.prefix == "/api/closet/env"


def test_latest_no_rows(client):
    assert client.get("/api/closet/env/latest").json() == {"ok": False, "age_sec": None, "sample": None}


def test_latest_fresh_row(client, monkeypatch):
    async def latest(*, node: str):
        return _row()

    monkeypatch.setattr(closet_env_routes, "_latest_query", latest)
    body = client.get("/api/closet/env/latest").json()
    assert body["ok"] is True
    assert body["sample"] == {
        "ts": "2026-10-27T20:00:00Z",
        "sensor_online": True,
        "closet_co2_ppm": 812.0,
        "closet_temp_c": 27.3,
        "closet_humidity_pct": 40.0,
    }
    assert body["sample_age_sec"] == 1.0


def test_latest_stale_row_has_no_readings(client, monkeypatch):
    async def latest(*, node: str):
        return _row(closet_co2_ppm=None, closet_temp_c=None, closet_humidity_pct=None, sensor_online=False, stale=True)

    monkeypatch.setattr(closet_env_routes, "_latest_query", latest)
    body = client.get("/api/closet/env/latest").json()
    assert body["ok"] is False and body["sensor_stale"] is True
    assert body["sample"] == {"ts": "2026-10-27T20:00:00Z", "sensor_online": False}


def test_latest_publisher_dead_hides_the_old_reading(client, monkeypatch):
    async def latest(*, node: str):
        return _row(ts=datetime(2026, 10, 27, 18, 0, 0, tzinfo=timezone.utc))

    monkeypatch.setattr(closet_env_routes, "_latest_query", latest)
    body = client.get("/api/closet/env/latest").json()
    assert body["ok"] is False and body["publisher_stale"] is True
    assert "closet_co2_ppm" not in body["sample"]


def test_history_points_and_stats(client, monkeypatch):
    async def rows(*, node: str, hours: int):
        assert hours == 72
        return [
            {"t": SAMPLE_TS, "closet_co2_ppm": 700, "closet_temp_c": 27.0, "closet_humidity_pct": None},
            {"t": SAMPLE_TS + timedelta(seconds=30), "closet_co2_ppm": 900, "closet_temp_c": None, "closet_humidity_pct": 41.0},
        ]

    monkeypatch.setattr(closet_env_routes, "_history_query", rows)
    body = client.get("/api/closet/env/history?window=3d").json()
    assert body["ok"] is True and body["window"] == "3d"
    assert body["points"][0] == {"t": "2026-10-27T20:00:00Z", "closet_co2_ppm": 700.0, "closet_temp_c": 27.0}
    stats = body["stats"]
    assert stats["co2_min"] == 700.0 and stats["co2_max"] == 900.0
    assert stats["temp_min"] == 27.0 and stats["humidity_max"] == 41.0
    assert "eco2_min" not in stats  # closet stats, not office stats


def test_history_rejects_unknown_window(client):
    assert client.get("/api/closet/env/history?window=1h").status_code == 400


def test_history_db_failure_is_ok_false(client, monkeypatch):
    async def boom(*, node: str, hours: int):
        raise RuntimeError("db down")

    monkeypatch.setattr(closet_env_routes, "_history_query", boom)
    body = client.get("/api/closet/env/history?window=24h").json()
    assert body["ok"] is False and body["error"] == "closet_env_history_unavailable"


# --- Deploy-order safety: closet_env_sample not created yet by the sql-writer ---


class _MissingTableConnection:
    async def fetchrow(self, sql, *args):
        raise asyncpg.exceptions.UndefinedTableError('relation "closet_env_sample" does not exist')

    fetch = fetchrow

    async def close(self):
        return None


@pytest.fixture
def missing_table(monkeypatch):
    async def connect(dsn):
        return _MissingTableConnection()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setattr(asyncpg, "connect", connect)


def test_missing_table_reads_as_no_samples_not_an_error(client, missing_table, monkeypatch):
    monkeypatch.setattr(closet_env_routes, "_latest_query", closet_env_routes.query_latest_row)
    monkeypatch.setattr(closet_env_routes, "_history_query", closet_env_routes.query_history_rows)
    latest = client.get("/api/closet/env/latest")
    assert latest.status_code == 200
    assert latest.json() == {"ok": False, "age_sec": None, "sample": None}
    history = client.get("/api/closet/env/history?window=24h")
    assert history.status_code == 200
    body = history.json()
    assert body["ok"] is True and body["points"] == [] and body["stats"]["n_raw"] == 0


def test_other_db_errors_still_surface_as_errors(client, monkeypatch):
    class _Down(_MissingTableConnection):
        async def fetchrow(self, sql, *args):
            raise asyncpg.exceptions.UndefinedColumnError("column x does not exist")

        fetch = fetchrow

    async def connect(dsn):
        return _Down()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setattr(asyncpg, "connect", connect)
    monkeypatch.setattr(closet_env_routes, "_latest_query", closet_env_routes.query_latest_row)
    body = client.get("/api/closet/env/latest").json()
    assert body["error"] == "closet_env_latest_unavailable"
