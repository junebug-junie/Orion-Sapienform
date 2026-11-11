"""Tests for the Hub Cabinet / Office climate overlay API."""

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

from scripts import cabinet_climate_routes  # noqa: E402

T0 = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)


def _client(monkeypatch, results):
    calls = []

    async def fake_query(*, hours: int, grain_sec: int):
        calls.append((hours, grain_sec))
        if isinstance(results, Exception):
            raise results
        return results

    monkeypatch.setattr(cabinet_climate_routes, "_series_query", fake_query)
    app = FastAPI()
    app.include_router(cabinet_climate_routes.router)
    return TestClient(app), calls


def test_router_is_mounted_by_hub_api_routes():
    text = (HUB_ROOT / "scripts" / "api_routes.py").read_text(encoding="utf-8")
    assert "router.include_router(cabinet_climate_router)" in text
    assert cabinet_climate_routes.router.prefix == "/api/cabinet/climate"


def test_rows_to_points_skips_absent_values_and_keeps_sources():
    rows = [
        {"t": T0, "v": 21.5},
        {"t": T0 + timedelta(minutes=5), "v": None},
        {"t": T0 + timedelta(minutes=10), "v": 1500.0, "sources": 4},
    ]
    assert cabinet_climate_routes.rows_to_points(rows) == [
        {"t": "2026-10-28T12:00:00Z", "v": 21.5},
        {"t": "2026-10-28T12:10:00Z", "v": 1500.0, "sources": 4},
    ]


def test_overlay_returns_every_series_with_window_grain(monkeypatch):
    results = {
        "office_temp_c": [{"t": T0, "v": 26.1}],
        "cabinet_temp_c": [{"t": T0, "v": 28.2}],
        "closet_temp_c": [{"t": T0, "v": 27.1}],
        "outdoor_temp_c": [{"t": T0, "v": 14.3}],
        "orion_watts": [{"t": T0, "v": 1570.0, "sources": 4}],
    }
    client, calls = _client(monkeypatch, results)
    body = client.get("/api/cabinet/climate/overlay?window=3d").json()
    assert calls == [(72, 900)]
    assert body["ok"] is True
    assert body["errors"] == {}
    assert body["grain_sec"] == 900
    assert body["series_grain_sec"]["outdoor_temp_c"] == 3600
    assert body["series_grain_sec"]["orion_watts"] == 900
    assert body["series"]["orion_watts"] == [{"t": "2026-10-28T12:00:00Z", "v": 1570.0, "sources": 4}]
    assert body["series"]["closet_temp_c"] == [{"t": "2026-10-28T12:00:00Z", "v": 27.1}]
    assert body["series_grain_sec"]["closet_temp_c"] == 900
    assert set(body["series"]) == set(cabinet_climate_routes.SERIES)


def test_one_failed_series_empties_only_that_series(monkeypatch):
    results = {
        "office_temp_c": [{"t": T0, "v": 26.1}],
        "cabinet_temp_c": RuntimeError("boom"),
        "closet_temp_c": [{"t": T0, "v": 27.1}],
        "outdoor_temp_c": [],
        "orion_watts": [{"t": T0, "v": 1570.0, "sources": 3}],
    }
    client, _ = _client(monkeypatch, results)
    body = client.get("/api/cabinet/climate/overlay").json()
    assert body["ok"] is True
    assert body["errors"] == {"cabinet_temp_c": "unavailable"}
    assert body["series"]["cabinet_temp_c"] == []
    assert body["series"]["outdoor_temp_c"] == []  # empty, not an error and not zero-filled


def test_database_down_is_not_ok(monkeypatch):
    client, _ = _client(monkeypatch, RuntimeError("DATABASE_URL is not configured"))
    body = client.get("/api/cabinet/climate/overlay?window=7d").json()
    assert body["ok"] is False
    assert all(points == [] for points in body["series"].values())
    assert set(body["errors"]) == set(cabinet_climate_routes.SERIES)


def test_unsupported_window_is_rejected(monkeypatch):
    client, _ = _client(monkeypatch, {})
    assert client.get("/api/cabinet/climate/overlay?window=1y").status_code == 400


def test_watts_sql_does_not_count_a_zero_watt_node_as_a_source():
    assert "chassis_watts')::double precision > 0" in cabinet_climate_routes._WATTS_SQL


def test_closet_series_is_closet_temperature_only():
    sql = cabinet_climate_routes._CLOSET_SQL
    assert "FROM closet_env_sample" in sql
    assert "avg(closet_temp_c)" in sql and "closet_temp_c IS NOT NULL" in sql
    assert "co2" not in sql.lower()


class _FakeConnection:
    def __init__(self, failures):
        self.failures = failures

    async def fetch(self, sql, *args):
        for marker, exc in self.failures.items():
            if marker in sql:
                raise exc
        return [{"t": T0, "v": 1.0}]

    async def close(self):
        pass


def _run_query_series(monkeypatch, failures):
    import asyncio

    import asyncpg

    async def fake_connect(dsn):
        return _FakeConnection(failures)

    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(asyncpg, "connect", fake_connect)
    return asyncio.run(cabinet_climate_routes.query_series_rows(hours=24, grain_sec=300))


def test_missing_closet_table_reads_as_no_rows_not_an_error(monkeypatch):
    import asyncpg

    out = _run_query_series(monkeypatch, {"closet_env_sample": asyncpg.exceptions.UndefinedTableError("missing")})
    assert out["closet_temp_c"] == []
    assert out["office_temp_c"] == [{"t": T0, "v": 1.0}]


def test_other_closet_failures_still_surface_as_errors(monkeypatch):
    out = _run_query_series(monkeypatch, {"closet_env_sample": RuntimeError("boom")})
    assert isinstance(out["closet_temp_c"], RuntimeError)
