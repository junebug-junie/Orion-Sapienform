"""Tests for Hub cabinet ambient latest and history APIs."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
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

from scripts import cabinet_ambient_routes  # noqa: E402


NOW = datetime(2026, 8, 26, 3, 0, 5, tzinfo=timezone.utc)


def _snapshot(**overrides) -> dict:
    payload = {
        "schema": "orion.ambient_audio.v1",
        "status": "ok",
        "received_at": "2026-08-26T03:00:00Z",
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": 0.5,
        "sample_rate": 16000,
        "channels": 1,
        "rms": 5055.4,
        "peak": 19725,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    path = tmp_path / "latest.json"
    monkeypatch.setattr(cabinet_ambient_routes.settings, "AMBIENT_AUDIO_PATH", str(path))
    monkeypatch.setattr(cabinet_ambient_routes.settings, "AMBIENT_AUDIO_STALE_AFTER_SEC", 5.0)
    monkeypatch.setattr(cabinet_ambient_routes.settings, "CABINET_AMBIENT_HISTORY_NODE", "athena")
    monkeypatch.setattr(cabinet_ambient_routes.settings, "CABINET_AMBIENT_HISTORY_MAX_POINTS", 800)
    monkeypatch.setattr(cabinet_ambient_routes, "_now_utc", lambda: NOW)

    async def no_rows(*, node: str, hours: int):
        assert node == "athena"
        assert hours == 24
        return []

    monkeypatch.setattr(cabinet_ambient_routes, "_history_query", no_rows)
    app = FastAPI()
    app.include_router(cabinet_ambient_routes.router)
    return TestClient(app), path


def test_parse_window_accepts_only_supported_windows():
    assert cabinet_ambient_routes.parse_window("24h") == 24
    assert cabinet_ambient_routes.parse_window("3d") == 72
    assert cabinet_ambient_routes.parse_window("7d") == 168
    with pytest.raises(ValueError):
        cabinet_ambient_routes.parse_window("1h")


def test_rows_to_points_omits_missing_optional_values_instead_of_zero_filling():
    rows = [
        {
            "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
            "rms": "7457.8",
            "peak": None,
            "activity": None,
        }
    ]
    assert cabinet_ambient_routes.rows_to_points(rows) == [
        {"t": "2026-08-26T02:54:09Z", "rms": 7457.8}
    ]


def test_rows_to_points_preserves_peak_as_integer():
    points = cabinet_ambient_routes.rows_to_points(
        [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "rms": 7457.8,
                "peak": "19148",
                "activity": 0.3,
            }
        ]
    )
    assert points[0]["peak"] == 19148
    assert isinstance(points[0]["peak"], int)


def test_rows_to_points_accepts_float_peak_from_biometrics_json():
    """Biometrics stores cabinet_ambient_peak as float; Postgres ->> yields '16213.0'."""
    points = cabinet_ambient_routes.rows_to_points(
        [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "rms": 4132.5,
                "peak": 16213.0,
                "activity": 0.3,
            }
        ]
    )
    assert points[0]["peak"] == 16213


def test_rows_to_points_accepts_sql_writer_varchar_timestamp():
    points = cabinet_ambient_routes.rows_to_points(
        [
            {
                "t": "2026-08-26 02:41:42.067071+00",
                "rms": 3806.35,
                "peak": 16213.0,
                "activity": 0.3,
            }
        ]
    )
    assert points[0]["t"] == "2026-08-26T02:41:42.067071Z"


def test_parse_db_timestamp_pads_variable_fraction_for_python310():
    parsed = cabinet_ambient_routes._parse_db_timestamp("2026-08-26 02:46:53.36052+00")
    assert parsed.isoformat() == "2026-08-26T02:46:53.360520+00:00"


def test_downsample_points_averages_each_bucket_and_respects_cap():
    points = [
        {"t": f"2026-08-26T00:00:0{i}Z", "rms": float(i), "activity": i / 10}
        for i in range(6)
    ]
    sampled = cabinet_ambient_routes.downsample_points(points, 3)
    assert sampled == [
        {"t": "2026-08-26T00:00:00Z", "rms": 0.5, "activity": 0.05},
        {"t": "2026-08-26T00:00:02Z", "rms": 2.5, "activity": 0.25},
        {"t": "2026-08-26T00:00:04Z", "rms": 4.5, "activity": 0.45},
    ]


def test_history_query_uses_index_compatible_timestamp_range(monkeypatch):
    captured = {}

    class FakeConnection:
        async def fetch(self, sql, *args):
            captured["sql"] = sql
            captured["args"] = args
            return []

        async def close(self):
            captured["closed"] = True

    class FakeAsyncpg:
        @staticmethod
        async def connect(*, dsn):
            captured["dsn"] = dsn
            return FakeConnection()

    monkeypatch.setenv("DATABASE_URL", "postgresql://example/db")
    monkeypatch.setitem(sys.modules, "asyncpg", FakeAsyncpg)
    monkeypatch.setattr(cabinet_ambient_routes, "_now_utc", lambda: NOW)

    asyncio.run(cabinet_ambient_routes.query_history_rows(node="athena", hours=72))

    assert captured["args"] == ("athena", "2026-08-23T03:00:05Z")
    assert "timestamp >= $2" in captured["sql"]
    assert "ORDER BY timestamp ASC" in captured["sql"]
    assert "timestamp::timestamptz" not in captured["sql"]
    assert "cabinet_ambient_peak')::double precision" in captured["sql"]
    assert "cabinet_ambient_peak')::bigint" not in captured["sql"]
    assert captured["closed"] is True


def test_latest_missing_returns_ok_false(client):
    tc, path = client
    assert not path.exists()
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body == {"ok": False, "age_sec": None, "snapshot": None}


def test_latest_fresh_valid_snapshot_returns_age(client):
    tc, path = client
    payload = _snapshot()
    path.write_text(json.dumps(payload), encoding="utf-8")
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body["ok"] is True
    assert body["age_sec"] == pytest.approx(5.0)
    assert body["snapshot"] == payload


def test_latest_stale_keeps_last_snapshot(client):
    tc, path = client
    payload = _snapshot(received_at="2026-08-26T02:59:50Z")
    path.write_text(json.dumps(payload), encoding="utf-8")
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body["ok"] is False
    assert body["age_sec"] == pytest.approx(15.0)
    assert body["snapshot"] == payload


def test_latest_invalid_schema_is_unreadable(client):
    tc, path = client
    path.write_text(json.dumps(_snapshot(schema="wrong.v1")), encoding="utf-8")
    body = tc.get("/api/cabinet/ambient/latest").json()
    assert body == {"ok": False, "age_sec": None, "snapshot": None}


def test_history_defaults_to_24h_and_returns_empty_points(client):
    tc, _path = client
    body = tc.get("/api/cabinet/ambient/history").json()
    assert body == {
        "ok": True,
        "node": "athena",
        "window": "24h",
        "grain_sec": 30,
        "points": [],
        "stats": {
            "n_raw": 0,
            "n": 0,
            "rms_min": None,
            "rms_max": None,
            "activity_max": None,
        },
    }


def test_history_uses_query_rows_and_reports_stats(client, monkeypatch):
    tc, _path = client

    async def rows(*, node: str, hours: int):
        assert (node, hours) == ("athena", 72)
        return [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "rms": 1200.0,
                "peak": 19000,
                "activity": 0.2,
            },
            {
                "t": datetime(2026, 8, 26, 2, 54, 39, tzinfo=timezone.utc),
                "rms": 9000.0,
                "peak": 20000,
                "activity": 0.85,
            },
        ]

    monkeypatch.setattr(cabinet_ambient_routes, "_history_query", rows)
    body = tc.get("/api/cabinet/ambient/history?window=3d").json()
    assert body["ok"] is True
    assert body["window"] == "3d"
    assert len(body["points"]) == 2
    assert body["stats"] == {
        "n_raw": 2,
        "n": 2,
        "rms_min": 1200.0,
        "rms_max": 9000.0,
        "activity_max": 0.85,
    }


def test_history_invalid_window_returns_400(client):
    tc, _path = client
    response = tc.get("/api/cabinet/ambient/history?window=1h")
    assert response.status_code == 400


def test_history_db_failure_returns_ok_false(client, monkeypatch):
    tc, _path = client

    async def failed(*, node: str, hours: int):
        raise OSError("db unavailable")

    monkeypatch.setattr(cabinet_ambient_routes, "_history_query", failed)
    body = tc.get("/api/cabinet/ambient/history").json()
    assert body["ok"] is False
    assert body["points"] == []
    assert body["error"] == "ambient_history_unavailable"


def test_router_registered_on_api_routes():
    from scripts import api_routes

    paths = {getattr(route, "path", None) for route in api_routes.router.routes}
    assert "/api/cabinet/ambient/latest" in paths
    assert "/api/cabinet/ambient/history" in paths
