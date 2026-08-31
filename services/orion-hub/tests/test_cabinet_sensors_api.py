"""Tests for Hub GET /api/cabinet/sensors/latest (cabinet Nano snapshot)."""

from __future__ import annotations

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

# Required Hub Settings fields (no defaults) for import without a live .env.
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
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import cabinet_sensors_routes  # noqa: E402


NOW = datetime(2026, 8, 24, 12, 0, 5, tzinfo=timezone.utc)
FRESH_RECEIVED_AT = "2026-08-24T12:00:00.000Z"
STALE_AGE_RECEIVED_AT = "2026-08-24T11:59:40.000Z"


def _frame(**overrides):
    base = {
        "schema": "orion.sensor_frame.v1",
        "seq": 12,
        "uptime_ms": 1000,
        "environment": {
            "temp_c": 24.6,
            "humidity_pct": 45.0,
            "pressure_hpa": 900.0,
            "gas_resistance_ohm": 1000.0,
        },
        "uv": {"raw": 17.0, "als_raw": 67.0},
        "magnetic": {"x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0, "magnitude_ut": 78.01},
        "particulate": {"pm1_ug_m3": 2.0, "pm25_ug_m3": 4.0, "pm10_ug_m3": 5.0},
        "lidar": {"distance_mm": 438.0, "status": 0},
        "imu": {
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 9.80665,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
        },
    }
    base.update(overrides)
    return base


def _write_snapshot(path: Path, **overrides) -> dict:
    payload = {
        "status": "ok",
        "received_at": FRESH_RECEIVED_AT,
        "device": "/dev/serial/by-id/usb-Arduino_Nano_ESP32_test-if01",
        "frame": _frame(),
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_boot(path: Path, **overrides) -> dict:
    payload = {
        "schema": "orion.sensor_boot.v1",
        "i2c": {"primary": "A4/A5", "addresses": ["0x30", "0x53"]},
        "sensors": {"mmc5603": {"ok": True, "addr": "0x30"}},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    sensors_path = tmp_path / "latest.json"
    boot_path = tmp_path / "boot.json"
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_SENSORS_PATH", str(sensors_path))
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_BOOT_PATH", str(boot_path))
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_SENSORS_B_PATH", "")
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_BOOT_B_PATH", "")
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_SENSORS_STALE_AFTER_SEC", 10.0)
    monkeypatch.setattr(cabinet_sensors_routes, "_TRACKER", cabinet_sensors_routes.CabinetSensorTracker(
        cabinet_sensors_routes.CabinetPressureConfig()
    ))
    monkeypatch.setattr(cabinet_sensors_routes, "_now_utc", lambda: NOW)

    app = FastAPI()
    app.include_router(cabinet_sensors_routes.router)
    return TestClient(app), sensors_path, boot_path


def test_missing_snapshot_returns_ok_false_null_snapshot(client):
    tc, sensors_path, _boot = client
    assert not sensors_path.exists()

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["snapshot"] is None
    assert body["boot"] is None
    assert body["measurements"] == {}
    assert body["pressures"] == {}
    assert body["age_sec"] is None
    assert "cabinet_sensor_staleness" not in body["pressures"]


def test_unreadable_json_returns_ok_false_null_snapshot(client):
    tc, sensors_path, _boot = client
    sensors_path.write_text("{not json", encoding="utf-8")

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["snapshot"] is None
    assert body["measurements"] == {}
    assert body["pressures"] == {}


def test_fresh_frame_includes_magnetic_and_uv_measurements(client):
    tc, sensors_path, boot_path = client
    snapshot = _write_snapshot(sensors_path)
    boot = _write_boot(boot_path)

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["snapshot"]["frame"] == snapshot["frame"]
    assert body["sources"]["a"]["snapshot"] == snapshot
    assert body["boot"] == boot
    assert body["age_sec"] == pytest.approx(5.0)
    assert body["measurements"]["cabinet_magnetic_ut"] == pytest.approx(78.01)
    assert body["measurements"]["cabinet_als_raw"] == pytest.approx(67.0)
    assert body["measurements"]["cabinet_uv_raw"] == pytest.approx(17.0)
    assert body["pressures"]["cabinet_sensor_staleness"] == pytest.approx(0.0)
    assert "cabinet_em_activity" in body["pressures"]
    assert "cabinet_uv_activity" in body["pressures"]


def test_stale_status_returns_ok_false_but_keeps_snapshot(client):
    tc, sensors_path, _boot = client
    snapshot = _write_snapshot(sensors_path, status="stale")

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["snapshot"]["frame"] == snapshot["frame"]
    assert body["snapshot"]["status"] == "stale"
    assert body["measurements"] == {}
    assert body["pressures"]["cabinet_sensor_staleness"] == pytest.approx(1.0)


def test_stale_age_returns_ok_false_but_keeps_snapshot(client):
    tc, sensors_path, _boot = client
    snapshot = _write_snapshot(sensors_path, status="ok", received_at=STALE_AGE_RECEIVED_AT)

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["snapshot"]["frame"] == snapshot["frame"]
    assert body["snapshot"]["status"] == "stale"
    assert body["age_sec"] == pytest.approx(25.0)
    assert body["pressures"]["cabinet_sensor_staleness"] == pytest.approx(1.0)


def test_missing_boot_returns_boot_null(client):
    tc, sensors_path, boot_path = client
    _write_snapshot(sensors_path)
    assert not boot_path.exists()

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["boot"] is None
    assert body["snapshot"] is not None


def test_absent_sensor_keys_not_zero_filled(client):
    tc, sensors_path, _boot = client
    # Only environment present — no magnetic / uv / particulate / lidar / imu.
    _write_snapshot(
        sensors_path,
        frame={
            "schema": "orion.sensor_frame.v1",
            "seq": 1,
            "uptime_ms": 100,
            "environment": {"temp_c": 22.0},
        },
    )

    r = tc.get("/api/cabinet/sensors/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["measurements"]["cabinet_temp_c"] == pytest.approx(22.0)
    for absent_key in (
        "cabinet_magnetic_ut",
        "cabinet_uv_raw",
        "cabinet_als_raw",
        "cabinet_pm1_ug_m3",
        "cabinet_lidar_mm",
        "cabinet_vibration_g",
    ):
        assert absent_key not in body["measurements"]
    assert "cabinet_em_activity" not in body["pressures"]
    assert "cabinet_uv_activity" not in body["pressures"]
    assert body["pressures"]["cabinet_sensor_staleness"] == pytest.approx(0.0)


def test_router_registered_on_api_routes():
    from scripts import api_routes

    paths = {getattr(route, "path", None) for route in api_routes.router.routes}
    assert "/api/cabinet/sensors/latest" in paths
    assert "/api/cabinet/sensors/history" in paths


def test_rows_to_sensor_series_skips_nulls_and_zero_fills_nothing():
    rows = [
        {
            "t": datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc),
            "temp_c": 24.5,
            "humidity_pct": 45.0,
            "lidar_mm": 18.0,
            "als_raw": None,
            "uv_raw": 0.0,
            "magnetic_ut": 55.0,
            "vibration_g": 0.02,
            "imu_yaw_deg": -42.0,
            "imu_pitch_deg": 0.1,
            "imu_roll_deg": -0.2,
            "climate_activity": 0.1,
            "proximity_activity": None,
            "em_activity": 0.3,
            "vibration_activity": 0.15,
            "uv_activity": 0.05,
        },
        {
            "t": datetime(2026, 8, 29, 12, 0, 30, tzinfo=timezone.utc),
            "temp_c": 24.7,
            "humidity_pct": None,
            "lidar_mm": 17.0,
            "als_raw": 120.0,
            "uv_raw": 1.0,
            "magnetic_ut": 52.0,
            "vibration_g": 0.04,
            "imu_yaw_deg": -41.5,
            "climate_activity": 0.2,
            "proximity_activity": 0.4,
            "em_activity": 0.25,
            "vibration_activity": 0.12,
            "uv_activity": None,
        },
    ]
    series = cabinet_sensors_routes.rows_to_sensor_series(rows)
    assert len(series["temp_c"]) == 2
    assert series["temp_c"][0]["v"] == pytest.approx(24.5)
    assert series["humidity_pct"] == [{"t": "2026-08-29T12:00:00Z", "v": pytest.approx(45.0)}]
    assert len(series["lidar_mm"]) == 2
    assert len(series["als_raw"]) == 1
    assert series["als_raw"][0]["v"] == pytest.approx(120.0)
    assert len(series["proximity_activity"]) == 1
    assert len(series["magnetic_ut"]) == 2
    assert series["imu_yaw_deg"][0]["v"] == pytest.approx(-42.0)


def test_history_defaults_to_24h_and_returns_empty_series(client, monkeypatch):
    tc, _sensors, _boot = client

    async def no_rows(*, node: str, hours: int):
        assert (node, hours) == ("athena", 24)
        return []

    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_AMBIENT_HISTORY_NODE", "athena")
    monkeypatch.setattr(cabinet_sensors_routes, "_history_query", no_rows)
    body = tc.get("/api/cabinet/sensors/history").json()
    assert body["ok"] is True
    assert body["window"] == "24h"
    assert body["grain_sec"] == 30
    assert body["series"]["temp_c"] == []
    assert body["stats"] == {}


def test_history_uses_query_rows_and_reports_stats(client, monkeypatch):
    tc, _sensors, _boot = client

    async def rows(*, node: str, hours: int):
        assert (node, hours) == ("athena", 72)
        return [
            {
                "t": datetime(2026, 8, 26, 2, 54, 9, tzinfo=timezone.utc),
                "temp_c": 24.0,
                "humidity_pct": 40.0,
                "lidar_mm": 50.0,
                "als_raw": 100.0,
                "climate_activity": 0.2,
                "proximity_activity": 0.1,
                "uv_activity": 0.05,
            },
            {
                "t": datetime(2026, 8, 26, 2, 54, 39, tzinfo=timezone.utc),
                "temp_c": 26.0,
                "humidity_pct": 42.0,
                "lidar_mm": 18.0,
                "als_raw": 120.0,
                "climate_activity": 0.85,
                "proximity_activity": 0.9,
                "uv_activity": 0.15,
            },
        ]

    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_AMBIENT_HISTORY_NODE", "athena")
    monkeypatch.setattr(cabinet_sensors_routes, "_history_query", rows)
    body = tc.get("/api/cabinet/sensors/history?window=3d").json()
    assert body["ok"] is True
    assert body["window"] == "3d"
    assert len(body["series"]["temp_c"]) == 2
    assert body["stats"]["temp_c"] == {"n_raw": 2, "min": 24.0, "max": 26.0}
    assert body["stats"]["lidar_mm"]["min"] == pytest.approx(18.0)


def test_history_invalid_window_returns_400(client):
    tc, _sensors, _boot = client
    response = tc.get("/api/cabinet/sensors/history?window=1h")
    assert response.status_code == 400


def test_history_db_failure_returns_ok_false(client, monkeypatch):
    tc, _sensors, _boot = client

    async def failed(*, node: str, hours: int):
        raise OSError("db unavailable")

    monkeypatch.setattr(cabinet_sensors_routes, "_history_query", failed)
    body = tc.get("/api/cabinet/sensors/history").json()
    assert body["ok"] is False
    assert body["series"]["temp_c"] == []
    assert body["error"] == "sensor_history_unavailable"


def test_dual_nano_merge_in_latest(tmp_path: Path, monkeypatch) -> None:
    sensors_a = tmp_path / "a.json"
    sensors_b = tmp_path / "b.json"
    boot_a = tmp_path / "boot-a.json"
    boot_b = tmp_path / "boot-b.json"
    _write_snapshot(
        sensors_a,
        frame={
            "schema": "orion.sensor_frame.v1",
            "seq": 1,
            "uptime_ms": 100,
            "environment": {"temp_c": 28.0},
            "lidar": {"distance_mm": 50.0, "status": 0},
        },
    )
    _write_snapshot(
        sensors_b,
        frame={
            "schema": "orion.sensor_frame.v1",
            "seq": 2,
            "uptime_ms": 200,
            "magnetic": {"magnitude_ut": 42.0, "x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0},
            "imu": {"accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.8, "yaw_deg": 3.0},
        },
    )
    _write_boot(boot_a)
    _write_boot(boot_b)

    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_SENSORS_PATH", str(sensors_a))
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_BOOT_PATH", str(boot_a))
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_SENSORS_B_PATH", str(sensors_b))
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_BOOT_B_PATH", str(boot_b))
    monkeypatch.setattr(cabinet_sensors_routes.settings, "CABINET_SENSORS_STALE_AFTER_SEC", 10.0)
    monkeypatch.setattr(
        cabinet_sensors_routes,
        "_TRACKER",
        cabinet_sensors_routes.CabinetSensorTracker(
            cabinet_sensors_routes.CabinetPressureConfig()
        ),
    )
    monkeypatch.setattr(cabinet_sensors_routes, "_now_utc", lambda: NOW)

    app = FastAPI()
    app.include_router(cabinet_sensors_routes.router)
    tc = TestClient(app)

    r = tc.get("/api/cabinet/sensors/latest")
    body = r.json()
    assert body["ok"] is True
    frame = body["snapshot"]["frame"]
    assert frame["environment"]["temp_c"] == pytest.approx(28.0)
    assert frame["lidar"]["distance_mm"] == pytest.approx(50.0)
    assert frame["magnetic"]["magnitude_ut"] == pytest.approx(42.0)
    assert frame["imu"]["yaw_deg"] == pytest.approx(3.0)
    assert "a" in body["sources"]
    assert "b" in body["sources"]
