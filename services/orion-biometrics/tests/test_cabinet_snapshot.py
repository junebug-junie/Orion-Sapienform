"""Tests for host cabinet sensor snapshot ingest (app/cabinet_snapshot.py)."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_SERVICE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _name in list(sys.modules):
    if _name == "app" or _name.startswith("app."):
        del sys.modules[_name]
sys.path.insert(0, str(_SERVICE_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from app.cabinet_snapshot import load_cabinet_sensors_snapshot  # noqa: E402
from orion.telemetry.cabinet_sensors import extract_cabinet_measurements  # noqa: E402


def _frame(**overrides):
    base = {
        "schema": "orion.sensor_frame.v1",
        "seq": 1,
        "uptime_ms": 1000,
        "environment": {
            "temp_c": 24.6,
            "humidity_pct": 45.0,
            "pressure_hpa": 900.0,
            "gas_resistance_ohm": 1000.0,
        },
        "uv": {"raw": 17.0, "als_raw": 1292.0},
        "magnetic": {"x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0, "magnitude_ut": 53.0},
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


def _write_snapshot(path: Path, **overrides) -> None:
    payload = {
        "status": "ok",
        "received_at": "2026-08-23T12:00:00.000Z",
        "device": "/dev/serial/by-id/usb-Arduino_Nano_ESP32_test-if01",
        "frame": _frame(),
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


NOW = datetime(2026, 8, 23, 12, 0, 5, tzinfo=timezone.utc)


def test_missing_file_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_cabinet_sensors_snapshot(missing, stale_after_sec=10.0, now=NOW) is None


def test_invalid_json_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "latest.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_cabinet_sensors_snapshot(bad, stale_after_sec=10.0, now=NOW) is None


def test_snapshot_without_frame_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, frame=None, status="missing")
    assert load_cabinet_sensors_snapshot(path, stale_after_sec=10.0, now=NOW) is None


def test_fresh_ok_snapshot_not_stale(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, status="ok", received_at="2026-08-23T12:00:00.000Z")

    sensors = load_cabinet_sensors_snapshot(path, stale_after_sec=10.0, now=NOW)

    assert sensors is not None
    assert sensors["stale"] is False
    assert sensors["received_at"] == "2026-08-23T12:00:00.000Z"
    assert isinstance(sensors["frame"], dict)
    assert sensors["frame"]["schema"] == "orion.sensor_frame.v1"


@pytest.mark.parametrize("status", ["stale", "error", "missing"])
def test_reader_status_marks_stale(tmp_path: Path, status: str) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, status=status, received_at="2026-08-23T12:00:00.000Z")

    sensors = load_cabinet_sensors_snapshot(path, stale_after_sec=10.0, now=NOW)

    assert sensors is not None
    assert sensors["stale"] is True


def test_age_over_threshold_marks_stale(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, status="ok", received_at="2026-08-23T11:59:40.000Z")

    sensors = load_cabinet_sensors_snapshot(path, stale_after_sec=10.0, now=NOW)

    assert sensors is not None
    assert sensors["stale"] is True


def test_stale_sensors_produce_no_cabinet_measurements(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, status="stale")

    sensors = load_cabinet_sensors_snapshot(path, stale_after_sec=10.0, now=NOW)
    assert sensors is not None
    assert extract_cabinet_measurements(sensors) == {}


def test_fresh_sensors_produce_measurements_without_zero_fill(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    _write_snapshot(path, status="ok")

    sensors = load_cabinet_sensors_snapshot(path, stale_after_sec=10.0, now=NOW)
    assert sensors is not None
    measurements = extract_cabinet_measurements(sensors)

    assert "cabinet_temp_c" in measurements
    assert measurements["cabinet_temp_c"] == pytest.approx(24.6)
    assert all(value != 0.0 for key, value in measurements.items() if key.endswith("_raw"))


def test_dual_nano_merge_in_biometrics_loader(tmp_path: Path) -> None:
    primary = tmp_path / "a.json"
    secondary = tmp_path / "b.json"
    _write_snapshot(
        primary,
        status="ok",
        frame={
            "schema": "orion.sensor_frame.v1",
            "seq": 1,
            "uptime_ms": 1000,
            "environment": {"temp_c": 28.0, "humidity_pct": 20.0, "pressure_hpa": 900.0, "gas_resistance_ohm": 1000.0},
            "lidar": {"distance_mm": 100.0, "status": 0},
        },
    )
    _write_snapshot(
        secondary,
        status="ok",
        frame={
            "schema": "orion.sensor_frame.v1",
            "seq": 2,
            "uptime_ms": 2000,
            "magnetic": {"x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0, "magnitude_ut": 40.0},
            "imu": {
                "accel_x": 0.0,
                "accel_y": 0.0,
                "accel_z": 9.80665,
                "yaw_deg": 2.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
            },
        },
    )

    sensors = load_cabinet_sensors_snapshot(
        primary,
        secondary_path=secondary,
        stale_after_sec=10.0,
        now=NOW,
    )
    assert sensors is not None
    assert sensors["stale"] is False
    measurements = extract_cabinet_measurements(sensors)
    assert measurements["cabinet_temp_c"] == pytest.approx(28.0)
    assert measurements["cabinet_magnetic_ut"] == pytest.approx(40.0)
    assert measurements["cabinet_lidar_mm"] == pytest.approx(100.0)
