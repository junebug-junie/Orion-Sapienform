from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.telemetry.cabinet_sensor_frame import (
    FRAME_SCHEMA_V1,
    CabinetSensorFrameV1,
)
from orion.telemetry.biometrics_pipeline import BiometricsPipeline, PipelineConfig
from orion.telemetry.cabinet_sensors import (
    CabinetPressureConfig,
    CabinetSensorTracker,
    compute_cabinet_pressures,
    extract_cabinet_measurements,
)

CFG = CabinetPressureConfig()


def _frame(**overrides):
    base = {
        "schema": "orion.sensor_frame.v1",
        "seq": 1,
        "uptime_ms": 1000,
        "environment": {"temp_c": 24.6, "humidity_pct": 45.0, "pressure_hpa": 900.0, "gas_resistance_ohm": 1000.0},
        "uv": {"raw": 17.0, "als_raw": 1292.0},
        "magnetic": {"x_ut": 1.0, "y_ut": 2.0, "z_ut": 3.0, "magnitude_ut": 53.0},
        "particulate": {"pm1_ug_m3": 2.0, "pm25_ug_m3": 4.0, "pm10_ug_m3": 5.0},
        "lidar": {"distance_mm": 438.0, "status": 0},
        "imu": {"accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.80665, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
    }
    base.update(overrides)
    return base


def _sensors(**frame_overrides):
    return {"frame": _frame(**frame_overrides), "received_at": "2026-08-23T00:00:01Z", "stale": False}


# --- frame schema -----------------------------------------------------------


def test_frame_schema_accepts_valid_ndjson():
    frame = CabinetSensorFrameV1.model_validate(_frame())
    assert frame.schema_ == FRAME_SCHEMA_V1
    assert frame.seq == 1
    assert frame.environment is not None
    assert frame.environment.temp_c == 24.6


def test_frame_schema_rejects_missing_schema():
    bad = _frame()
    del bad["schema"]
    with pytest.raises(Exception):
        CabinetSensorFrameV1.model_validate(bad)


def test_frame_schema_has_no_audio_block():
    frame = CabinetSensorFrameV1.model_validate(_frame())
    assert not hasattr(frame, "audio") or getattr(frame, "audio", None) is None


# --- extract_cabinet_measurements: absent-is-not-zero ---------------------


def test_measurements_absent_when_sensors_none():
    assert extract_cabinet_measurements(None) == {}


def test_measurements_absent_when_sensors_not_dict():
    assert extract_cabinet_measurements("not-a-dict") == {}


def test_measurements_absent_when_stale():
    sensors = _sensors()
    sensors["stale"] = True
    assert extract_cabinet_measurements(sensors) == {}


def test_measurements_absent_when_frame_missing():
    assert extract_cabinet_measurements({"received_at": "x", "stale": False}) == {}


def test_lidar_excluded_when_status_not_zero():
    m = extract_cabinet_measurements(_sensors(lidar={"distance_mm": 438.0, "status": 4}))
    assert "cabinet_lidar_mm" not in m


def test_lidar_included_when_status_zero():
    m = extract_cabinet_measurements(_sensors())
    assert m["cabinet_lidar_mm"] == 438.0


def test_missing_subpayload_omits_its_keys_not_zero():
    frame = _frame()
    del frame["particulate"]
    m = extract_cabinet_measurements({"frame": frame, "stale": False})
    assert "cabinet_pm25_ug_m3" not in m
    assert "cabinet_pm1_ug_m3" not in m
    assert m["cabinet_temp_c"] == 24.6


def test_vibration_g_at_rest_is_zero():
    m = extract_cabinet_measurements(_sensors())
    assert m["cabinet_vibration_g"] == 0.0


def test_vibration_g_hand_computed_deviation():
    from orion.telemetry.cabinet_sensors import GRAVITY_MPS2

    m = extract_cabinet_measurements(_sensors(imu={"accel_x": 0.0, "accel_y": 0.0, "accel_z": 2 * GRAVITY_MPS2}))
    assert abs(m["cabinet_vibration_g"] - 1.0) < 1e-9


def test_no_audio_measurements_extracted():
    frame = _frame()
    frame["audio"] = {"rms": 0.1, "peak": 0.4}
    m = extract_cabinet_measurements({"frame": frame, "stale": False})
    assert "cabinet_audio_rms" not in m
    assert "cabinet_audio_peak" not in m


# --- compute_cabinet_pressures ---------------------------------------------


def test_pressures_empty_when_measurements_empty():
    tracker = CabinetSensorTracker(CFG)
    assert compute_cabinet_pressures({}, tracker) == {}


def test_pressures_bounded_zero_to_one():
    tracker = CabinetSensorTracker(CFG)
    measurements = extract_cabinet_measurements(_sensors())
    for _ in range(20):
        out = compute_cabinet_pressures(measurements, tracker)
        for key, val in out.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"
        measurements = dict(measurements)
        measurements["cabinet_magnetic_ut"] = measurements.get("cabinet_magnetic_ut", 53.0) + 10.0


def test_activity_signal_rests_at_zero_for_constant_input():
    """A raw magnitude that never changes must read exactly 0.0 activity."""
    tracker = CabinetSensorTracker(CFG)
    measurements = {"cabinet_magnetic_ut": 53.0}
    for _ in range(10):
        out = compute_cabinet_pressures(measurements, tracker)
        assert out["cabinet_em_activity"] == 0.0


def test_activity_signal_responds_to_a_real_change():
    tracker = CabinetSensorTracker(CFG)
    for _ in range(5):
        compute_cabinet_pressures({"cabinet_magnetic_ut": 53.0}, tracker)
    out = compute_cabinet_pressures({"cabinet_magnetic_ut": 500.0}, tracker)
    assert out["cabinet_em_activity"] > 0.0


def test_pressure_keys_match_design_spec():
    tracker = CabinetSensorTracker(CFG)
    out = compute_cabinet_pressures(extract_cabinet_measurements(_sensors()), tracker)
    expected = {
        "cabinet_climate_activity",
        "cabinet_particulate_activity",
        "cabinet_em_activity",
        "cabinet_uv_activity",
        "cabinet_vibration_activity",
        "cabinet_proximity_activity",
    }
    assert expected <= set(out.keys())
    assert "cabinet_thermal_pressure" not in out
    assert "em_activity" not in out
    assert "uv_exposure" not in out


def test_climate_constant_input_rests_at_zero():
    tracker = CabinetSensorTracker(CFG)
    measurements = {"cabinet_temp_c": 24.6, "cabinet_humidity_pct": 45.0}
    for _ in range(10):
        out = compute_cabinet_pressures(measurements, tracker)
        assert out["cabinet_climate_activity"] == 0.0


# --- pipeline: host peak_pressure unchanged by cabinet ----------------------


def _host_sample(**overrides):
    base = {
        "timestamp": datetime.now(timezone.utc),
        "node": "athena",
        "cpu": {"util": 80.0},
        "gpu": {"gpus": [{"utilization_gpu": 50.0, "memory_used_mb": 4000, "memory_total_mb": 8000}]},
        "memory": {"used_pct": 60.0},
        "disk": {"read_bytes_per_sec": 0, "write_bytes_per_sec": 0},
        "network": {"rx_bytes_per_sec": 0, "tx_bytes_per_sec": 0},
        "temps": {"max_c": 70.0},
        "power": {"power_draw_watts": 200.0},
    }
    base.update(overrides)
    return base


def test_host_peak_pressure_unchanged_when_cabinet_present():
    pipe = BiometricsPipeline(PipelineConfig())
    without, _ = pipe.update(_host_sample())
    with_cab, _ = pipe.update(_host_sample(sensors=_sensors()))
    assert without.peak_pressure == with_cab.peak_pressure
    assert without.peak_pressure_channel == with_cab.peak_pressure_channel
    assert without.constraint == with_cab.constraint
    assert any(k.startswith("cabinet_") for k in with_cab.pressures)


def test_host_peak_ignores_high_cabinet_activity():
    pipe = BiometricsPipeline(PipelineConfig())
    sample = _host_sample(sensors=_sensors())
    summary, _ = pipe.update(sample)
    host_peak = summary.peak_pressure
    host_channel = summary.peak_pressure_channel
    for _ in range(30):
        frame = _frame(
            magnetic={"x_ut": 0, "y_ut": 0, "z_ut": 0, "magnitude_ut": 5000.0 + _ * 100},
            imu={"accel_x": 0, "accel_y": 0, "accel_z": 9.80665 * (2 + _ * 0.1)},
        )
        summary, _ = pipe.update(_host_sample(sensors={"frame": frame, "stale": False, "received_at": "x"}))
    assert summary.peak_pressure == host_peak
    assert summary.peak_pressure_channel == host_channel
    assert max(v for k, v in summary.pressures.items() if k.startswith("cabinet_")) > 0.0
