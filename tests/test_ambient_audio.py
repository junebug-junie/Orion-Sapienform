from __future__ import annotations

import pytest

from orion.schemas.telemetry.ambient_audio import (
    AMBIENT_AUDIO_SCHEMA_V1,
    AmbientAudioSnapshotV1,
)
from orion.telemetry.ambient_audio import (
    AmbientAudioPressureConfig,
    AmbientAudioTracker,
    compute_ambient_audio_pressures,
    extract_ambient_audio_measurements,
)

CFG = AmbientAudioPressureConfig()


def _snapshot(**overrides):
    base = {
        "schema": "orion.ambient_audio.v1",
        "status": "ok",
        "received_at": "2026-08-25T05:00:00.123Z",
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": 0.5,
        "sample_rate": 16000,
        "channels": 1,
        "rms": 412.3,
        "peak": 1820,
    }
    base.update(overrides)
    return base


def _ambient_audio(**overrides):
    base = {
        "rms": 412.3,
        "peak": 1820,
        "received_at": "2026-08-25T05:00:00.123Z",
        "stale": False,
        "device": "plughw:CARD=CMTECK,DEV=0",
        "window_sec": 0.5,
    }
    base.update(overrides)
    return base


# --- snapshot schema --------------------------------------------------------


def test_snapshot_schema_accepts_valid_json():
    snap = AmbientAudioSnapshotV1.model_validate(_snapshot())
    assert snap.schema_ == AMBIENT_AUDIO_SCHEMA_V1
    assert snap.status == "ok"
    assert snap.rms == 412.3
    assert snap.peak == 1820


def test_snapshot_schema_rejects_missing_schema():
    bad = _snapshot()
    del bad["schema"]
    with pytest.raises(Exception):
        AmbientAudioSnapshotV1.model_validate(bad)


# --- extract_ambient_audio_measurements: absent-is-not-zero -----------------


def test_measurements_absent_when_ambient_audio_none():
    assert extract_ambient_audio_measurements(None) == {}


def test_measurements_absent_when_ambient_audio_not_dict():
    assert extract_ambient_audio_measurements("not-a-dict") == {}


def test_measurements_absent_when_stale():
    assert extract_ambient_audio_measurements(_ambient_audio(stale=True)) == {}


def test_measurements_present_when_fresh():
    m = extract_ambient_audio_measurements(_ambient_audio())
    assert m["cabinet_ambient_rms"] == 412.3
    assert m["cabinet_ambient_peak"] == 1820.0


def test_measurements_omit_missing_rms_or_peak():
    m = extract_ambient_audio_measurements(_ambient_audio(rms=None))
    assert "cabinet_ambient_rms" not in m
    assert m["cabinet_ambient_peak"] == 1820.0


# --- compute_ambient_audio_pressures ----------------------------------------


def test_pressures_empty_when_measurements_empty():
    tracker = AmbientAudioTracker(CFG)
    assert compute_ambient_audio_pressures({}, tracker) == {}


def test_pressures_bounded_zero_to_one():
    tracker = AmbientAudioTracker(CFG)
    measurements = {"cabinet_ambient_rms": 412.3}
    for i in range(20):
        out = compute_ambient_audio_pressures(measurements, tracker)
        for key, val in out.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"
        measurements = {"cabinet_ambient_rms": 412.3 + i * 50.0}


def test_activity_signal_rests_at_zero_for_constant_rms():
    """Constant RMS must read exactly 0.0 activity (same invariant as cabinet)."""
    tracker = AmbientAudioTracker(CFG)
    measurements = {"cabinet_ambient_rms": 412.3}
    for _ in range(10):
        out = compute_ambient_audio_pressures(measurements, tracker)
        assert out["cabinet_ambient_audio_activity"] == 0.0


def test_activity_signal_responds_to_rms_change():
    tracker = AmbientAudioTracker(CFG)
    for _ in range(5):
        compute_ambient_audio_pressures({"cabinet_ambient_rms": 412.3}, tracker)
    out = compute_ambient_audio_pressures({"cabinet_ambient_rms": 5000.0}, tracker)
    assert out["cabinet_ambient_audio_activity"] > 0.0


def test_peak_does_not_drive_activity():
    """Activity is RMS-only; peak alone must not produce a pressure."""
    tracker = AmbientAudioTracker(CFG)
    out = compute_ambient_audio_pressures({"cabinet_ambient_peak": 9000.0}, tracker)
    assert out == {}
