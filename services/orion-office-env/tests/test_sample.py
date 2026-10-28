from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.sample import build_sample
from orion.schemas.telemetry.office_env import OfficeEnvFrameV1

NOW = datetime(2026, 10, 27, 20, 0, 0, tzinfo=timezone.utc)


def _frame(**env) -> OfficeEnvFrameV1:
    body = {"schema": "orion.office_env_frame.v1", "seq": 7, "uptime_ms": 9000}
    if env is not None:
        body["environment"] = env
    return OfficeEnvFrameV1.model_validate(body)


def _build(frame, age_sec: float | None, stale_after: float = 10.0):
    received = NOW - timedelta(seconds=age_sec) if age_sec is not None else None
    return build_sample(
        frame=frame, received_at=received, now=NOW, stale_after_sec=stale_after,
        node="pan-1", sensor_id="sht40-office",
    )


def test_fresh_frame_carries_readings() -> None:
    s = _build(_frame(temp_c=21.4, humidity_pct=33.0), age_sec=2.0)
    assert s.state.stale is False
    assert s.sensor.online is True
    assert s.measurements.office_temp_c == 21.4
    assert s.measurements.office_humidity_pct == 33.0
    assert s.provenance.frame_seq == 7
    assert s.provenance.sample_age_sec == 2.0


def test_old_frame_is_stale_with_no_readings() -> None:
    s = _build(_frame(temp_c=21.4, humidity_pct=33.0), age_sec=60.0)
    assert s.state.stale is True
    assert s.sensor.online is False
    assert s.measurements.office_temp_c is None
    assert s.measurements.office_humidity_pct is None
    assert s.provenance.sample_age_sec == 60.0


def test_no_frame_ever_is_stale_not_zero() -> None:
    s = _build(None, age_sec=None)
    assert s.state.stale is True
    assert s.measurements.model_dump(exclude_none=True) == {}
    assert s.provenance.frame_seq is None


def test_fresh_frame_with_no_sensor_data_is_stale() -> None:
    """SHT40 failed init on the Nano: firmware omits both keys. Not a 0.0."""
    s = _build(_frame(), age_sec=1.0)
    assert s.state.stale is True
    assert s.measurements.model_dump(exclude_none=True) == {}


def test_partial_frame_keeps_only_what_was_read() -> None:
    s = _build(_frame(temp_c=19.0), age_sec=1.0)
    assert s.state.stale is False
    assert s.measurements.office_temp_c == 19.0
    assert s.measurements.office_humidity_pct is None


def test_payload_round_trips_with_schema_alias() -> None:
    s = _build(_frame(temp_c=21.4, humidity_pct=33.0), age_sec=2.0)
    dumped = s.model_dump(mode="json", by_alias=True)
    assert dumped["schema"] == "office.env.sample.v1"
    assert dumped["node"] == "pan-1"
    assert dumped["role"] == "office_environment"
