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


# --- CCS811 air quality -------------------------------------------------------

WARM_MS = 20 * 60 * 1000


def _air_frame(env: dict | None = None, **air) -> OfficeEnvFrameV1:
    body = {"schema": "orion.office_env_frame.v1", "seq": 8, "uptime_ms": 9000, "air": air}
    if env is not None:
        body["environment"] = env
    return OfficeEnvFrameV1.model_validate(body)


def test_conditioned_air_reading_is_published() -> None:
    f = _air_frame({"temp_c": 21.0, "humidity_pct": 30.0}, eco2_ppm=650, tvoc_ppb=37, since_init_ms=WARM_MS)
    s = _build(f, age_sec=1.0)
    assert s.state.stale is False
    assert s.state.air_warming is False
    assert s.measurements.office_eco2_ppm == 650
    assert s.measurements.office_tvoc_ppb == 37
    assert s.measurements.office_temp_c == 21.0
    assert s.provenance.air_since_init_ms == WARM_MS


def test_warming_air_reading_is_withheld_but_temp_kept() -> None:
    f = _air_frame({"temp_c": 21.0, "humidity_pct": 30.0}, eco2_ppm=400, tvoc_ppb=0, since_init_ms=WARM_MS - 1)
    s = _build(f, age_sec=1.0)
    assert s.state.air_warming is True
    assert s.state.stale is False
    assert s.measurements.office_eco2_ppm is None
    assert s.measurements.office_tvoc_ppb is None
    assert s.measurements.office_temp_c == 21.0


def test_air_without_init_clock_counts_as_warming() -> None:
    s = _build(_air_frame({"temp_c": 21.0}, eco2_ppm=650, tvoc_ppb=37), age_sec=1.0)
    assert s.state.air_warming is True
    assert s.measurements.office_eco2_ppm is None


def test_out_of_range_air_values_are_dropped_not_clamped() -> None:
    """eCO2 below the 400 ppm floor is a not-yet-valid register, not clean air."""
    f = _air_frame({"temp_c": 21.0}, eco2_ppm=0, tvoc_ppb=40000, since_init_ms=WARM_MS)
    s = _build(f, age_sec=1.0)
    assert s.measurements.office_eco2_ppm is None
    assert s.measurements.office_tvoc_ppb is None
    assert s.state.air_warming is False


def test_air_alone_keeps_sample_fresh_when_sht40_missing() -> None:
    s = _build(_air_frame(None, eco2_ppm=500, tvoc_ppb=12, since_init_ms=WARM_MS), age_sec=1.0)
    assert s.state.stale is False
    assert s.measurements.office_temp_c is None
    assert s.measurements.office_eco2_ppm == 500


def test_warming_air_alone_is_stale() -> None:
    s = _build(_air_frame(None, eco2_ppm=500, tvoc_ppb=12, since_init_ms=1000), age_sec=1.0)
    assert s.state.stale is True
    assert s.measurements.model_dump(exclude_none=True) == {}


def test_stale_frame_drops_air_and_warming_flag() -> None:
    f = _air_frame({"temp_c": 21.0}, eco2_ppm=650, tvoc_ppb=37, since_init_ms=1000)
    s = _build(f, age_sec=60.0)
    assert s.state.stale is True
    assert s.state.air_warming is False
    assert s.measurements.office_eco2_ppm is None


def test_odd_air_register_does_not_reject_frame() -> None:
    """Frame bounds are the uint16 register; the SHT40 values must survive any air value."""
    f = _air_frame({"temp_c": 21.0}, eco2_ppm=65535, tvoc_ppb=65535, since_init_ms=WARM_MS)
    assert f.environment.temp_c == 21.0
