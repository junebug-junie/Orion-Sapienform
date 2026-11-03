from __future__ import annotations

"""Frame -> bus sample. Pure: no I/O, no clock reads, so freshness is testable."""

from datetime import datetime
from typing import Optional

from orion.schemas.telemetry.office_env import (
    OfficeEnvFrameAirV1,
    OfficeEnvFrameV1,
    OfficeEnvMeasurementsV1,
    OfficeEnvProvenanceV1,
    OfficeEnvSampleV1,
    OfficeEnvSensorV1,
    OfficeEnvStateV1,
)

# CCS811 datasheet: readings are not valid until 20 minutes after each sensor start.
CCS811_CONDITIONING_MS = 20 * 60 * 1000
# CCS811 datasheet output ranges; anything outside is not a reading.
ECO2_RANGE_PPM = (400, 29206)
TVOC_RANGE_PPB = (0, 32768)


def _in_range(value: Optional[int], bounds: tuple[int, int]) -> Optional[int]:
    if value is None or not bounds[0] <= value <= bounds[1]:
        return None
    return value


def _air_readings(air: Optional[OfficeEnvFrameAirV1]) -> tuple[Optional[int], Optional[int], bool]:
    """(eco2, tvoc, warming). Warming readings are withheld, not passed on as data.

    A frame without ``since_init_ms`` cannot prove the sensor is past conditioning,
    so it counts as warming too.
    """
    if air is None:
        return None, None, False
    if air.since_init_ms is None or air.since_init_ms < CCS811_CONDITIONING_MS:
        return None, None, True
    return _in_range(air.eco2_ppm, ECO2_RANGE_PPM), _in_range(air.tvoc_ppb, TVOC_RANGE_PPB), False


def build_sample(
    *,
    frame: Optional[OfficeEnvFrameV1],
    received_at: Optional[datetime],
    now: datetime,
    stale_after_sec: float,
    node: str,
    sensor_id: str,
) -> OfficeEnvSampleV1:
    """A stale or empty frame yields NO measurements -- never a repeated or zero value.

    "Stale" covers: no frame ever, last frame older than ``stale_after_sec``, and a
    fresh frame carrying no usable reading from either sensor (SHT40 and CCS811 both
    absent, or the CCS811 still warming with no SHT40). Each sensor's readings are
    otherwise independent: one failing does not drop the other.
    """
    age = max(0.0, (now - received_at).total_seconds()) if received_at is not None else None
    env = frame.environment if frame is not None else None
    temp = env.temp_c if env is not None else None
    humidity = env.humidity_pct if env is not None else None
    air = frame.air if frame is not None else None
    eco2, tvoc, warming = _air_readings(air)
    stale = (
        frame is None
        or age is None
        or age > stale_after_sec
        or all(v is None for v in (temp, humidity, eco2, tvoc))
    )
    if stale:
        measurements = OfficeEnvMeasurementsV1()
    else:
        measurements = OfficeEnvMeasurementsV1(
            office_temp_c=temp,
            office_humidity_pct=humidity,
            office_eco2_ppm=eco2,
            office_tvoc_ppb=tvoc,
        )
    return OfficeEnvSampleV1(
        ts=now,
        node=node,
        sensor=OfficeEnvSensorV1(id=sensor_id, online=not stale),
        measurements=measurements,
        # A stale frame says nothing current about the sensor, warming or not.
        state=OfficeEnvStateV1(stale=stale, air_warming=warming and not stale),
        provenance=OfficeEnvProvenanceV1(
            frame_seq=frame.seq if frame is not None else None,
            firmware_uptime_ms=frame.uptime_ms if frame is not None else None,
            sample_age_sec=age,
            air_since_init_ms=air.since_init_ms if air is not None else None,
        ),
    )
