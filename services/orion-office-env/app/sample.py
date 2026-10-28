from __future__ import annotations

"""Frame -> bus sample. Pure: no I/O, no clock reads, so freshness is testable."""

from datetime import datetime
from typing import Optional

from orion.schemas.telemetry.office_env import (
    OfficeEnvFrameV1,
    OfficeEnvMeasurementsV1,
    OfficeEnvProvenanceV1,
    OfficeEnvSampleV1,
    OfficeEnvSensorV1,
    OfficeEnvStateV1,
)


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
    fresh frame whose firmware reported neither temperature nor humidity (SHT40
    failed to initialise on the Nano).
    """
    age = max(0.0, (now - received_at).total_seconds()) if received_at is not None else None
    env = frame.environment if frame is not None else None
    temp = env.temp_c if env is not None else None
    humidity = env.humidity_pct if env is not None else None
    stale = (
        frame is None
        or age is None
        or age > stale_after_sec
        or (temp is None and humidity is None)
    )
    if stale:
        measurements = OfficeEnvMeasurementsV1()
    else:
        measurements = OfficeEnvMeasurementsV1(office_temp_c=temp, office_humidity_pct=humidity)
    return OfficeEnvSampleV1(
        ts=now,
        node=node,
        sensor=OfficeEnvSensorV1(id=sensor_id, online=not stale),
        measurements=measurements,
        state=OfficeEnvStateV1(stale=stale),
        provenance=OfficeEnvProvenanceV1(
            frame_seq=frame.seq if frame is not None else None,
            firmware_uptime_ms=frame.uptime_ms if frame is not None else None,
            sample_age_sec=age,
        ),
    )
