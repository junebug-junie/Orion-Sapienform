from __future__ import annotations

"""Server-closet environment sample (pan-1 closet Nano ESP32 + SCD40).

The closet Nano sits above the server cabinet and runs the same firmware as the
office Nano, so its serial frame is ``OfficeEnvFrameV1`` (only the ``co2`` object
is present). This module holds the bus payload ``orion-office-env`` publishes for
it on ``orion:closet:env:sample``. It is a separate sample, not an office sample,
so closet readings never show up under ``office_*`` names.

Absent is not zero: a stale frame or a missing SCD40 yields NO measurement keys.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SAMPLE_KIND_V1 = "closet.env.sample.v1"


class ClosetEnvSensorV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model: str = "SCD40"
    # True when the serial link produced a valid frame with an SCD40 reading inside
    # the stale window.
    online: bool


class ClosetEnvMeasurementsV1(BaseModel):
    """Only fields actually read. Omit unknowns -- never zero-fill."""

    model_config = ConfigDict(extra="forbid")

    # SCD4x output range. Accuracy is specified only for 400..2000 ppm (SCD40).
    closet_co2_ppm: Optional[int] = Field(default=None, ge=0, le=40000)
    # SCD40 operating range -10..60 C, 0..100 %RH.
    closet_temp_c: Optional[float] = Field(default=None, ge=-10.0, le=60.0)
    closet_humidity_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class ClosetEnvStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stale: bool = False


class ClosetEnvProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "nano-esp32-usb-serial"
    frame_seq: Optional[int] = Field(default=None, ge=0)
    firmware_uptime_ms: Optional[int] = Field(default=None, ge=0)
    sample_age_sec: Optional[float] = Field(default=None, ge=0.0)


class ClosetEnvSampleV1(BaseModel):
    """Read-only server-closet CO2/temperature/humidity sample."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(default=SAMPLE_KIND_V1, alias="schema")
    ts: datetime
    node: str = "pan-1"
    role: str = "closet_environment"
    sensor: ClosetEnvSensorV1
    measurements: ClosetEnvMeasurementsV1
    state: ClosetEnvStateV1 = Field(default_factory=ClosetEnvStateV1)
    provenance: ClosetEnvProvenanceV1 = Field(default_factory=ClosetEnvProvenanceV1)

    @field_validator("ts")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
