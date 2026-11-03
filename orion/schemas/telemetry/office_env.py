from __future__ import annotations

"""Office environment contracts (pan-1 Nano ESP32 + SHT40 + CCS811).

Two shapes live here:

* ``OfficeEnvFrameV1`` -- host-local NDJSON the Nano firmware writes over USB
  serial (``orion.office_env_frame.v1``). NOT a bus payload; deliberately not in
  ``orion/schemas/registry.py``. Same role as ``cabinet_sensor_frame.py``.
* ``OfficeEnvSampleV1`` -- the bus payload ``orion-office-env`` publishes on
  ``orion:office:env:sample``.

Absent is not zero: a sensor that failed to initialise, or a frame older than
the producer's stale window, yields NO measurement keys. Never a 0.0.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

FRAME_SCHEMA_V1 = "orion.office_env_frame.v1"
SAMPLE_KIND_V1 = "office.env.sample.v1"


class OfficeEnvFrameEnvironmentV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # SHT40 datasheet range: -40..125 C, 0..100 %RH.
    temp_c: Optional[float] = Field(default=None, ge=-40.0, le=125.0)
    humidity_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class OfficeEnvFrameAirV1(BaseModel):
    """CCS811 raw registers. Bounds are the uint16 register width, not the datasheet
    range, so one odd reading cannot reject the whole frame (and drop the SHT40 values
    with it). ``build_sample`` applies the datasheet range."""

    model_config = ConfigDict(extra="ignore")

    eco2_ppm: Optional[int] = Field(default=None, ge=0, le=65535)
    tvoc_ppb: Optional[int] = Field(default=None, ge=0, le=65535)
    # Time since the firmware last (re)started the CCS811. Drives the warm-up gate.
    since_init_ms: Optional[int] = Field(default=None, ge=0)


class OfficeEnvFrameV1(BaseModel):
    """One firmware tick. Every sub-payload is optional."""

    model_config = ConfigDict(extra="ignore")

    schema_name: str = Field(alias="schema")
    seq: int = Field(ge=0)
    uptime_ms: int = Field(ge=0)
    environment: Optional[OfficeEnvFrameEnvironmentV1] = None
    air: Optional[OfficeEnvFrameAirV1] = None

    @field_validator("schema_name")
    @classmethod
    def _schema_matches(cls, value: str) -> str:
        if value != FRAME_SCHEMA_V1:
            raise ValueError(f"unsupported frame schema {value!r}")
        return value


class OfficeEnvSensorV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    model: str = "SHT40"  # primary sensor; the same node also carries the CCS811
    # True when the serial link produced a valid frame inside the stale window carrying
    # a usable reading from either sensor (SHT40 or CCS811).
    online: bool


class OfficeEnvMeasurementsV1(BaseModel):
    """Only fields actually read. Omit unknowns -- never zero-fill."""

    model_config = ConfigDict(extra="forbid")

    office_temp_c: Optional[float] = Field(default=None, ge=-40.0, le=125.0)
    office_humidity_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    # CCS811 datasheet output ranges. eCO2 is estimated from VOCs, not measured CO2.
    office_eco2_ppm: Optional[int] = Field(default=None, ge=400, le=29206)
    office_tvoc_ppb: Optional[int] = Field(default=None, ge=0, le=32768)


class OfficeEnvStateV1(BaseModel):
    """``stale=True``: no valid frame within the producer's stale window, so every
    cached reading is omitted from the sample."""

    model_config = ConfigDict(extra="forbid")

    stale: bool = False
    # True while the CCS811 is inside its 20-minute conditioning period after a start:
    # its readings exist but are not yet valid, so they are omitted from measurements.
    air_warming: bool = False


class OfficeEnvProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = "nano-esp32-usb-serial"
    frame_seq: Optional[int] = Field(default=None, ge=0)
    firmware_uptime_ms: Optional[int] = Field(default=None, ge=0)
    sample_age_sec: Optional[float] = Field(default=None, ge=0.0)
    air_since_init_ms: Optional[int] = Field(default=None, ge=0)


class OfficeEnvSampleV1(BaseModel):
    """Read-only office temperature/humidity/air-quality sample."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(default=SAMPLE_KIND_V1, alias="schema")
    ts: datetime
    node: str = "pan-1"
    role: str = "office_environment"
    sensor: OfficeEnvSensorV1
    measurements: OfficeEnvMeasurementsV1
    state: OfficeEnvStateV1 = Field(default_factory=OfficeEnvStateV1)
    provenance: OfficeEnvProvenanceV1 = Field(default_factory=OfficeEnvProvenanceV1)

    @field_validator("ts")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
