from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CoolingControllerV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool
    driver: str = "zwave-js"
    device_path: Optional[str] = None


class CoolingDeviceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    product: Optional[str] = None
    online: bool


class CoolingMeasurementsV1(BaseModel):
    """Only include fields that were actually read. Omit unknowns — never zero-fill."""

    model_config = ConfigDict(extra="forbid")

    cooling_watts: Optional[float] = Field(default=None, ge=0.0)
    cooling_volts: Optional[float] = Field(default=None, ge=0.0)
    cooling_amps: Optional[float] = Field(default=None, ge=0.0)
    cooling_power_factor: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class CoolingObservedStateV1(BaseModel):
    """Observational only — not an actuator affordance."""

    model_config = ConfigDict(extra="forbid")

    switch_on: Optional[bool] = None


class CoolingProvenanceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zwave_node_id: int = Field(ge=1)
    source: str = "zwave-js"
    sample_age_sec: Optional[float] = Field(default=None, ge=0.0)


class HomeCoolingSampleV1(BaseModel):
    """Read-only cabinet cooling sample (portable AC on Shelly Wave)."""

    model_config = ConfigDict(extra="forbid")

    schema_name: str = Field(default="home.cooling.sample.v1", alias="schema")
    ts: datetime
    node: str = "athena"
    role: str = "cabinet_cooling"
    controller: CoolingControllerV1
    device: CoolingDeviceV1
    measurements: CoolingMeasurementsV1
    state: CoolingObservedStateV1 = Field(default_factory=CoolingObservedStateV1)
    provenance: CoolingProvenanceV1

    @field_validator("ts")
    @classmethod
    def _ensure_tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
