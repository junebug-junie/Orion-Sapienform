"""Hardware watch incidents: the watcher's open/resolve facts, and the pool's shed input.

Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md (Part 4).
Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md.

``orion-hardware-watch`` publishes one ``HardwareWatchIncidentV1`` on
``orion:hardware:watch:incident`` for every transition (``opened``, ``resolved``) and re-publishes
each open incident every refresh interval (``refresh``), so a consumer that restarted re-learns the
open set within one interval. ``orion-gpu-pool`` reads ``shed`` from cooling incidents: while an
open incident's ``shed.requested`` is true and ``shed.valid_until`` has not passed, the pool's
``cooling_incident`` shed reason is set (scheduler rule U4).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import to_json

HARDWARE_WATCH_INCIDENT_CHANNEL = "orion:hardware:watch:incident"
HARDWARE_WATCH_INCIDENT_KIND = "hardware.watch.incident.v1"
HARDWARE_WATCH_EVIDENCE_MAX_BYTES = 16_000

Rule = Literal["cooling", "cpu_heat", "gpu_heat"]
Transition = Literal["opened", "refresh", "resolved"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class HardwareWatchShedV1(BaseModel):
    """The watcher's shed request for one cooling incident. Latched: once ``requested`` it stays
    requested until the incident resolves."""

    model_config = ConfigDict(extra="forbid")

    requested: bool = False
    # Why: cabinet_rising (>= rise over the window), cabinet_elevated (>= thermal_gate elevated),
    # cabinet_unreadable (no fresh cabinet reading), or disabled (HARDWARE_WATCH_SHED_ENABLED=false).
    reason: str | None = Field(default=None, max_length=64)
    requested_at: datetime | None = None
    cabinet_temp_c: float | None = None
    cabinet_rise_c: float | None = None
    # The pool ignores this request after valid_until (the watcher re-publishes before it lapses).
    valid_until: datetime | None = None


class HardwareWatchIncidentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["hardware.watch.incident.v1"] = HARDWARE_WATCH_INCIDENT_KIND
    incident_id: str = Field(pattern=r"^[0-9a-f]{12,32}$")
    rule: Rule
    # "cabinet_ac", "athena", "circe", "circe/gpu1".
    subject: str = Field(min_length=1, max_length=120)
    transition: Transition
    status: Literal["open", "resolved"]
    # Opening arm: low_power, no_fresh_sample, device_offline, controller_not_ready, no_samples,
    # frozen, above_p95, above_ceiling, simulated.
    open_reason: str = Field(max_length=64)
    opened_at: datetime
    resolved_at: datetime | None = None
    # recovered | operator
    resolve_reason: str | None = Field(default=None, max_length=64)
    resolved_by: str | None = Field(default=None, max_length=64)
    shed: HardwareWatchShedV1 | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    emitted_at: datetime = Field(default_factory=_now)

    @field_validator("evidence")
    @classmethod
    def _bounded(cls, value: dict[str, Any]) -> dict[str, Any]:
        size = len(to_json(value))
        if size > HARDWARE_WATCH_EVIDENCE_MAX_BYTES:
            raise ValueError(f"evidence is {size} bytes; limit {HARDWARE_WATCH_EVIDENCE_MAX_BYTES}")
        return value
