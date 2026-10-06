"""Hardware watch incidents: the watcher's open/resolve facts, and the pool's shed input.

Spec: docs/superpowers/specs/2026-09-28-urgent-curiosity-and-hardware-watch-design.md (Part 4).
Plan: docs/superpowers/plans/2026-09-29-urgent-curiosity-plan-4-5-hardware-watch-and-shedding.md.

``orion-hardware-watch`` publishes one ``HardwareWatchIncidentV1`` on
``orion:hardware:watch:incident`` for every transition (``opened``, ``resolved``) and re-publishes
each open incident every refresh interval (``refresh``), so a consumer that restarted re-learns the
open set within one interval. ``orion-gpu-pool`` reads ``shed`` from cooling incidents: while an
open incident's ``shed.requested`` is true and ``shed.valid_until`` has not passed, the pool's
``cooling_incident`` shed reason is set (scheduler rule U4).

Thermal controller v2 (docs/superpowers/specs/2026-10-06-thermal-controller-redesign-design.md, D2):
with ``HARDWARE_WATCH_HEAT_CONTROLLER=v2`` the reflex no longer rides an incident. Every tick the
watcher publishes one ``HardwareWatchReflexShedV1`` on ``orion:hardware:watch:reflex_shed`` while the
cabinet is critical (>= 34 C) or unreadable past grace, with ``valid_until = now + 3 ticks``; the pool
sets shed reason ``cabinet_hot`` / ``cabinet_unknown`` from it. When the state drops the watcher sends
one ``active=false`` clear and stops; a lost clear still lapses at ``valid_until`` (fail-open). v2
cooling incidents carry ``shed=None``: they alert, they never shed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import to_json

HARDWARE_WATCH_INCIDENT_CHANNEL = "orion:hardware:watch:incident"
HARDWARE_WATCH_INCIDENT_KIND = "hardware.watch.incident.v1"
HARDWARE_WATCH_EVIDENCE_MAX_BYTES = 16_000
HARDWARE_WATCH_REFLEX_SHED_CHANNEL = "orion:hardware:watch:reflex_shed"
HARDWARE_WATCH_REFLEX_SHED_KIND = "hardware.watch.reflex_shed.v1"
# The reasons the reflex may assert on the pool's shed board (orion/gpu_pool/shed.py).
ReflexShedReason = Literal["cabinet_hot", "cabinet_unknown"]

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


class HardwareWatchReflexShedV1(BaseModel):
    """One tick of the v2 reflex: assert (``active``) or clear one shed board reason.

    ``source_id`` names the producer instance (one cabinet today: ``hardware-watch:cabinet``); the pool
    keeps at most one reflex reason per source, so an ``active`` signal for ``cabinet_hot`` replaces a
    ``cabinet_unknown`` from the same source."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["hardware.watch.reflex_shed.v1"] = HARDWARE_WATCH_REFLEX_SHED_KIND
    source_id: str = Field(min_length=1, max_length=120)
    active: bool
    reason: ReflexShedReason | None = None
    # Required while active; the pool caps it (SHED_MAX_VALID_SEC) so a bad clock never latches.
    valid_until: datetime | None = None
    cabinet: dict[str, Any] = Field(default_factory=dict)   # CabinetHeatReading.as_dict() at emit
    emitted_at: datetime = Field(default_factory=_now)

    @field_validator("cabinet")
    @classmethod
    def _small(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(to_json(value)) > 4_000:
            raise ValueError("cabinet snapshot over 4000 bytes")
        return value

    @model_validator(mode="after")
    def _active_is_complete(self) -> "HardwareWatchReflexShedV1":
        if self.active and (self.reason is None or self.valid_until is None):
            raise ValueError("an active reflex shed needs reason and valid_until")
        return self
