"""Orion's body: one reading of every constraint Orion lives under, in its own unit.

Body economy design, Organ 1 "Body" (docs/superpowers/specs/2026-10-23-organism-body-economy-
design.md, PR #2855), Patch A3: the Body organ only. No prices, no forecasts, and no reader
decides anything from it yet.

One writer: the ``physiology`` node of the ``temporal_self.update`` durable graph in
orion-durable-runs, at the regulation thread's cadence (every ~120 s; chat-turn steps inside
``ORION_PHYSIOLOGY_MIN_INTERVAL_SEC`` are skipped). The pure reducer is
``orion.physiology.body.build_physiology_state``; durable-runs only does the reads.

* Redis ``orion:physiology:latest`` (TTL, three ticks): the latest state.
* Bus ``orion:physiology:state``: one event per tick, persisted by orion-sql-writer to
  ``physiology_state`` (one row per tick). This is the strain/arousal history the design notes
  is missing today.

``no_reading`` IS NOT ZERO. A constraint whose source is missing, stale or failed carries
``level=None`` and a reason in ``no_reading``. A reader must treat it as "unknown" and never as
calm or as empty. A level of 0 means the source was read fresh and really says 0.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

PHYSIOLOGY_STATE_KIND = "physiology.state.v1"
PHYSIOLOGY_STATE_CHANNEL = "orion:physiology:state"
PHYSIOLOGY_STATE_REDIS_KEY = "orion:physiology:latest"

# Constraint ids. ``lane:<role>`` is one per GPU pool seat (e.g. ``lane:fast``).
HEAT = "heat"
ENERGY = "energy"
ATTENTION_JUNIPER = "attention:juniper"
CLAUDE = "claude"
LANE_PREFIX = "lane:"

DetailValue = Union[float, int, str, bool, None]


class PhysiologyConstraintReadingV1(BaseModel):
    """One constraint at one tick. ``level`` and ``budget`` share ``unit``."""

    model_config = ConfigDict(extra="forbid")

    constraint: str                      # heat | energy | lane:<role> | attention:juniper | claude
    unit: str                            # c | usd | slot_s | messages | limit_events
    level: Optional[float] = None        # None exactly when no_reading is set
    # The line the level is held against, in the same unit: heat soft line, monthly energy
    # envelope, lane capacity (slots x window), the outreach daily cap. None: no line is set
    # (energy until Juniper sets an envelope) or it could not be read (lane capacity).
    budget: Optional[float] = None
    budget_kind: Optional[Literal["soft_line", "envelope", "capacity", "daily_cap", "not_limited"]] = None
    # The claude constraint only: clear | limited | unknown (ClaudeLimitObservationV1.state).
    state: Optional[str] = None
    # When the source itself was last fresh (newest row / observation), not when we read it.
    source_at: Optional[AwareDatetime] = None
    # Why there is no level: "read_failed:<source>", "stale:<source>", "no_observation", ...
    no_reading: Optional[str] = None
    detail: Dict[str, DetailValue] = Field(default_factory=dict)


class PhysiologyStateV1(BaseModel):
    """One per physiology step. Bus ``orion:physiology:state``, Redis ``orion:physiology:latest``,
    table ``physiology_state``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["physiology.state.v1"] = PHYSIOLOGY_STATE_KIND
    # Deterministic: ``physiology:<generated_at iso>``. sql-writer's primary key, so a redelivered
    # event is one row.
    tick_id: str
    generated_at: AwareDatetime
    constraints: List[PhysiologyConstraintReadingV1] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def parse_physiology_state(raw: object) -> Optional[PhysiologyStateV1]:
    """Redis value -> state, or None for anything absent or malformed (readers: unknown)."""
    if raw is None:
        return None
    try:
        if isinstance(raw, PhysiologyStateV1):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return PhysiologyStateV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return PhysiologyStateV1.model_validate(raw)
    except Exception:  # noqa: BLE001 -- malformed is unknown, never a reading
        return None
    return None


__all__ = [
    "ATTENTION_JUNIPER",
    "CLAUDE",
    "ENERGY",
    "HEAT",
    "LANE_PREFIX",
    "PHYSIOLOGY_STATE_CHANNEL",
    "PHYSIOLOGY_STATE_KIND",
    "PHYSIOLOGY_STATE_REDIS_KEY",
    "PhysiologyConstraintReadingV1",
    "PhysiologyStateV1",
    "parse_physiology_state",
]
