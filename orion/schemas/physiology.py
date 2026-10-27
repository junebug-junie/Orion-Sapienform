"""Orion's body: one reading of every constraint Orion lives under, in its own unit.

Body economy design, Organ 1 "Body" (docs/superpowers/specs/2026-10-23-organism-body-economy-
design.md, PR #2855), Patch A3: the Body organ. Patch B2 adds, per reading, an optional SHADOW
``price`` (Organ 2 Budget) and on the state an optional ``strain`` dial (Organ 5). No reader
decides anything from either yet.

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

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

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


BIND_STATES = ("slack", "binding", "exogenous_bind", "stuck_at_ceiling", "no_reading")


class PhysiologyPriceV1(BaseModel):
    """Organ 2 Budget (Patch B2), SHADOW: computed and published, read by nothing that decides.

    The price says what one unit of this constraint is worth giving up right now, in want-value
    per unit (``orion/physiology/prices.py``, dual ascent with anti-windup). ``value`` is held where
    it was while the constraint has no reading (unknown is not "nothing used")."""

    model_config = ConfigDict(extra="forbid")

    value: float = Field(ge=0.0)
    ceiling: float = Field(gt=0.0)
    # provisional until Appetite (Patch C) has graded wants to size it against (prices.ceiling_for).
    ceiling_source: Literal["provisional", "sized"] = "provisional"
    # slack: use <= budget (the price decays toward 0). binding: forecast use > budget.
    # exogenous_bind: over the line even with Orion's share at 0; clamped at a bounded ceiling.
    # stuck_at_ceiling: at the ceiling, unflagged, longer than one horizon (R3; a defect signal).
    # no_reading: the use or the budget is unknown; the value is the last known one.
    bind_state: Literal["slack", "binding", "exogenous_bind", "stuck_at_ceiling", "no_reading"]
    # The price's own inputs, in the reading's unit (heat: Orion's share in C vs the room's headroom).
    use: Optional[float] = None
    budget: Optional[float] = None
    excess: Optional[float] = None
    horizon_sec: Optional[float] = None
    why: str = ""


class PhysiologyStrainV1(BaseModel):
    """Organ 5 Felt state, the ``strain`` dial (Patch B2, SHADOW): the largest price / its ceiling
    over constraints that are live (not exogenous_bind, not no_reading, not self-spent), and which one it is.
    A monotonic rendering of the prices, not new signal. ``value`` None: no constraint priced live."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    constraint: Optional[str] = None
    # Out of Orion's hands (anti-windup clamp), listed apart and NOT in the dial.
    exogenous_bind: List[str] = Field(default_factory=list)
    # Patch C1: constraints whose use is Orion spending its OWN budget (attention:juniper: the
    # sends are Orion's own outreach against Juniper's cap). Strain is the world pushing back, so
    # these are listed apart and NOT in the dial; their price still charges wants (Appetite /
    # Choice). Consumer-first: sql-writer must run code with this field before durable-runs emits it.
    self_spent: List[str] = Field(default_factory=list)
    stuck: List[str] = Field(default_factory=list)
    no_reading: List[str] = Field(default_factory=list)
    why: str = ""


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
    # Patch B2 shadow price (additive, optional; None before B2 or with prices off).
    price: Optional[PhysiologyPriceV1] = None

    @model_validator(mode="after")
    def _no_reading_means_no_level(self) -> "PhysiologyConstraintReadingV1":
        # The contract, not a convention: a level of 0 is a fresh real 0, never a missing reading.
        if (self.no_reading is None) == (self.level is None):
            raise ValueError("exactly one of level / no_reading must be set")
        return self


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
    # Patch B2 shadow strain dial (additive, optional). Not a table column: it is a pure function
    # of the stored prices (orion.physiology.budget.strain_dial), so the physiology_state row's
    # ``constraints`` carries everything needed to recompute it.
    strain: Optional[PhysiologyStrainV1] = None


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
    "BIND_STATES",
    "PhysiologyPriceV1",
    "PhysiologyStrainV1",
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
