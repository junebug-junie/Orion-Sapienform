"""Orion's regulation state: one slow arousal reading plus the latest drive readings.

Temporal Self rev 4 (PR #2369), section R3, order 3 ("R2/R3 core").

One writer: the ``regulate`` node of the ``temporal_self.update`` durable graph in
orion-durable-runs (thread ``temporal_self:orion:<local date>``), every 120 s and on every
chat turn. The latest state is in Redis ``orion:regulation:latest`` (TTL) and at
``GET /regulation/state``. Not a bus channel: no bus consumer exists yet. Every level change
is one ``arousal_transition`` row in ``temporal_self_event``.

NO READER YET. Spec order 6 wires the dials one PR at a time. A reader must treat a missing,
expired, unparseable, stale or ``unknown`` state as "arousal unavailable" and behave exactly as
it did before arousal existed.

Inputs, per the spec's metric gate (R3):

* E1 -- minutes since Juniper's last turn: ``chat_history_log`` rows that carry a prompt and are
  not marked unsolicited (``orion.regulation.juniper_turns``). Orion's own outreach never counts.
* S1 -- the cabinet-heat reflex verdict (``orion.autonomy.cabinet_heat``), ``cabinet_hot`` only.
* S2 -- GPU pool queue depth (``GpuPoolStateV1.queue_depth``, summed) at or above a floor for a
  sustained time, from the saved ``gpu_pool_state_history`` snapshots. ``backlog_depth`` (the
  spec's first choice) is excluded: it was empty in every saved snapshot (replay #2576).
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from orion.schemas.drive_reading import DriveReadingV1

REGULATION_STATE_KIND = "regulation.state.v1"
REGULATION_STATE_REDIS_KEY = "orion:regulation:latest"

ArousalLevel = Literal["engaged", "idle", "strained", "unknown"]


class ArousalInputsV1(BaseModel):
    """One step's raw evidence, as the regulate node read it. The reducer's only input."""

    model_config = ConfigDict(extra="forbid")

    observed_at: AwareDatetime
    # E1. read_ok=False: the query failed (stale). read_ok=True with minutes=None: no Juniper
    # turn has ever been recorded, which is a real, fresh "not engaged".
    juniper_turn_read_ok: bool = False
    minutes_since_juniper_turn: Optional[float] = Field(default=None, ge=0.0)
    # S1. CabinetHeatReading.reflex verbatim ("cabinet_hot", "cabinet_unknown" or None) and the
    # thermal state it came from. None for both: the cabinet query itself failed.
    cabinet_read_ok: bool = False
    cabinet_reflex: Optional[str] = None
    cabinet_thermal_state: Optional[str] = None
    cabinet_critical: bool = False
    cabinet_temp_c: Optional[float] = None
    # S2. Age of the newest saved GPU pool snapshot (None: none in the window, or the read
    # failed), its summed queue depth, and how long the depth has held at or above the floor
    # without a gap in the snapshots.
    gpu_state_age_sec: Optional[float] = Field(default=None, ge=0.0)
    gpu_queue_depth: Optional[int] = Field(default=None, ge=0)
    gpu_queue_sustained_sec: Optional[float] = Field(default=None, ge=0.0)


class ArousalReadingV1(BaseModel):
    """One slow reading of whether Orion is with Juniper, alone and free, or under load."""

    model_config = ConfigDict(extra="forbid")

    arousal_level: ArousalLevel
    # When this level began. Carried across steps (and across a short restart) while the level
    # holds; resets on every change.
    since: AwareDatetime
    observed_at: AwareDatetime
    minutes_since_juniper_turn: Optional[float] = None   # E1, user turns only
    cabinet_reflex: Optional[str] = None                 # S1, CabinetHeatVerdict.reflex verbatim
    cabinet_temp_c: Optional[float] = None
    gpu_queue_depth: Optional[int] = None                # S2
    gpu_queue_sustained_sec: Optional[float] = None
    # Hysteresis memory: strain holds until every strain input has read fresh and clear for the
    # clear time. Stale time never earns clear time.
    strain_latched: bool = False
    strain_clear_since: Optional[AwareDatetime] = None
    # Which input decided, in plain words (e.g. "cabinet_hot", "gpu_queue_sustained",
    # "juniper_turn_recent", "no_juniper_turn_45m", "stale:cabinet", "disabled").
    reasons: List[str] = Field(default_factory=list)


class RegulationStateV1(BaseModel):
    """One per regulate step. Redis ``orion:regulation:latest`` and ``GET /regulation/state``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["regulation.state.v1"] = REGULATION_STATE_KIND
    generated_at: AwareDatetime
    arousal: ArousalReadingV1
    # The latest reading of every admitted drive, verbatim from its owner's Redis key (only
    # "rest" in v1). Embedded for the trace only: arousal never reads a drive (that would be a
    # loop, since the rest drive is meant to read arousal).
    drives: List[DriveReadingV1] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def parse_regulation_state(raw: object) -> Optional[RegulationStateV1]:
    """Redis value -> state, or None for anything absent or malformed (readers: unknown)."""
    if raw is None:
        return None
    try:
        if isinstance(raw, RegulationStateV1):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return RegulationStateV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return RegulationStateV1.model_validate(raw)
    except Exception:  # noqa: BLE001 -- malformed is unknown, never a reading
        return None
    return None


__all__ = [
    "ArousalInputsV1",
    "ArousalLevel",
    "ArousalReadingV1",
    "REGULATION_STATE_KIND",
    "REGULATION_STATE_REDIS_KEY",
    "RegulationStateV1",
    "parse_regulation_state",
]
