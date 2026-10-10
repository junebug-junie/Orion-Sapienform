"""Orion's rest drive: how tired Orion is, readable outside the dream service.

Temporal Self rev 4 (PR #2369), section R2. The dream service already keeps a
real sleep pressure (novelty-based since #2557): it starts at 0 when a sleep
begins, rises as new unprocessed material arrives, and Orion sleeps once it
crosses the threshold (or the 48 h window fills) and Juniper has been quiet.

`DriveReadingV1` is that same pressure, published by the service that owns the
discharge action (orion-dream) at every 600 s check, built from the very values
its own sleep gate uses -- one copy of "due", never a mirror that could
disagree. Producer: services/orion-dream/app/cycle.py `run_cycle_once`, via
`orion.regulation.rest_drive.read_rest_drive`.

Transport: the latest reading in Redis `orion:drive:rest:latest`, with a TTL
equal to the readers' staleness bound (Hub curiosity and outreach, which ease
off when the drive is `due`). Not a bus channel: nothing subscribes, and an
unconsumed channel is an orphan. History is the dream's own per-check row,
`dream_pressure_observation`, whose `check_id` is this reading's `source_ref`
(except the reading published right after a sleep, `dp-postsleep-*`, which
has no row). A missing, expired, unparseable or `no_reading` value is
UNKNOWN, and every reader then behaves exactly as it did before this drive
existed. Missing is never read as rested, and never as tired.

Distinct from the retired, producer-less `DriveStateV1`
(orion/core/schemas/drives.py) -- see orion/inner_state_registry.py.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

REST_DRIVE_REDIS_KEY = "orion:drive:rest:latest"

DriveName = Literal["rest"]
DriveState = Literal["resting", "building", "due", "refractory", "no_reading"]
DueReason = Literal["threshold", "overdue"]


class DriveReadingV1(BaseModel):
    """One check of one drive. `level` is None only when `state == "no_reading"`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["drive.reading.v1"] = "drive.reading.v1"
    # Widened only by a drive that passes the CLAUDE.md metric gate (spec R2).
    drive: DriveName = "rest"
    observed_at: AwareDatetime
    # The dream's SleepPressureV1.pressure at this check.
    level: Optional[float] = Field(default=None, ge=0.0)
    threshold: float = Field(ge=0.0)
    # Declared rest level. Live: pressure reads exactly 0.0 at the first check
    # after a sleep (dream_pressure_observation, 2026-10-10 01:32 UTC).
    rest_level: float = Field(default=0.0, ge=0.0)
    state: DriveState
    # Why `due`: crossed the threshold, or the window reached its lookback cap
    # with material to replay (the dream's overdue backstop). None otherwise.
    due_reason: Optional[DueReason] = None
    # Start of the window the level is summed over (the last good sleep's start,
    # floored at the lookback cap).
    accumulating_since: Optional[AwareDatetime] = None
    # End of the last sleep attempt (dream_cycle.ended_at), the refractory clock.
    last_discharge_at: Optional[AwareDatetime] = None
    refractory_until: Optional[AwareDatetime] = None
    # Joins to dream_pressure_observation.check_id for the same check; the
    # post-sleep reading (`dp-postsleep-*`) has no observation row.
    source_ref: str = Field(min_length=1, max_length=200)
    no_reading_reason: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _level_matches_state(self) -> "DriveReadingV1":
        if self.state == "no_reading":
            if self.level is not None:
                raise ValueError("no_reading carries no level")
            if not self.no_reading_reason:
                raise ValueError("no_reading needs a no_reading_reason")
        elif self.level is None:
            raise ValueError(f"state {self.state!r} needs a level")
        if (self.state == "due") != (self.due_reason is not None):
            raise ValueError("due_reason is set exactly when state is due")
        return self


def parse_drive_reading(raw: object) -> Optional[DriveReadingV1]:
    """Redis/bus value -> reading, or None for anything absent or malformed."""
    if raw is None:
        return None
    try:
        if isinstance(raw, DriveReadingV1):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            return DriveReadingV1.model_validate_json(raw)
        if isinstance(raw, dict):
            return DriveReadingV1.model_validate(raw)
    except Exception:  # noqa: BLE001 -- malformed is unknown, never a reading
        return None
    return None


__all__ = [
    "REST_DRIVE_REDIS_KEY",
    "DriveReadingV1",
    "parse_drive_reading",
]
