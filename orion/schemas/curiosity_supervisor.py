"""Patch 1 + the arming patch of
`docs/superpowers/specs/2026-09-09-curiosity-supervisor-design.md`.

`HopReadingV1` is what the supervisor THINKS one hop was doing -- a READING,
not a fact, and not a write to `orion_worldview`. It exists so a human can
sample fifty of these and check whether the supervisor is reading Orion's own
prose correctly, per the spec's own acceptance check 1 and 7. Field names and
types match the spec's contract verbatim.

Registered in `orion/schemas/registry.py` / `orion/bus/channels.yaml` as of
2026-09-19 (`READING_CHANNEL` / `READING_KIND` below), persisted by
`orion-sql-writer`. No live producer yet -- publishing is opt-in and manual,
via `scripts/report_curiosity_supervisor_readings.py --publish`; there is no
`orion/curiosity/supervisor_loop.py` or any always-on service that runs this
on a cadence. Registration ONLY arms the reading -- no intervention (`end_run_early`,
`hand_off_to_claude`, `reorder_next_offer`) is wired by this. Those change
what Orion does next and stay their own, separately-approved patch per the
design doc's own Danger section and CLAUDE.md's "Proposal mode before
invasive cognition changes".

`extra="forbid"` on both models: nothing downstream but the sql-writer
persistence layer consumes these, so there is no compatibility reason to
accept unknown fields, and forbidding them makes a schema-drifted LLM
response fail loudly during parsing rather than silently dropping a field
nobody noticed was missing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

READING_CHANNEL = "orion:curiosity:supervisor:reading"
READING_KIND = "curiosity.supervisor.reading.v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HopReadingV1(BaseModel):
    """What the supervisor thinks one hop was doing. A READING, not a fact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.supervisor.reading.v1"] = READING_KIND
    # Row identity for the writer, same convention as DurableRunStateV1.
    # entry_id / PeerBriefV1.brief_id: `(hop_run_id, hop_n)` is NOT a safe
    # key -- it is exactly the pair this arc's own hop-identity patch found
    # colliding on real data (run 58b638778228: six real hops, three hop_n
    # values). Never the model's to fill; stamped at generation time.
    reading_id: str = Field(default_factory=lambda: uuid4().hex)
    generated_at: datetime = Field(default_factory=_utc_now)
    hop_run_id: str
    hop_n: int
    # The hop's own graph clock (ms), copied from the `Hop` node -- never
    # the model's to fill; `parse_reading_batch` overwrites it. None for a
    # hop written before 2026-09-19, when `Hop` carried no timestamp and `n`
    # restarted at 1 on every retried turn (so two readings can share an
    # `hop_n` under one run and only this field, when present, orders them).
    hop_written_at: Optional[int] = None
    # Derived from the note's prose. Recorded so a human can check fifty of
    # these and see whether it is reading them right -- the supervisor must
    # not be the only witness to its own accuracy. `None` is a real answer:
    # a hop can be about no live-or-closed prior at all (e.g. bookkeeping,
    # or a claim formed and never persisted).
    about_prior_id: Optional[str] = None
    # Free-form on purpose -- CLAUDE.md 0A bans fixing a taxonomy up front.
    # Let the kinds fall out of what the supervisor actually writes.
    kind: str
    # None = could not tell, a real third state alongside True/False.
    moved_the_claim: Optional[bool] = None
    reading_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class HopReadingBatchV1(BaseModel):
    """The structured-output wrapper for one run's worth of hops.

    One LLM call answers every hop in a run at once -- cheaper than one call
    per hop, and it gives the model the run's full context (a run usually
    investigates one thread) instead of one isolated note at a time.
    """

    model_config = ConfigDict(extra="forbid")

    readings: list[HopReadingV1]
