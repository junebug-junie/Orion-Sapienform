"""Dream cycle v2 — sleep as offline work with a measurable consequence.

A `DreamCycleV1` records one sleep: why it started (sleep pressure), what it
chose to replay, and which hypotheses REM recombination produced. The old
`DreamResultV1` is a narrative; nothing downstream acted on it. This artifact
exists so a dream can be *scored*: every hypothesis carries the two memory refs
it linked and which experimental arm produced it.

Arms:
  dream    pairs chosen from the replay set (salient, distant).
  control  pairs drawn at random from the whole candidate pool, same prompt.

Hub offers both arms to Orion's curiosity loop blind (arm is never shown). If
dream-arm hypotheses are not adopted/supported more often than control-arm
ones, recombination is theater and the scorecard says so.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# §cap-all-collections.
MAX_REPLAY_ITEMS = 24
MAX_HYPOTHESES = 12

# A prior Orion forms from a dream hypothesis carries this prefix in its
# `formed_from` property. The scorecard joins on it; the Hub kickoff section
# teaches it. One constant, three readers.
FORMED_FROM_PREFIX = "dream_hypothesis:"

ReplaySourceKind = Literal["metacog", "compaction_request", "resonance", "crystallization"]
HypothesisArm = Literal["dream", "control"]
CycleStatus = Literal["completed", "empty", "failed"]
CycleTrigger = Literal["pressure", "manual"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SleepPressureV1(BaseModel):
    """How much the day left unprocessed since the last cycle.

    `pressure` is the sum of candidate weights — the same numbers replay ranks
    on, so pressure and replay cannot drift apart. It returns to exactly 0 after
    a cycle because every source is windowed on `since` (the last cycle's end).
    """

    model_config = ConfigDict(extra="forbid")

    since: datetime
    computed_at: datetime = Field(default_factory=_utc_now)
    pressure: float = Field(ge=0.0)
    counts: dict[str, int] = Field(default_factory=dict)
    idle_minutes: Optional[float] = None
    threshold: float = Field(ge=0.0)
    idle_required_minutes: float = Field(ge=0.0)

    @property
    def is_idle(self) -> bool:
        # Unknown idleness (no chat log readable) is NOT idle: never sleep
        # through a conversation because a read failed.
        return self.idle_minutes is not None and self.idle_minutes >= self.idle_required_minutes

    @property
    def should_sleep(self) -> bool:
        return self.pressure >= self.threshold and self.is_idle


class ReplayItemV1(BaseModel):
    """One memory fragment chosen for replay, with the reason it was chosen."""

    model_config = ConfigDict(extra="forbid")

    ref_id: str
    source_kind: ReplaySourceKind
    text: str
    weight: float = Field(ge=0.0, le=1.0)
    reason: str
    tags: list[str] = Field(default_factory=list, max_length=32)


class DreamHypothesisV1(BaseModel):
    """A testable claim linking two memory fragments. Not a belief.

    Orion decides whether to hold it (by writing a :Prior whose `formed_from`
    is `dream_hypothesis:<hypothesis_id>`). The dream never writes priors.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    cycle_id: str
    arm: HypothesisArm
    claim: str = Field(min_length=20, max_length=400)
    why: str = Field(default="", max_length=400)
    ref_a: str
    ref_b: str
    created_at: datetime = Field(default_factory=_utc_now)
    expires_at: datetime


class DreamCycleV1(BaseModel):
    """One sleep. Persisted whole as dream_cycle.cycle_json (no bus channel:
    Hub reads hypotheses from Postgres, and a channel with no consumer is an
    orphan the metric-lineage gate rightly refuses)."""

    model_config = ConfigDict(extra="forbid")

    cycle_id: str
    trigger: CycleTrigger
    status: CycleStatus
    started_at: datetime
    ended_at: datetime
    pressure: SleepPressureV1
    replay: list[ReplayItemV1] = Field(default_factory=list, max_length=MAX_REPLAY_ITEMS)
    hypotheses: list[DreamHypothesisV1] = Field(default_factory=list, max_length=MAX_HYPOTHESES)
    # Pairs the LLM declined to link (answered "no link"). Counted, not hidden:
    # a dream that links everything is manufacturing significance.
    no_link_count: int = 0
    llm_failures: int = 0
    compaction_delta_id: Optional[str] = None
    note: str = ""
