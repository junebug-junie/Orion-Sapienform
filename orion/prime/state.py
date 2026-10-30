"""Prime's persisted state and the tick (design section 8.1): CPU only, never calls a model.

The tick reads a ``PrimeSnapshotV1`` (what the body, attention and curiosity look like now),
compares it with what Prime last saw, and records each difference as a pending change. A thought
is worth bidding for only while there is at least one pending change: "nothing changed" is the
normal tick and costs nothing.

What counts as a change (each one is a fact Prime can be shown, never a score):

* attention moved: Temporal Self's ``active_arc`` is a different arc than last tick;
* the set of BINDING body constraints changed (``prices[*].bind_state == "binding"`` from GET
  /physiology/state: the body's own label; slack and "out of my hands" (exogenous) are not news);
* one of Prime's own acts settled (written by the runtime when a nomination closes);
* a concern faded: no evidence for ``CONCERN_TTL`` (an arc it points to was not active);
* first wake: Prime has never thought.

The model's hunger (``WantV1.intensity``) is NOT computed here: the body-economy market fills it
from the ``prime`` family's own act history (orion/physiology/appetite.py). This module only
answers "is there anything to think about at all".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Optional
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

MAX_CONCERNS = 3
MAX_PENDING_CHANGES = 20
MAX_RECENT_OUTCOMES = 10
CONCERN_TEXT_CAP = 300
# A concern with no evidence for a day fades. One day, because Temporal Self's arcs are per day
# (``/temporal-self/frame`` day_id): an arc not seen for a whole day is no longer part of today.
CONCERN_TTL = timedelta(hours=24)

Arm = Literal["treated", "control"]


class PrimeConcernV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concern_id: str
    text: str = Field(min_length=1, max_length=CONCERN_TEXT_CAP)
    # The Temporal Self arc this concern is about, when it is about one. Evidence for the concern is
    # that arc being Orion's active arc again.
    arc_id: Optional[str] = None
    opened_at: AwareDatetime
    last_evidence_at: AwareDatetime


class PrimeNominationV1(BaseModel):
    """One question Prime asked curiosity to look at (or held back), until its act closes."""

    model_config = ConfigDict(extra="forbid")

    act_id: str
    thought_id: str
    focus_id: str
    question: str
    why: str
    arm: Arm
    opened_at: AwareDatetime
    concern_id: Optional[str] = None
    # Treated only: what Hub answered. None until it answered; False with queue_error on refusal.
    queued: Optional[bool] = None
    queue_error: Optional[str] = None


class PrimeStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["prime.state.v1"] = "prime.state.v1"
    concerns: list[PrimeConcernV1] = Field(default_factory=list)
    open_nominations: list[PrimeNominationV1] = Field(default_factory=list)
    pending_changes: list[str] = Field(default_factory=list)
    recent_outcomes: list[str] = Field(default_factory=list)
    last_thought_at: Optional[AwareDatetime] = None
    last_reading: str = ""
    thoughts_total: int = 0
    # Cursors: what the last tick saw, so the next one can tell what moved.
    seen_active_arc_id: Optional[str] = None
    seen_binding: Optional[list[str]] = None


class PrimeSnapshotV1(BaseModel):
    """What the tick reads. ``None`` fields are unknown (the read failed), never "calm"."""

    model_config = ConfigDict(extra="forbid")

    active_arc_id: Optional[str] = None
    active_arc_label: Optional[str] = None
    # Body constraints whose bind_state is "binding", sorted. None when the body was not read.
    binding: Optional[list[str]] = None
    temporal_self_ok: bool = False
    body_ok: bool = False


def _push(items: list[str], text: str, cap: int) -> list[str]:
    out = [*items, text]
    return out[-cap:]


def note_change(state: PrimeStateV1, text: str) -> PrimeStateV1:
    return state.model_copy(update={"pending_changes": _push(state.pending_changes, text, MAX_PENDING_CHANGES)})


def note_outcome(state: PrimeStateV1, text: str) -> PrimeStateV1:
    """An own act settled: something to think about, and a line for the next frames."""
    state = note_change(state, text)
    return state.model_copy(update={"recent_outcomes": _push(state.recent_outcomes, text, MAX_RECENT_OUTCOMES)})


def observe(state: PrimeStateV1, snap: PrimeSnapshotV1, now: datetime) -> PrimeStateV1:
    """One tick: record what moved since the last one. Unknown reads change nothing."""
    if state.last_thought_at is None and not state.pending_changes:
        state = note_change(state, "first wake: I have not thought yet")

    if snap.temporal_self_ok and snap.active_arc_id and snap.active_arc_id != state.seen_active_arc_id:
        label = snap.active_arc_label or snap.active_arc_id
        state = note_change(state, f"attention moved to {label!r} (arc {snap.active_arc_id})")
        concerns = [
            c.model_copy(update={"last_evidence_at": now}) if c.arc_id == snap.active_arc_id else c
            for c in state.concerns
        ]
        state = state.model_copy(update={"seen_active_arc_id": snap.active_arc_id, "concerns": concerns})

    if snap.body_ok and snap.binding is not None and snap.binding != state.seen_binding:
        now_binding = ", ".join(snap.binding) or "nothing"
        state = note_change(state, f"body: binding constraints now {now_binding}")
        state = state.model_copy(update={"seen_binding": snap.binding})

    kept: list[PrimeConcernV1] = []
    for concern in state.concerns:
        if now - concern.last_evidence_at > CONCERN_TTL:
            state = note_change(state, f"concern faded with no evidence for a day: {concern.text!r}")
        else:
            kept.append(concern)
    return state.model_copy(update={"concerns": kept})


def wants_to_think(state: PrimeStateV1, now: datetime, *, refractory: timedelta) -> bool:
    """True when there is something to think about and the refractory interval has passed.
    Price, heat and lane contention are not decided here: the bid decides those."""
    if not state.pending_changes:
        return False
    return state.last_thought_at is None or now - state.last_thought_at >= refractory


def new_id() -> str:
    return uuid4().hex[:16]


__all__ = [
    "Arm",
    "CONCERN_TTL",
    "MAX_CONCERNS",
    "PrimeConcernV1",
    "PrimeNominationV1",
    "PrimeSnapshotV1",
    "PrimeStateV1",
    "new_id",
    "note_change",
    "note_outcome",
    "observe",
    "wants_to_think",
]
