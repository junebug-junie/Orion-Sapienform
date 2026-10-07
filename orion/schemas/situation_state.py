"""Orion's running situation: what is true right now (spec 2026-10-07-situation-graph-design.md).

One writer: the ``situation.update`` durable graph in orion-durable-runs. It re-derives the facts
from ``episode_memory`` on every event, carries revision / lapsed / primed memories across events
in its LangGraph checkpoint, and projects this model to Redis (``orion:situation:latest``) and the
bus (``orion:situation:state``).

SHADOW (step 2 of the spec): nothing reads it yet. Chat starts reading it in step 3.

Every field here has a producer in that graph. Fields the spec lists for later steps (Orion's
self-view, recent reading, themes, open threads, presence) are added when their producer lands.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SITUATION_WORKFLOW = "situation.update"
SITUATION_STATE_KIND = "situation.state.v1"
SITUATION_STATE_CHANNEL = "orion:situation:state"
SITUATION_STATE_REDIS_KEY = "orion:situation:latest"
SITUATION_THREAD_PREFIX = "situation:juniper"

FactSlot = Literal["whereabouts", "doing", "waiting_on", "recent"]
UntilSource = Literal["juniper_words", "default_ttl", "follow_up"]
EventKind = Literal["chat_turn", "episode_distilled", "tick", "boot"]


class SituationFactV1(BaseModel):
    """One currently-true fact, pointing back to its episode memory."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    slot: FactSlot
    gist: str = Field(..., max_length=240)
    valid_from: datetime
    valid_until: datetime
    # juniper_words: expires_at grounded by her own quote (validator v5). default_ttl: no stated
    # end, shown "as of" valid_from. follow_up: the distiller's follow_up window.
    until_source: UntilSource
    voice: str
    confirmation: str
    referents: List[str] = Field(default_factory=list)


class SituationPrimedV1(BaseModel):
    """A memory prepared for the next turn because it shares a referent with the situation."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    gist: str = Field(..., max_length=240)
    voice: str
    confirmation: str
    why: str  # the cue referent key that brought it back
    score: float


class SituationLapsedV1(BaseModel):
    """A fact that stopped being current, so "Orion forgot" differs from "never told"."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str
    slot: FactSlot
    gist: str = Field(..., max_length=240)
    lapsed_at: datetime


class SituationEventRefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    kind: EventKind
    correlation_id: Optional[str] = None


class SituationJuniperV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Current state needs a stated duration: whereabouts / doing only hold facts whose end date
    # Juniper gave (until_source juniper_words). A `happened` memory with no end date is "recent",
    # never current state: occurred_at is when she TOLD Orion, so an old trip retold today would
    # otherwise read as where she is now (live check 2026-10-07: Austin karaoke as whereabouts).
    whereabouts: Optional[SituationFactV1] = None
    doing: List[SituationFactV1] = Field(default_factory=list, max_length=3)
    waiting_on: List[SituationFactV1] = Field(default_factory=list, max_length=3)
    recent: List[SituationFactV1] = Field(default_factory=list, max_length=3)


class SituationRecallV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cues: List[str] = Field(default_factory=list)
    # The cues the primed set was actually built from. Differs from ``cues`` after a failed or
    # timed-out priming, so the next step re-primes instead of trusting a stale set.
    primed_cues: List[str] = Field(default_factory=list)
    primed: List[SituationPrimedV1] = Field(default_factory=list, max_length=6)
    primed_at: Optional[datetime] = None
    # < revision means priming is behind (or failed); the turn would still use the last set.
    primed_revision: int = 0


class SituationStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["situation.state.v1"] = "situation.state.v1"
    thread_id: str
    revision: int = 0
    updated_at: datetime
    last_event: Optional[SituationEventRefV1] = None
    juniper: SituationJuniperV1 = Field(default_factory=SituationJuniperV1)
    recall: SituationRecallV1 = Field(default_factory=SituationRecallV1)
    lapsed: List[SituationLapsedV1] = Field(default_factory=list, max_length=3)
