"""Curiosity contractor peer contracts — HelpRequest + PeerBrief.

Design: docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md

Not substrate FrontierInvocation / frontier_buddy. Peer never authors
:Prior / :Finding / :SelfDefinition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HELP_REQUEST_CHANNEL = "orion:curiosity:help:request"
HELP_REQUEST_KIND = "curiosity.help.request.v1"
PEER_BRIEF_CHANNEL = "orion:curiosity:peer:brief"
PEER_BRIEF_KIND = "curiosity.peer.brief.v1"
PEER_BRIEF_CONSUMED_CHANNEL = "orion:curiosity:peer:brief:consumed"
PEER_BRIEF_CONSUMED_KIND = "curiosity.peer.brief.consumed.v1"

CuriosityPeerModeV1 = Literal["world_curiosity", "self_inquiry"]
CuriosityPeerNameV1 = Literal["cursor_auto", "claude_room"]
CuriosityPeerBriefStatusV1 = Literal["ok", "failed", "refused_budget", "empty"]

MAX_QUESTION_CHARS = 2000
MAX_SUMMARY_CHARS = 4000
MAX_POINTER_CHARS = 500
MAX_LIST_ITEMS = 32


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clip(text: object, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"


class PeerAskExpectationV1(BaseModel):
    """Orion's forecast and actual alternatives, authored before hiring."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    expected_reply: str = Field(min_length=1, max_length=2000)
    if_not_asked: str = Field(min_length=1, max_length=2000)
    alternatives: List[str] = Field(min_length=2, max_length=8)
    chosen_action: Literal["hire_peer"] = "hire_peer"
    within_seconds: int = Field(ge=1, le=86400)

    @field_validator("alternatives")
    @classmethod
    def _alternatives(cls, values: List[str]) -> List[str]:
        values = [v.strip() for v in values]
        if "hire_peer" not in values or any(not v or len(v) > 200 for v in values) or len(set(values)) != len(values):
            raise ValueError("record distinct nonempty alternatives including hire_peer")
        return values


class HelpRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.help.request.v1"] = "curiosity.help.request.v1"
    help_id: str
    run_id: str
    prior_id: Optional[str] = None
    mode: CuriosityPeerModeV1
    question: str
    tried_summary: str
    success_criteria: str
    written_at: datetime = Field(default_factory=_utc_now)
    expectation: Optional[PeerAskExpectationV1] = None

    @field_validator("question", "tried_summary", "success_criteria", mode="before")
    @classmethod
    def _clip_text(cls, v: object) -> str:
        return clip(v, MAX_QUESTION_CHARS)


class PeerBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.peer.brief.v1"] = "curiosity.peer.brief.v1"
    brief_id: str
    help_id: str
    run_id: str
    prior_id: Optional[str] = None
    peer: CuriosityPeerNameV1
    status: CuriosityPeerBriefStatusV1
    summary: str = ""
    evidence_pointers: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    suggested_next_looks: List[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    written_at: datetime = Field(default_factory=_utc_now)

    @field_validator("summary", mode="before")
    @classmethod
    def _clip_summary(cls, v: object) -> str:
        return clip(v, MAX_SUMMARY_CHARS)

    @field_validator(
        "evidence_pointers", "open_questions", "suggested_next_looks", mode="before"
    )
    @classmethod
    def _clip_lists(cls, v: object) -> list[str]:
        if not v:
            return []
        out: list[str] = []
        for item in list(v)[:MAX_LIST_ITEMS]:
            text = clip(item, MAX_POINTER_CHARS)
            if text:
                out.append(text)
        return out

    @field_validator("refusal_reason", mode="before")
    @classmethod
    def _clip_refusal(cls, v: object) -> object:
        if v is None:
            return None
        return clip(v, MAX_SUMMARY_CHARS)


class PeerBriefConsumedV1(BaseModel):
    """Hub signals that soft-nudge injected these briefs; peer MERGEs consumed."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["curiosity.peer.brief.consumed.v1"] = (
        "curiosity.peer.brief.consumed.v1"
    )
    brief_ids: List[str] = Field(default_factory=list)
    consumer_run_id: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{6,32}$")
    phase: Literal["offered", "completed"] = "offered"

    @model_validator(mode="after")
    def _completion_run(self):
        if self.phase == "completed" and self.consumer_run_id is None:
            raise ValueError("completion requires consumer_run_id")
        return self

    @field_validator("brief_ids", mode="before")
    @classmethod
    def _clip_ids(cls, v: object) -> list[str]:
        if not v:
            return []
        out: list[str] = []
        for item in list(v)[:MAX_LIST_ITEMS]:
            text = clip(item, MAX_POINTER_CHARS)
            if text:
                out.append(text)
        return out
