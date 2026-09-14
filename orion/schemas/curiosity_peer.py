"""Curiosity contractor peer contracts — HelpRequest + PeerBrief.

Design: docs/superpowers/specs/2026-09-14-orion-contractor-peer-design.md

Not substrate FrontierInvocation / frontier_buddy. Peer never authors
:Prior / :Finding / :SelfDefinition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
