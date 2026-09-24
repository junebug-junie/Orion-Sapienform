"""Orion asking Juniper a question and keeping the answer against it.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 3.
Before this, outreach was one-way: Orion could say something but had no place
to receive an answer. An ask stays open until it is answered, dismissed, or
expires, and the answer is stored on the same row as the question.

Producers open asks by inserting into ``orion_ask`` and publishing
``OrionAskV1`` on ``orion:ask:opened``. The Hub renders open asks and, on
answer, updates the row and publishes ``OrionAskAnsweredV1`` on
``orion:ask:answered``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

AskStatus = Literal["open", "answered", "dismissed", "expired"]


class OrionAskV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["orion.ask.v1"] = "orion.ask.v1"
    ask_id: str
    asked_of: str = "juniper"
    question: str = Field(min_length=1)
    evidence_refs: List[str] = Field(default_factory=list)
    image_ref: Optional[str] = None
    status: AskStatus = "open"
    answer: Optional[str] = None
    answered_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    # What the question is about, so the answer can be routed back to it.
    # e.g. source_kind="vision_individual", source_ref="<individual_id>".
    source_kind: str
    source_ref: str


class OrionAskAnsweredV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["orion.ask.answered.v1"] = "orion.ask.answered.v1"
    ask_id: str
    status: Literal["answered", "dismissed"]
    answer: Optional[str] = None
    answered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_kind: str
    source_ref: str
