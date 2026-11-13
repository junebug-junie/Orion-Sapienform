"""Orion asking Juniper a question and keeping the answer against it.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md idea 3.
Before this, outreach was one-way: Orion could say something but had no place
to receive an answer. An ask stays open until it is answered, dismissed, or
expires, and the answer is stored on the same row as the question.

Producers open asks by inserting ``OrionAskV1``-shaped rows into
``orion_ask`` (no bus event: the Hub polls the table). The Hub renders open
asks and, on answer, updates the row and publishes ``OrionAskAnsweredV1`` on
``orion:ask:answered``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

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
    # Every picture the card shows, in order (2026-10-15). A "same person?" ask
    # shows two faces. Producers set image_ref = image_refs[0] so readers that
    # only know image_ref still show the first one. Rows from before this field
    # have only image_ref.
    image_refs: List[str] = Field(default_factory=list)
    status: AskStatus = "open"
    answer: Optional[str] = None
    answered_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    # What the question is about, so the answer can be routed back to it.
    # e.g. source_kind="vision_individual", source_ref="<individual_id>";
    # source_kind="memory_confirmation", source_ref="memory-confirm-<memory_id>" (2026-10-06,
    # orion.memory.episode.confirmation: answered only through the Hub's /resolve route, which also
    # writes the attention_loop_outcome row);
    # source_kind="vision_same_individual", source_ref="<individual_a>|<individual_b>" (sorted
    # ids, 2026-10-15): "are these the same person?", answer "yes" or "no";
    # source_kind="vision_story_person", source_ref="<individual_id>" (2026-10-18): written by
    # the Hub already answered, "someone else" / "not sure" about one person on a walkway story
    # card (answer None, answer_detail.pictures). A name typed on that card is a plain
    # "vision_individual" answer with answer_detail.story_id;
    # source_kind="vision_person_confirm", source_ref="<individual_id>|<name>" (2026-10-20,
    # orion-sql-writer person learning): "Is this <name>?", answer "yes" or "no" ("can't tell" =
    # dismiss); yes names the individual, no adds "not <name>" album exemplars.
    # source_kind="person_fact", source_ref="<person_id>|<category>" (2026-10-22,
    # orion-sql-writer app/person_record.py): a question about a person ("Who is X to you?",
    # pronouns, gender, ...; orion/person_model/categories.yaml). Free-text answer, stored
    # verbatim as a person_fact row; dismiss = do not ask that category again.
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
    # Names Juniper typed on a "same person?" card (2026-10-15): individual_id ->
    # name, for one or both faces. Each is a direct "who is this?" answer for that
    # individual. Additive: the Hub leaves the key out of the payload when it is
    # empty, so a consumer built before this field still validates every event
    # that carries no names. Deploy consumers (orion-substrate-runtime) first.
    names: Dict[str, str] = Field(default_factory=dict)
