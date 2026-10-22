"""Retro play session summary: what one Zelda session leaves in Orion's memory.

Orion plays Zelda, section F (docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md).
Producer: orion-hub (``scripts/retro_memory_export.py``), at the end of every co-play or leisure
session. Stored twice: one ``memory_cards`` row (provenance/trust_source ``retro_play``, cross-day
recall) and one ``retro_play_sessions`` row (the Temporal Self ``retro_play`` source, same day).

The summary is facts and counts only, built deterministically from the session journal and the
atlas delta. No model writes it. Not exported: frames, RAM, ROM, raw journals. It feeds no phi,
arousal, reward or training signal.

Inputs (``RetroJournalEntryV1``, ``RetroAtlasDeltaV1``) are written by the Retro lanes (A: atlas,
goal_finished; B: the session runner). They ignore fields this module does not read, so a lane
adding a field does not break the export; every field read here is listed on the model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

SUMMARY_MAX = 300  # orion/schemas/temporal_self.py LABEL_MAX: the summary is the arc label.

SessionKind = Literal["co_play", "leisure"]
Controller = Literal["human", "orion"]
Verdict = Literal["met", "refuted", "unknown"]


class RetroJournalEntryV1(BaseModel):
    """One journal line. Kinds read here:

    - ``session_started``: ``at`` (required once).
    - ``session_ended``: ``at`` (required once), ``reason``.
    - ``control``: ``port``, ``controller`` (human | orion), ``frames`` -- frames that controller
      drove on that port since the previous ``control`` line for the port. Summed per port.
    - ``goal_finished`` (spec section B): ``decision_id``, ``verdict`` (met | refuted | unknown),
      ``human_intervention``.

    Any other kind is ignored."""

    model_config = ConfigDict(extra="ignore")

    kind: str
    at: AwareDatetime | None = None
    reason: str | None = None
    port: int | None = Field(None, ge=0, le=7)
    controller: Controller | None = None
    frames: int | None = Field(None, ge=0)
    decision_id: str | None = None
    verdict: Verdict | None = None
    human_intervention: bool | None = None


class RetroAtlasItemV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    item: str = Field(min_length=1)
    value: int | None = None
    room: str | None = None


class RetroAtlasDeathV1(BaseModel):
    model_config = ConfigDict(extra="ignore")

    room: str | None = None


class RetroAtlasDeltaV1(BaseModel):
    """What this session added to the lineage's atlas (spec section C), observed only. ``rooms_new``
    and ``caves_new`` are room/cave keys first seen in this session (``level:room``)."""

    model_config = ConfigDict(extra="ignore")

    rooms_new: list[str] = Field(default_factory=list)
    caves_new: list[str] = Field(default_factory=list)
    items_gained: list[RetroAtlasItemV1] = Field(default_factory=list)
    deaths: list[RetroAtlasDeathV1] = Field(default_factory=list)


class RetroPortControlV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int = Field(ge=0, le=7)
    human_frames: int = Field(ge=0)
    orion_frames: int = Field(ge=0)


class RetroPlaySessionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["retro_play_session.v1"] = "retro_play_session.v1"
    session_id: str = Field(min_length=1, max_length=256)
    lineage_id: str | None = Field(None, max_length=256)
    game_id: str = Field(min_length=1, max_length=64)
    game_title: str = Field(min_length=1, max_length=120)
    kind: SessionKind
    # Set for a leisure session the motor allocator admitted: closes its `play` act.
    dispatch_id: str | None = Field(None, max_length=256)
    started_at: AwareDatetime
    ended_at: AwareDatetime
    duration_sec: float = Field(ge=0.0)
    end_reason: str | None = Field(None, max_length=200)
    control: list[RetroPortControlV1] = Field(default_factory=list)
    human_frames: int = Field(ge=0)
    orion_frames: int = Field(ge=0)
    # None when no controlled frame was recorded: no share can be claimed either way.
    human_share: float | None = Field(None, ge=0.0, le=1.0)
    # Spec section F (#2751 section 7.4): a human drove more than half of the frames. Never
    # recorded as Orion's achievement.
    played_together: bool
    rooms_new: int = Field(ge=0)
    caves_new: int = Field(ge=0)
    items_gained: list[str] = Field(default_factory=list)
    deaths: int = Field(ge=0)
    death_rooms: list[str] = Field(default_factory=list)
    decisions: int = Field(ge=0)
    predictions_met: int = Field(ge=0)
    predictions_refuted: int = Field(ge=0)
    predictions_unknown: int = Field(ge=0)
    human_interventions: int = Field(ge=0)
    summary: str = Field(min_length=1, max_length=SUMMARY_MAX)


__all__ = [
    "Controller",
    "RetroAtlasDeathV1",
    "RetroAtlasDeltaV1",
    "RetroAtlasItemV1",
    "RetroJournalEntryV1",
    "RetroPlaySessionSummaryV1",
    "RetroPortControlV1",
    "SUMMARY_MAX",
    "SessionKind",
    "Verdict",
]
