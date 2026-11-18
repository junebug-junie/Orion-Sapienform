"""Retro play session summary: what one Zelda session leaves in Orion's memory.

Orion plays Zelda, section F (docs/superpowers/specs/2026-10-22-orion-plays-zelda-design.md).
Producer: orion-hub (``scripts/retro_memory_export.py``), at the end of every co-play or leisure
session. Stored twice: one ``memory_cards`` row (provenance/trust_source ``retro_play``, cross-day
recall) and one ``retro_play_sessions`` row (the Temporal Self ``retro_play`` source, same day).

The summary is facts and counts only, built deterministically from the session journal and the
atlas delta. No model writes it. Not exported in this summary: frames, RAM, ROM, raw journals. It feeds
no phi, arousal, reward or training signal. (This is a statement about the summary only. Since game
company patch 2 Hub's Retro gateway also LOOKS at the game: up to 4 stills per look go to the percept
store for one hour, one model call writes a note, and only the note is kept in ``retro_watch_look``;
see services/orion-hub/scripts/retro_watch.py. ``looks`` below is the count of those notes.)

Inputs (``RetroJournalEntryV1``, ``RetroAtlasDeltaV1``) come from the Retro lanes (A: the atlas
session record ``zelda.atlas.session`` on ``GET /sessions/{id}``; B: the leisure runner's
``RetroSessionEnded``). They ignore fields this module does not read, so a lane adding a field does
not break the export; every field read here is listed on the model. A count the lanes did not
record is None and is left out of the summary, never written as 0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

SUMMARY_MAX = 300  # orion/schemas/temporal_self.py LABEL_MAX: the summary is the arc label.

# human_play (2026-11-17, game company patch 1): a sitting where a person played and Orion held no
# controller. Written by services/orion-hub/scripts/retro_sitting.py; credits Orion with nothing.
SessionKind = Literal["co_play", "leisure", "human_play"]
# Where the control share comes from:
#   frames          -- per-port frame counts from ``control`` journal lines;
#   leisure_sponsor -- a leisure session with no frame lines: Retro's arbiter lets any human acquire
#                      or takeover preempt the leisure grant and the runner then ends the session
#                      (``human_preempted``), so every frame it ran was Orion's (human share 0);
#   unknown         -- a co-play session with no frame lines: no share is claimed either way.
#   human_input     -- a human_play sitting: who played is known from the browser's input messages
#                      (orion/retro/sitting.py); frames are not counted, so human_frames and
#                      orion_frames are 0 and human_share is None.
ControlBasis = Literal["frames", "leisure_sponsor", "unknown", "human_input"]
# What the game page was told about who plays. Never a guess, never a name from a camera.
PlayerStated = Literal["juniper", "guest"]
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
    """What this session added to the lineage's atlas (spec section C), observed only. Accepts Lane A's
    atlas session record as is (``rooms_new``, ``deaths`` counts; its other keys are ignored). A list
    of room/cave keys is also accepted for ``rooms_new``/``caves_new`` and counted (distinct keys).
    None = not recorded."""

    model_config = ConfigDict(extra="ignore")

    rooms_new: int | None = Field(None, ge=0)
    caves_new: int | None = Field(None, ge=0)
    deaths: int | None = Field(None, ge=0)
    items_gained: list[RetroAtlasItemV1] = Field(default_factory=list)
    death_rooms: list[str] = Field(default_factory=list)

    @field_validator("rooms_new", "caves_new", mode="before")
    @classmethod
    def _count_keys(cls, value):
        if isinstance(value, (list, tuple, set)):
            return len({str(v) for v in value})
        return value


class RetroPortControlV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: int = Field(ge=0, le=7)
    human_frames: int = Field(ge=0)
    orion_frames: int = Field(ge=0)


class RetroPlaySessionSummaryV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["retro_play_session.v1"] = "retro_play_session.v1"
    # The row key: the Retro session id for leisure, "<retro session id>:<handoff start>" for a
    # co-play handoff (one Retro session can hold several).
    session_id: str = Field(min_length=1, max_length=256)
    retro_session_id: str | None = Field(None, max_length=256)
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
    control_basis: ControlBasis
    # Lane B's flag: a human acted during the session, or presence was unreadable.
    human_may_be_in_control: bool = False
    human_frames: int = Field(ge=0)
    orion_frames: int = Field(ge=0)
    # None when no controlled frame was recorded: no share can be claimed either way.
    human_share: float | None = Field(None, ge=0.0, le=1.0)
    # Spec section F (#2751 section 7.4): a human drove more than half of the frames. Never
    # recorded as Orion's achievement.
    played_together: bool
    rooms_new: int | None = Field(None, ge=0)
    caves_new: int | None = Field(None, ge=0)
    items_gained: list[str] = Field(default_factory=list)
    deaths: int | None = Field(None, ge=0)
    death_rooms: list[str] = Field(default_factory=list)
    # None: no lane recorded a decision count (not claimed as 0).
    decisions: int | None = Field(None, ge=0)
    predictions_met: int = Field(ge=0)
    predictions_refuted: int = Field(ge=0)
    predictions_unknown: int | None = Field(None, ge=0)
    human_interventions: int = Field(ge=0)
    summary: str = Field(min_length=1, max_length=SUMMARY_MAX)
    # human_play rows only. None: nobody told the page who plays (treated as a guest everywhere).
    # Only Hub builds this model (retro_memory_export.py, retro_sitting.py); readers take the table row.
    player_stated: PlayerStated | None = None
    # human_play rows only (game company patch 2): how many notes Orion stored while watching this sitting
    # (retro_watch_look rows with outcome 'stored'). None: the watch did not run for it. Counts only.
    looks: int | None = Field(None, ge=0)


__all__ = [
    "ControlBasis",
    "Controller",
    "RetroAtlasDeathV1",
    "RetroAtlasDeltaV1",
    "RetroAtlasItemV1",
    "RetroJournalEntryV1",
    "PlayerStated",
    "RetroPlaySessionSummaryV1",
    "RetroPortControlV1",
    "SUMMARY_MAX",
    "SessionKind",
    "Verdict",
]
