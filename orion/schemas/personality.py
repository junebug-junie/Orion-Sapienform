"""Personality evidence: one choice Orion made, plus the options Orion did not pick.

Dynamic personality graph design (PR #2726), section 1 "Evidence". The evidence unit is a
choice: what was picked, what else was on the table, what situation it happened in, and what
it cost. A tendency ("Orion goes off the suggested order", "Orion's goals pull attention
toward the body") can only be read off many of these, never off one.

P0 scope: a READ-ONLY reconstruction from two lanes that already persist chosen + losers.
No bus channel, no table, no writer yet. Producer: ``orion.personality.choices`` (pure
reducers over stored rows). Consumer: ``scripts/analysis/measure_personality_choices.py``.

* ``curiosity_prior_selection`` -- Orion (the LLM) picks which priors to test in a curiosity
  investigation. Source rows: ``curiosity_offer_decisions`` + ``curiosity_run_outcomes``.
* ``voluntary_attention_override`` -- Orion's active goal biases the attention formula and
  flips the bottom-up winner. Source rows: ``substrate_attention_broadcast_log``
  (``frame.voluntary_override``). The chooser is a formula, not the LLM; it is labelled so.

Lists are capped at ``MAX_CHOICE_LIST``. The reducer truncates; the model refuses more.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

PERSONALITY_CHOICE_SCHEMA_VERSION = "personality.choice.v1"
PERSONALITY_CHOICE_KIND = "personality.choice.v1"
CHOICE_OPTION_KIND = "personality.choice.option.v1"
MAX_CHOICE_LIST = 50

ChoiceLane = Literal["curiosity_prior_selection", "voluntary_attention_override"]
ChoiceChooser = Literal["orion_llm", "goal_biased_formula"]


class ChoiceOptionV1(BaseModel):
    """One option: picked or not. ``rank`` is its position in the order Orion was shown
    (1 = first); None when it was not in that ordered list. ``score`` is the lane's own
    number for the option (curiosity: ``expected_nats``; attention: bottom-up salience)."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    rank: Optional[int] = Field(default=None, ge=1)
    score: Optional[float] = None


class PersonalityChoiceV1(BaseModel):
    """One choice and its alternatives. ``choice_id`` is ``f"{lane}:{source_id}"`` so a
    re-run over the same rows yields the same ids."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["personality.choice.v1"] = PERSONALITY_CHOICE_SCHEMA_VERSION
    choice_id: str = Field(min_length=1)
    lane: ChoiceLane
    chooser: ChoiceChooser
    occurred_at: AwareDatetime
    chosen: List[ChoiceOptionV1] = Field(min_length=1, max_length=MAX_CHOICE_LIST)
    alternatives: List[ChoiceOptionV1] = Field(default_factory=list, max_length=MAX_CHOICE_LIST)
    # Lowest shown rank among the chosen options. 1 = Orion took the first thing in the order;
    # None = nothing chosen was in the ordered list (off-menu or from an unranked section).
    chosen_best_rank: Optional[int] = Field(default=None, ge=1)
    cost_seconds: Optional[float] = Field(default=None, ge=0.0)
    cost_nats: Optional[float] = None
    # Attention lane only: the effort budget the goal bias spent to flip the winner
    # (``voluntary_override.effort_spent``). Not seconds and not nats, so it has its own field.
    cost_effort: Optional[float] = None
    situation_refs: List[str] = Field(default_factory=list, max_length=MAX_CHOICE_LIST)
    correlation_id: Optional[str] = None
    source_ref: str = Field(min_length=1)


# --- Personality graph loop (rev 2, bottom-up) -------------------------------------------------
# Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 2).
# Single writer: orion-durable-runs' personality reducer (Postgres personality_* tables).
# Frame: GET /personality/frame on durable-runs. Readers render it with
# ``orion.personality.cue.project_personality_cue`` (pure) into chat stance, reverie, curiosity
# kickoff and outreach -- as Orion's own recurring concerns, with evidence, never tone rules.

PersonalityLane = Literal[
    "dream", "journal", "reverie", "reading", "day_letter", "chat", "outreach",
    "curiosity_choice", "self_inquiry_seed",
]
DispositionOrigin = Literal["seed_self_inquiry", "recurrence"]
DispositionStatus = Literal["candidate", "active", "retired"]
PersonalityCueSkipReason = Literal["disabled", "fetch_failed", "invalid_frame", "stale", "empty", "render_failed"]
DISPOSITION_LABEL_MAX_CHARS = 160
FRAME_MAX_DISPOSITIONS = 12
# Room for the header, five shortened node lines and the choice line (live 2026-10-20: at 600 only
# the top node fit, so the one non-prediction-error theme never reached Orion).
PERSONALITY_CUE_MAX_CHARS = 900


class DispositionV1(BaseModel):
    """One node of the graph: something Orion keeps returning to.

    ``label`` is a short excerpt of Orion's own words from the evidence closest to the node's
    centroid (no model writes it). ``lane_counts`` is evidence per lane; a node seen in several
    independent lanes is stronger than one repeated in one lane. ``weight`` already includes fade.
    """

    model_config = ConfigDict(extra="forbid")

    disposition_id: str = Field(min_length=1)
    label: str = Field(min_length=1, max_length=DISPOSITION_LABEL_MAX_CHARS)
    origin: DispositionOrigin
    status: DispositionStatus
    weight: float = Field(ge=0.0)
    evidence_count: int = Field(ge=0)
    lane_counts: dict[str, int] = Field(default_factory=dict)
    chosen_count: int = Field(default=0, ge=0)  # curiosity picks that reinforced it
    first_seen_at: AwareDatetime
    last_reinforced_at: AwareDatetime


class ChoiceStyleV1(BaseModel):
    """How Orion has been choosing in curiosity over ``window_days``: how often Orion took the
    top-ranked suggestion versus going their own way (off the order, or off the list)."""

    model_config = ConfigDict(extra="forbid")

    window_days: int = Field(ge=1)
    choices: int = Field(ge=0)
    took_top: int = Field(ge=0)
    went_own_way: int = Field(ge=0)


class PersonalityFrameV1(BaseModel):
    """The read-back projection. ``dispositions`` holds active nodes only, strongest first."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["personality.frame.v1"] = "personality.frame.v1"
    generated_at: AwareDatetime
    dispositions: List[DispositionV1] = Field(default_factory=list, max_length=FRAME_MAX_DISPOSITIONS)
    choice_style: Optional[ChoiceStyleV1] = None
    active_count: int = Field(default=0, ge=0)
    candidate_count: int = Field(default=0, ge=0)
    retired_count: int = Field(default=0, ge=0)
    evidence_rows: int = Field(default=0, ge=0)
    warnings: List[str] = Field(default_factory=list, max_length=20)


class PersonalityCueV1(BaseModel):
    """The rendered read-back line. ``text`` is "" when skipped."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["personality.cue.v1"] = "personality.cue.v1"
    text: str = Field(default="", max_length=PERSONALITY_CUE_MAX_CHARS)
    rendered_at: AwareDatetime
    frame_generated_at: Optional[AwareDatetime] = None
    disposition_ids: List[str] = Field(default_factory=list)
    skip_reason: Optional[PersonalityCueSkipReason] = None
