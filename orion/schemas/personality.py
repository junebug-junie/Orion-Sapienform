"""Personality evidence: one choice Orion made, plus the options Orion turned down.

Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md, rev 4 ("one
organ: choices in, values out") and rev 3 ("only choices are evidence"). A choice point shows Orion
the top options a funded act would take and Orion picks one, or declines them all ("no move", "say
nothing"). Each pick is one ``PersonalityChoiceV1``.

Flow: a producer (orion-mind Prime, Hub outreach, ...) publishes the model on bus channel
``orion:personality:choice`` (kind ``personality.choice.v1``). orion-durable-runs is the single
writer: it inserts the row into ``personality_choice`` (``choice_id`` primary key, insert-only, a
re-delivery is a no-op), and its personality reducer folds those rows, and only those, into the
disposition graph. The curiosity lane is written by durable-runs itself from the curiosity
offer/outcome tables (``orion.personality.choices``).

A rule, a gate, a cap or a formula is not a choice: only an act where Orion (the model) picked among
real alternatives may write one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

PERSONALITY_CHOICE_SCHEMA_VERSION = "personality.choice.v1"
PERSONALITY_CHOICE_KIND = "personality.choice.v1"
PERSONALITY_CHOICE_CHANNEL = "orion:personality:choice"
MAX_CHOICE_LIST = 50
CHOICE_OPTION_TEXT_MAX = 400
CHOICE_SITUATION_MAX_KEYS = 16
CHOICE_SITUATION_KEY_MAX = 64
CHOICE_SITUATION_TEXT_MAX = 200

ChoiceLane = Literal["prime", "outreach", "curiosity", "reading", "chat"]


class ChoiceOptionV1(BaseModel):
    """One option Orion was shown. ``key`` is the producer's stable id for it (a concern id, a
    topic id, a prior id). ``text`` is what the option says, as Orion saw it; the reducer embeds it,
    so it must describe the option, not Orion's prose about it. ``market_rank`` is its position in
    the order the market (or the producer's ranking) offered it, 1 = first; None when the option was
    shown unranked."""

    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=CHOICE_OPTION_TEXT_MAX)
    market_rank: Optional[int] = Field(default=None, ge=1)


class PersonalityChoiceV1(BaseModel):
    """One choice and the options turned down.

    * ``chosen``: the option Orion picked. None only when ``declined_all``.
    * ``alternatives``: every other option shown. With ``declined_all`` these are the options
      Orion declined (at least one).
    * ``chosen_market_rank``: the chosen option's market rank (1 = Orion followed the order; it
      weighs 0.25 in the reducer, any other rank or None weighs 1.0). Left None, it is copied
      from ``chosen.market_rank``; set, it must equal it.
    * ``declined_all``: Orion picked "none" / "no move" / "say nothing" although options existed.
    * ``care_promoted``: the option reached the shown set only because the care term (rev 4 move 2)
      lifted it. Counts at half weight (runaway guard).
    * ``picks_in_decision``: how many options this same decision picked (a curiosity run can test
      several priors; each pick is one row). The reducer divides the weight by it, so a decision
      weighs the same however many picks it made. Single-pick producers leave it at 1.
    * ``situation``: a small FLAT dict of what Orion saw (arousal, binding constraint, company):
      at most 16 keys of at most 64 chars, scalar values, strings at most 200 chars.
    * ``act_id`` / ``correlation_id``: the act and the model call that made the pick.

    ``choice_id`` must start with ``<lane>:`` and be deterministic per decision (and pick), e.g.
    ``prime:<thought_id>``, so a re-published choice is the same row.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["personality.choice.v1"] = PERSONALITY_CHOICE_SCHEMA_VERSION
    choice_id: str = Field(min_length=1, max_length=300)
    lane: ChoiceLane
    occurred_at: AwareDatetime
    chosen: Optional[ChoiceOptionV1] = None
    alternatives: List[ChoiceOptionV1] = Field(default_factory=list, max_length=MAX_CHOICE_LIST)
    chosen_market_rank: Optional[int] = Field(default=None, ge=1)
    declined_all: bool = False
    care_promoted: bool = False
    picks_in_decision: int = Field(default=1, ge=1, le=MAX_CHOICE_LIST)
    situation: Dict[str, Any] = Field(default_factory=dict)
    act_id: Optional[str] = Field(default=None, max_length=300)
    correlation_id: Optional[str] = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _a_real_choice(self) -> "PersonalityChoiceV1":
        if not self.choice_id.startswith(f"{self.lane}:"):
            raise ValueError(f"choice_id must start with '{self.lane}:' (the lane's id namespace)")
        if not self.alternatives:
            raise ValueError("a choice needs at least one alternative (declined option) Orion was shown")
        if self.declined_all:
            if self.chosen is not None or self.chosen_market_rank is not None:
                raise ValueError("declined_all: chosen and chosen_market_rank must be None")
            if self.picks_in_decision != 1:
                raise ValueError("declined_all: picks_in_decision must be 1")
        elif self.chosen is None:
            raise ValueError("chosen is required unless declined_all")
        keys = [o.key for o in self.alternatives] + ([self.chosen.key] if self.chosen else [])
        if len(set(keys)) != len(keys):
            raise ValueError("option keys must be unique across chosen and alternatives")
        if self.chosen is not None:
            if self.chosen_market_rank is None:
                self.chosen_market_rank = self.chosen.market_rank
            elif self.chosen_market_rank != self.chosen.market_rank:
                raise ValueError("chosen_market_rank must equal chosen.market_rank")
        if len(self.situation) > CHOICE_SITUATION_MAX_KEYS:
            raise ValueError(f"situation holds at most {CHOICE_SITUATION_MAX_KEYS} keys")
        for k, v in self.situation.items():
            if not isinstance(k, str) or not k or len(k) > CHOICE_SITUATION_KEY_MAX:
                raise ValueError(f"situation keys are 1-{CHOICE_SITUATION_KEY_MAX} char strings")
            if v is not None and not isinstance(v, (str, int, float, bool)):
                raise ValueError("situation values are scalars (str, int, float, bool, None): keep it flat")
            if isinstance(v, str) and len(v) > CHOICE_SITUATION_TEXT_MAX:
                raise ValueError(f"situation strings hold at most {CHOICE_SITUATION_TEXT_MAX} chars")
        return self


# --- Personality graph (rev 4: choices only) ---------------------------------------------------
# Spec: docs/superpowers/specs/2026-10-16-orion-dynamic-personality-graph-design.md (rev 4).
# Single writer: orion-durable-runs' personality reducer (Postgres personality_* tables), folding
# only personality_choice rows. Frame: GET /personality/frame on durable-runs.

# Rev 4: the graph's evidence lanes are the choice lanes (``ChoiceLane``). Rev 2's prose lanes
# (dream, journal, reverie, reading text, day letter, chat/outreach text, self-inquiry seeds) are
# retired: their evidence rows stay in personality_evidence for audit, nothing folds them.
PersonalityLane = ChoiceLane
# "choice": a rev 4 node, born from a chosen-over-declined direction. The two rev 2 origins remain
# only so retired rows still validate; the reducer never loads or creates them.
DispositionOrigin = Literal["choice", "seed_self_inquiry", "recurrence"]
DispositionStatus = Literal["candidate", "active", "retired"]
PersonalityCueSkipReason = Literal["disabled", "fetch_failed", "invalid_frame", "stale", "empty", "render_failed"]
DISPOSITION_LABEL_MAX_CHARS = 160
FRAME_MAX_DISPOSITIONS = 12
NAMING_NAME_MAX_CHARS = 40
NAMING_NAME_MAX_WORDS = 6
NAMING_SENTENCE_MIN_CHARS = 20
NAMING_SENTENCE_MAX_CHARS = 240
NAMING_PACKET_MAX_EVIDENCE = 5
NAMING_PACKET_MAX_NEIGHBORS = 3
PERSONALITY_NAMING_KIND = "personality.naming.v1"
# Room for the header, five shortened node lines and the choice line (live 2026-10-20: at 600 only
# the top node fit, so the one non-prediction-error theme never reached Orion).
PERSONALITY_CUE_MAX_CHARS = 900


class DispositionV1(BaseModel):
    """One node of the graph: a tendency in what Orion picks over what Orion turns down.

    ``label`` is computed from the choices, never quoted from prose: "<chosen gist> over <declined
    gist>, n of m" (``orion.personality.graph.choice_label``). ``lane_counts`` is choices per lane.
    ``weight`` is the faded sum of the choices' evidence weights (0.25 for following the market's
    order, 1.0 for departing from it). Rev 2 nodes (origin recurrence / seed) are retired.
    """

    model_config = ConfigDict(extra="forbid")

    disposition_id: str = Field(min_length=1)
    label: str = Field(min_length=1, max_length=DISPOSITION_LABEL_MAX_CHARS)
    origin: DispositionOrigin
    status: DispositionStatus
    weight: float = Field(ge=0.0)
    evidence_count: int = Field(ge=0)
    lane_counts: dict[str, int] = Field(default_factory=dict)
    chosen_count: int = Field(default=0, ge=0)  # choices folded into it
    first_seen_at: AwareDatetime
    last_reinforced_at: AwareDatetime
    # Orion's own name for the node (self-naming, docs/.../2026-10-17-personality-orion-names-
    # dispositions-design.md). None until Orion names it. ``label`` keeps its meaning (computed).
    name: Optional[str] = Field(default=None, max_length=NAMING_NAME_MAX_CHARS)
    name_sentence: Optional[str] = Field(default=None, max_length=NAMING_SENTENCE_MAX_CHARS)


class ChoiceStyleV1(BaseModel):
    """How Orion has been choosing over ``window_days`` (every choice lane): how often Orion took
    the market's top option versus going their own way (another rank, unranked, or none)."""

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



# --- Self-naming (Orion names their own dispositions) ------------------------------------------
# Spec: docs/superpowers/specs/2026-10-17-personality-orion-names-dispositions-design.md (PR #2742),
# decisions 2026-10-20: names reach every lane at once; evidence quoting a node's own name adds no
# weight; grounding is recorded, never a refusal; no Juniper veto.
# Flow: durable-runs GET /personality/naming/next -> DispositionNamingPacketV1 -> Hub self-inquiry
# naming run -> DispositionNamingV1 on orion:personality:naming -> orion-sql-writer ->
# orion_disposition_naming -> durable-runs applies it (single writer of personality_*).

NamingVerdict = Literal["name", "reject", "merge", "no_answer"]
NamingHubCheck = Literal["accepted", "refused"]
NamingRefusedReason = Literal[
    "unknown_disposition", "packet_mismatch", "merge_target_not_shown", "evidence_not_shown",
    "name_too_long", "name_equals_label", "empty_sentence", "missing_name", "missing_merge_target",
    "no_evidence_refs", "sentence_too_long",
]


class NamingEvidenceV1(BaseModel):
    """One evidence row shown to Orion in a naming packet."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    lane: str = Field(min_length=1)
    occurred_at: AwareDatetime
    excerpt: str = Field(min_length=1, max_length=DISPOSITION_LABEL_MAX_CHARS)


class NamingNeighborV1(BaseModel):
    """A nearby non-retired node: the only valid MERGE targets."""

    model_config = ConfigDict(extra="forbid")

    disposition_id: str = Field(min_length=1)
    label: str = Field(min_length=1, max_length=DISPOSITION_LABEL_MAX_CHARS)  # name if named, else excerpt
    cosine: float = Field(ge=-1.0, le=1.0)


class NamingPriorVerdictV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: NamingVerdict
    name: Optional[str] = Field(default=None, max_length=NAMING_NAME_MAX_CHARS)
    sentence: Optional[str] = Field(default=None, max_length=NAMING_SENTENCE_MAX_CHARS)
    at: AwareDatetime


class DispositionNamingPacketV1(BaseModel):
    """What Orion is shown about one node. Built by durable-runs from its own rows. Weight is not
    shown: it is not a fact Orion can check against the excerpts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["personality.naming.packet.v1"] = "personality.naming.packet.v1"
    packet_id: str = Field(min_length=16, max_length=16)  # sha1(disposition_id, evidence ids, generated_at)[:16]
    disposition_id: str = Field(min_length=1)
    label_excerpt: str = Field(min_length=1, max_length=DISPOSITION_LABEL_MAX_CHARS)
    origin: DispositionOrigin
    evidence_count: int = Field(ge=0)
    lane_counts: dict[str, int] = Field(default_factory=dict)
    first_seen_at: AwareDatetime
    last_reinforced_at: AwareDatetime
    evidence: List[NamingEvidenceV1] = Field(min_length=1, max_length=NAMING_PACKET_MAX_EVIDENCE)
    # {"arousal": {"strained": 10}, "day_phase": {...}, "company": {"true": 3, "false": 7}}
    situation_mix: dict[str, dict[str, int]] = Field(default_factory=dict)
    situation_coverage: float = Field(ge=0.0, le=1.0)
    neighbors: List[NamingNeighborV1] = Field(default_factory=list, max_length=NAMING_PACKET_MAX_NEIGHBORS)
    prior_verdict: Optional[NamingPriorVerdictV1] = None
    generated_at: AwareDatetime


class DispositionNamingV1(BaseModel):
    """Orion's answer to one packet. Exactly one per naming run (``no_answer`` when nothing was
    written). Hub checks structure (``hub_check``); it never repairs an id."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["personality.naming.v1"] = "personality.naming.v1"
    naming_id: str = Field(min_length=1)  # uuid5(NAMESPACE_URL, f"{run_id}:{packet_id}")
    run_id: str = Field(min_length=1)
    packet_id: str = Field(min_length=16, max_length=16)
    disposition_id: str = Field(min_length=1)
    verdict: NamingVerdict
    name: Optional[str] = Field(default=None, max_length=200)  # raw; length rules are hub_check reasons
    sentence: Optional[str] = Field(default=None, max_length=2000)
    merge_with: Optional[str] = None
    evidence_refs: List[str] = Field(default_factory=list, max_length=NAMING_PACKET_MAX_EVIDENCE * 4)
    shown: DispositionNamingPacketV1
    hub_check: NamingHubCheck
    refused_reasons: List[NamingRefusedReason] = Field(default_factory=list)
    written_at: AwareDatetime
