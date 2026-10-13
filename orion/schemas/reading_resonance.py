"""Reading resonance: the frozen evidence packet a judge sees, and the judge's answer.

Design: docs/superpowers/specs/2026-10-13-reading-resonance-design.md (PR #2642).

Patch A (this file's first version) is read-only: the packet is built and the judge
contract is checked offline by services/orion-hub/evals/run_reading_resonance_eval.py.
Nothing here is published on the bus yet.

Similarity is NOT evidence. ``ResonanceCandidateV1.score`` is the retrieval rank score
(centered, hub-corrected cosine; see orion/world_pulse_read/resonance_similarity.py),
kept for the eval only. It is never shown to the judge and never decides "interesting".
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ResonanceRefKind = Literal[
    "self_prior",
    "lived_question",
    "lived_answer",
    "atlas_concept",
    "crystallization",
    "curiosity_finding",
    "dream",
    "open_thread",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResonanceNeighborV1(_Base):
    """One piece of a candidate's graph neighbourhood, with its own stored text.

    For a prior: a Finding that SUPPORTS/CONTRADICTS/is ABOUT it, or a PriorRevision.
    For a lived question: one of its LivedAnswers. For an atlas concept: a 1-hop
    substrate neighbour over a substrate edge, or an Assertion it is subject/object of.
    """

    relation: str = Field(min_length=1, max_length=80)
    ref_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=600)


class ResonanceCandidateV1(_Base):
    ref_kind: ResonanceRefKind
    ref_id: str = Field(min_length=1)
    store: str = Field(min_length=1)  # e.g. "falkor:orion_worldview:Prior"
    excerpt: str = Field(min_length=1, max_length=600)  # re-read from the record store
    neighborhood: list[ResonanceNeighborV1] = Field(default_factory=list, max_length=6)
    score: float  # retrieval rank only; never shown to the judge, never a gate on interest
    recent_packet_count: int = Field(default=0, ge=0)  # attractor guard


class ResonancePacketV1(_Base):
    packet_id: str = Field(min_length=1)
    seed_id: str = Field(min_length=1)
    root_request_id: str | None = None
    read_title: str = ""
    read_url: str = ""
    read_summary: str = ""  # Stage 2 summary
    read_learned: str = ""  # Stage 1 what_i_learned: what the article itself says
    read_text_sha256: list[str] = Field(default_factory=list)
    candidates: list[ResonanceCandidateV1] = Field(default_factory=list, max_length=24)
    built_at: datetime


class ResonanceCitationV1(_Base):
    ref_kind: ResonanceRefKind
    ref_id: str = Field(min_length=1)
    how: str = Field(min_length=1, max_length=400)


class ReadingResonanceJudgmentV1(_Base):
    """What the judge must return. Checked deterministically by
    orion.world_pulse_read.resonance.check_judgment before anything is accepted."""

    article_summary: str = Field(min_length=1, max_length=1200)  # 3-5 sentences, the article only
    read_quote: str = ""  # verbatim from retained text, Stage 2 summary, or what_i_learned
    connection: str = Field(default="", max_length=900)  # Orion's words, first person
    citations: list[ResonanceCitationV1] = Field(default_factory=list, max_length=6)
    bears_on_prior: Literal["supports", "strains"] | None = None
    strained_prior_id: str | None = None  # required when bears_on_prior == "strains"
    decision: Literal["share", "note", "none"]
    explore_next: str | None = Field(default=None, max_length=400)
    confidence: float = Field(ge=0.0, le=1.0)


ResonanceRejectReason = Literal[
    "citation_not_in_packet",
    "quote_not_in_read",
    "empty_connection",
    "prompt_echo",
    "prior_relation_without_prior",
    "strained_prior_not_cited",
    "summary_sentence_count",
    "unparseable",
]


class ReadingResonanceCheckV1(_Base):
    packet_id: str
    accepted: bool
    decision: Literal["share", "note", "none"]
    reasons: list[ResonanceRejectReason] = Field(default_factory=list)
