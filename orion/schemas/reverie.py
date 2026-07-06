"""Reverie schemas — the spontaneous-thought mode of orion-thought.

A `SpontaneousThoughtV1` is an *unprompted* narration of the current rung-3
winning coalition. It is the sibling of the *evoked* `ThoughtEventV1`
(`orion/schemas/thought.py`): same coalition grounding (`CoalitionSnapshotV1`),
different trigger (self-driven tick, no `user_message`) and destination
(`orion:reverie:thought`, later a `ProposalFrameV1`).

Governing constraint (§0A): this must never ship as hollow cognition. The
`is_hollow()` guard is load-bearing — a spontaneous thought with no user
question anchoring relevance is the real fail-fast risk, so the guard rejects
short *and* un-anchored text (interpretation not backed by coalition ids).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.thought import CoalitionSnapshotV1

# Minimum grounded interpretation length. Below this, a "thought" is drivel.
MIN_INTERPRETATION_CHARS = 40
# Cap on audit evidence refs (§ cap-all-collections).
MAX_EVIDENCE_REFS = 50


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SpontaneousThoughtV1(BaseModel):
    """Unprompted narration of the current winning coalition.

    Grounding is the *same* `CoalitionSnapshotV1` the evoked path uses, so dream
    and evals treat evoked and spontaneous thoughts as one evidence stream.
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["reverie.thought.v1"] = "reverie.thought.v1"
    thought_id: str
    correlation_id: str
    created_at: datetime = Field(default_factory=_utc_now)

    # Grounding — shared vocabulary with the evoked path. Optional so an absent
    # coalition degrades to a hollow-marked thought instead of raising.
    coalition: CoalitionSnapshotV1 | None = None

    # Voice (latent — LLM writes these).
    interpretation: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)

    # Audit.
    evidence_refs: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)
    hollow: bool = False
    hollow_reason: str | None = None

    llm_profile: str = "brain"
    producer: str = "reverie_narrate_v1"
    model_id: str | None = None

    # Phase C chain linkage — additive, optional; set only when emitted inside a
    # chain. next_focus/drift are the LLM's forward pointer for the next step.
    chain_id: str | None = None
    thought_index: int | None = None
    next_focus: str | None = None
    drift: str | None = None

    # Phase D grounding — read-only refs into Layer 11 motifs (consolidation
    # frame ids) + rung-4 episode ids. Empty unless ORION_REVERIE_GROUND_CONSOLIDATION.
    motif_refs: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)
    episode_summary_refs: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)

    def grounding_ids(self) -> set[str]:
        """The set of coalition ids that legitimately anchor this thought."""
        if self.coalition is None:
            return set()
        ids: set[str] = set(self.coalition.attended_node_ids)
        ids.update(self.coalition.open_loop_ids)
        if self.coalition.selected_open_loop_id:
            ids.add(self.coalition.selected_open_loop_id)
        return ids

    def is_hollow(self) -> bool:
        """True if this thought is empty-shell cognition (§0A).

        Rejects three failure modes:
          - no coalition (nothing to narrate);
          - too-short interpretation (drivel);
          - un-anchored — evidence_refs empty or not a subset of coalition ids
            (the harder bar now that there is no user question to anchor it).
        """
        return self.hollow_reason_for() is not None

    def hollow_reason_for(self) -> str | None:
        if self.coalition is None:
            return "absent_coalition"
        if len(self.interpretation.strip()) < MIN_INTERPRETATION_CHARS:
            return "interpretation_too_short"
        grounding = self.grounding_ids()
        if not grounding:
            return "zero_grounding"
        if not self.evidence_refs:
            return "unanchored_no_evidence"
        if not set(self.evidence_refs).issubset(grounding):
            return "unanchored_evidence_outside_coalition"
        return None

    def marked_hollow(self) -> "SpontaneousThoughtV1":
        """Return a copy with the hollow flag + reason stamped from the guard."""
        reason = self.hollow_reason_for()
        return self.model_copy(update={"hollow": reason is not None, "hollow_reason": reason})


# --- Phase C: reverie chain ---------------------------------------------------

# Cap the verbatim thought window per chain (§ cap-all-collections). The wide-n
# memory is the lossy EMA, never a growing verbatim window.
MAX_CHAIN_THOUGHTS = 50

TerminalReason = Literal[
    "pressure_discharged",
    "max_steps",
    "no_coalition",
    "refractory",
    "low_salience",
]


class ReverieRefractoryEntry(BaseModel):
    """A resolved theme suppressed as a chain trigger until a cooldown expires."""

    schema_version: Literal["reverie.refractory.entry.v1"] = "reverie.refractory.entry.v1"
    theme_key: str
    suppressed_until: datetime


class ReverieChainTriggerV1(BaseModel):
    pressure_kind: str = "unspecified"
    magnitude: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_payload: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)


class ReverieChainV1(BaseModel):
    """Readout of one train of thought — successive climbs of the ladder.

    Continuity is the last-n verbatim `thought_ids`; the wide-n memory is the
    lossy `ema_salience` low-pass (never a verbatim wide window). A chain reads
    coalitions and prior thoughts; it never reads a dream (ouroboros safety).
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["reverie.chain.v1"] = "reverie.chain.v1"
    chain_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    theme_key: str | None = None
    trigger: ReverieChainTriggerV1 | None = None
    thought_ids: list[str] = Field(default_factory=list, max_length=MAX_CHAIN_THOUGHTS)
    ema_salience: float = Field(default=0.0, ge=0.0, le=1.0)
    ema_summary: str = ""
    terminal_reason: TerminalReason = "max_steps"
    committed_proposal_id: str | None = None


# --- Phase E: compaction request (reverie → dream queue, applied by nothing) ---

CompactionOpHint = Literal["consolidate", "downscale", "prune"]


class CompactionRequestV1(BaseModel):
    """A typed *ask* from the awake reverie (reasoning) to the offline dream
    (storage): "this theme feels settled — consider compacting it." A request,
    not an act — queued for a later, different process. Applied by nothing here.
    """

    schema_version: Literal["dream.compaction.request.v1"] = "dream.compaction.request.v1"
    request_id: str
    theme: str
    reason: str = ""
    op_hint: CompactionOpHint = "consolidate"
    evidence_refs: list[str] = Field(default_factory=list, max_length=MAX_EVIDENCE_REFS)
    origin_chain_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
