"""Memory-compaction delta — the dream's *proposed* housekeeping (Phase F).

A `MemoryCompactionDeltaV1` is what REM sleep would do to memory: consolidate a
few episodes into gist cards, downscale some edge weights, prune some low-salience
episodics. Phase F **produces this delta and applies nothing** — `proposal_marked`
is hard-`True`, and the applier (Phase G) is a separate, hard-gated process.

Governing constraints:
  - §0A no empty-shell cognition: a delta with no ops is still a valid *readout*
    (nothing to compact tonight) but carries explicit zero metrics, not a fake win;
  - §cap-all-collections: every op list is capped;
  - "awake proposes, substrate disposes": this is a proposal artifact, never an act.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Caps (§cap-all-collections). A single night should never emit an unbounded delta.
MAX_CONSOLIDATE = 50
MAX_DOWNSCALE = 200
MAX_PRUNE = 200
# Cap on evidence/superseded refs per op.
MAX_OP_REFS = 50


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConsolidateEntryV1(BaseModel):
    """A proposed gist card that would *supersede* a batch of episodes.

    The card text is latent (narrator-written); `evidence_refs`/`supersedes` are
    deterministic ids so the applier (and reviewer) can see exactly what folds in.
    """

    gist_card: str = ""
    evidence_refs: list[str] = Field(default_factory=list, max_length=MAX_OP_REFS)
    supersedes: list[str] = Field(default_factory=list, max_length=MAX_OP_REFS)


class DownscaleEntryV1(BaseModel):
    """A proposed edge/weight downscale (renormalize, never delete)."""

    target_id: str
    old_w: float = Field(default=0.0, ge=0.0, le=1.0)
    new_w: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class PruneEntryV1(BaseModel):
    """A proposed episodic prune (only ever applied after downscale, and gated)."""

    episodic_id: str
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    ttl_reason: str = ""


class CompactionMetricsV1(BaseModel):
    """Deterministic counts describing the delta — code owns these, not the LLM."""

    cards_out: int = 0
    edges_downscaled: int = 0
    rows_pruned: int = 0
    bytes_reclaimed_est: int = 0


class MemoryCompactionDeltaV1(BaseModel):
    """What sleep *would* do to memory — a proposal, applied by nothing in Phase F.

    `proposal_marked` is hard-`True` here: the delta can enter Layer 7 as a
    proposal, but only the separately-gated Phase-G applier may ever touch memory,
    and only after Layer-8 policy approval. Consuming this delta must never be
    treated as an apply.
    """

    model_config = ConfigDict(protected_namespaces=())

    schema_version: Literal["dream.compaction.delta.v1"] = "dream.compaction.delta.v1"
    delta_id: str
    dream_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)

    consolidate: list[ConsolidateEntryV1] = Field(default_factory=list, max_length=MAX_CONSOLIDATE)
    downscale: list[DownscaleEntryV1] = Field(default_factory=list, max_length=MAX_DOWNSCALE)
    prune: list[PruneEntryV1] = Field(default_factory=list, max_length=MAX_PRUNE)

    metrics: CompactionMetricsV1 = Field(default_factory=CompactionMetricsV1)

    # Provenance of the ask (the Phase-E requests this delta responds to).
    source_request_ids: list[str] = Field(default_factory=list, max_length=MAX_OP_REFS)

    # Hard invariant: a Phase-F delta is always a proposal, never an applied fact.
    proposal_marked: Literal[True] = True

    def op_count(self) -> int:
        return len(self.consolidate) + len(self.downscale) + len(self.prune)

    def is_empty(self) -> bool:
        """True when there is nothing to compact tonight (an honest zero, not a
        fake win). Callers surface this explicitly rather than as cognition."""
        return self.op_count() == 0
