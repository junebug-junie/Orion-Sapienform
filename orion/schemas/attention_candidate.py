"""One shape for everything that can compete for Orion's attention.

Spec: docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md,
section "A. The attention seam" (PR #2528, approved 2026-10-07 / 2026-10-10).

A candidate is either ``internal`` (the body: a substrate node's prediction
error) or ``external`` (the world: chat activity, camera surprise). Both carry
the SAME unusualness reading -- ``PredictionErrorMagnitudeV1``, reused as-is:
the current value's percentile against that source's own 7-day history, a
band, a trend and an explicit ``insufficient_history``. Because every source
is scored against its own normal, internal and external candidates are on one
scale and no exchange rate between them is needed.

``absent`` is separate from the reading on purpose: a source that cannot
measure right now (camera frames stale, chat log unreadable, a body node that
has not reported for half an hour) is NOT a calm source, and must never be
ranked as one. The ranking rule lives in ``orion.attention.world_first``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.attention_frame import PredictionErrorMagnitudeV1

AttentionSourceKindV1 = Literal["internal", "external"]

# World sources. Kept in this light schema module so thin consumers (reverie
# proposals) can recognise a world target without importing the ranking code.
WORLD_CHAT_SOURCE_ID = "world:chat"
PERCEPTION_NODE_ID = "node:substrate.perception"
WORLD_SOURCE_PREFIX = "world:"


def is_world_source_id(source_id: str | None) -> bool:
    """True for an external (world) source id: chat, camera surprise, or any
    future ``world:*`` source. Everything else is the body."""
    sid = str(source_id or "")
    return sid.startswith(WORLD_SOURCE_PREFIX) or sid == PERCEPTION_NODE_ID


# Polarity comes from the semantic layer (orion.metrics.semantics.
# derived_channel_polarity), never invented per candidate. None = the layer
# declares no direction (e.g. a trigger, whose value is "it fired").
AttentionPolarityV1 = Literal["higher_is_better", "higher_is_worse"]


class AttentionCandidateV1(BaseModel):
    """One source's bid for attention this tick, with its own sense of scale."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["attention.candidate.v1"] = "attention.candidate.v1"
    candidate_id: str
    # What is being attended to: a substrate node id (``node:substrate.*``) or
    # a world source id (``world:chat``).
    source_id: str
    source_kind: AttentionSourceKindV1
    label: str
    unusualness: PredictionErrorMagnitudeV1
    # Mid-rank percentile (share below + half the share equal) in the RAW
    # value direction, when the builder had the history. Ranking key only:
    # eligibility uses unusualness.percentile_now (strict-below) so an
    # all-zero history with a current 0 still reads as rest.
    rank_percentile: float | None = Field(default=None, ge=0.0, le=1.0)
    observed_at: datetime | None = None
    # Silence is not calm: True when the source could not measure this tick.
    absent: bool = False
    absent_reason: str | None = None
    # From the semantic layer (glossary value_kind / derived polarity), copied
    # here so a stored frame says why a candidate was or was not eligible.
    value_kind: str | None = None
    polarity: AttentionPolarityV1 | None = None
    evidence_refs: list[str] = Field(default_factory=list)
