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

# Camera surprise for the outdoor walkway camera (2026-10-15). A separate node
# so a busy street never overwrites the room cameras' reading in
# PERCEPTION_NODE_ID. Underscore, not a dot: the metric registry builds URNs as
# "<node>.<channel>", so a dotted suffix would be ambiguous.
PERCEPTION_WALKWAY_NODE_ID = "node:substrate.perception_walkway"

# Per-consumer fences: substrate nodes that are written and recorded (graph
# node, receipt, prediction-error history) but kept out of one cognition
# candidate builder. Endogenous curiosity, the substrate attention broadcast and
# the attention runtime's world-first candidates each pick up ANY
# node:substrate.* node carrying prediction_error, so "not this consumer yet"
# needs an explicit fence. Removing an id from a set is the go-live decision for
# that node in that consumer, made after its live-data sanity check (AGENTS.md
# metric quality gate, step 4).
#
# The walkway camera node, 2026-10-15: curiosity reads it (Juniper's decision;
# gate record in docs/superpowers/pr-reports/2026-10-15-walkway-street-into-mind-pr.md).
# Attention still does not: before it does, decide whether is_world_source_id
# below should call it world (it is an outdoor camera) and give it the
# camera-absent check PERCEPTION_NODE_ID gets.
NO_ATTENTION_CONSUMER_NODE_IDS: frozenset[str] = frozenset({PERCEPTION_WALKWAY_NODE_ID})

# Street EVENTS (doorbell ring, graded cross-camera handoffs), 2026-10-20:
# docs/superpowers/specs/2026-10-20-street-continuity-handoff-design.md
# section 6, Juniper's rule in PR #2763 (answer 7). Written by
# orion-substrate-runtime from orion:vision:street:events through the gate in
# orion/substrate/street_attention.py. Admitted to attention (deliberately NOT
# in NO_ATTENTION_CONSUMER_NODE_IDS: its value comes from gated events, not
# pixels) and labelled as the outside world (is_world_source_id). Fenced from
# endogenous curiosity until Juniper decides that consumer after live data, the
# same go-live rule the walkway node followed. Rollback: add it to
# NO_ATTENTION_CONSUMER_NODE_IDS, or SUBSTRATE_STREET_ATTENTION_ENABLED=false.
STREET_NODE_ID = "node:substrate.street"

NO_CURIOSITY_CONSUMER_NODE_IDS: frozenset[str] = frozenset({STREET_NODE_ID})

# Substrate nodes that measure the outside world, not Orion's body.
_WORLD_NODE_IDS: frozenset[str] = frozenset({PERCEPTION_NODE_ID, STREET_NODE_ID})


def is_world_source_id(source_id: str | None) -> bool:
    """True for an external (world) source id: chat, camera surprise, street
    events, or any future ``world:*`` source. Everything else is the body."""
    sid = str(source_id or "")
    return sid.startswith(WORLD_SOURCE_PREFIX) or sid in _WORLD_NODE_IDS


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
    # Set when the semantic layer says this source is EVENT-WRITTEN: one
    # reading per event, carried forward unchanged until the next event
    # (orion.attention.world_first.prediction_error_is_event_written). The
    # reading is news for this many seconds after the event that wrote it
    # (its unusualness.age_sec), then stops competing even though the carried
    # value is unchanged. None = a level read every tick, or decay disabled.
    event_window_sec: float | None = Field(default=None, gt=0.0)
    # Set for an INTERNAL source the semantic layer marks as read every tick
    # (biometrics, bus_synaptic: orion.attention.world_first.
    # node_prediction_error_continuous). Such a source is high for itself ~10%
    # of the time by construction, so it competes only when it has been hot for
    # more than orion.attention.sustained.SUSTAIN_DWELL_SEC of the trailing
    # ``sustain_window_sec`` and is hot now. ``hot_dwell_sec`` is that dwell,
    # measured over the source's own readings. Both None = the rule does not
    # apply (not a per-tick body source, no history, or
    # ATTENTION_SUSTAINED_BODY_ENABLED=false): today's verdict exactly.
    sustain_window_sec: float | None = Field(default=None, gt=0.0)
    hot_dwell_sec: float | None = Field(default=None, ge=0.0)
    evidence_refs: list[str] = Field(default_factory=list)
