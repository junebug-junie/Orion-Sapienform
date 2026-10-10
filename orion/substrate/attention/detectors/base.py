from __future__ import annotations

from typing import Any, Protocol

from orion.schemas.attention_frame import AttentionSignalV1


# World-first audit (spec docs/superpowers/specs/2026-10-07-orion-self-
# calibration-design.md, section A, 2026-10-10). The per-turn chat frame's
# three detectors keep FIXED salience priors -- named in each module
# (CURRENT_TURN_SALIENCE, CONCEPT_SALIENCE_*, SITUATION_SALIENCE_*) -- rather
# than becoming calibrated AttentionCandidateV1s, for stated reasons, not by
# default:
#
# 1. There is nothing to calibrate them against. A calibrated candidate's
#    unusualness is its percentile against its OWN stored history; none of
#    these detectors' outputs (LLM-picked phrases, concept-profile buckets,
#    situation affordances) is persisted per tick, so a percentile would be
#    invented.
# 2. The values only act as a rank order. build_open_loops turns
#    salience x confidence into evidence_strength and Borda-ranks it
#    (orion/substrate/attention/salience.py), so 0.72 vs 0.62 only means
#    "current-turn phrases outrank concept tensions"; the absolute numbers
#    are never read as a measurement.
# 3. World-first already holds here by construction: this frame is built
#    only when a chat turn arrives (a fresh world event), and no body
#    prediction-error node competes in it (those live in the field contest
#    and the substrate broadcast, which ARE world-first now).
#
# Each detector tags provenance["source_kind"] (current_turn and situation:
# external -- they read Juniper's message and the conversation; concept
# induction: internal -- Orion's own concept profile), so the trace says
# which side of the seam a loop came from. Becoming calibrated needs a
# per-detector history table first; that is the named next step, not this
# patch.


class AttentionSignalDetector(Protocol):
    detector_id: str

    def detect(
        self,
        ctx: dict[str, Any],
        inputs: dict[str, Any],
        belief_lineage: list[str],
    ) -> list[AttentionSignalV1]:
        ...
