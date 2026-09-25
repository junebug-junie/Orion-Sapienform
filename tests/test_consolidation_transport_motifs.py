"""The two transport motifs were deleted 2026-09-25 (chore/transport-lattice-
semantics). These tests pin the deletion and the bug that justified it.

Live shape: capability attention targets are novelty-scored and always carry
`dominant_channels={}` (orion/attention/field_attention/selectors.py). The old
`transport_healthy_idle` detector read the missing keys as 0.0, so an attended
capability:transport with NO channel data at all produced a "healthy idle"
motif and a `transport_stable` expectation. The old
`transport_contract_drift_loop` could never fire on that same shape.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.consolidation.expectation import _MOTIF_TO_EXPECTATION
from orion.consolidation.motif import _DETECTORS, detect_motifs
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import ExpectationV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 9, 25, 4, 30, tzinfo=timezone.utc)
POLICY = load_consolidation_policy(REPO_ROOT / "config" / "consolidation" / "consolidation_policy.v1.yaml")
DELETED = ("transport_contract_drift_loop", "transport_healthy_idle")


def _live_shaped_transport_frame(i: int) -> FieldAttentionFrameV1:
    """What production actually emits for an attended capability:transport."""
    target = FieldAttentionTargetV1(
        target_id="capability:transport",
        target_kind="capability",
        salience_score=0.4,
        pressure_score=0.12,
        novelty_score=0.4,
        urgency_score=0.0,
        confidence_score=1.0,
        dominant_channels={},
        reasons=["Candidate B novelty-only salience"],
        suggested_observation_mode="inspect",
    )
    return FieldAttentionFrameV1(
        frame_id=f"attention.frame:{i}",
        generated_at=NOW,
        source_field_tick_id=f"tick_{i}",
        source_field_generated_at=NOW,
        attention_policy_id="field_attention_policy.v1",
        overall_salience=0.4,
        dominant_targets=[target],
        capability_targets=[target],
    )


def test_transport_motifs_are_gone_from_policy_detectors_and_expectations() -> None:
    labels = {rule.label for rule in POLICY.motif_rules.values()}
    for label in DELETED:
        assert label not in labels
        assert label not in _DETECTORS
        assert label not in _MOTIF_TO_EXPECTATION


def test_attended_transport_with_no_channel_data_yields_no_transport_motif() -> None:
    """Regression for absence-read-as-calm: before the deletion this window
    produced `transport_healthy_idle`."""
    window = ConsolidationWindowData(
        window_start=NOW,
        window_end=NOW,
        attention_frames=[_live_shaped_transport_frame(i) for i in range(10)],
        proposal_frames=[],
        policy_frames=[],
        dispatch_frames=[],
        feedback_frames=[],
    )
    labels = {m.label for m in detect_motifs(window=window, policy=POLICY)}
    assert not labels & set(DELETED)


def test_every_policy_motif_rule_has_a_detector() -> None:
    """A rule with no detector is silently skipped by detect_motifs()."""
    for rule in POLICY.motif_rules.values():
        assert rule.label in _DETECTORS, rule.label


def test_historical_transport_stable_expectation_still_parses() -> None:
    """One stored row carries it; the Literal must keep accepting it."""
    ExpectationV1(
        expectation_id="expectation:motif:transport_healthy_idle:transport_stable",
        trigger_motif_id="motif:transport_healthy_idle",
        expected_outcome_kind="transport_stable",
        confidence_score=0.5,
        support_count=130,
    )
