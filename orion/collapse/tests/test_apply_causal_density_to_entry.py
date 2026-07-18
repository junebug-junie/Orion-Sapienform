from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.collapse.service import apply_causal_density_to_entry
from orion.schemas.collapse_mirror import (
    CollapseMirrorCausalDensity,
    CollapseMirrorEntryV2,
    CollapseMirrorStateSnapshot,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1


def _metacog_entry(*, score: float = 0.4, state_snapshot: CollapseMirrorStateSnapshot | None = None) -> CollapseMirrorEntryV2:
    return CollapseMirrorEntryV2(
        event_id="evt-1",
        observer="Orion",
        trigger="baseline",
        type="reflection",
        emergent_entity="orion",
        summary="test summary",
        mantra="test mantra",
        snapshot_kind="baseline",
        causal_density=CollapseMirrorCausalDensity(label="low", score=score, rationale="test"),
        tag_scores={"shift": score},
        is_causally_dense=False,
        state_snapshot=state_snapshot or CollapseMirrorStateSnapshot(),
    )


def _relational_state_snapshot(*, novelty: float = 0.9, confidence: float = 0.85) -> CollapseMirrorStateSnapshot:
    return CollapseMirrorStateSnapshot(
        telemetry={
            "trigger_payload": {
                "trigger_kind": "relational",
                "upstream": {
                    "shift_kind": "REPAIR",
                    "novelty_score": novelty,
                    "confidence": confidence,
                },
            }
        }
    )


def _unstable_self_state() -> SelfStateV1:
    now = datetime.now(timezone.utc)
    return SelfStateV1(
        self_state_id="ss-2",
        generated_at=now,
        source_field_tick_id="tick-2",
        source_field_generated_at=now,
        source_attention_frame_id="frame-2",
        source_attention_generated_at=now,
        overall_condition="unstable",
        overall_intensity=0.9,
        overall_confidence=0.7,
        dimensions={
            "execution_pressure": SelfStateDimensionV1(
                dimension_id="execution_pressure", score=0.85, confidence=0.7
            )
        },
        prediction_error_scores={"execution_pressure": 0.62},
        trajectory_condition="degrading",
        overall_surprise=0.7,
    )


def test_apply_causal_density_blends_self_state_for_metacog_lane():
    entry = _metacog_entry(score=0.3)
    out = apply_causal_density_to_entry(entry, self_state=_unstable_self_state())
    assert out.causal_density is not None
    assert out.causal_density.score > 0.3
    assert out.is_causally_dense is True


def test_apply_causal_density_self_report_only_without_self_state():
    entry = _metacog_entry(score=0.2)
    out = apply_causal_density_to_entry(entry, self_state=None)
    assert out.causal_density is not None
    assert out.causal_density.score == 0.2
    assert out.is_causally_dense is False


def test_apply_causal_density_blends_relational_evidence_for_relational_trigger():
    entry = _metacog_entry(score=0.2, state_snapshot=_relational_state_snapshot())
    out = apply_causal_density_to_entry(entry, self_state=None)
    # self_report=0.2 (weight 0.35) blended with relational_evidence=0.875 (weight 0.65)
    expected = (0.35 * 0.2 + 0.65 * 0.875) / (0.35 + 0.65)
    assert out.causal_density.score == pytest.approx(expected, abs=1e-6)
    assert out.is_causally_dense is True


def test_apply_causal_density_ignores_relational_upstream_for_non_relational_trigger():
    """A dense/pulse (substrate) trigger's telemetry has no relational upstream --
    entries not fired by a relational trigger must not pick up phantom relational evidence."""
    state_snapshot = CollapseMirrorStateSnapshot(
        telemetry={"trigger_payload": {"trigger_kind": "dense", "upstream": {"substrate_score": 0.9}}}
    )
    entry = _metacog_entry(score=0.2, state_snapshot=state_snapshot)
    out = apply_causal_density_to_entry(entry, self_state=None)
    assert out.causal_density.score == 0.2


def test_apply_causal_density_blends_all_three_evidence_sources_when_present():
    entry = _metacog_entry(score=0.2, state_snapshot=_relational_state_snapshot())
    out = apply_causal_density_to_entry(entry, self_state=_unstable_self_state())
    # _phi_evidence_score for _unstable_self_state: prediction_error=0.62,
    # severity_norm=1.0 ("unstable" is the top of _SEVERITY_ORDER), degrading_bump=0.15
    phi_score = min(1.0, 0.5 * 0.62 + 0.5 * 1.0 + 0.15)
    relational_score = 0.875
    expected = (0.35 * 0.2 + 0.65 * phi_score + 0.65 * relational_score) / (0.35 + 0.65 + 0.65)
    assert out.causal_density.score == pytest.approx(expected, abs=1e-6)
