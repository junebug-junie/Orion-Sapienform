from __future__ import annotations

from datetime import datetime, timezone

from orion.collapse.service import apply_causal_density_to_entry
from orion.schemas.collapse_mirror import CollapseMirrorEntryV2, CollapseMirrorCausalDensity
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1


def _metacog_entry(*, score: float = 0.4) -> CollapseMirrorEntryV2:
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
