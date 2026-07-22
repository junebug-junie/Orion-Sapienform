from __future__ import annotations

import pytest

from orion.collapse.service import apply_causal_density_to_entry
from orion.schemas.collapse_mirror import (
    CollapseMirrorCausalDensity,
    CollapseMirrorEntryV2,
    CollapseMirrorStateSnapshot,
)


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


def test_apply_causal_density_self_report_only():
    entry = _metacog_entry(score=0.2)
    out = apply_causal_density_to_entry(entry)
    assert out.causal_density is not None
    assert out.causal_density.score == 0.2
    assert out.is_causally_dense is False


def test_apply_causal_density_blends_relational_evidence_for_relational_trigger():
    entry = _metacog_entry(score=0.2, state_snapshot=_relational_state_snapshot())
    out = apply_causal_density_to_entry(entry)
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
    out = apply_causal_density_to_entry(entry)
    assert out.causal_density.score == 0.2
