from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from orion.consolidation.motif import detect_motifs
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def _attention_frame(*, contract: float) -> FieldAttentionFrameV1:
    target = FieldAttentionTargetV1(
        target_id="capability:transport",
        target_kind="capability",
        salience_score=0.8,
        pressure_score=contract,
        novelty_score=0.0,
        urgency_score=0.2,
        confidence_score=0.9,
        dominant_channels={"contract_pressure": contract},
        reasons=["capability contract_pressure is elevated"],
        suggested_observation_mode="inspect",
    )
    return FieldAttentionFrameV1(
        frame_id=f"attention.frame:{contract}",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        attention_policy_id="field_attention_policy.v1",
        overall_salience=0.8,
        dominant_targets=[target],
        capability_targets=[target],
    )


def test_transport_contract_drift_motif() -> None:
    policy = load_consolidation_policy(REPO_ROOT / "config" / "consolidation" / "consolidation_policy.v1.yaml")
    window = ConsolidationWindowData(
        window_start=NOW,
        window_end=NOW,
        self_states=[],
        attention_frames=[_attention_frame(contract=1.0) for _ in range(3)],
        proposal_frames=[],
        policy_frames=[],
        dispatch_frames=[],
        feedback_frames=[],
    )
    motifs = detect_motifs(window=window, policy=policy)
    labels = {m.label for m in motifs}
    assert "transport_contract_drift_loop" in labels
