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
        attention_frames=[_attention_frame(contract=1.0) for _ in range(3)],
        proposal_frames=[],
        policy_frames=[],
        dispatch_frames=[],
        feedback_frames=[],
    )
    motifs = detect_motifs(window=window, policy=policy)
    labels = {m.label for m in motifs}
    assert "transport_contract_drift_loop" in labels


def _transport_frame(frame_id: str, dominant_channels: dict[str, float]) -> FieldAttentionFrameV1:
    target = FieldAttentionTargetV1(
        target_id="capability:transport",
        target_kind="capability",
        salience_score=0.5,
        pressure_score=max(dominant_channels.values()),
        novelty_score=0.0,
        urgency_score=0.1,
        confidence_score=0.9,
        dominant_channels=dominant_channels,
        reasons=["capability pressure"],
        suggested_observation_mode="inspect",
    )
    return FieldAttentionFrameV1(
        frame_id=frame_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        attention_policy_id="field_attention_policy.v1",
        overall_salience=0.5,
        dominant_targets=[target],
        capability_targets=[target],
    )


def _labels_for(frames: list[FieldAttentionFrameV1]) -> set[str]:
    policy = load_consolidation_policy(REPO_ROOT / "config" / "consolidation" / "consolidation_policy.v1.yaml")
    window = ConsolidationWindowData(
        window_start=NOW,
        window_end=NOW,
        attention_frames=frames,
        proposal_frames=[],
        policy_frames=[],
        dispatch_frames=[],
        feedback_frames=[],
    )
    return {m.label for m in detect_motifs(window=window, policy=policy)}


def test_transport_healthy_idle_is_keyed_on_pressure_and_can_read_both_ways() -> None:
    """2026-09-25 (fix/bus-observer-scope): the motif reads
    capability:transport.pressure (bus_synaptic) only, under the rekeyed
    `max_pressure` condition. Calm pressure fires it; busy pressure does not."""
    calm = [_transport_frame(f"calm{i}", {"pressure": 0.02}) for i in range(3)]
    busy = [_transport_frame(f"busy{i}", {"pressure": 0.8}) for i in range(3)]
    assert "transport_healthy_idle" in _labels_for(calm)
    assert "transport_healthy_idle" not in _labels_for(busy)


def test_transport_healthy_idle_ignores_retired_stream_backlog_pressure() -> None:
    """A stale frame still carrying the retired channel must not decide the
    motif: busy bus_synaptic pressure stays busy regardless of it."""
    frames = [
        _transport_frame(f"f{i}", {"pressure": 0.8, "stream_backlog_pressure": 0.0}) for i in range(3)
    ]
    assert "transport_healthy_idle" not in _labels_for(frames)


def test_policy_has_no_retired_stream_backlog_condition_keys() -> None:
    raw = (REPO_ROOT / "config" / "consolidation" / "consolidation_policy.v1.yaml").read_text(encoding="utf-8")
    assert "max_stream_backlog_pressure:" not in raw
    assert "min_stream_backlog_health:" not in raw
