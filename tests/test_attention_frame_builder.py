from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_exec_attention",
        node_vectors={
            "node:athena": {
                "execution_load": 1.0,
                "reasoning_load": 0.35,
                "availability": 1.0,
            },
            "node:prometheus": {
                "cpu_pressure": 0.02,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=["state_delta:exec_1", "state_delta:exec_2"],
    )


def test_builder_selects_athena_and_orchestration() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    node_ids = {t.target_id for t in frame.node_targets}
    cap_ids = {t.target_id for t in frame.capability_targets}
    assert "node:athena" in node_ids
    assert "capability:orchestration" in cap_ids


def test_dominant_channels_present() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    athena = next(t for t in frame.node_targets if t.target_id == "node:athena")
    orch = next(t for t in frame.capability_targets if t.target_id == "capability:orchestration")
    assert "execution_load" in athena.dominant_channels
    assert "execution_pressure" in orch.dominant_channels


def test_overall_salience_positive() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    assert frame.overall_salience > 0.0


def test_low_salience_suppressed() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    dominant_ids = {t.target_id for t in frame.dominant_targets}
    assert "node:prometheus" not in dominant_ids


def test_targets_sorted_desc() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    scores = [t.salience_score for t in frame.dominant_targets]
    assert scores == sorted(scores, reverse=True)


def test_frame_id_stable() -> None:
    field = _synthetic_field()
    a = build_attention_frame(field=field, policy=POLICY, now=NOW)
    b = build_attention_frame(field=field, policy=POLICY, now=NOW)
    assert a.frame_id == b.frame_id


def test_source_field_tick_id() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    assert frame.source_field_tick_id == "tick_exec_attention"


def test_recent_perturbations_carried() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    assert frame.recent_perturbations == ["state_delta:exec_1", "state_delta:exec_2"]
