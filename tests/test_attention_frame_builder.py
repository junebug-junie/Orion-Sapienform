from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    # node_vectors/capability_vectors are still part of FieldStateV1's real
    # schema (other consumers read them) but 2026-07-30's attention rewrite
    # no longer scores them directly -- only `prediction_error_histories`
    # (passed separately, see below) drives node_targets now, and
    # capability_targets is always []. Included here to confirm the killed
    # hand-weighted path really produces nothing, not just that it's untested.
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_exec_attention",
        node_vectors={
            "node:athena": {
                "cortex_exec_step_load": 1.0,
                "reasoning_load": 0.35,
                "availability": 1.0,
            },
            "node:substrate.execution": {"prediction_error": 0.9},
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=["state_delta:exec_1", "state_delta:exec_2"],
    )


def _histories() -> dict[str, list[float]]:
    # Calm baseline (small, real variance) then a real spike on the current
    # tick -- a genuine, non-degenerate precision-weighted-salience case.
    return {
        "node:substrate.execution": [0.05, 0.06, 0.04, 0.05, 0.9],
    }


def test_builder_selects_only_prediction_error_native_targets() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_histories=_histories(), now=NOW
    )
    node_ids = {t.target_id for t in frame.node_targets}
    # The physical host node (node:athena) has no real prediction-error
    # history and is NOT a candidate-A target -- it must not appear, even
    # though its old hand-weighted vector would have scored it highly.
    assert node_ids == {"node:substrate.execution"}
    # Capability attention is killed outright -- always empty.
    assert frame.capability_targets == []


def test_target_with_no_real_history_is_excluded_not_zero_scored() -> None:
    field = _synthetic_field()
    frame = build_attention_frame(
        field=field, policy=POLICY, prediction_error_histories={}, now=NOW
    )
    assert frame.node_targets == []
    assert frame.capability_targets == []


def test_dominant_channels_present() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_histories=_histories(), now=NOW
    )
    execution = next(t for t in frame.node_targets if t.target_id == "node:substrate.execution")
    assert "prediction_error" in execution.dominant_channels


def test_overall_salience_positive() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_histories=_histories(), now=NOW
    )
    assert frame.overall_salience > 0.0


def test_targets_sorted_desc() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_histories=_histories(), now=NOW
    )
    scores = [t.salience_score for t in frame.dominant_targets]
    assert scores == sorted(scores, reverse=True)


def test_frame_id_stable() -> None:
    field = _synthetic_field()
    histories = _histories()
    a = build_attention_frame(field=field, policy=POLICY, prediction_error_histories=histories, now=NOW)
    b = build_attention_frame(field=field, policy=POLICY, prediction_error_histories=histories, now=NOW)
    assert a.frame_id == b.frame_id


def test_source_field_tick_id() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_histories=_histories(), now=NOW
    )
    assert frame.source_field_tick_id == "tick_exec_attention"


def test_recent_perturbations_carried() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_histories=_histories(), now=NOW
    )
    assert frame.recent_perturbations == ["state_delta:exec_1", "state_delta:exec_2"]
