from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.candidate_precision_weighted import (
    NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
    NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    PrecisionEwmaBaseline,
    advance_precision_baseline,
)
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    # node_vectors/capability_vectors are still part of FieldStateV1's real
    # schema (other consumers read them) but 2026-07-30's attention rewrite
    # no longer scores them directly -- only `prediction_error_baselines`
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


def _baseline(values: list[float]) -> PrecisionEwmaBaseline:
    baseline = PrecisionEwmaBaseline()
    return advance_precision_baseline(
        baseline,
        values,
        alpha=NODE_TARGET_PREDICTION_ERROR_EWMA_ALPHA,
        min_variance=NODE_TARGET_PREDICTION_ERROR_MIN_VARIANCE,
    )


def _baselines() -> dict[str, PrecisionEwmaBaseline]:
    # Calm history (small, real variance) then a real spike on the current
    # tick -- a genuine, non-degenerate precision-weighted-salience case.
    return {
        "node:substrate.execution": _baseline([0.05, 0.06, 0.04, 0.05, 0.9]),
    }


def test_builder_selects_only_prediction_error_native_targets() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_baselines=_baselines(), now=NOW
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
        field=field, policy=POLICY, prediction_error_baselines={}, now=NOW
    )
    assert frame.node_targets == []
    assert frame.capability_targets == []


def test_dominant_channels_present() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_baselines=_baselines(), now=NOW
    )
    execution = next(t for t in frame.node_targets if t.target_id == "node:substrate.execution")
    assert "prediction_error" in execution.dominant_channels


def test_overall_salience_positive() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_baselines=_baselines(), now=NOW
    )
    assert frame.overall_salience > 0.0


def test_targets_sorted_desc() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_baselines=_baselines(), now=NOW
    )
    scores = [t.salience_score for t in frame.dominant_targets]
    assert scores == sorted(scores, reverse=True)


def test_frame_id_stable() -> None:
    field = _synthetic_field()
    baselines = _baselines()
    a = build_attention_frame(field=field, policy=POLICY, prediction_error_baselines=baselines, now=NOW)
    b = build_attention_frame(field=field, policy=POLICY, prediction_error_baselines=baselines, now=NOW)
    assert a.frame_id == b.frame_id


def test_source_field_tick_id() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_baselines=_baselines(), now=NOW
    )
    assert frame.source_field_tick_id == "tick_exec_attention"


def test_recent_perturbations_carried() -> None:
    frame = build_attention_frame(
        field=_synthetic_field(), policy=POLICY, prediction_error_baselines=_baselines(), now=NOW
    )
    assert frame.recent_perturbations == ["state_delta:exec_1", "state_delta:exec_2"]


def _capability_field(tick_id: str, pressure: float, n: int = 7) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id=tick_id,
        capability_vectors={
            f"capability:c{i}": {"execution_pressure": pressure} for i in range(n)
        },
    )


def test_over_cap_active_targets_are_recorded_and_do_not_fake_novelty() -> None:
    # 2026-09-25 (D1, second path): 7 capabilities go active on the same tick,
    # the per-kind cap keeps 5. The other 2 used to land in NO bucket, so on
    # the next steady tick their prior read 0.0 and their whole pressure
    # scored as fresh novelty. They are now kept in suppressed_targets with a
    # reason, and a steady tick reads zero novelty for all 7.
    cap = POLICY.limits.max_capability_targets
    assert cap < 7
    frame1 = build_attention_frame(field=_capability_field("t1", 0.1), policy=POLICY, now=NOW)
    frame2 = build_attention_frame(
        field=_capability_field("t2", 0.9), policy=POLICY, previous_frame=frame1, now=NOW
    )
    assert len(frame2.capability_targets) == cap
    over_cap = [t for t in frame2.suppressed_targets if "over the per-kind target cap" in " ".join(t.reasons)]
    assert len(over_cap) == 7 - cap
    ids_in_frame = {
        t.target_id
        for bucket in (frame2.capability_targets, frame2.suppressed_targets)
        for t in bucket
    }
    assert ids_in_frame == {f"capability:c{i}" for i in range(7)}

    frame3 = build_attention_frame(
        field=_capability_field("t3", 0.9), policy=POLICY, previous_frame=frame2, now=NOW
    )
    every = [
        t
        for bucket in (frame3.capability_targets, frame3.suppressed_targets)
        for t in bucket
    ]
    assert len(every) == 7
    assert all(t.novelty_score == 0.0 for t in every)
    assert all(t.confidence_score == 1.0 for t in every)


def test_suppressed_targets_are_strongest_first_like_every_other_bucket() -> None:
    # Review finding on D1: over-cap targets were appended after the
    # below-threshold ones, so the panel listed 0.08 and 0.01 above 0.8.
    def field(tick: str, hi: float, low: float, mid: float) -> FieldStateV1:
        return FieldStateV1(
            generated_at=NOW,
            tick_id=tick,
            capability_vectors={
                **{f"capability:c{i}": {"execution_pressure": hi} for i in range(7)},
                "capability:low": {"execution_pressure": low},
                "capability:mid": {"execution_pressure": mid},
            },
        )

    frame1 = build_attention_frame(field=field("t1", 0.1, 0.1, 0.1), policy=POLICY, now=NOW)
    frame2 = build_attention_frame(
        field=field("t2", 0.9, 0.11, 0.18), policy=POLICY, previous_frame=frame1, now=NOW
    )
    saliences = [t.salience_score for t in frame2.suppressed_targets]
    assert saliences == sorted(saliences, reverse=True)
    over_cap = ["over the per-kind target cap" in " ".join(t.reasons) for t in frame2.suppressed_targets]
    assert over_cap == sorted(over_cap, reverse=True) and any(over_cap) and not all(over_cap)
