from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    VoluntaryOverrideV1,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.substrate.attention_self_model import (
    DEFAULT_BROADCAST_STALE_THRESHOLD_SEC,
    reduce_attention_self_model,
)

NOW = datetime(2026, 7, 18, 12, 0, 0, tzinfo=timezone.utc)


def _broadcast(
    *,
    generated_at: datetime = NOW,
    override: VoluntaryOverrideV1 | None = None,
    selected_open_loop_id: str = "loop-1",
) -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(generated_at=generated_at, voluntary_override=override)
    return AttentionBroadcastProjectionV1(
        generated_at=generated_at,
        frame=frame,
        selected_action_type="watch",
        selected_open_loop_id=selected_open_loop_id,
        selected_description="a dispatched open loop",
        attended_node_ids=["node-a", "node-b"],
        dwell_ticks=2,
        coalition_stability_score=0.8,
    )


def _override(**overrides: object) -> VoluntaryOverrideV1:
    base = dict(
        goal_artifact_id="goal-123",
        goal_drive_origin="curiosity",
        chosen_loop_id="loop-1",
        beat_loop_id="loop-2",
        chosen_bottom_up=0.4,
        beat_bottom_up=0.6,
        applied_bias=0.3,
        effort_spent=0.1,
    )
    base.update(overrides)
    return VoluntaryOverrideV1(**base)


def _field_frame(*, generated_at: datetime = NOW, overall_salience: float = 0.7) -> FieldAttentionFrameV1:
    target = FieldAttentionTargetV1(
        target_id="node-a",
        target_kind="node",
        salience_score=0.7,
        pressure_score=0.5,
        novelty_score=0.3,
        urgency_score=0.4,
        confidence_score=0.65,
    )
    return FieldAttentionFrameV1(
        frame_id="field-frame-1",
        generated_at=generated_at,
        source_field_tick_id="tick-1",
        source_field_generated_at=generated_at,
        overall_salience=overall_salience,
        dominant_targets=[target],
    )


def _prediction_error_by_domain(**overrides: float) -> dict[str, float]:
    base = {
        "execution": 0.0001,
        "transport": 0.0,
        "biometrics": 0.0459,
        "chat": 0.0,
        "route": 0.0,
    }
    base.update(overrides)
    return base


class TestVoluntaryOverridePresent:
    def test_override_present_narrates_top_down_reason(self) -> None:
        override = _override()
        broadcast = _broadcast(override=override)
        model = reduce_attention_self_model(broadcast, _field_frame(), now=NOW)

        assert model.attention_reason == "top_down_override"
        assert model.voluntary_override is not None
        assert model.voluntary_override.chosen_loop_id == "loop-1"
        assert "Top-down goal bias flipped the winner" in model.reason_narrative
        assert "loop-1" in model.reason_narrative
        assert "loop-2" in model.reason_narrative
        assert model.confidence == pytest.approx(0.8)
        assert model.confidence_basis.startswith("broadcast.coalition_stability_score")

    def test_override_absent_narrates_bottom_up_reason(self) -> None:
        broadcast = _broadcast(override=None)
        model = reduce_attention_self_model(broadcast, _field_frame(), now=NOW)

        assert model.attention_reason == "bottom_up_salience"
        assert model.voluntary_override is None
        assert "Pure bottom-up dispatch" in model.reason_narrative
        assert "no active goal override" in model.reason_narrative


class TestCadenceMismatch:
    def test_fresh_broadcast_is_not_stale(self) -> None:
        broadcast = _broadcast(generated_at=NOW - timedelta(seconds=5))
        model = reduce_attention_self_model(broadcast, _field_frame(), now=NOW)

        assert model.broadcast_lane_stale is False
        assert model.broadcast_lane_age_sec == pytest.approx(5.0, abs=0.01)
        assert model.attention_reason == "bottom_up_salience"

    def test_stale_broadcast_is_distinct_from_nothing_salient(self) -> None:
        stale_age = DEFAULT_BROADCAST_STALE_THRESHOLD_SEC + 30.0
        broadcast = _broadcast(generated_at=NOW - timedelta(seconds=stale_age))
        model = reduce_attention_self_model(
            broadcast, _field_frame(), now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(),
        )

        assert model.broadcast_lane_stale is True
        assert model.broadcast_lane_age_sec == pytest.approx(stale_age, abs=0.01)
        # Distinct state: broadcast lane WAS present (a real snapshot exists)
        # but it's stale -- this must not be conflated with "no data at all".
        assert model.broadcast_lane_present is True
        assert model.attention_reason == "field_salience_only"
        assert "no new GWT-dispatch-lane activity" in model.reason_narrative

    def test_broadcast_from_the_future_relative_to_reference_tick_is_treated_as_absent(self) -> None:
        # The broadcast lane is a singleton upsert row (confirmed live
        # 2026-07-18: substrate_attention_broadcast_projection has exactly
        # one row, PK on projection_id). A historical replay tick that
        # predates the single available snapshot has no honest broadcast
        # data for that moment -- must not silently reuse a snapshot from
        # later in time as if it applied retroactively.
        broadcast = _broadcast(generated_at=NOW + timedelta(seconds=10))
        model = reduce_attention_self_model(broadcast, _field_frame(), now=NOW)

        assert model.broadcast_lane_present is False
        assert model.broadcast_lane_age_sec is None
        assert model.attention_reason == "field_salience_only"
        assert "no GWT-dispatch-lane data available" in model.reason_narrative


class TestPartialData:
    def test_all_none_is_no_data_not_a_crash(self) -> None:
        model = reduce_attention_self_model(None, None)

        assert model.attention_reason == "no_data"
        assert model.field_lane_present is False
        assert model.broadcast_lane_present is False
        assert model.voluntary_override is None
        assert model.reason_narrative == "No attention data available from either lane."

    def test_only_broadcast_present(self) -> None:
        broadcast = _broadcast(override=_override())
        model = reduce_attention_self_model(broadcast, None, now=NOW)

        assert model.field_lane_present is False
        assert model.attention_reason == "top_down_override"
        assert model.predicted_shift is None  # no trend data -> honestly absent, not invented

    def test_only_field_frame_present(self) -> None:
        model = reduce_attention_self_model(None, _field_frame(), now=NOW)

        assert model.broadcast_lane_present is False
        assert model.field_lane_present is True
        assert model.attention_reason == "field_salience_only"
        assert model.confidence_basis.startswith(
            "mean(field_attention_frame.dominant_targets"
        )

    def test_only_prediction_error_by_domain_present(self) -> None:
        model = reduce_attention_self_model(
            None, None, now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(),
        )

        assert model.field_lane_present is False
        assert model.attention_reason == "field_salience_only"
        assert model.confidence_basis.startswith("1 - mean(prediction_error)")

    def test_only_prediction_error_trend_present_is_not_no_data(self) -> None:
        """Regression guard (found in code review): trend-only input used to
        leave attention_reason == "no_data" (reason_narrative claiming "No
        attention data available from either lane") while predicted_shift
        was simultaneously populated with a real, substantive prediction --
        a self-contradictory output CLAUDE.md's "no empty-shell cognition"
        rule bans. Must now honestly reflect that real data is present."""
        model = reduce_attention_self_model(
            None, None, now=NOW,
            prediction_error_trend_by_domain={"biometrics": 0.3},
        )

        assert model.attention_reason != "no_data"
        assert model.predicted_shift is not None
        assert "No attention data available" not in model.reason_narrative


class TestPredictionErrorConfidence:
    def test_low_stable_prediction_error_yields_high_confidence(self) -> None:
        model = reduce_attention_self_model(
            None, _field_frame(), now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(),
        )
        # mean of {0.0001, 0.0, 0.0459, 0.0, 0.0} = 0.0092
        assert model.confidence == pytest.approx(1.0 - 0.0092, abs=1e-4)
        assert "5 domains" in model.confidence_basis
        assert "biometrics" in model.confidence_basis

    def test_several_domains_surprising_at_once_lowers_confidence(self) -> None:
        calm = reduce_attention_self_model(
            None, _field_frame(), now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(),
        )
        surprised = reduce_attention_self_model(
            None, _field_frame(), now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(
                execution=0.6, chat=0.5, route=0.4,
            ),
        )
        assert surprised.confidence < calm.confidence

    def test_confidence_never_goes_negative_at_extreme_error(self) -> None:
        model = reduce_attention_self_model(
            None, _field_frame(), now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(
                execution=5.0, transport=5.0, biometrics=5.0, chat=5.0, route=5.0,
            ),
        )
        assert model.confidence == pytest.approx(0.0)

    def test_empty_prediction_error_dict_falls_back_to_field_frame_confidence(self) -> None:
        """An empty dict (falsy) must behave exactly like omitting the
        argument -- no ZeroDivisionError from dividing by len({})."""
        model = reduce_attention_self_model(
            None, _field_frame(), now=NOW,
            prediction_error_by_domain={},
        )
        assert model.confidence_basis.startswith("mean(field_attention_frame.dominant_targets")

    def test_negative_input_does_not_push_confidence_above_one(self) -> None:
        """Regression guard (found in code review): every real producer in
        prediction_error.py only ever emits non-negative values, but this
        function takes an arbitrary caller-supplied dict with no input-range
        assertion -- a negative value must not silently produce
        confidence > 1.0 (AttentionSelfModelV1.confidence's own declared
        range, which Pydantic v2 does not re-check on plain attribute
        assignment)."""
        model = reduce_attention_self_model(
            None, _field_frame(), now=NOW,
            prediction_error_by_domain=_prediction_error_by_domain(biometrics=-0.5),
        )
        assert model.confidence <= 1.0


class TestPredictedShift:
    def test_uses_real_prediction_error_trend_not_invented(self) -> None:
        model = reduce_attention_self_model(
            None, None, now=NOW,
            prediction_error_trend_by_domain={
                "execution": 0.001, "biometrics": 0.18, "chat": -0.02,
            },
        )

        assert model.predicted_shift_basis == (
            "prediction_error_trend_by_domain (Active Inference, argmax by |trend|)"
        )
        assert "biometrics" in model.predicted_shift
        assert "rising" in model.predicted_shift

    def test_falling_trend_is_named_as_falling(self) -> None:
        model = reduce_attention_self_model(
            None, None, now=NOW,
            prediction_error_trend_by_domain={"route": -0.3, "execution": 0.01},
        )

        assert "route" in model.predicted_shift
        assert "falling" in model.predicted_shift

    def test_no_trend_data_yields_no_prediction(self) -> None:
        model = reduce_attention_self_model(None, None, now=NOW)

        assert model.predicted_shift is None
        assert model.predicted_shift_basis == ""

    def test_all_zero_trend_yields_no_prediction(self) -> None:
        model = reduce_attention_self_model(
            None, None, now=NOW,
            prediction_error_trend_by_domain={"execution": 0.0, "chat": 0.0},
        )

        assert model.predicted_shift is None

    def test_predicted_shift_computed_independent_of_attention_reason_branch(self) -> None:
        """The old self_state-driven predicted_shift was computed
        unconditionally, before the attention_reason branching -- the new
        trend-based version must preserve that positioning: it should be set
        even when a top_down_override wins the reason branch."""
        broadcast = _broadcast(override=_override())
        model = reduce_attention_self_model(
            broadcast, _field_frame(), now=NOW,
            prediction_error_trend_by_domain={"biometrics": 0.2},
        )

        assert model.attention_reason == "top_down_override"
        assert model.predicted_shift is not None
        assert "biometrics" in model.predicted_shift


def test_reference_tick_defaults_to_field_frame_generated_at() -> None:
    field_frame = _field_frame(generated_at=NOW)
    model = reduce_attention_self_model(None, field_frame)

    assert model.generated_at == NOW


def test_reference_tick_falls_back_to_broadcast_when_no_field_frame() -> None:
    broadcast = _broadcast(generated_at=NOW)
    model = reduce_attention_self_model(broadcast, None)

    assert model.generated_at == NOW


class TestHarnessClosureSignal:
    """harness_closure_signal only enriches the field_salience_only branch's
    reason_narrative -- see reduce_attention_self_model's own docstring. Not
    touched: top_down_override/bottom_up_salience branches, attention_reason
    itself, or any other field."""

    def test_harness_closure_signal_appends_narrative_clause(self) -> None:
        model = reduce_attention_self_model(
            None,
            _field_frame(),
            now=NOW,
            harness_closure_signal={
                "prediction_error": 0.42,
                "contributing_turn_ids": ["turn-1", "turn-2", "turn-3"],
            },
        )

        assert model.attention_reason == "field_salience_only"
        assert "sustained prediction-error surprise across 3 recent turn(s)" in model.reason_narrative
        assert "current magnitude=0.42" in model.reason_narrative
        # attention_reason itself must not change -- enrichment is narrative-only.
        assert model.attention_reason == "field_salience_only"

    def test_harness_closure_signal_absent_is_byte_identical_to_default(self) -> None:
        """Regression guard: omitting harness_closure_signal (the default,
        every existing caller) must reproduce today's narrative exactly."""
        baseline = reduce_attention_self_model(None, _field_frame(), now=NOW)
        with_none = reduce_attention_self_model(
            None, _field_frame(), now=NOW, harness_closure_signal=None
        )

        assert baseline.reason_narrative == with_none.reason_narrative
        assert "sustained prediction-error surprise" not in baseline.reason_narrative

    def test_harness_closure_signal_zero_prediction_error_does_not_enrich(self) -> None:
        model = reduce_attention_self_model(
            None,
            _field_frame(),
            now=NOW,
            harness_closure_signal={
                "prediction_error": 0.0,
                "contributing_turn_ids": ["turn-1"],
            },
        )

        assert "sustained prediction-error surprise" not in model.reason_narrative

    def test_harness_closure_signal_empty_contributing_turn_ids_does_not_enrich(self) -> None:
        model = reduce_attention_self_model(
            None,
            _field_frame(),
            now=NOW,
            harness_closure_signal={
                "prediction_error": 0.75,
                "contributing_turn_ids": [],
            },
        )

        assert "sustained prediction-error surprise" not in model.reason_narrative

    def test_harness_closure_signal_ignored_outside_field_salience_only_branch(self) -> None:
        """A fresh, non-stale broadcast with no override takes the
        bottom_up_salience branch -- harness_closure_signal must not leak
        its narrative clause in there even if it's provided."""
        broadcast = _broadcast(override=None)
        model = reduce_attention_self_model(
            broadcast,
            _field_frame(),
            now=NOW,
            harness_closure_signal={
                "prediction_error": 0.9,
                "contributing_turn_ids": ["turn-1"],
            },
        )

        assert model.attention_reason == "bottom_up_salience"
        assert "sustained prediction-error surprise" not in model.reason_narrative
