"""Deterministic unit tests for measure_ast_hot_reducer.py.

No DB. The replay layer calls the REAL `reduce_attention_self_model`
(pure, no I/O -- see orion/substrate/attention_self_model.py), so it is
exercised directly with synthetic field/field-state/broadcast payload dicts,
same fixture-loading pattern as
scripts/analysis/tests/test_measure_origination_gate.py.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "measure_ast_hot_reducer.py"
_spec = importlib.util.spec_from_file_location("measure_ast_hot_reducer", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_ast_hot_reducer"] = mod
_spec.loader.exec_module(mod)

from orion.schemas.attention_frame import (  # noqa: E402
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    VoluntaryOverrideV1,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1  # noqa: E402

UTC = timezone.utc
BASE = datetime(2026, 7, 1, 0, 0, 0, tzinfo=UTC)


def _field_payload(ts: datetime) -> dict:
    frame = FieldAttentionFrameV1(
        frame_id=f"frame-{ts.isoformat()}",
        generated_at=ts,
        source_field_tick_id="tick",
        source_field_generated_at=ts,
        overall_salience=0.6,
        dominant_targets=[
            FieldAttentionTargetV1(
                target_id="node-1", target_kind="node", salience_score=0.6,
                pressure_score=0.4, novelty_score=0.2, urgency_score=0.3, confidence_score=0.5,
            )
        ],
    )
    return frame.model_dump(mode="json")


def _field_state_payload(ts: datetime, **prediction_errors: float) -> dict:
    """Minimal FieldStateV1-shaped dict -- only the node_vectors keys this
    script's extract_prediction_error_by_domain() actually reads, since the
    full schema isn't needed to exercise this pure extraction/parsing logic.
    """
    domain_nodes = mod.PREDICTION_ERROR_DOMAIN_NODES
    node_vectors = {
        domain_nodes[domain]: {"prediction_error": value}
        for domain, value in prediction_errors.items()
    }
    return {"generated_at": ts.isoformat(), "node_vectors": node_vectors}


def _broadcast_payload(ts: datetime, *, override: VoluntaryOverrideV1 | None = None) -> dict:
    frame = AttentionFrameV1(generated_at=ts, voluntary_override=override)
    projection = AttentionBroadcastProjectionV1(
        generated_at=ts, frame=frame, selected_action_type="watch",
        selected_open_loop_id="loop-1", coalition_stability_score=0.7,
    )
    return projection.model_dump(mode="json")


class TestExtractPredictionErrorByDomain:
    def test_extracts_present_domains_only(self) -> None:
        payload = _field_state_payload(BASE, execution=0.1, biometrics=0.5)
        out = mod.extract_prediction_error_by_domain(payload)
        assert out == {"execution": 0.1, "biometrics": 0.5}

    def test_missing_node_vectors_yields_empty_dict(self) -> None:
        assert mod.extract_prediction_error_by_domain({}) == {}

    def test_malformed_value_is_skipped_not_raised(self) -> None:
        payload = {
            "node_vectors": {
                "node:substrate.execution": {"prediction_error": "not-a-number"},
                "node:substrate.biometrics": {"prediction_error": 0.3},
            }
        }
        out = mod.extract_prediction_error_by_domain(payload)
        assert out == {"biometrics": 0.3}


class TestComputePredictionErrorTrend:
    def test_reversion_predicted_after_recent_climb(self) -> None:
        """This is deliberately the OPPOSITE sign of a naive "continue the
        recent direction" formula -- validated against real historical data
        to be the empirically correct one (see the function's own
        docstring). A domain whose recent half climbed above its prior half
        is predicted to FALL next (negative trend), not keep climbing."""
        window = [
            {"biometrics": 0.01}, {"biometrics": 0.02},
            {"biometrics": 0.3}, {"biometrics": 0.4},
        ]
        trend = mod.compute_prediction_error_trend(window)
        assert trend["biometrics"] < 0

    def test_reversion_predicted_after_recent_drop(self) -> None:
        window = [
            {"biometrics": 0.4}, {"biometrics": 0.3},
            {"biometrics": 0.02}, {"biometrics": 0.01},
        ]
        trend = mod.compute_prediction_error_trend(window)
        assert trend["biometrics"] > 0

    def test_domain_missing_from_either_half_is_excluded(self) -> None:
        window = [
            {"biometrics": 0.1}, {"execution": 0.1},
            {"biometrics": 0.2}, {"biometrics": 0.3},
        ]
        trend = mod.compute_prediction_error_trend(window)
        assert "execution" not in trend
        assert "biometrics" in trend

    def test_fewer_than_two_ticks_yields_empty_dict(self) -> None:
        assert mod.compute_prediction_error_trend([]) == {}
        assert mod.compute_prediction_error_trend([{"biometrics": 0.1}]) == {}


def test_replay_reducer_joins_field_state_by_nearest_preceding_timestamp() -> None:
    field_rows = [
        (BASE, _field_payload(BASE)),
        (BASE + timedelta(seconds=10), _field_payload(BASE + timedelta(seconds=10))),
    ]
    field_state_rows = [
        (BASE - timedelta(seconds=1), _field_state_payload(BASE - timedelta(seconds=1), biometrics=0.2)),
    ]

    ticks, field_skipped, fs_skipped, bc_skipped = mod.replay_reducer(field_rows, [], field_state_rows)

    assert field_skipped == 0
    assert fs_skipped == 0
    assert bc_skipped == 0
    assert len(ticks) == 2
    assert all(t.confidence is not None for t in ticks)


def test_replay_reducer_skips_unparseable_rows_without_raising() -> None:
    field_rows = [(BASE, {"not": "a valid frame"})]
    ticks, field_skipped, fs_skipped, bc_skipped = mod.replay_reducer(field_rows, [], [])

    assert ticks == []
    assert field_skipped == 1
    assert fs_skipped == 0
    assert bc_skipped == 0


def test_replay_reducer_surfaces_real_voluntary_override() -> None:
    override = VoluntaryOverrideV1(
        goal_artifact_id="goal-1", goal_drive_origin="curiosity",
        chosen_loop_id="loop-1", beat_loop_id="loop-2",
        chosen_bottom_up=0.3, beat_bottom_up=0.5, applied_bias=0.4, effort_spent=0.1,
    )
    broadcast_rows = [(BASE, _broadcast_payload(BASE, override=override))]
    field_rows = [(BASE + timedelta(seconds=1), _field_payload(BASE + timedelta(seconds=1)))]

    ticks, _, _, bc_skipped = mod.replay_reducer(field_rows, broadcast_rows, [])

    assert bc_skipped == 0
    assert len(ticks) == 1
    assert ticks[0].has_voluntary_override is True
    assert ticks[0].attention_reason == "top_down_override"
    assert "loop-1" in ticks[0].reason_narrative


def test_replay_reducer_honestly_absent_when_broadcast_predates_nothing_reference() -> None:
    """A broadcast row dated AFTER the field tick must not be joined to it --
    the reducer's own absent/stale logic (see TestCadenceMismatch in
    orion/substrate/tests/test_attention_self_model.py) must not treat a
    future snapshot as if it applied retroactively."""
    broadcast_rows = [(BASE + timedelta(hours=1), _broadcast_payload(BASE + timedelta(hours=1)))]
    field_rows = [(BASE, _field_payload(BASE))]

    ticks, _, _, _ = mod.replay_reducer(field_rows, broadcast_rows, [])

    assert len(ticks) == 1
    assert ticks[0].broadcast_lane_present is False


def test_replay_reducer_joins_broadcast_by_nearest_preceding_timestamp_per_tick() -> None:
    """Real per-tick history join (the fix this test module covers): each
    field tick should see the most-recent broadcast row at or before its own
    timestamp, not a single static row pinned to every call. Two broadcast
    rows straddle two field ticks; each tick must join to its own nearest-
    preceding broadcast, proving the two-pointer join advances independently
    per tick rather than reusing the first (or only) row."""
    override_early = VoluntaryOverrideV1(
        goal_artifact_id="goal-early", goal_drive_origin="curiosity",
        chosen_loop_id="loop-early", beat_loop_id="loop-x",
        chosen_bottom_up=0.3, beat_bottom_up=0.5, applied_bias=0.4, effort_spent=0.1,
    )
    override_late = VoluntaryOverrideV1(
        goal_artifact_id="goal-late", goal_drive_origin="curiosity",
        chosen_loop_id="loop-late", beat_loop_id="loop-y",
        chosen_bottom_up=0.3, beat_bottom_up=0.5, applied_bias=0.4, effort_spent=0.1,
    )
    broadcast_rows = [
        (BASE, _broadcast_payload(BASE, override=override_early)),
        (BASE + timedelta(seconds=20), _broadcast_payload(BASE + timedelta(seconds=20), override=override_late)),
    ]
    field_rows = [
        (BASE + timedelta(seconds=5), _field_payload(BASE + timedelta(seconds=5))),
        (BASE + timedelta(seconds=25), _field_payload(BASE + timedelta(seconds=25))),
    ]

    ticks, _, _, bc_skipped = mod.replay_reducer(field_rows, broadcast_rows, [])

    assert bc_skipped == 0
    assert len(ticks) == 2
    assert "loop-early" in ticks[0].reason_narrative
    assert "loop-late" in ticks[1].reason_narrative


def test_replay_reducer_predicted_shift_tracks_rolling_trend_window() -> None:
    """A domain whose prediction_error climbs steadily across field_state
    ticks should surface as the named predicted_shift once enough ticks
    have accumulated to compute a trend (>=2) -- predicting reversion
    (falling), the empirically-validated direction, not naive continuation.
    """
    field_state_rows = [
        (BASE + timedelta(seconds=i), _field_state_payload(BASE + timedelta(seconds=i), biometrics=0.01 * i))
        for i in range(1, 6)
    ]
    field_rows = [(BASE + timedelta(seconds=6), _field_payload(BASE + timedelta(seconds=6)))]

    ticks, _, fs_skipped, _ = mod.replay_reducer(field_rows, [], field_state_rows)

    assert fs_skipped == 0
    assert len(ticks) == 1
    assert ticks[0].predicted_shift is not None
    assert "biometrics" in ticks[0].predicted_shift
    assert "falling" in ticks[0].predicted_shift


def test_reason_histogram_counts_each_category() -> None:
    ticks = [
        mod.ReplayTick(BASE, "top_down_override", 0.5, None, True, False, 1.0, True, None, True, ""),
        mod.ReplayTick(BASE, "bottom_up_salience", 0.5, None, True, False, 1.0, True, None, False, ""),
        mod.ReplayTick(BASE, "bottom_up_salience", 0.5, None, True, False, 1.0, True, None, False, ""),
    ]
    hist = mod.reason_histogram(ticks)
    assert hist["top_down_override"] == 1
    assert hist["bottom_up_salience"] == 2


def test_find_override_examples_returns_context_window() -> None:
    ticks = [
        mod.ReplayTick(BASE, "field_salience_only", None, None, False, True, None, True, None, False, "a"),
        mod.ReplayTick(BASE, "top_down_override", 0.9, None, True, False, 1.0, True, None, True, "override!"),
        mod.ReplayTick(BASE, "bottom_up_salience", 0.7, None, True, False, 1.0, True, None, False, "c"),
    ]
    examples = mod.find_override_examples(ticks, context=1)
    assert len(examples) == 1
    assert len(examples[0]) == 3
    assert examples[0][1].has_voluntary_override is True


def test_find_override_examples_empty_when_no_override_present() -> None:
    ticks = [
        mod.ReplayTick(BASE, "field_salience_only", None, None, False, True, None, True, None, False, "a"),
    ]
    assert mod.find_override_examples(ticks) == []
