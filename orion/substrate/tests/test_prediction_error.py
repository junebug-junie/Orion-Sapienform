"""Unit tests for the field-native prediction-error instruments in
``orion/substrate/prediction_error.py`` -- 0-1 surprise scores diffing successive
reducer-projection snapshots (not ``SelfStateV1``, not ``tensions.py``'s bucket
vocabulary; see the Sentience Striving Program charter §9b item 3)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.biometrics_projection import (
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.schemas.chat_projection import ChatSessionProjectionV1, ChatTurnStateV1
from orion.schemas.route_projection import (
    RouteArbitrationProjectionV1,
    RouteArbitrationRunStateV1,
)
from orion.substrate.prediction_error import (
    biometrics_prediction_error,
    chat_prediction_error,
    route_prediction_error,
)

_NOW = datetime(2026, 7, 21, 0, 0, 0, tzinfo=timezone.utc)


def _node(node_id: str, pressure_hints: dict) -> NodeBiometricsStateV1:
    return NodeBiometricsStateV1(node_id=node_id, pressure_hints=pressure_hints)


def _projection(nodes: dict[str, NodeBiometricsStateV1]) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="node_biometrics_projection",
        generated_at=_NOW,
        nodes=nodes,
    )


def test_biometrics_prediction_error_zero_when_no_change() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.8, "strain": 0.1})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 0.8, "strain": 0.1})})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_biometrics_prediction_error_scales_with_delta_magnitude() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.5})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 0.8})})
    # |0.8 - 0.5| = 0.3 == _THRESHOLD -> saturates at 1.0
    assert biometrics_prediction_error(prev, curr) == pytest.approx(1.0)


def test_biometrics_prediction_error_partial_delta_below_threshold() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.5})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 0.65})})
    # |0.65 - 0.5| = 0.15 -> 0.15 / 0.30 = 0.5
    assert biometrics_prediction_error(prev, curr) == pytest.approx(0.5)


def test_biometrics_prediction_error_zero_when_node_not_in_prev() -> None:
    """A brand-new node in curr with no prev counterpart contributes no delta --
    mirrors execution_prediction_error's ``prev_run is None: continue`` skip."""
    prev = _projection({})
    curr = _projection({"circe": _node("circe", {"gpu": 0.9})})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_biometrics_prediction_error_zero_when_projections_empty() -> None:
    prev = _projection({})
    curr = _projection({})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_biometrics_prediction_error_handles_disjoint_pressure_hint_keys() -> None:
    """Real biometrics nodes carry different pressure_hints key sets depending on
    node role (GPU nodes: gpu/strain; orchestration nodes: disk/memory/thermal
    pressure) -- confirmed live 2026-07-21 against substrate_node_biometrics_
    projection. A key appearing only on one side of the diff (e.g. a pressure
    signal that newly starts or stops firing) must still be diffed against an
    implicit 0.0, not silently dropped."""
    prev = _projection({"athena": _node("athena", {"disk_pressure": 0.1})})
    curr = _projection(
        {"athena": _node("athena", {"disk_pressure": 0.1, "thermal_pressure": 0.3})}
    )
    # thermal_pressure delta: |0.3 - 0.0| = 0.3; disk_pressure delta: 0.0
    # mean(0.3, 0.0) = 0.15 -> 0.15 / 0.30 = 0.5
    assert biometrics_prediction_error(prev, curr) == pytest.approx(0.5)


def test_biometrics_prediction_error_averages_across_multiple_nodes() -> None:
    prev = _projection(
        {
            "atlas": _node("atlas", {"gpu": 0.5}),
            "athena": _node("athena", {"memory_pressure": 0.2}),
        }
    )
    curr = _projection(
        {
            "atlas": _node("atlas", {"gpu": 0.8}),  # delta 0.3
            "athena": _node("athena", {"memory_pressure": 0.2}),  # delta 0.0
        }
    )
    # mean(0.3, 0.0) = 0.15 -> 0.15 / 0.30 = 0.5
    assert biometrics_prediction_error(prev, curr) == pytest.approx(0.5)


def test_biometrics_prediction_error_clamps_to_one() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.0})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 1.0})})
    assert biometrics_prediction_error(prev, curr) == pytest.approx(1.0)


def test_biometrics_prediction_error_fail_open_on_non_numeric_value() -> None:
    """pressure_hints is dict[str, Any] (unlike execution's dict[str, float]) --
    a malformed/non-numeric value for one key must not raise, just be skipped,
    while a real numeric key on the same node still contributes its delta."""
    prev = _projection(
        {"athena": _node("athena", {"thermal_pressure": "not-a-number", "gpu": 0.5})}
    )
    curr = _projection(
        {"athena": _node("athena", {"thermal_pressure": "still-not-a-number", "gpu": 0.8})}
    )
    # thermal_pressure delta skipped (non-numeric); gpu delta: |0.8-0.5| = 0.3 -> 1.0
    assert biometrics_prediction_error(prev, curr) == pytest.approx(1.0)


# -- chat_prediction_error ---------------------------------------------------


def _chat_turn(
    turn_id: str,
    *,
    word_count: int = 0,
    repair_pressure_level: float = 0.0,
) -> ChatTurnStateV1:
    return ChatTurnStateV1(
        trace_id=f"hub.chat:athena:{turn_id}",
        turn_id=turn_id,
        session_id="orion_journal",
        node_id="athena",
        observed_at=_NOW,
        word_count=word_count,
        repair_pressure_level=repair_pressure_level,
        last_updated_at=_NOW,
    )


def _chat_projection(turns: dict[str, ChatTurnStateV1]) -> ChatSessionProjectionV1:
    return ChatSessionProjectionV1(
        projection_id="chat_session_projection",
        generated_at=_NOW,
        turns=turns,
    )


def test_chat_prediction_error_zero_when_no_change() -> None:
    prev = _chat_projection({"t1": _chat_turn("t1", word_count=20, repair_pressure_level=0.1)})
    curr = _chat_projection({"t1": _chat_turn("t1", word_count=20, repair_pressure_level=0.1)})
    assert chat_prediction_error(prev, curr) == 0.0


def test_chat_prediction_error_zero_when_turn_not_in_prev() -> None:
    """A brand-new turn_id in curr with no prev counterpart contributes no delta --
    mirrors execution_prediction_error's ``prev_run is None: continue`` skip."""
    prev = _chat_projection({})
    curr = _chat_projection({"t1": _chat_turn("t1", word_count=50)})
    assert chat_prediction_error(prev, curr) == 0.0


def test_chat_prediction_error_zero_when_projections_empty() -> None:
    prev = _chat_projection({})
    curr = _chat_projection({})
    assert chat_prediction_error(prev, curr) == 0.0


def test_chat_prediction_error_scales_with_conversation_load_delta() -> None:
    # word_count 0 -> conversation_load 0.0; word_count 45 -> conversation_load 0.30
    prev = _chat_turn("t1", word_count=0)
    curr = _chat_turn("t1", word_count=45)
    prev_proj = _chat_projection({"t1": prev})
    curr_proj = _chat_projection({"t1": curr})
    # conversation_load delta = |0.30 - 0.0| = 0.30; repair_pressure delta = 0.0;
    # topic_coherence delta = 0.0 (both 1.0). mean(0.30, 0, 0) = 0.10 -> 0.10/0.30 = 1/3
    assert chat_prediction_error(prev_proj, curr_proj) == pytest.approx(0.30 / 3 / 0.30)


def test_chat_prediction_error_scales_with_repair_pressure_delta() -> None:
    # repair_pressure and topic_coherence both move by the same magnitude when
    # repair_pressure_level changes (topic_coherence = 1 - repair_pressure_level).
    prev = _chat_turn("t1", repair_pressure_level=0.0)
    curr = _chat_turn("t1", repair_pressure_level=0.30)
    prev_proj = _chat_projection({"t1": prev})
    curr_proj = _chat_projection({"t1": curr})
    # conversation_load delta = 0.0; repair_pressure delta = 0.30;
    # topic_coherence delta = |0.70 - 1.0| = 0.30. mean(0, 0.30, 0.30) = 0.20 -> 0.20/0.30 = 2/3
    assert chat_prediction_error(prev_proj, curr_proj) == pytest.approx(0.20 / 0.30)


def test_chat_prediction_error_averages_across_multiple_turns() -> None:
    prev = _chat_projection(
        {
            "t1": _chat_turn("t1", repair_pressure_level=0.0),
            "t2": _chat_turn("t2", repair_pressure_level=0.0),
        }
    )
    curr = _chat_projection(
        {
            "t1": _chat_turn("t1", repair_pressure_level=0.30),  # non-zero delta alone
            "t2": _chat_turn("t2", repair_pressure_level=0.0),  # zero delta
        }
    )
    deltas = [0.0, 0.30, 0.30, 0.0, 0.0, 0.0]  # t1: cl/rp/tc, t2: cl/rp/tc
    expected = min(1.0, (sum(deltas) / len(deltas)) / 0.30)
    assert chat_prediction_error(prev, curr) == pytest.approx(expected)


# -- route_prediction_error ---------------------------------------------------


def _route_run(
    trace_id: str,
    *,
    lane: str = "background",
    lane_reason: str = "verb_background",
    output_mode: str = "direct_answer",
    mind_requested: bool = False,
) -> RouteArbitrationRunStateV1:
    return RouteArbitrationRunStateV1(
        trace_id=trace_id,
        correlation_id=trace_id,
        session_id="orion_journal",
        node_id="athena",
        lane=lane,
        lane_reason=lane_reason,
        mind_requested=mind_requested,
        output_mode=output_mode,
        last_updated_at=_NOW,
    )


def _route_projection(
    runs: dict[str, RouteArbitrationRunStateV1],
) -> RouteArbitrationProjectionV1:
    return RouteArbitrationProjectionV1(
        projection_id="route_arbitration_projection",
        generated_at=_NOW,
        runs=runs,
    )


def test_route_prediction_error_zero_when_no_change() -> None:
    prev = _route_projection({"r1": _route_run("r1")})
    curr = _route_projection({"r1": _route_run("r1")})
    assert route_prediction_error(prev, curr) == 0.0


def test_route_prediction_error_zero_when_run_not_in_prev() -> None:
    prev = _route_projection({})
    curr = _route_projection({"r1": _route_run("r1")})
    assert route_prediction_error(prev, curr) == 0.0


def test_route_prediction_error_zero_when_projections_empty() -> None:
    prev = _route_projection({})
    curr = _route_projection({})
    assert route_prediction_error(prev, curr) == 0.0


def test_route_prediction_error_one_field_flip_is_quarter() -> None:
    prev = _route_projection({"r1": _route_run("r1", lane="background")})
    curr = _route_projection({"r1": _route_run("r1", lane="chat")})
    # 1 of 4 compared fields differs -> 0.25 (no _THRESHOLD scaling applied)
    assert route_prediction_error(prev, curr) == pytest.approx(0.25)


def test_route_prediction_error_all_fields_flip_is_one() -> None:
    prev = _route_projection(
        {
            "r1": _route_run(
                "r1",
                lane="background",
                lane_reason="verb_background",
                output_mode="direct_answer",
                mind_requested=False,
            )
        }
    )
    curr = _route_projection(
        {
            "r1": _route_run(
                "r1",
                lane="spark",
                lane_reason="explicit_options",
                output_mode="mind_escalation",
                mind_requested=True,
            )
        }
    )
    assert route_prediction_error(prev, curr) == pytest.approx(1.0)


def test_route_prediction_error_not_saturated_by_threshold_scaling() -> None:
    """Explicit regression guard for the documented deviation: a single-field flip
    (mismatch rate 0.25) must NOT be scaled by ``_THRESHOLD`` (0.30) the way the
    other three instruments scale their deltas -- 0.25 / 0.30 would round up to a
    different, wrong value. This must equal 0.25 exactly, not min(1.0, 0.25/0.30)."""
    prev = _route_projection({"r1": _route_run("r1", output_mode="direct_answer")})
    curr = _route_projection({"r1": _route_run("r1", output_mode="mind_escalation")})
    result = route_prediction_error(prev, curr)
    assert result == pytest.approx(0.25)
    assert result != pytest.approx(min(1.0, 0.25 / 0.30))


def test_route_prediction_error_averages_across_multiple_runs() -> None:
    prev = _route_projection(
        {
            "r1": _route_run("r1", lane="background"),
            "r2": _route_run("r2", lane="background"),
        }
    )
    curr = _route_projection(
        {
            "r1": _route_run("r1", lane="chat"),  # 1/4 fields flip -> 0.25
            "r2": _route_run("r2", lane="background"),  # no flip -> 0.0
        }
    )
    assert route_prediction_error(prev, curr) == pytest.approx(0.125)
