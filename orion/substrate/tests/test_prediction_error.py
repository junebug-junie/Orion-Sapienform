"""Unit tests for the field-native prediction-error instruments in
``orion/substrate/prediction_error.py`` -- 0-1 surprise scores diffing successive
reducer-projection snapshots (not ``SelfStateV1``, not ``tensions.py``'s bucket
vocabulary; see the Sentience Striving Program charter §9b item 3)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.biometrics_projection import (
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.schemas.chat_projection import ChatSessionProjectionV1, ChatTurnStateV1
from orion.schemas.execution_projection import (
    ExecutionRunStateV1,
    ExecutionTrajectoryProjectionV1,
)
from orion.schemas.route_projection import (
    RouteArbitrationProjectionV1,
    RouteArbitrationRunStateV1,
)
from orion.structural_mass.git_delta import GitChurnDelta
from orion.structural_mass.graph_delta import GraphStructuralDelta
from orion.structural_mass.pr_lifecycle import PrLifecycleDelta
from orion.substrate.prediction_error import (
    CodebaseMassBaseline,
    _DomainEwmaBaseline,
    biometrics_prediction_error,
    bus_synaptic_prediction_error,
    chat_prediction_error,
    codebase_prediction_error,
    execution_prediction_error,
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


# -- execution_prediction_error -----------------------------------------------


def _exec_run(
    trace_id: str,
    *,
    pressure_hints: dict | None = None,
    last_updated_at: datetime = _NOW,
) -> ExecutionRunStateV1:
    return ExecutionRunStateV1(
        trace_id=trace_id,
        correlation_id=trace_id,
        node_id="athena",
        pressure_hints=pressure_hints or {},
        last_updated_at=last_updated_at,
    )


def _exec_projection(
    runs: dict[str, ExecutionRunStateV1],
    *,
    baseline_ewma: float = 0.0,
    baseline_ewma_var: float = 0.0,
    baseline_ewma_n: int = 0,
) -> ExecutionTrajectoryProjectionV1:
    return ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=_NOW,
        runs=runs,
        prediction_error_baseline_ewma=baseline_ewma,
        prediction_error_baseline_ewma_var=baseline_ewma_var,
        prediction_error_baseline_ewma_n=baseline_ewma_n,
    )


def test_execution_prediction_error_zero_when_no_change() -> None:
    prev = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.25})})
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.25})})
    assert execution_prediction_error(prev, curr) == 0.0


def test_execution_prediction_error_first_tick_is_cold_start_but_seeds_baseline() -> None:
    """2026-07-28 baseline fix: live-confirmed the old fixed ``_THRESHOLD=0.30``
    divisor made this instrument read ~0 essentially always (real deltas run ~3
    orders of magnitude below it) -- the mirror-image of bus_synaptic's old calm-
    floor bug. Fix tracks an EWMA baseline instead. A tick with no established
    baseline yet (``prediction_error_baseline_ewma_n == 0``) must still return
    0.0 (no z-score to compute against -- 'no empty-shell cognition'), but must
    seed the baseline with this tick's real raw delta so the very next tick has
    something real to compare against."""
    prev = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.0})})
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.3})})
    result = execution_prediction_error(prev, curr)
    assert result == 0.0
    # mean of one key's delta (0.3) across the other three implicit-0.0 keys:
    # (0.3 + 0 + 0 + 0) / 4 = 0.075
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.075)
    assert curr.prediction_error_baseline_ewma_var == 0.0
    assert curr.prediction_error_baseline_ewma_n == 1


def test_execution_prediction_error_zero_when_prev_empty() -> None:
    """No prev runs at all -- no fallback reference exists either, must stay 0.0,
    not raise, and must not touch curr's baseline fields (no real delta was ever
    computed this tick)."""
    prev = _exec_projection({})
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.9})})
    assert execution_prediction_error(prev, curr) == 0.0
    assert curr.prediction_error_baseline_ewma_n == 0


def test_execution_prediction_error_falls_back_to_latest_prev_run_for_new_trace_id() -> None:
    """Regression test for the confirmed-live bug (2026-07-21): real cortex-exec runs
    are single-shot creates, a fresh trace_id every time, so an exact trace_id match
    structurally never occurs. Before that fix, this returned 0.0 unconditionally --
    permanently blind regardless of real execution volume. A brand-new trace_id in
    curr with no prev counterpart must diff against prev's most-recently-updated run
    instead of contributing nothing.

    The return value itself is 0.0 here regardless (this call has no established
    baseline yet, a separate and legitimate reason for 0.0 -- see the cold-start
    test above), so the regression guard is on the *baseline* the fallback-matched
    delta seeds, not the return value: it must reflect the real 0.3 delta against
    "prior-run", not 0.0 from an exact match that structurally never occurs."""
    prev = _exec_projection(
        {"prior-run": _exec_run("prior-run", pressure_hints={"cortex_exec_step_load": 0.0})}
    )
    curr = _exec_projection(
        {"new-run": _exec_run("new-run", pressure_hints={"cortex_exec_step_load": 0.3})}
    )
    assert execution_prediction_error(prev, curr) == 0.0
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.075)


def test_execution_prediction_error_fallback_uses_most_recently_updated_prev_run() -> None:
    prev = _exec_projection(
        {
            "older": _exec_run(
                "older",
                pressure_hints={"cortex_exec_step_load": 0.9},
                last_updated_at=_NOW - timedelta(minutes=5),
            ),
            "newer": _exec_run(
                "newer",
                pressure_hints={"cortex_exec_step_load": 0.0},
                last_updated_at=_NOW,
            ),
        }
    )
    curr = _exec_projection(
        {"brand-new": _exec_run("brand-new", pressure_hints={"cortex_exec_step_load": 0.3})}
    )
    execution_prediction_error(prev, curr)
    # Must have diffed against "newer" (delta 0.3 -> mean 0.075), not "older"
    # (delta 0.6 -> mean 0.15).
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.075)


def test_execution_prediction_error_prefers_exact_trace_id_match_over_fallback() -> None:
    """If a trace_id genuinely does recur (a run revised in place), that exact match
    must win over the most-recent-run fallback, even when a more recent, unrelated
    prev run exists."""
    prev = _exec_projection(
        {
            "r1": _exec_run(
                "r1",
                pressure_hints={"cortex_exec_step_load": 0.25},
                last_updated_at=_NOW - timedelta(minutes=5),
            ),
            "unrelated-newer": _exec_run(
                "unrelated-newer",
                pressure_hints={"cortex_exec_step_load": 0.9},
                last_updated_at=_NOW,
            ),
        }
    )
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.25})})
    execution_prediction_error(prev, curr)
    # If the fallback ("unrelated-newer", delta 0.65) won instead of the exact
    # match ("r1", delta 0.0), this would be nonzero.
    assert curr.prediction_error_baseline_ewma == 0.0


def test_execution_prediction_error_scores_deviation_from_established_baseline() -> None:
    """Once a baseline exists, this tick's raw delta is scored as a z-score against
    it (not divided by the old fixed ``_THRESHOLD``). Baseline mean/variance set
    directly here (rather than simulated via prior calls) to keep the expected
    z-score exactly checkable."""
    prev = _exec_projection(
        {"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.20})},
        baseline_ewma=0.022,
        baseline_ewma_var=0.00002,
        baseline_ewma_n=2,
    )
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.32})})
    # delta 0.12 across 4 keys -> raw_mean_delta = 0.03
    # zscore = (0.03 - 0.022) / sqrt(0.00002) = 1.7888543819998317
    # error = min(1.0, max(0.0, zscore) / 3.0)
    expected = min(1.0, max(0.0, (0.03 - 0.022) / math.sqrt(0.00002)) / 3.0)
    assert execution_prediction_error(prev, curr) == pytest.approx(expected)
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.2 * 0.03 + 0.8 * 0.022)
    assert curr.prediction_error_baseline_ewma_n == 3


def test_execution_prediction_error_clamps_below_baseline_tick_to_zero() -> None:
    """A tick calmer than its own recent baseline (negative z-score) must clamp to
    0.0, not go negative -- "surprising" means "more turbulent than usual," not
    "different from usual" in either direction."""
    prev = _exec_projection(
        {"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.20})},
        baseline_ewma=0.05,
        baseline_ewma_var=0.0001,
        baseline_ewma_n=5,
    )
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.20})})
    # delta 0.0 -> raw_mean_delta 0.0; zscore = (0.0-0.05)/sqrt(0.0001) = -5.0
    assert execution_prediction_error(prev, curr) == 0.0


def test_execution_prediction_error_uses_domain_specific_variance_floor() -> None:
    """Regression guard for the 2026-07-28 floor fix: live-confirmed this domain's
    real variance runs about five orders of magnitude below orion.bus.ewma's
    shared default ``_MIN_VARIANCE`` (1e-6, calibrated for a different domain).
    Replaying real historical execution receipts through that shared default
    left every z-score dominated by the borrowed constant instead of this
    domain's real spread (max error 0.015 across 120 real receipts) --
    reintroducing a milder version of the exact bug this whole patch exists to
    fix. This locks in that execution_prediction_error passes its own much
    smaller floor (``_EXECUTION_PREDICTION_ERROR_MIN_VARIANCE`` = 1e-10), not
    the shared default."""
    prev = _exec_projection(
        {"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.0})},
        baseline_ewma=0.0,
        baseline_ewma_var=1e-9,  # below the shared default (1e-6), above the domain floor (1e-10)
        baseline_ewma_n=5,
    )
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 4e-4})})
    # delta 4e-4 across 4 keys -> raw_mean_delta = 1e-4
    result = execution_prediction_error(prev, curr)
    # Domain floor (1e-10) loses to the real variance (1e-9): zscore = 1e-4 /
    # sqrt(1e-9) ~= 3.162 -> saturates at 1.0.
    assert result == pytest.approx(1.0)
    # Under the shared default floor (1e-6, which would win over 1e-9 instead):
    # zscore = 1e-4 / sqrt(1e-6) = 0.1 -> error 0.1/3.0 ~= 0.0333, nowhere near 1.0.
    assert result != pytest.approx(0.1 / 3.0)


def test_execution_prediction_error_saturates_at_one_for_large_zscore() -> None:
    prev = _exec_projection(
        {"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 0.0})},
        baseline_ewma=0.0,
        baseline_ewma_var=1e-8,
        baseline_ewma_n=5,
    )
    curr = _exec_projection({"r1": _exec_run("r1", pressure_hints={"cortex_exec_step_load": 1.0})})
    assert execution_prediction_error(prev, curr) == 1.0


# -- chat_prediction_error ---------------------------------------------------


def _chat_turn(
    turn_id: str,
    *,
    word_count: int = 0,
    repair_pressure_level: float = 0.0,
    last_updated_at: datetime | None = None,
) -> ChatTurnStateV1:
    return ChatTurnStateV1(
        trace_id=f"hub.chat:athena:{turn_id}",
        turn_id=turn_id,
        session_id="orion_journal",
        node_id="athena",
        observed_at=_NOW,
        word_count=word_count,
        repair_pressure_level=repair_pressure_level,
        last_updated_at=last_updated_at or _NOW,
    )


def _chat_projection(
    turns: dict[str, ChatTurnStateV1],
    *,
    baseline_ewma: float = 0.0,
    baseline_ewma_var: float = 0.0,
    baseline_ewma_n: int = 0,
) -> ChatSessionProjectionV1:
    return ChatSessionProjectionV1(
        projection_id="chat_session_projection",
        generated_at=_NOW,
        turns=turns,
        prediction_error_baseline_ewma=baseline_ewma,
        prediction_error_baseline_ewma_var=baseline_ewma_var,
        prediction_error_baseline_ewma_n=baseline_ewma_n,
    )


def test_chat_prediction_error_zero_when_no_change() -> None:
    prev = _chat_projection({"t1": _chat_turn("t1", word_count=20, repair_pressure_level=0.1)})
    curr = _chat_projection({"t1": _chat_turn("t1", word_count=20, repair_pressure_level=0.1)})
    assert chat_prediction_error(prev, curr) == 0.0


def test_chat_prediction_error_zero_when_turn_not_in_prev_and_no_fallback_exists() -> None:
    """A brand-new turn_id with an entirely empty prev (no fallback candidate either)
    still contributes no delta -- there is nothing to compare against, and the baseline
    must be left untouched (mirrors execution_prediction_error's identical case)."""
    prev = _chat_projection({})
    curr = _chat_projection({"t1": _chat_turn("t1", word_count=50)})
    assert chat_prediction_error(prev, curr) == 0.0
    assert curr.prediction_error_baseline_ewma_n == 0


def test_chat_prediction_error_first_tick_is_cold_start_but_seeds_baseline() -> None:
    """2026-08-19 baseline fix: live-confirmed the old fixed ``_THRESHOLD=0.30`` divisor
    made chat lose the predicted_shift argmax to execution/biometrics/bus_synaptic on
    every one of 19,426 real ticks over a 7-day window -- the same disease
    execution_prediction_error was fixed for on 2026-07-28. Fix tracks an EWMA baseline
    instead. A tick with no established baseline yet (``prediction_error_baseline_ewma_n
    == 0``) must still return 0.0 (no z-score to compute against -- 'no empty-shell
    cognition'), but must seed the baseline with this tick's real raw delta so the very
    next tick has something real to compare against."""
    prev = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.0)})
    curr = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.30)})
    result = chat_prediction_error(prev, curr)
    assert result == 0.0
    # repair_pressure delta 0.30, topic_coherence delta 0.30, conversation_load delta 0.0
    # -> mean(0.30, 0.30, 0.0) = 0.20
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.20)
    assert curr.prediction_error_baseline_ewma_var == 0.0
    assert curr.prediction_error_baseline_ewma_n == 1


def test_chat_prediction_error_uses_latest_prev_turn_as_fallback_for_new_turn() -> None:
    """2026-07-22 fix: a brand-new turn_id (the common case -- chat turns are single-shot
    bursts, never revised in place, so an exact turn_id match structurally never occurs)
    falls back to prev's most-recently-updated turn instead of being skipped. Without this
    fallback, chat_prediction_error was permanently 0.0 in production (confirmed live:
    node:substrate.chat was never written despite 241 real accumulated turns).

    This call has no established baseline yet (a separate, legitimate reason for a 0.0
    return -- see the cold-start test above), so the regression guard is on the *baseline*
    the fallback-matched delta seeds, not the return value."""
    prev = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.0)})
    curr = _chat_projection({"t2": _chat_turn("t2", repair_pressure_level=0.30)})
    assert chat_prediction_error(prev, curr) == 0.0
    # Same math as test_chat_prediction_error_first_tick_is_cold_start_but_seeds_baseline.
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.20)


def test_chat_prediction_error_exact_match_still_takes_priority_over_fallback() -> None:
    """When an exact turn_id match exists, it must be used even if a different, more-
    recently-updated turn also exists in prev -- the fallback only fires when no exact
    match is available."""
    prev = _chat_projection(
        {
            "t1": _chat_turn("t1", repair_pressure_level=0.0, last_updated_at=_NOW),
            "t_other": _chat_turn(
                "t_other",
                repair_pressure_level=0.9,
                last_updated_at=_NOW + timedelta(minutes=5),
            ),
        }
    )
    curr = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.30)})
    assert chat_prediction_error(prev, curr) == 0.0
    # If the fallback (t_other, repair_pressure_level=0.9) were used instead of the exact
    # match (t1, repair_pressure_level=0.0), this baseline seed would be different.
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.20)


def test_chat_prediction_error_zero_when_projections_empty() -> None:
    prev = _chat_projection({})
    curr = _chat_projection({})
    assert chat_prediction_error(prev, curr) == 0.0
    assert curr.prediction_error_baseline_ewma_n == 0


def test_chat_prediction_error_conversation_load_key_contributes_to_baseline_seed() -> None:
    # word_count 0 -> conversation_load 0.0; word_count 45 -> conversation_load 0.30
    prev_proj = _chat_projection({"t1": _chat_turn("t1", word_count=0)})
    curr_proj = _chat_projection({"t1": _chat_turn("t1", word_count=45)})
    # conversation_load delta = |0.30 - 0.0| = 0.30; repair_pressure delta = 0.0;
    # topic_coherence delta = 0.0 (both 1.0). mean(0.30, 0, 0) = 0.10
    assert chat_prediction_error(prev_proj, curr_proj) == 0.0  # cold start
    assert curr_proj.prediction_error_baseline_ewma == pytest.approx(0.30 / 3)


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
    assert chat_prediction_error(prev, curr) == 0.0  # cold start
    assert curr.prediction_error_baseline_ewma == pytest.approx(sum(deltas) / len(deltas))


def test_chat_prediction_error_scores_deviation_from_established_baseline() -> None:
    """Once a baseline exists, this tick's raw delta is scored as a z-score against it
    (not divided by the old fixed ``_THRESHOLD``). Baseline mean/variance set directly
    here (rather than simulated via prior calls) to keep the expected z-score exactly
    checkable, mirroring execution_prediction_error's identically-shaped test."""
    prev = _chat_projection(
        {"t1": _chat_turn("t1", repair_pressure_level=0.0)},
        baseline_ewma=0.05,
        baseline_ewma_var=0.01,
        baseline_ewma_n=4,
    )
    curr = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.33)})
    # repair_pressure delta 0.33, topic_coherence delta 0.33, conversation_load delta 0.0
    # -> raw_mean_delta = 0.22; zscore = (0.22 - 0.05) / sqrt(0.01) = 1.7
    expected = min(1.0, max(0.0, (0.22 - 0.05) / math.sqrt(0.01)) / 3.0)
    assert chat_prediction_error(prev, curr) == pytest.approx(expected)
    assert curr.prediction_error_baseline_ewma == pytest.approx(0.2 * 0.22 + 0.8 * 0.05)
    assert curr.prediction_error_baseline_ewma_n == 5


def test_chat_prediction_error_clamps_below_baseline_tick_to_zero() -> None:
    """A tick calmer than its own recent baseline (negative z-score) must clamp to 0.0,
    not go negative -- "surprising" means "more turbulent than usual," not "different
    from usual" in either direction."""
    prev = _chat_projection(
        {"t1": _chat_turn("t1", repair_pressure_level=0.20)},
        baseline_ewma=0.10,
        baseline_ewma_var=0.01,
        baseline_ewma_n=5,
    )
    curr = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.20)})
    # delta 0.0 -> raw_mean_delta 0.0; zscore = (0.0-0.10)/sqrt(0.01) = -1.0
    assert chat_prediction_error(prev, curr) == 0.0


def test_chat_prediction_error_uses_domain_specific_variance_floor() -> None:
    """Regression guard for the 2026-08-19 floor fix: live-confirmed 2026-08-19 that
    chat's real derived raw-delta variance (~5.24e-7, from a 7-day/19,425-tick window)
    is close to, but below, orion.bus.ewma's shared default ``_MIN_VARIANCE`` (1e-6,
    calibrated for a different domain) -- close enough that the shared default would
    still meaningfully flatten real z-scores, the same class of bug fixed harder for
    execution_prediction_error on 2026-07-28. This locks in that chat_prediction_error
    passes its own smaller floor (``_CHAT_PREDICTION_ERROR_MIN_VARIANCE`` = 5e-8), not
    the shared default."""
    prev = _chat_projection(
        {"t1": _chat_turn("t1", repair_pressure_level=0.0)},
        baseline_ewma=0.0,
        baseline_ewma_var=1e-7,  # below shared default (1e-6), above the domain floor (5e-8)
        baseline_ewma_n=5,
    )
    curr = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=0.0015)})
    # repair_pressure delta 0.0015, topic_coherence delta 0.0015, conversation_load delta 0.0
    # -> raw_mean_delta = 0.001
    result = chat_prediction_error(prev, curr)
    # Domain floor (5e-8) loses to the real variance (1e-7): zscore = 0.001 /
    # sqrt(1e-7) ~= 3.162 -> saturates at 1.0.
    assert result == pytest.approx(1.0)
    # Under the shared default floor (1e-6, which would win over 1e-7 instead):
    # zscore = 0.001 / sqrt(1e-6) = 1.0 -> error 1.0/3.0 ~= 0.333, nowhere near 1.0.
    assert result != pytest.approx(1.0 / 3.0)


def test_chat_prediction_error_saturates_at_one_for_large_zscore() -> None:
    prev = _chat_projection(
        {"t1": _chat_turn("t1", repair_pressure_level=0.0)},
        baseline_ewma=0.0,
        baseline_ewma_var=1e-8,
        baseline_ewma_n=5,
    )
    curr = _chat_projection({"t1": _chat_turn("t1", repair_pressure_level=1.0)})
    assert chat_prediction_error(prev, curr) == 1.0


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


def test_route_prediction_error_falls_back_to_latest_prev_run_for_new_trace_id() -> None:
    """Same defect and fix as execution_prediction_error: real route-arbitration runs
    are single-shot per turn, so an exact trace_id match structurally never occurs.
    A brand-new trace_id in curr must diff against prev's most-recently-updated run
    instead of contributing nothing."""
    prev = _route_projection({"prior-run": _route_run("prior-run", lane="background")})
    curr = _route_projection({"new-run": _route_run("new-run", lane="chat")})
    assert route_prediction_error(prev, curr) == pytest.approx(0.25)
    assert route_prediction_error(prev, curr) != 0.0


def test_route_prediction_error_zero_when_prev_empty_no_fallback() -> None:
    prev = _route_projection({})
    curr = _route_projection({"r1": _route_run("r1", lane="chat")})
    assert route_prediction_error(prev, curr) == 0.0


class TestBusSynapticPredictionError:
    """2026-07-30: this instrument counts the FRACTION of anomalous edges; it
    is no longer a magnitude. The whole prior test class encoded the retired
    mean-based formula (calm-floor subtraction, saturation transform) and was
    replaced wholesale rather than patched -- keeping assertions phrased in the
    old formula's terms would have quietly asserted the wrong contract."""

    def test_empty_list_is_zero(self) -> None:
        assert bus_synaptic_prediction_error([]) == 0.0

    def test_all_calm_edges_read_exactly_zero(self) -> None:
        """A genuinely calm mesh must read 0.0. The retired formula could not
        do this reliably -- its calm floor was calibrated for a standard normal
        population the real graph does not follow."""
        assert bus_synaptic_prediction_error([0.0, 0.5, 1.2, 2.9]) == 0.0

    def test_takes_absolute_value_of_negative_zscores(self) -> None:
        """A z-score of -3.5 is just as anomalous as +3.5 and must count."""
        assert bus_synaptic_prediction_error([-3.5]) == 1.0
        assert bus_synaptic_prediction_error([-3.5]) == bus_synaptic_prediction_error([3.5])

    def test_counts_at_the_saturation_boundary_inclusively(self) -> None:
        """>= 3.0, not > 3.0 -- the boundary is the documented anomaly bar
        (Hub's own zscore_threshold=3.0), so an edge sitting exactly on it is
        anomalous."""
        assert bus_synaptic_prediction_error([3.0]) == 1.0
        assert bus_synaptic_prediction_error([2.999999]) == 0.0

    def test_one_extreme_edge_cannot_saturate_a_calm_mesh(self) -> None:
        """The live failure this metric change exists to fix, reproduced.

        Real edge set: median |z| 0.399, p90 1.123, but ONE stale
        cortex-orch->Channel edge carrying |z| = 7087.8 dragged the old mean to
        29.278 and pinned the node at 1.0, driving continuous false "Bus Anomaly
        Detected" alerts. Under a counting metric that edge is worth exactly
        1/N, the same as any other anomalous edge -- the bug cannot recur by
        construction, not by calibration."""
        calm = [0.5] * 230
        assert bus_synaptic_prediction_error(calm) == 0.0

        with_outlier = calm + [7087.8]
        assert bus_synaptic_prediction_error(with_outlier) == pytest.approx(1 / 231)
        # Pre-fix this returned exactly 1.0.
        assert bus_synaptic_prediction_error(with_outlier) < 0.005

    def test_magnitude_beyond_the_bar_does_not_increase_the_reading(self) -> None:
        """The defining property of a counting metric, and the reason this is
        immune to the heavy tail: an edge at |z|=7087 and an edge at |z|=3.01
        contribute identically."""
        assert bus_synaptic_prediction_error([3.01] * 5 + [0.5] * 95) == pytest.approx(
            bus_synaptic_prediction_error([7087.8] * 5 + [0.5] * 95)
        )

    def test_reads_the_live_measured_baseline_for_a_realistic_mesh(self) -> None:
        """Live-measured rest point, 60 samples over 10 minutes of the real
        graph: median 0.026, mean 0.027, p95 0.072, max 0.094, 24 distinct
        values. A realistic mesh shape must land inside that band -- neither a
        suspiciously clean 0.0 nor saturated (CLAUDE.md metric quality gate
        step 4, which warns about BOTH directions)."""
        realistic = [0.5] * 229 + [3.5] * 6
        result = bus_synaptic_prediction_error(realistic)
        assert 0.0128 <= result <= 0.0936, "outside the measured live baseline band"
        assert result == pytest.approx(6 / 235)

    def test_threshold_sits_above_the_measured_baseline_ceiling(self) -> None:
        """Ties the consumer's 0.15 threshold to real data rather than a
        comment. The worst baseline sample observed across 10 minutes was
        0.094; the threshold must clear it, or the false "Bus Anomaly Detected"
        alerts this whole change exists to stop would simply return."""
        observed_baseline_max = 0.0936
        consumer_threshold = 0.15
        assert consumer_threshold > observed_baseline_max

        worst_calm = [3.5] * 22 + [0.5] * 213  # ~0.094, the measured worst case
        assert bus_synaptic_prediction_error(worst_calm) < consumer_threshold

    def test_whole_mesh_anomalous_saturates_at_one(self) -> None:
        assert bus_synaptic_prediction_error([3.5] * 50) == 1.0

    def test_scales_linearly_with_the_anomalous_fraction(self) -> None:
        """A proportion, so the response curve is exactly the fraction -- no
        hidden transform between the count and the reading."""
        assert bus_synaptic_prediction_error([3.5] * 25 + [0.5] * 75) == pytest.approx(0.25)
        assert bus_synaptic_prediction_error([3.5] * 50 + [0.5] * 50) == pytest.approx(0.50)

    def test_single_organ_failure_sits_below_the_consumers_threshold(self) -> None:
        """Documents the disclosed structural limit as an executable fact, so a
        future patch that "fixes" single-organ blindness by lowering the
        equilibrium threshold has to confront this test.

        Live per-organ counts: the busiest organ (orion-social-memory) holds 12
        of ~235 live edges, so its total failure reads ~0.051 -- inside the
        measured baseline ceiling of 0.043 and well under the consumer's 0.15.
        Three busiest organs together read ~0.136, just under it."""
        one_organ = bus_synaptic_prediction_error([3.5] * 12 + [0.5] * 223)
        top_three = bus_synaptic_prediction_error([3.5] * 32 + [0.5] * 203)
        assert one_organ == pytest.approx(0.051, abs=0.001)
        assert top_three == pytest.approx(0.136, abs=0.001)
        assert one_organ < 0.15
        assert top_three < 0.15


class TestCodebaseMassBaselineSerialization:
    """Wire-format round-trip for the consumer patch's
    substrate_codebase_mass_baseline table -- CodebaseMassBaseline.to_json_dict()/
    from_json_dict()."""

    def test_round_trip_preserves_all_three_sub_baselines(self) -> None:
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=1.5, variance=2.5, n=3),
            pr=_DomainEwmaBaseline(ewma=4.5, variance=5.5, n=6),
            graph=_DomainEwmaBaseline(ewma=7.5, variance=8.5, n=9),
        )
        restored = CodebaseMassBaseline.from_json_dict(baseline.to_json_dict())
        assert restored == baseline

    def test_round_trip_default_baseline(self) -> None:
        baseline = CodebaseMassBaseline()
        restored = CodebaseMassBaseline.from_json_dict(baseline.to_json_dict())
        assert restored == baseline

    def test_from_json_dict_tolerates_missing_keys(self) -> None:
        """A row from before some future field addition, or a hand-edited
        JSON blob missing a sub-domain entirely, must not crash -- defaults
        to a fresh cold-start sub-baseline for whatever's missing."""
        restored = CodebaseMassBaseline.from_json_dict({"git": {"ewma": 1.0, "variance": 2.0, "n": 3}})
        assert restored.git == _DomainEwmaBaseline(ewma=1.0, variance=2.0, n=3)
        assert restored.pr == _DomainEwmaBaseline()
        assert restored.graph == _DomainEwmaBaseline()

    def test_from_json_dict_empty_dict_is_fresh_baseline(self) -> None:
        assert CodebaseMassBaseline.from_json_dict({}) == CodebaseMassBaseline()


class TestCodebasePredictionError:
    """Contract patch (docs/superpowers/specs/2026-07-30-codebase-mass-signal-
    design.md) -- composite scoring across all three structural_mass producer
    domains (git/pr/graph), each on its own independent EWMA baseline since each
    producer runs on its own interval and a given tick may populate only some of
    them. Not wired into any tick/bus channel yet -- that's a separate, later
    patch; this only proves the scoring function itself is correct."""

    def _git(
        self,
        *,
        lines_added: int = 0,
        lines_removed: int = 0,
        commit_count: int = 1,
        files_added: int = 0,
        files_deleted: int = 0,
        files_modified: int = 0,
    ) -> GitChurnDelta:
        return GitChurnDelta(
            prev_sha="a" * 40,
            head_sha="b" * 40,
            commit_count=commit_count,
            files_added=files_added,
            files_deleted=files_deleted,
            files_modified=files_modified,
            lines_added=lines_added,
            lines_removed=lines_removed,
        )

    def _pr(
        self,
        *,
        submitted_count: int = 0,
        merged_count: int = 0,
        closed_without_merge_count: int = 0,
    ) -> PrLifecycleDelta:
        _now = datetime(2026, 7, 30, tzinfo=timezone.utc)
        return PrLifecycleDelta(
            since=_now,
            until=_now,
            submitted_count=submitted_count,
            merged_count=merged_count,
            closed_without_merge_count=closed_without_merge_count,
        )

    def _graph(
        self,
        *,
        node_count_delta: int = 0,
        edge_count_delta: int = 0,
        community_count_delta: int = 0,
        god_node_jaccard_similarity: float | None = 1.0,
    ) -> GraphStructuralDelta:
        return GraphStructuralDelta(
            node_count_delta=node_count_delta,
            edge_count_delta=edge_count_delta,
            community_count_delta=community_count_delta,
            god_node_jaccard_similarity=god_node_jaccard_similarity,
        )

    def test_no_producers_fired_is_explicit_no_op_and_baseline_untouched(self) -> None:
        """All three domains None (no producer fired this tick) must read 0.0
        without mutating any sub-baseline -- a real 'nothing happened,' not a
        computed reading against absent data."""
        baseline = CodebaseMassBaseline()
        result = codebase_prediction_error(
            git_delta=None, pr_delta=None, graph_delta=None, baseline=baseline
        )
        assert result.score == 0.0
        assert result.baseline == baseline
        assert result.baseline.git.n == 0

    def test_first_tick_is_cold_start_but_seeds_baseline(self) -> None:
        """No established baseline yet for the one domain that fired -- must
        return 0.0 (no z-score to compute against), but must seed that domain's
        sub-baseline so the next tick has something real to compare against.
        Other two domains' sub-baselines stay untouched (n=0)."""
        baseline = CodebaseMassBaseline()
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=400, lines_removed=200),
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        assert result.score == 0.0
        assert result.baseline.git.ewma == pytest.approx(600.0)
        assert result.baseline.git.n == 1
        assert result.baseline.pr.n == 0
        assert result.baseline.graph.n == 0

    def test_single_domain_scores_against_its_own_established_baseline(self) -> None:
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=300, lines_removed=220),  # magnitude 520
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        # zscore = (520 - 500) / sqrt(10_000) = 0.2 -> mean of just this one zscore
        assert result.score == pytest.approx(0.2 / 3.0)
        assert result.baseline.git.n == 6

    def test_two_domains_average_their_independent_zscores(self) -> None:
        """git and pr both fire this tick; graph doesn't. Composite score is the
        mean of exactly the two available z-scores, not diluted by a phantom
        third 0.0 for the domain that didn't observe anything."""
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5),
            pr=_DomainEwmaBaseline(ewma=2.0, variance=1.0, n=5),
        )
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=300, lines_removed=220),  # magnitude 520, z=0.2
            pr_delta=self._pr(submitted_count=3, merged_count=2),  # magnitude 5, z=(5-2)/1=3.0
            graph_delta=None,
            baseline=baseline,
        )
        expected_mean_z = (0.2 + 3.0) / 2
        assert result.score == pytest.approx(min(1.0, expected_mean_z / 3.0))
        assert result.baseline.git.n == 6
        assert result.baseline.pr.n == 6
        assert result.baseline.graph.n == 0

    def test_all_three_domains_average_together(self) -> None:
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=5),
            pr=_DomainEwmaBaseline(ewma=2.0, variance=1.0, n=5),
            graph=_DomainEwmaBaseline(ewma=100.0, variance=400.0, n=5),
        )
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=300, lines_removed=220),  # z=0.2
            pr_delta=self._pr(submitted_count=3, merged_count=2),  # z=3.0
            graph_delta=self._graph(node_count_delta=120, edge_count_delta=0),  # magnitude 120, z=(120-100)/20=1.0
            baseline=baseline,
        )
        expected_mean_z = (0.2 + 3.0 + 1.0) / 3
        assert result.score == pytest.approx(min(1.0, expected_mean_z / 3.0))

    def test_saturates_exactly_at_zscore_saturation_boundary(self) -> None:
        """No off-by-one at the saturation ceiling: a single domain's zscore of
        exactly 3.0 must score exactly 1.0."""
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=100.0, n=10))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=530, lines_removed=0),  # magnitude 530, z=3.0 exactly
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        assert result.score == pytest.approx(1.0)

    def test_large_catch_up_diff_spanning_missed_ticks_scores_high(self) -> None:
        """A big diff accumulated across several missed ticks should read as a
        real spike relative to the domain's own established baseline."""
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=500.0, variance=10_000.0, n=20))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=9000, lines_removed=6000, commit_count=14, files_modified=40),
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        # zscore = (15_000 - 500) / sqrt(10_000) = 145.0 -> far past saturation
        assert result.score == pytest.approx(1.0)

    def test_below_baseline_tick_clamps_to_zero_before_averaging(self) -> None:
        """A quieter-than-usual git tick (negative z-score) clamps to 0.0 before
        averaging with a genuinely surprising PR tick -- the PR spike must not
        be diluted toward 0 by an unclamped negative contribution from git."""
        baseline = CodebaseMassBaseline(
            git=_DomainEwmaBaseline(ewma=5000.0, variance=1_000_000.0, n=10),
            pr=_DomainEwmaBaseline(ewma=2.0, variance=1.0, n=10),
        )
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=10, lines_removed=5),  # magnitude 15, well below baseline
            pr_delta=self._pr(submitted_count=3, merged_count=2),  # magnitude 5, z=3.0
            graph_delta=None,
            baseline=baseline,
        )
        # If git's negative zscore weren't clamped, mean would pull below 3.0/3.0=1.0.
        # Clamped to 0.0: mean(0.0, 3.0) / 3.0 = 0.5.
        assert result.score == pytest.approx(0.5)

    def test_git_only_below_baseline_tick_is_exactly_zero(self) -> None:
        baseline = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=5000.0, variance=1_000_000.0, n=10))
        result = codebase_prediction_error(
            git_delta=self._git(lines_added=10, lines_removed=5),
            pr_delta=None,
            graph_delta=None,
            baseline=baseline,
        )
        assert result.score == 0.0

    def test_baseline_state_flows_independently_of_module_globals(self) -> None:
        """Two independent baseline threads must not interfere with each other
        -- state is entirely caller-owned, no hidden module-level mutation."""
        baseline_a = CodebaseMassBaseline()
        baseline_b = CodebaseMassBaseline(git=_DomainEwmaBaseline(ewma=9999.0, variance=1.0, n=50))
        result_a = codebase_prediction_error(
            git_delta=self._git(lines_added=100, lines_removed=0), pr_delta=None, graph_delta=None,
            baseline=baseline_a,
        )
        result_b = codebase_prediction_error(
            git_delta=self._git(lines_added=100, lines_removed=0), pr_delta=None, graph_delta=None,
            baseline=baseline_b,
        )
        assert result_a.baseline.git.n == 1
        assert result_b.baseline.git.n == 51
        assert result_a.baseline.git.ewma != result_b.baseline.git.ewma
