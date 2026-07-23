from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.state_delta import StateDeltaV1

from app.ingest.state_deltas import delta_to_perturbations

FIXED_TS = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def _make_execution_run_delta(
    *,
    node_id: str = "athena",
    pressure_hints: dict | None = None,
    llm_serving_node: str | None = None,
    lane: str | None = None,
) -> StateDeltaV1:
    """`lane=None` (the default) produces a bare cortex-exec-shaped trace_id -- the
    realistic shape for the majority of execution_run deltas in production. Pass
    lane="harness_motor" or lane="hub_turn_timeout" to simulate the specific traces
    the FCC-motor-only channels are gated on."""
    after: dict = {"node_id": node_id, "pressure_hints": pressure_hints or {}}
    if llm_serving_node is not None:
        after["llm_serving_node"] = llm_serving_node
    target_id = f"cortex.exec:{node_id}:corr-1"
    if lane:
        target_id = f"{target_id}:{lane}"
    return StateDeltaV1(
        delta_id="delta_exec_1",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id=target_id,
        operation="update",
        after=after,
        caused_by_event_ids=["gev_exec_1"],
        reducer_id="execution_trajectory_reducer",
    )


def test_execution_run_without_llm_serving_node_targets_run_node() -> None:
    """Backward-compat: a run that never called an LLM keeps every channel,
    including reasoning_load, attributed to the orchestrating node -- exactly
    today's behavior."""
    delta = _make_execution_run_delta(
        pressure_hints={
            "execution_load": 0.5,
            "execution_friction": 0.1,
            "reasoning_load": 0.05,
            "failure_pressure": 0.0,
        },
    )
    perturbations = delta_to_perturbations(delta)
    by_channel = {p.channel: p for p in perturbations}
    assert by_channel["reasoning_load"].node_id == "node:athena"
    assert by_channel["execution_load"].node_id == "node:athena"


def test_execution_run_with_llm_serving_node_routes_reasoning_load_only() -> None:
    """A run whose LLM call was served by atlas attributes reasoning_load to
    node:atlas specifically, while execution_load/execution_friction/
    failure_pressure stay on the orchestrating node (athena) -- see
    services/orion-field-digester/README.md's reasoning_pressure glossary
    entry for why node:atlas.reasoning_load was permanently 0.0 before this.
    llm_serving_node lives on the top-level ExecutionRunStateV1 field (not
    inside pressure_hints, which must stay float-only -- see
    orion/substrate/relational/adapters/execution_ctx.py's max(hints.values())
    over the same dict)."""
    delta = _make_execution_run_delta(
        pressure_hints={
            "execution_load": 0.5,
            "execution_friction": 0.1,
            "reasoning_load": 0.35,
            "failure_pressure": 0.0,
        },
        llm_serving_node="atlas",
    )
    perturbations = delta_to_perturbations(delta)
    by_channel = {p.channel: p for p in perturbations}
    assert by_channel["reasoning_load"].node_id == "node:atlas"
    assert by_channel["reasoning_load"].intensity == 0.35
    assert by_channel["execution_load"].node_id == "node:athena"
    assert by_channel["execution_friction"].node_id == "node:athena"
    assert by_channel["failure_pressure"].node_id == "node:athena"


def test_execution_run_new_fcc_channels_attributed_to_orchestrating_node() -> None:
    """harness_step_load/tool_failure_streak_pressure/avg_step_chars_pressure/
    compliance_deficit are FCC-motor-process signals, not LLM-serving-node signals --
    they must stay on node_key even when llm_serving_node is set, unlike reasoning_load.
    Requires lane="harness_motor" -- see test_execution_run_fcc_channels_ignored_off_lane
    below for why."""
    delta = _make_execution_run_delta(
        lane="harness_motor",
        pressure_hints={
            "harness_step_load": 0.62,
            "tool_failure_streak_pressure": 0.67,
            "avg_step_chars_pressure": 0.3,
            "compliance_deficit": 0.333,
        },
        llm_serving_node="atlas",
    )
    perturbations = delta_to_perturbations(delta)
    by_channel = {p.channel: p for p in perturbations}
    for channel in (
        "harness_step_load",
        "tool_failure_streak_pressure",
        "avg_step_chars_pressure",
        "compliance_deficit",
    ):
        assert by_channel[channel].node_id == "node:athena"
    assert by_channel["harness_step_load"].intensity == 0.62


def test_execution_run_turn_incompletion_attributed_to_hub_node() -> None:
    """turn_incompletion comes from an orion-hub-sourced exec_turn_timeout event whose
    trace_id lane is under Hub's own node identity, not the governor's -- node_key here
    is Hub's node, distinct from what harness-governor would report for the same
    correlation_id (there is none, in this failure mode). Requires
    lane="hub_turn_timeout" -- see test_execution_run_fcc_channels_ignored_off_lane."""
    delta = _make_execution_run_delta(
        node_id="athena",
        lane="hub_turn_timeout",
        pressure_hints={"turn_incompletion": 1.0},
    )
    perturbations = delta_to_perturbations(delta)
    by_channel = {p.channel: p for p in perturbations}
    assert by_channel["turn_incompletion"].node_id == "node:athena"
    assert by_channel["turn_incompletion"].intensity == 1.0


def test_execution_run_fcc_channels_ignored_off_lane() -> None:
    """LIVE-CONFIRMED BUG regression test (2026-07-23): compute_pressure_hints()
    unconditionally includes harness_step_load/tool_failure_streak_pressure/
    avg_step_chars_pressure/compliance_deficit/turn_incompletion in EVERY
    execution_run's pressure_hints dict, including cortex-exec-only runs. Before this
    lane gate, ANY execution_run delta -- a bare cortex-exec trace, or even the SAME
    turn's own :harness_finalize_reflect/:orion_voice_finalize cortex-exec sub-lanes
    -- emitted a mode="replace" perturbation for these channels targeting the same
    node_key, silently resetting a real harness-motor value back to whatever that
    unrelated delta happened to carry (0.0/"unknown" defaults). Confirmed live: a
    real harness_step_load=0.6892 was reset to 0.0 within 15 seconds by the same
    turn's own :harness_finalize_reflect delta. These 5 channels must ONLY ever
    produce perturbations from their own specific lane."""
    off_lane_deltas = [
        _make_execution_run_delta(lane=None, pressure_hints={"harness_step_load": 0.5}),
        _make_execution_run_delta(
            lane="harness_finalize_reflect", pressure_hints={"harness_step_load": 0.5}
        ),
        _make_execution_run_delta(
            lane="orion_voice_finalize", pressure_hints={"harness_step_load": 0.5}
        ),
        _make_execution_run_delta(lane=None, pressure_hints={"turn_incompletion": 1.0}),
        _make_execution_run_delta(
            lane="harness_motor", pressure_hints={"turn_incompletion": 1.0}
        ),
    ]
    for delta in off_lane_deltas:
        perturbations = delta_to_perturbations(delta)
        channels = {p.channel for p in perturbations}
        assert "harness_step_load" not in channels
        assert "turn_incompletion" not in channels


def test_execution_run_harness_step_load_stays_bounded_through_apply_perturbations() -> None:
    """Regression test for a code-review finding: apply_perturbations() hard-clamps
    every mode="replace" channel to [0,1] (app/digestion/perturbation.py), which an
    earlier (fixed) version of harness_step_load's formula could exceed at a
    harness_started_step_count as low as 2 -- silently saturating the channel for
    virtually every real run. compute_pressure_hints() now keeps the value <=1.0 on
    its own, so this asserts the value survives the real write path unchanged, not
    just that the write path happens to clamp a bad input."""
    from app.digestion.perturbation import apply_perturbations
    from orion.schemas.field_state import FieldStateV1

    delta = _make_execution_run_delta(lane="harness_motor", pressure_hints={"harness_step_load": 0.92})
    state = FieldStateV1(generated_at=FIXED_TS, tick_id="tick_x", node_vectors={}, edges=[])
    state = apply_perturbations(state, delta_to_perturbations(delta), now=FIXED_TS)
    assert state.node_vectors["node:athena"]["harness_step_load"] == 0.92


def test_execution_run_off_lane_delta_does_not_reset_prior_harness_value() -> None:
    """End-to-end reproduction of the live bug via apply_perturbations(): a real
    harness_motor value, once set, must survive a subsequent off-lane delta (the
    same turn's own cortex-exec sub-lane, or any later unrelated cortex-exec-only
    turn) rather than being reset to that delta's inert default."""
    from app.digestion.perturbation import apply_perturbations
    from orion.schemas.field_state import FieldStateV1

    state = FieldStateV1(generated_at=FIXED_TS, tick_id="tick_x", node_vectors={}, edges=[])
    real_delta = _make_execution_run_delta(
        lane="harness_motor", pressure_hints={"harness_step_load": 0.6892}
    )
    state = apply_perturbations(state, delta_to_perturbations(real_delta), now=FIXED_TS)
    assert state.node_vectors["node:athena"]["harness_step_load"] == 0.6892

    off_lane_delta = _make_execution_run_delta(
        lane="harness_finalize_reflect", pressure_hints={"harness_step_load": 0.0}
    )
    state = apply_perturbations(state, delta_to_perturbations(off_lane_delta), now=FIXED_TS)
    assert state.node_vectors["node:athena"]["harness_step_load"] == 0.6892


def test_execution_run_noop_delta_skipped() -> None:
    delta = _make_execution_run_delta(
        pressure_hints={"reasoning_load": 0.35},
        llm_serving_node="atlas",
    )
    delta = delta.model_copy(update={"operation": "noop"})
    assert delta_to_perturbations(delta) == []
