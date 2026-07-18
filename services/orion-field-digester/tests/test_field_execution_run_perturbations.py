from __future__ import annotations

from orion.schemas.state_delta import StateDeltaV1

from app.ingest.state_deltas import delta_to_perturbations


def _make_execution_run_delta(
    *,
    node_id: str = "athena",
    pressure_hints: dict | None = None,
    llm_serving_node: str | None = None,
) -> StateDeltaV1:
    after: dict = {"node_id": node_id, "pressure_hints": pressure_hints or {}}
    if llm_serving_node is not None:
        after["llm_serving_node"] = llm_serving_node
    return StateDeltaV1(
        delta_id="delta_exec_1",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id=f"cortex.exec:{node_id}:corr-1",
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


def test_execution_run_noop_delta_skipped() -> None:
    delta = _make_execution_run_delta(
        pressure_hints={"reasoning_load": 0.35},
        llm_serving_node="atlas",
    )
    delta = delta.model_copy(update={"operation": "noop"})
    assert delta_to_perturbations(delta) == []
