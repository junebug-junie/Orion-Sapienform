from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    ActiveNodePressureStateV1,
)
from orion.schemas.execution_projection import (
    ExecutionRunStateV1,
    ExecutionTrajectoryProjectionV1,
)
from orion.schemas.transport_projection import (
    TransportBusProjectionV1,
    TransportBusStateV1,
)
from orion.substrate.relational.adapters.biometrics_ctx import (
    map_biometrics_ctx_to_substrate,
)
from orion.substrate.relational.adapters.execution_ctx import (
    map_execution_ctx_to_substrate,
)
from orion.substrate.relational.adapters.transport_ctx import (
    map_transport_ctx_to_substrate,
)

NOW = datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #
def _execution_projection(n: int = 1) -> ExecutionTrajectoryProjectionV1:
    runs: dict[str, ExecutionRunStateV1] = {}
    for i in range(n):
        trace_id = f"trace{i:04d}abcdef"
        runs[trace_id] = ExecutionRunStateV1(
            trace_id=trace_id,
            correlation_id=f"corr{i}",
            node_id="orion-exec",
            verb="chat",
            mode="stream",
            status="running",
            step_count=3 + i,
            pressure_hints={
                "cortex_exec_step_load": 0.4,
                "execution_friction": 0.9,
                "failure_pressure": 0.1,
                "reasoning_load": 0.2,
            },
            last_updated_at=NOW + timedelta(seconds=i),
        )
    return ExecutionTrajectoryProjectionV1(projection_id="exec1", generated_at=NOW, runs=runs)


def test_execution_adapter_emits_run_nodes() -> None:
    proj = _execution_projection(1)
    record = map_execution_ctx_to_substrate({"execution_trajectory_projection": proj})
    assert record is not None
    assert record.anchor_scope == "orion"
    node = record.nodes[0]
    assert node.label.startswith("execution:chat:")
    assert node.anchor_scope == "orion"
    assert node.subject_ref == "entity:orion"
    # salience = max pressure hint = execution_friction 0.9
    assert node.signals.salience == 0.9
    assert node.metadata["source_kind"] == "execution_trajectory"
    assert node.metadata["verb"] == "chat"
    assert node.metadata["status"] == "running"
    assert node.metadata["pressure_hints"]["execution_friction"] == 0.9


def test_execution_adapter_accepts_dict_json_absent_and_garbage() -> None:
    proj = _execution_projection(1)
    assert map_execution_ctx_to_substrate(
        {"execution_trajectory_projection": proj.model_dump(mode="json")}
    ) is not None
    assert map_execution_ctx_to_substrate(
        {"execution_trajectory_projection": proj.model_dump_json()}
    ) is not None
    assert map_execution_ctx_to_substrate({}) is None
    assert map_execution_ctx_to_substrate({"execution_trajectory_projection": "not json"}) is None


def test_execution_adapter_caps_at_20() -> None:
    record = map_execution_ctx_to_substrate(
        {"execution_trajectory_projection": _execution_projection(35)}
    )
    assert record is not None
    assert len(record.nodes) <= 20


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def _transport_projection(n: int = 1) -> TransportBusProjectionV1:
    buses: dict[str, TransportBusStateV1] = {}
    for i in range(n):
        key = f"bus{i}"
        buses[key] = TransportBusStateV1(
            target_id=f"target{i}",
            node_id=f"node{i}",
            sample_window_id="w1",
            source_trace_id="t1",
            redis_ping_ok=True,
            reliability_pressure=0.2,
            contract_pressure=0.1 + (i % 5) * 0.1,
            observed_at=NOW,
        )
    return TransportBusProjectionV1(updated_at=NOW, buses=buses)


def test_transport_adapter_emits_bus_nodes() -> None:
    proj = _transport_projection(1)
    record = map_transport_ctx_to_substrate({"transport_bus_projection": proj})
    assert record is not None
    assert record.anchor_scope == "orion"
    node = record.nodes[0]
    assert node.label == "transport:node0"
    assert node.anchor_scope == "orion"
    assert node.subject_ref == "entity:orion"
    # salience = max(reliability 0.2, contract 0.1)
    assert node.signals.salience == pytest.approx(0.2)
    # confidence = 1 - reliability_pressure (what the retired
    # delivery_confidence always equalled)
    assert node.signals.confidence == pytest.approx(0.8)
    assert node.metadata["redis_ping_ok"] is True
    for retired in ("stream_backlog_health", "delivery_confidence", "stream_backlog_pressure"):
        assert retired not in node.metadata
    assert node.metadata["source_kind"] == "transport_bus"
    assert node.metadata["target_id"] == "target0"
    assert node.metadata["node_id"] == "node0"


def test_transport_adapter_accepts_dict_json_absent_and_garbage() -> None:
    proj = _transport_projection(1)
    assert map_transport_ctx_to_substrate(
        {"transport_bus_projection": proj.model_dump(mode="json")}
    ) is not None
    assert map_transport_ctx_to_substrate(
        {"transport_bus_projection": proj.model_dump_json()}
    ) is not None
    assert map_transport_ctx_to_substrate({}) is None
    assert map_transport_ctx_to_substrate({"transport_bus_projection": "not json"}) is None


def test_transport_adapter_caps_at_20() -> None:
    record = map_transport_ctx_to_substrate(
        {"transport_bus_projection": _transport_projection(40)}
    )
    assert record is not None
    assert len(record.nodes) <= 20


# --------------------------------------------------------------------------- #
# biometrics
# --------------------------------------------------------------------------- #
def _biometrics_projection(n: int = 1, include_quiet: bool = False) -> ActiveNodePressureProjectionV1:
    nodes: dict[str, ActiveNodePressureStateV1] = {}
    for i in range(n):
        key = f"n{i}"
        nodes[key] = ActiveNodePressureStateV1(
            node_id=key,
            availability_status="online",
            active_pressures=["cpu_saturation"],
            capability_impacts=["chat_latency"],
            pressure_score=0.5 + (i % 4) * 0.1,
            last_updated_at=NOW,
        )
    if include_quiet:
        nodes["quiet"] = ActiveNodePressureStateV1(
            node_id="quiet",
            availability_status="online",
            active_pressures=[],
            pressure_score=0.0,
            last_updated_at=NOW,
        )
    return ActiveNodePressureProjectionV1(projection_id="bio1", generated_at=NOW, nodes=nodes)


def test_biometrics_adapter_emits_pressure_nodes() -> None:
    proj = _biometrics_projection(1)
    record = map_biometrics_ctx_to_substrate({"active_node_pressure_projection": proj})
    assert record is not None
    assert record.anchor_scope == "orion"
    node = record.nodes[0]
    assert node.label == "biometrics:n0"
    assert node.anchor_scope == "orion"
    assert node.subject_ref == "entity:orion"
    assert node.signals.salience == 0.5
    assert node.metadata["source_kind"] == "biometrics_pressure"
    assert node.metadata["node_id"] == "n0"
    assert node.metadata["pressure_score"] == 0.5
    assert node.metadata["active_pressures"] == ["cpu_saturation"]


def test_biometrics_adapter_excludes_quiet_nodes() -> None:
    proj = _biometrics_projection(1, include_quiet=True)
    record = map_biometrics_ctx_to_substrate({"active_node_pressure_projection": proj})
    assert record is not None
    labels = {n.label for n in record.nodes}
    assert "biometrics:quiet" not in labels
    assert "biometrics:n0" in labels


def test_biometrics_adapter_accepts_dict_json_absent_and_garbage() -> None:
    proj = _biometrics_projection(1)
    assert map_biometrics_ctx_to_substrate(
        {"active_node_pressure_projection": proj.model_dump(mode="json")}
    ) is not None
    assert map_biometrics_ctx_to_substrate(
        {"active_node_pressure_projection": proj.model_dump_json()}
    ) is not None
    assert map_biometrics_ctx_to_substrate({}) is None
    assert map_biometrics_ctx_to_substrate({"active_node_pressure_projection": "not json"}) is None


def test_biometrics_adapter_caps_at_20() -> None:
    record = map_biometrics_ctx_to_substrate(
        {"active_node_pressure_projection": _biometrics_projection(30)}
    )
    assert record is not None
    assert len(record.nodes) <= 20


# --------------------------------------------------------------------------- #
# registry wiring
# --------------------------------------------------------------------------- #
def test_registry_registers_three_reducer_lanes() -> None:
    from orion.cognition.projection_builder import build_projection_unification_registry

    reg = build_projection_unification_registry()
    ids = [p.producer_id for p in reg.producers]
    # 15 -> 14 (2026-07-22, self_state_ctx burn, landed independently on main)
    # -> 13 (2026-07-22, this patch): "orionmem" producer removed, dead code
    # reading test-fixture-polluted Fuseki content -- see approve.py's docstring.
    # -> 12 (2026-09-09): "self_study" producer removed, dead SPARQL read
    # against the retired RDF store (SELF_STUDY_NAMED_GRAPH was empty everywhere).
    assert len(reg.producers) == 12
    assert "orionmem" not in ids
    assert "self_study" not in ids
    assert {"biometrics", "execution", "transport", "attention", "episodes", "curiosity"} <= set(ids)


@pytest.mark.parametrize(
    ("ping_ok", "observer_failures", "expected_confidence"),
    [(True, 0, 1.0), (None, 0, 0.5), (False, 0, 0.7), (True, 1, 0.7)],
)
def test_transport_adapter_confidence_unchanged_by_delivery_confidence_retirement(
    ping_ok, observer_failures, expected_confidence
) -> None:
    """Old: _clamp(delivery_confidence) or 0.7. New: _clamp(1 - reliability_pressure)
    or 0.7. Uses reducer-produced states so every real reliability value is covered."""
    from orion.substrate.transport_loop.extract import compute_transport_pressures

    state = TransportBusStateV1(
        target_id="bus:athena", node_id="athena", sample_window_id="w", source_trace_id="t",
        redis_ping_ok=ping_ok, observer_failure_count=observer_failures,
    )
    state = state.model_copy(update=compute_transport_pressures(state))
    proj = TransportBusProjectionV1(updated_at=NOW, buses={"bus:athena": state})
    record = map_transport_ctx_to_substrate({"transport_bus_projection": proj})
    assert record is not None
    assert record.nodes[0].signals.confidence == pytest.approx(expected_confidence)
