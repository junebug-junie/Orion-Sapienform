from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
                "execution_load": 0.4,
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
            bus_health=0.5,
            delivery_confidence=0.8,
            transport_pressure=0.3 + (i % 5) * 0.1,
            backpressure=0.6,
            reliability_pressure=0.2,
            contract_pressure=0.1,
            stream_depth_pressure=0.05,
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
    # salience = max of the pressures = backpressure 0.6
    assert node.signals.salience == 0.6
    # confidence = delivery_confidence
    assert node.signals.confidence == 0.8
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
    assert len(reg.producers) == 12
    assert {"biometrics", "execution", "transport"} <= set(ids)
