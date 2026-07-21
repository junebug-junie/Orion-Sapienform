from __future__ import annotations

from orion.schemas.biometrics_projection import NodeBiometricsProjectionV1
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.transport_projection import TransportBusProjectionV1

_THRESHOLD = 0.30


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def execution_prediction_error(
    prev: ExecutionTrajectoryProjectionV1,
    curr: ExecutionTrajectoryProjectionV1,
) -> float:
    """0-1 surprise score: how much did execution pressure hints change this batch?"""
    deltas: list[float] = []
    for trace_id, curr_run in curr.runs.items():
        prev_run = prev.runs.get(trace_id)
        if prev_run is None:
            continue
        for key in ("execution_load", "execution_friction", "failure_pressure", "reasoning_load"):
            pv = prev_run.pressure_hints.get(key, 0.0)
            cv = curr_run.pressure_hints.get(key, 0.0)
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0


def transport_prediction_error(
    prev: TransportBusProjectionV1,
    curr: TransportBusProjectionV1,
) -> float:
    """0-1 surprise score: how much did transport bus health change this batch?"""
    deltas: list[float] = []
    for bus_id, curr_bus in curr.buses.items():
        prev_bus = prev.buses.get(bus_id)
        if prev_bus is None:
            continue
        for field in ("bus_health", "delivery_confidence", "transport_pressure"):
            pv = getattr(prev_bus, field, 0.0)
            cv = getattr(curr_bus, field, 0.0)
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0


def biometrics_prediction_error(
    prev: NodeBiometricsProjectionV1,
    curr: NodeBiometricsProjectionV1,
) -> float:
    """0-1 surprise score: how much did node biometrics pressure hints change this batch?

    Unlike ``execution_prediction_error``'s fixed four-key set, biometrics
    ``pressure_hints`` keys are not enumerable in advance -- they are populated
    conditionally per node role by ``orion/substrate/biometrics_loop/
    grammar_extract.py::extract_node_state_from_events()`` (``strain`` always when a
    body_state atom carries salience, ``gpu`` only for ``local_llm_heavy`` nodes,
    ``memory_pressure``/``thermal_pressure``/``disk_pressure`` only when the
    matching pressure-signal atom is present). Confirmed live against real
    ``substrate_node_biometrics_projection`` data 2026-07-21: a GPU node (``atlas``)
    carries ``{"gpu", "strain"}`` while an orchestration node (``athena``) carries
    ``{"strain", "disk_pressure", "memory_pressure", "thermal_pressure"}`` -- no
    single fixed key list covers every node. So this diffs the union of keys
    present on either side of a given node, defaulting a missing key to 0.0 the
    same way ``execution_prediction_error`` defaults a missing fixed key.
    """
    deltas: list[float] = []
    for node_id, curr_node in curr.nodes.items():
        prev_node = prev.nodes.get(node_id)
        if prev_node is None:
            continue
        keys = set(prev_node.pressure_hints) | set(curr_node.pressure_hints)
        for key in keys:
            # pressure_hints is typed dict[str, Any] (unlike execution's pydantic-
            # enforced dict[str, float]), since node role gates which keys ever get
            # set -- coerce defensively rather than let a malformed/non-numeric
            # value raise out of a poll tick.
            try:
                pv = float(prev_node.pressure_hints.get(key, 0.0) or 0.0)
                cv = float(curr_node.pressure_hints.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            deltas.append(abs(cv - pv))
    return min(1.0, _mean(deltas) / _THRESHOLD) if deltas else 0.0
