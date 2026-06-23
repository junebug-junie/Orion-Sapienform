from __future__ import annotations

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
