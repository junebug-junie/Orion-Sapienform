from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

PRESSURE_CHANNELS = {
    "staleness",
    "cpu_pressure",
    "memory_pressure",
    "gpu_pressure",
    "thermal_pressure",
    "disk_pressure",
}


def apply_decay(state: FieldStateV1, *, decay_rate: float) -> None:
    for vec in state.node_vectors.values():
        for ch in PRESSURE_CHANNELS:
            if ch in vec:
                vec[ch] = vec[ch] * decay_rate
    for vec in state.capability_vectors.values():
        if "pressure" in vec:
            vec["pressure"] = vec["pressure"] * decay_rate
        if "available_capacity" in vec:
            vec["available_capacity"] = min(1.0, 1.0 - vec.get("pressure", 0.0))
