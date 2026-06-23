from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

NODE_DECAY_CHANNELS = {
    # hardware / biometrics
    "staleness",
    "cpu_pressure",
    "memory_pressure",
    "gpu_pressure",
    "thermal_pressure",
    "disk_pressure",
    # execution
    "execution_load",
    "execution_friction",
    "reasoning_load",
    "failure_pressure",
    "egress_confidence_deficit",
    "repair_pressure",
    "conversation_load",
    # transport
    "transport_pressure",
    "contract_pressure",
    "catalog_drift_pressure",
    "observer_failure_pressure",
    "reliability_pressure",
    "field_coherence_warning",
    "prediction_error",
}

CAPABILITY_DECAY_CHANNELS = {
    "pressure",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "transport_pressure",
    "contract_pressure",
}


def apply_decay(state: FieldStateV1, *, decay_rate: float) -> None:
    for vec in state.node_vectors.values():
        for ch in NODE_DECAY_CHANNELS:
            if ch in vec:
                vec[ch] = vec[ch] * decay_rate
    for vec in state.capability_vectors.values():
        for ch in CAPABILITY_DECAY_CHANNELS:
            if ch in vec:
                vec[ch] = vec[ch] * decay_rate
        if "available_capacity" in vec:
            vec["available_capacity"] = min(1.0, 1.0 - vec.get("pressure", 0.0))
