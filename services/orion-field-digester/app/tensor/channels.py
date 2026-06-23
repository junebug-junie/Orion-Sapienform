NODE_CHANNELS = [
    "availability",
    "staleness",
    "cpu_pressure",
    "memory_pressure",
    "gpu_pressure",
    "thermal_pressure",
    "disk_pressure",
    "expected_offline_suppression",
    "execution_load",
    "execution_friction",
    "reasoning_load",
    "failure_pressure",
    "egress_confidence_deficit",
    "repair_pressure",
    "conversation_load",
    "transport_pressure",
    "contract_pressure",
    "catalog_drift_pressure",
    "delivery_confidence",
    "bus_health",
    "observer_failure_pressure",
    "field_coherence_warning",
    "prediction_error",
]
CAPABILITY_CHANNELS = [
    "pressure",
    "confidence",
    "available_capacity",
    "execution_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "transport_pressure",
    "contract_pressure",
]

DEFAULT_NODE_VECTOR = {ch: 0.0 for ch in NODE_CHANNELS}
DEFAULT_NODE_VECTOR["availability"] = 1.0

DEFAULT_CAPABILITY_VECTOR = {ch: 0.0 for ch in CAPABILITY_CHANNELS}
DEFAULT_CAPABILITY_VECTOR["confidence"] = 1.0
DEFAULT_CAPABILITY_VECTOR["available_capacity"] = 1.0
