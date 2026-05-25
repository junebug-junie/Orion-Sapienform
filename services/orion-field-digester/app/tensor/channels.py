NODE_CHANNELS = [
    "availability",
    "staleness",
    "cpu_pressure",
    "memory_pressure",
    "gpu_pressure",
    "thermal_pressure",
    "disk_pressure",
    "expected_offline_suppression",
]
CAPABILITY_CHANNELS = ["pressure", "confidence", "available_capacity"]

DEFAULT_NODE_VECTOR = {ch: 0.0 for ch in NODE_CHANNELS}
DEFAULT_NODE_VECTOR["availability"] = 1.0

DEFAULT_CAPABILITY_VECTOR = {
    "pressure": 0.0,
    "confidence": 1.0,
    "available_capacity": 1.0,
}
