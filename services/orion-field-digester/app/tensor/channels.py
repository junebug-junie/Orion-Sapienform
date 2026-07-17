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
# Merge-polarity fix follow-up (2026-07-17, live post-deploy verification).
# bus_health/delivery_confidence are HIGHER_IS_BETTER_CHANNELS
# (orion/self_state/scoring.py) -- collect_field_channel_pressures() merges
# them via min() (worst node wins), same as availability above. Only
# node:athena (the transport-bus observer node) ever actually reports these;
# every other lattice node inherited the blanket 0.0 default every other
# channel in NODE_CHANNELS gets. Under the OLD max()-merge this was inert (a
# real report always beat 0.0). Under min()-merge it's actively wrong: any
# untouched node's meaningless 0.0 always wins over athena's real 1.0,
# permanently masking real bus-health data with a false "unhealthy" reading
# -- confirmed live: substrate_transport_bus_projection showed
# bus_health=1.0/delivery_confidence=1.0 for node:athena while the merged
# field_channel_corpus.v1 row read 0.0 for both, 100% of rows, for the
# entire post-deploy window. Match availability's existing precedent: a node
# that has never reported is "presumed healthy," not "worst possible."
DEFAULT_NODE_VECTOR["bus_health"] = 1.0
DEFAULT_NODE_VECTOR["delivery_confidence"] = 1.0

DEFAULT_CAPABILITY_VECTOR = {ch: 0.0 for ch in CAPABILITY_CHANNELS}
DEFAULT_CAPABILITY_VECTOR["confidence"] = 1.0
DEFAULT_CAPABILITY_VECTOR["available_capacity"] = 1.0
