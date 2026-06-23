from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

# Each rule is (channel_a, channel_b, description).
# Suspicion fires when channel_a is HIGH and channel_b is LOW, suggesting
# the two reducers disagree about the node's actual state.
_RULES: tuple[tuple[str, str], ...] = (
    ("execution_load", "cpu_pressure"),
    ("execution_load", "gpu_pressure"),
    ("failure_pressure", "availability"),
    ("transport_pressure", "bus_health"),
    ("reasoning_load", "cpu_pressure"),
)

_HIGH = 0.6
_LOW = 0.25


def _rule_suspicion(vec: dict[str, float]) -> float:
    """Return 0-1 incoherence score for one node vector."""
    hits = 0
    applicable = 0
    for high_ch, low_ch in _RULES:
        if high_ch not in vec and low_ch not in vec:
            continue
        applicable += 1
        high_val = vec.get(high_ch, 0.0)
        low_val = vec.get(low_ch, 1.0)
        if high_val >= _HIGH and low_val <= _LOW:
            hits += 1
    if applicable == 0:
        return 0.0
    return hits / applicable


def check_field_coherence(state: FieldStateV1) -> dict[str, float]:
    """Return per-node incoherence scores (0-1). Empty if no suspicion found."""
    scores: dict[str, float] = {}
    for node_id, vec in state.node_vectors.items():
        s = _rule_suspicion(vec)
        if s > 0.0:
            scores[node_id] = round(s, 4)
    return scores
