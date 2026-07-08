"""Organ layer taxonomy for Organ Signals mesh filters (Milestone B0)."""
from __future__ import annotations

from typing import Dict, List

# Layer keys exposed to Hub UI (B7 dropdown).
ORGAN_LAYER_OPTIONS: List[str] = [
    "runtime",
    "cognition",
    "memory",
    "infra",
    "social",
    "vision",
    "persistence",
]

ORGAN_LAYER: Dict[str, str] = {
    # Runtime execution path
    "cortex_exec": "runtime",
    "llm_gateway": "runtime",
    "cortex_gateway": "runtime",
    "cortex_orch": "runtime",
    "hub": "runtime",
    # Cognition / stance
    "graph_cognition": "cognition",
    "chat_stance": "cognition",
    "recall": "cognition",
    "mind": "cognition",
    "spark_introspector": "cognition",
    "autonomy": "cognition",
    "concept_induction": "cognition",
    "topic_foundry": "cognition",
    # Memory / journal
    "journaler": "memory",
    "state_journaler": "memory",
    "dream": "memory",
    "collapse_mirror": "memory",
    # Infra / health
    "biometrics": "infra",
    "equilibrium": "infra",
    "power_guard": "infra",
    "security_watcher": "infra",
    # Social
    "social_memory": "social",
    "social_room_bridge": "social",
    # Vision
    "vision": "vision",
    # World / environment hybrid
    "world_pulse": "vision",
    # Persistence writers (Milestone B5)
    "sql_writer": "persistence",
    "rdf_writer": "persistence",
    "vector_writer": "persistence",
}


def organ_layer(organ_id: str) -> str:
    """Return layer for organ_id; unknown organs default to cognition."""
    return ORGAN_LAYER.get(organ_id, "cognition")


def layers_export() -> Dict[str, object]:
    """JSON-serializable layer map for Hub API."""
    return {
        "options": ["all", *ORGAN_LAYER_OPTIONS],
        "organs": dict(ORGAN_LAYER),
    }
