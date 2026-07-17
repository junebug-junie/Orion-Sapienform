from __future__ import annotations

from orion.schemas.field_state import FieldStateV1


def apply_suppression(state: FieldStateV1) -> None:
    for node_id, vec in state.node_vectors.items():
        if vec.get("expected_offline_suppression", 0.0) >= 1.0:
            vec["availability"] = max(vec.get("availability", 1.0), 0.85)
            vec["staleness"] = 0.0
            # staleness is a NODE_DECAY_CHANNELS entry (decay.py) but is
            # written here directly rather than through apply_perturbations()
            # -- record the same node_vector_updated_at stamp so a suppressed
            # node's staleness reset participates in the 2026-07-17
            # decay-hold fix instead of silently falling back to unconditional
            # per-tick decay (found by code review; currently inert only
            # because decaying 0.0 is a no-op, but a latent risk if this
            # reset value or apply_decay's math ever changes).
            state.node_vector_updated_at.setdefault(node_id, {})["staleness"] = state.generated_at
