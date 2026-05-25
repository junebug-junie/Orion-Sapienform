from __future__ import annotations

from orion.schemas.field_state import FieldStateV1


def apply_suppression(state: FieldStateV1) -> None:
    for vec in state.node_vectors.values():
        if vec.get("expected_offline_suppression", 0.0) >= 1.0:
            vec["availability"] = max(vec.get("availability", 1.0), 0.85)
            vec["staleness"] = 0.0
