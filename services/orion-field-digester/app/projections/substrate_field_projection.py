from __future__ import annotations

from orion.schemas.field_state import FieldStateV1


def build_substrate_field_projection(state: FieldStateV1) -> dict:
    return state.model_dump(mode="json")
