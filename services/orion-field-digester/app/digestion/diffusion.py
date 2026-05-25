from __future__ import annotations

from orion.schemas.field_state import FieldStateV1


def apply_diffusion(state: FieldStateV1, *, diffusion_rate: float) -> None:
    for edge in state.edges:
        src = state.node_vectors.get(edge.source_id, {})
        tgt = state.capability_vectors.setdefault(edge.target_id, {})
        for src_ch, tgt_ch in edge.channel_map.items():
            src_val = float(src.get(src_ch, 0.0))
            tgt[tgt_ch] = min(1.0, tgt.get(tgt_ch, 0.0) + src_val * edge.weight * diffusion_rate)
        if "available_capacity" in tgt:
            tgt["available_capacity"] = max(0.0, 1.0 - tgt.get("pressure", 0.0))
        if "confidence" in tgt:
            tgt["confidence"] = max(0.0, 1.0 - 0.5 * tgt.get("pressure", 0.0))
