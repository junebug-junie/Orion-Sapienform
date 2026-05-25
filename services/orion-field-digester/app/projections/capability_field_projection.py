from __future__ import annotations

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1


def _capability_key(raw: str) -> str:
    cid = raw.strip().lower()
    return cid if cid.startswith("capability:") else f"capability:{cid}"


def _node_pressure_for_edge(state: FieldStateV1, edge: FieldEdgeV1) -> float:
    node_vec = state.node_vectors.get(edge.source_id, {})
    if not edge.channel_map:
        return 0.0
    return max(float(node_vec.get(src_ch, 0.0)) for src_ch in edge.channel_map)


def build_capability_field_projection(state: FieldStateV1, capability_id: str) -> dict:
    cid = _capability_key(capability_id)
    connected = []
    for edge in state.edges:
        if edge.target_id != cid:
            continue
        connected.append(
            {
                "node_id": edge.source_id.replace("node:", ""),
                "pressure": _node_pressure_for_edge(state, edge),
                "edge_weight": edge.weight,
            }
        )
    return {
        "capability_id": cid.replace("capability:", ""),
        "field_vector": dict(state.capability_vectors.get(cid, {})),
        "connected_nodes": connected,
        "recent_perturbations": list(state.recent_perturbations),
    }
