from __future__ import annotations

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1


def _node_key(raw: str) -> str:
    nid = raw.strip().lower()
    return nid if nid.startswith("node:") else f"node:{nid}"


def build_node_field_projection(state: FieldStateV1, node_id: str) -> dict:
    nid = _node_key(node_id)
    connected = []
    for edge in state.edges:
        if edge.source_id != nid:
            continue
        cap = edge.target_id.replace("capability:", "")
        connected.append(
            {
                "capability_id": cap,
                "pressure": state.capability_vectors.get(edge.target_id, {}).get("pressure", 0.0),
                "edge_weight": edge.weight,
            }
        )
    return {
        "node_id": nid.replace("node:", ""),
        "field_vector": dict(state.node_vectors.get(nid, {})),
        "connected_capabilities": connected,
        "recent_perturbations": list(state.recent_perturbations),
    }
