"""Read-only debug API for substrate field digestion state."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

router = APIRouter(prefix="/api/substrate/field", tags=["substrate-field"])


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def _load_latest_field() -> FieldStateV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT field_json FROM substrate_field_state
                ORDER BY generated_at DESC
                LIMIT 1
                """
            ),
        ).mappings().first()
    if not row:
        return None
    payload = row["field_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return FieldStateV1.model_validate(payload)


def _node_key(raw: str) -> str:
    nid = raw.strip().lower()
    return nid if nid.startswith("node:") else f"node:{nid}"


def _capability_key(raw: str) -> str:
    cid = raw.strip().lower()
    return cid if cid.startswith("capability:") else f"capability:{cid}"


def _node_pressure_for_edge(state: FieldStateV1, edge: FieldEdgeV1) -> float:
    node_vec = state.node_vectors.get(edge.source_id, {})
    if not edge.channel_map:
        return 0.0
    return max(float(node_vec.get(src_ch, 0.0)) for src_ch in edge.channel_map)


def _build_node_field_projection(state: FieldStateV1, node_id: str) -> dict[str, Any]:
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


def _build_capability_field_projection(state: FieldStateV1, capability_id: str) -> dict[str, Any]:
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


@router.get("/latest")
async def field_latest() -> dict[str, Any]:
    state = _load_latest_field()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return state.model_dump(mode="json")


@router.get("/node/{node_id}")
async def field_node(node_id: str) -> dict[str, Any]:
    state = _load_latest_field()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return _build_node_field_projection(state, node_id)


@router.get("/capability/{capability_id}")
async def field_capability(capability_id: str) -> dict[str, Any]:
    state = _load_latest_field()
    if state is None:
        raise HTTPException(status_code=404, detail="not_found")
    return _build_capability_field_projection(state, capability_id)
