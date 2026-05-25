"""Read-only debug API for biometrics substrate closed loop."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.substrate.biometrics_loop.constants import (
    ACTIVE_NODE_PRESSURE_PROJECTION_ID,
    NODE_BIOMETRICS_PROJECTION_ID,
)
from orion.substrate.biometrics_loop.lineage import (
    emission_touches_node,
    receipt_touches_node,
    state_deltas_for_node,
)

router = APIRouter(prefix="/api/substrate", tags=["substrate-biometrics"])

EVENT_CHAIN = [
    "biometrics grammar event",
    "node projection delta",
    "organ emission",
    "pressure candidate event",
    "pressure reduction receipt",
    "active pressure projection update",
]

_LINEAGE_SCAN_LIMIT = 50


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


def _load_projection(table: str, projection_id: str) -> dict[str, Any] | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(f"SELECT projection_json FROM {table} WHERE projection_id = :pid"),
            {"pid": projection_id},
        ).mappings().first()
    if not row:
        return None
    payload = row["projection_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return payload


def _latest_trace_for_node(node_id: str) -> str | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT trace_id FROM grammar_events
                WHERE source_service = 'orion-biometrics'
                  AND trace_id LIKE :pattern
                ORDER BY created_at DESC
                LIMIT 1
                """
            ),
            {"pattern": f"biometrics.node:{node_id}:%"},
        ).mappings().first()
    return row["trace_id"] if row else None


def _recent_emissions() -> list[OrganEmissionV1]:
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT emission_json FROM substrate_organ_emissions
                WHERE organ_id = 'biometrics_pressure'
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": _LINEAGE_SCAN_LIMIT},
        ).mappings().all()
    emissions: list[OrganEmissionV1] = []
    for row in rows:
        payload = row["emission_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        emissions.append(OrganEmissionV1.model_validate(payload))
    return emissions


def _recent_receipts() -> list[ReductionReceiptV1]:
    with _engine().connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT receipt_json FROM substrate_reduction_receipts
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": _LINEAGE_SCAN_LIMIT},
        ).mappings().all()
    receipts: list[ReductionReceiptV1] = []
    for row in rows:
        payload = row["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        receipts.append(ReductionReceiptV1.model_validate(payload))
    return receipts


def _latest_emission_for_node(node_id: str) -> dict[str, Any] | None:
    for emission in _recent_emissions():
        if emission_touches_node(emission, node_id):
            return emission.model_dump(mode="json")
    return None


def _latest_receipt_for_node(node_id: str) -> dict[str, Any] | None:
    for receipt in _recent_receipts():
        if receipt_touches_node(receipt, node_id):
            return receipt.model_dump(mode="json")
    return None


def _load_receipt(receipt_id: str) -> ReductionReceiptV1 | None:
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT receipt_json FROM substrate_reduction_receipts
                WHERE receipt_id = :receipt_id
                """
            ),
            {"receipt_id": receipt_id},
        ).mappings().first()
    if not row:
        return None
    payload = row["receipt_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return ReductionReceiptV1.model_validate(payload)


@router.get("/biometrics-node/{node_id}/latest")
async def biometrics_node_latest(node_id: str) -> dict[str, Any]:
    nid = node_id.strip().lower()
    node_bio = _load_projection(
        "substrate_node_biometrics_projection",
        NODE_BIOMETRICS_PROJECTION_ID,
    )
    pressure = _load_projection(
        "substrate_active_node_pressure_projection",
        ACTIVE_NODE_PRESSURE_PROJECTION_ID,
    )
    latest_receipt_payload = _latest_receipt_for_node(nid)
    committed_state_deltas: list[dict[str, Any]] = []
    if latest_receipt_payload:
        receipt = ReductionReceiptV1.model_validate(latest_receipt_payload)
        committed_state_deltas = state_deltas_for_node(receipt, nid)

    return {
        "node_id": nid,
        "latest_biometrics_trace": _latest_trace_for_node(nid),
        "node_biometrics_projection": (node_bio or {}).get("nodes", {}).get(nid, {}),
        "latest_organ_emission": _latest_emission_for_node(nid) or {},
        "latest_reduction_receipt": latest_receipt_payload or {},
        "committed_state_deltas": committed_state_deltas,
        "active_node_pressure_projection": (pressure or {}).get("nodes", {}).get(nid, {}),
        "event_chain": EVENT_CHAIN,
    }


@router.get("/receipts/{receipt_id}")
async def substrate_receipt_by_id(receipt_id: str) -> dict[str, Any]:
    receipt = _load_receipt(receipt_id.strip())
    if receipt is None:
        raise HTTPException(status_code=404, detail="not_found")
    return receipt.model_dump(mode="json")


@router.get("/biometrics/latest")
async def biometrics_projection_latest() -> dict[str, Any]:
    payload = _load_projection(
        "substrate_node_biometrics_projection",
        NODE_BIOMETRICS_PROJECTION_ID,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="not_found")
    return payload


@router.get("/node-pressure/latest")
async def node_pressure_latest() -> dict[str, Any]:
    payload = _load_projection(
        "substrate_active_node_pressure_projection",
        ACTIVE_NODE_PRESSURE_PROJECTION_ID,
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="not_found")
    return payload
