"""Read-only Hub API for substrate lattice tuning console."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

router = APIRouter(prefix="/api/substrate-lattice", tags=["substrate-lattice"])

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config" / "substrate-lattice"

_LANES: list[dict[str, Any]] = [
    {
        "lane_id": "transport",
        "producer_id": "orion-bus",
        "source_service": "orion-bus",
        "trace_prefix": "bus.transport:",
        "field_capability_id": "capability:transport",
        "attention_target_id": "capability:transport",
        "self_state_dimension_id": "transport_integrity",
        "status": "live",
    },
    {
        "lane_id": "biometrics",
        "producer_id": "orion-biometrics",
        "source_service": "orion-biometrics",
        "status": "planned",
    },
    {
        "lane_id": "execution",
        "producer_id": "orion-cortex-exec",
        "source_service": "orion-cortex-exec",
        "status": "planned",
    },
]


def _engine():
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
    return create_engine(uri, pool_pre_ping=True)


# Used by gate and simulate endpoints added in later tasks.
def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _first_json(engine, table: str, col: str, order_col: str = "generated_at") -> dict | None:
    """Load the latest JSON payload from a substrate table. Table/col names are module-level constants."""
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT {col} FROM {table} ORDER BY {order_col} DESC LIMIT 1")
        ).mappings().first()
    if not row:
        return None
    payload = row[col]
    return json.loads(payload) if isinstance(payload, str) else payload


def _load_transport_proof_chain() -> dict[str, Any] | None:
    """Aggregate M3-L11 tables into a single proof chain dict. Returns None if no projection exists."""
    engine = _engine()

    # M3 transport projection
    proj_row = None
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT projection_json FROM substrate_transport_bus_projection"
                " ORDER BY updated_at DESC LIMIT 1"
            )
        ).mappings().first()
    if row:
        proj_row = row["projection_json"]
        if isinstance(proj_row, str):
            proj_row = json.loads(proj_row)

    if proj_row is None:
        return None

    buses = proj_row.get("buses", {})
    first_bus: dict[str, Any] = next(iter(buses.values()), {}) if buses else {}

    # M3 latest reducer receipts (transport only)
    with engine.connect() as conn:
        receipt_rows = conn.execute(
            text(
                """
                SELECT receipt_json FROM substrate_reduction_receipts
                WHERE reducer_name LIKE '%transport%'
                ORDER BY created_at DESC
                LIMIT 5
                """
            )
        ).mappings().all()
    receipts = []
    for r in receipt_rows:
        payload = r["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        receipts.append(payload)

    # M4 field: capability:transport vector
    field_raw = _first_json(engine, "substrate_field_state", "field_json")
    field_vector: dict[str, Any] = {}
    if field_raw:
        cap_vectors = field_raw.get("capability_vectors", {})
        field_vector = cap_vectors.get("capability:transport", {})

    # M5 attention
    attn_raw = _first_json(engine, "substrate_attention_frames", "frame_json")
    attention: dict[str, Any] = {}
    if attn_raw:
        attention = {
            "frame_id": attn_raw.get("frame_id"),
            "generated_at": attn_raw.get("generated_at"),
            "dominant_targets": attn_raw.get("dominant_targets", []),
            "capability_targets": attn_raw.get("capability_targets", []),
            "suppressed_targets": attn_raw.get("suppressed_targets", []),
        }

    # L6 self-state: transport_integrity dimension
    ss_raw = _first_json(engine, "substrate_self_state", "self_state_json")
    self_state: dict[str, Any] = {}
    if ss_raw:
        dims = ss_raw.get("dimensions", {})
        self_state = {
            "self_state_id": ss_raw.get("self_state_id"),
            "generated_at": ss_raw.get("generated_at"),
            "overall_condition": ss_raw.get("overall_condition"),
            "overall_intensity": ss_raw.get("overall_intensity"),
            "transport_integrity": dims.get("transport_integrity"),
        }

    # L7 proposals
    prop_raw = _first_json(engine, "substrate_proposal_frames", "proposal_frame_json")
    proposals: dict[str, Any] = {}
    if prop_raw:
        candidates = prop_raw.get("candidates", [])
        transport_candidates = [
            c for c in candidates if "transport" in c.get("target_id", "")
        ]
        proposals = {
            "frame_id": prop_raw.get("frame_id"),
            "generated_at": prop_raw.get("generated_at"),
            "count": len(candidates),
            "transport_count": len(transport_candidates),
            "candidates": transport_candidates,
        }

    # L8 policy decisions
    pol_raw = _first_json(engine, "substrate_policy_decision_frames", "policy_decision_frame_json")
    policy: dict[str, Any] = {}
    if pol_raw:
        policy = {
            "frame_id": pol_raw.get("frame_id"),
            "generated_at": pol_raw.get("generated_at"),
            "approved_count": pol_raw.get("approved_count", 0),
            "rejected_count": pol_raw.get("rejected_count", 0),
            "policy_mode": pol_raw.get("policy_mode"),
        }

    # L9 execution dispatch
    disp_raw = _first_json(engine, "substrate_execution_dispatch_frames", "dispatch_frame_json")
    dispatch: dict[str, Any] = {}
    if disp_raw:
        dispatch = {
            "frame_id": disp_raw.get("frame_id"),
            "generated_at": disp_raw.get("generated_at"),
            "dispatch_mode": disp_raw.get("dispatch_mode"),
            "dispatch_count": disp_raw.get("dispatch_count", 0),
            "blocked_count": disp_raw.get("blocked_count", 0),
        }

    # L10 feedback
    fb_raw = _first_json(engine, "substrate_feedback_frames", "feedback_frame_json")
    feedback: dict[str, Any] = {}
    if fb_raw:
        feedback = {
            "frame_id": fb_raw.get("frame_id"),
            "generated_at": fb_raw.get("generated_at"),
            "outcome_status": fb_raw.get("outcome_status"),
            "feedback_kind": fb_raw.get("feedback_kind"),
        }

    # L11 consolidation motifs
    consol_raw = _first_json(engine, "substrate_consolidation_frames", "consolidation_frame_json")
    motifs: list[dict[str, Any]] = []
    if consol_raw:
        obs = consol_raw.get("motif_observations", [])
        motifs = [
            {
                "motif_id": m.get("motif_id"),
                "label": m.get("label"),
                "recurrence_count": m.get("recurrence_count", 0),
            }
            for m in obs
        ]

    return {
        "projection": proj_row,
        # flattened subset of projection['buses'] for quick dashboard reads
        "bus_summary": {
            "bus_id": next(iter(buses.keys()), None) if buses else None,
            "bus_health": first_bus.get("bus_health"),
            "transport_pressure": first_bus.get("transport_pressure"),
            "contract_pressure": first_bus.get("contract_pressure"),
            "catalog_drift_pressure": first_bus.get("catalog_drift_pressure"),
            "observer_failure_pressure": first_bus.get("observer_failure_pressure"),
            "delivery_confidence": first_bus.get("delivery_confidence"),
            "observed_at": first_bus.get("observed_at"),
        },
        "receipts": receipts,
        "field_vector": field_vector,
        "attention": attention,
        "self_state": self_state,
        "proposals": proposals,
        "policy": policy,
        "dispatch": dispatch,
        "feedback": feedback,
        "motifs": motifs,
    }


@router.get("/lanes")
async def lattice_lanes() -> list[dict[str, Any]]:
    return _LANES


@router.get("/transport/latest")
async def transport_latest() -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")
    return chain
