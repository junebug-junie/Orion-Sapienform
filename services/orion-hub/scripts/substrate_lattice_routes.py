"""Read-only Hub API for substrate lattice tuning console."""

from __future__ import annotations

import copy
import difflib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
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


# Channel definitions for in-memory salience computation.
# IMPORTANT: These values MIRROR config/substrate-lattice/transport_lattice_policy.v1.yaml.
# If you edit dimension_weights or watch_at in the YAML, update this dict to match.
_TRANSPORT_CHANNELS: dict[str, dict] = {
    "transport_pressure": {
        "dimension": "delivery_integrity",
        "dimension_weight": 0.35,
        "watch_at": 0.25,
        "action_ceiling": "read_only",
    },
    "contract_pressure": {
        "dimension": "contract_integrity",
        "dimension_weight": 0.30,
        "watch_at": 0.50,
        "action_ceiling": "summarize",
    },
    "catalog_drift_pressure": {
        "dimension": "topology_integrity",
        "dimension_weight": 0.15,
        "watch_at": 0.50,
        "action_ceiling": "watch",
    },
    "observer_failure_pressure": {
        "dimension": "observability_integrity",
        "dimension_weight": 0.20,
        "watch_at": 0.25,
        "action_ceiling": "summarize",
    },
}

_CEILING_RANK = [
    "ignore", "no_op_motif", "watch", "summarize", "read_only", "dry_run", "request_operator"
]


def _compute_salience(
    bus_summary: dict[str, Any],
    threshold_overrides: dict[str, float],
) -> dict[str, Any]:
    """Compute attention bucket, salience, and action ceiling from bus values and thresholds.

    No DB access. Pure computation on already-loaded data.
    """
    total_salience = 0.0
    promoted_ceilings: list[str] = []

    for ch_id, ch_def in _TRANSPORT_CHANNELS.items():
        value = float(bus_summary.get(ch_id) or 0.0)
        watch_at_key = f"{ch_id}_watch_at"
        watch_at = threshold_overrides.get(watch_at_key, ch_def["watch_at"])

        if value >= watch_at:
            total_salience += value * ch_def["dimension_weight"]
            promoted_ceilings.append(ch_def["action_ceiling"])

    if promoted_ceilings:
        action_ceiling = max(
            promoted_ceilings,
            key=lambda c: _CEILING_RANK.index(c) if c in _CEILING_RANK else -1,
        )
        bucket = "capability_targets"
    else:
        action_ceiling = "ignore"
        bucket = "suppressed_targets"

    return {
        "bucket": bucket,
        "salience": round(total_salience, 4),
        "action_ceiling": action_ceiling,
    }


def _compute_gates(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate transport gate overlay from current proof chain."""
    gate_policy = _load_yaml("gate_policy.v1.yaml")
    lattice_policy = _load_yaml("transport_lattice_policy.v1.yaml")

    bus = chain.get("bus_summary", {})
    dispatch = chain.get("dispatch", {})
    receipts = chain.get("receipts", [])
    proj = chain.get("projection", {})

    # --- freshness gate ---
    freshness_max_age = (
        gate_policy.get("gates", {}).get("freshness", {}).get("max_age_sec", 30)
    )
    observed_at_str = bus.get("observed_at") or proj.get("updated_at")
    freshness_state = "unknown"
    freshness_reason = "no observed_at available"
    if observed_at_str:
        try:
            observed_dt = datetime.fromisoformat(observed_at_str)
            if observed_dt.tzinfo is None:
                observed_dt = observed_dt.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - observed_dt).total_seconds()
            if age_sec <= freshness_max_age:
                freshness_state = "pass"
                freshness_reason = f"projection {age_sec:.1f}s old (max {freshness_max_age}s)"
            else:
                freshness_state = "blocked"
                freshness_reason = f"projection {age_sec:.1f}s old — exceeds {freshness_max_age}s"
        except Exception:
            freshness_reason = "could not parse observed_at"

    # --- evidence gate ---
    evidence_min = (
        gate_policy.get("gates", {}).get("evidence", {}).get("min_events", 1)
    )
    evidence_count = len(receipts)
    evidence_state = "pass" if evidence_count >= evidence_min else "blocked"
    evidence_reason = f"{evidence_count} reducer receipt(s) found (min {evidence_min})"

    # --- lineage gate ---
    buses = (proj or {}).get("buses", {})
    first_bus = next(iter(buses.values()), {}) if buses else {}
    has_trace = bool(first_bus.get("source_trace_id"))
    lineage_state = "pass" if has_trace else "blocked"
    lineage_reason = (
        f"source_trace_id={first_bus.get('source_trace_id')!r}"
        if has_trace
        else "no source_trace_id on bus state"
    )

    # --- pressure gate ---
    channels = lattice_policy.get("channels", {})
    transport_p = float(bus.get("transport_pressure") or 0.0)
    observer_p = float(bus.get("observer_failure_pressure") or 0.0)
    transport_watch_at = float(
        (channels.get("transport_pressure") or {}).get("watch_at", 0.25)
    )
    observer_watch_at = float(
        (channels.get("observer_failure_pressure") or {}).get("watch_at", 0.25)
    )
    pressure_active = transport_p >= transport_watch_at or observer_p >= observer_watch_at
    pressure_state = "watch" if pressure_active else "quiet"
    pressure_reason = (
        f"transport_pressure={transport_p:.2f} "
        f"observer_failure_pressure={observer_p:.2f} "
        f"(thresholds: transport={transport_watch_at}, observer={observer_watch_at})"
    )

    # --- contract gate ---
    contract_watch_at = float(
        (channels.get("contract_pressure") or {}).get("watch_at", 0.50)
    )
    contract_p = float(bus.get("contract_pressure") or 0.0)
    if contract_p == 0.0:
        contract_state = "quiet"
    elif contract_p >= contract_watch_at:
        contract_state = "watch"
    else:
        contract_state = "pass"
    contract_reason = f"contract_pressure={contract_p:.2f} (watch_at={contract_watch_at})"

    # --- action_ceiling gate ---
    dispatch_mode = dispatch.get("dispatch_mode") or "dry_run"
    _KNOWN_CEILING_STATES = {"dry_run", "prepare_only", "dispatch_read_only", "ignore", "watch", "summarize"}
    action_ceiling_state = dispatch_mode if dispatch_mode in _KNOWN_CEILING_STATES else "unknown"
    action_ceiling_reason = f"dispatch_mode={dispatch_mode}"

    return [
        {"gate_id": "freshness", "state": freshness_state, "reason": freshness_reason},
        {"gate_id": "evidence", "state": evidence_state, "reason": evidence_reason},
        {"gate_id": "lineage", "state": lineage_state, "reason": lineage_reason},
        {"gate_id": "pressure", "state": pressure_state, "reason": pressure_reason},
        {"gate_id": "contract", "state": contract_state, "reason": contract_reason},
        {"gate_id": "action_ceiling", "state": action_ceiling_state, "reason": action_ceiling_reason},
    ]


@router.get("/lanes")
async def lattice_lanes() -> list[dict[str, Any]]:
    return _LANES


@router.get("/transport/latest")
async def transport_latest() -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")
    return chain


@router.get("/transport/gates")
async def transport_gates() -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")
    return {
        "lane_id": "transport",
        "gates": _compute_gates(chain),
    }


class SimulateRequest(BaseModel):
    lane_id: str
    thresholds: dict[str, float] = Field(default_factory=dict)


@router.post("/transport/simulate")
async def transport_simulate(req: SimulateRequest) -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")

    bus = chain.get("bus_summary", {})

    lattice_policy = _load_yaml("transport_lattice_policy.v1.yaml")
    policy_channels = lattice_policy.get("channels", {})
    current_thresholds: dict[str, float] = {}
    for ch_id, ch_def in _TRANSPORT_CHANNELS.items():
        yaml_watch = (policy_channels.get(ch_id) or {}).get("watch_at", ch_def["watch_at"])
        current_thresholds[f"{ch_id}_watch_at"] = float(yaml_watch)

    simulated_thresholds = {**current_thresholds, **req.thresholds}

    current_result = _compute_salience(bus, current_thresholds)
    simulated_result = _compute_salience(bus, simulated_thresholds)

    changed = (
        current_result["bucket"] != simulated_result["bucket"]
        or current_result["salience"] != simulated_result["salience"]
        or current_result["action_ceiling"] != simulated_result["action_ceiling"]
    )

    return {
        "lane_id": req.lane_id,
        "current": current_result,
        "simulated": simulated_result,
        "changed": changed,
        "applied_thresholds": simulated_thresholds,
    }


class DraftPatchRequest(BaseModel):
    lane_id: str
    thresholds: dict[str, float] = Field(default_factory=dict)


@router.post("/transport/draft-policy-patch")
async def transport_draft_policy_patch(req: DraftPatchRequest) -> dict[str, Any]:
    """
    Generate a unified YAML diff of what would change in transport_lattice_policy.v1.yaml.
    Does not write any files. Returns diff text only.
    """
    policy_path = _CONFIG_DIR / "transport_lattice_policy.v1.yaml"
    if not policy_path.exists():
        raise HTTPException(status_code=503, detail="transport_lattice_policy_not_found")

    original_text = policy_path.read_text(encoding="utf-8")
    current_doc: dict[str, Any] = yaml.safe_load(original_text) or {}

    proposed_doc = copy.deepcopy(current_doc)
    channels = proposed_doc.setdefault("channels", {})

    applied: dict[str, float] = {}
    for key, value in req.thresholds.items():
        # Key convention: "{channel_id}{_watch_at|_summarize_at|_propose_at}"
        # Assumes channel IDs don't end with these suffixes (safe for current "_pressure" naming).
        for suffix in ("_watch_at", "_summarize_at", "_propose_at"):
            if key.endswith(suffix):
                ch_id = key[: -len(suffix)]
                field = suffix.lstrip("_")
                if ch_id in channels:
                    channels[ch_id][field] = value
                    applied[key] = value
                break

    # Normalize both docs through the same serializer so the diff reflects only
    # data changes (not YAML formatting / comment differences).
    current_normalized = yaml.dump(current_doc, default_flow_style=False, sort_keys=False)
    proposed_text = yaml.dump(proposed_doc, default_flow_style=False, sort_keys=False)

    # splitlines(keepends=True) preserves the \n from yaml.dump;
    # lineterm="" tells unified_diff not to append an extra newline per line.
    diff_lines = list(
        difflib.unified_diff(
            current_normalized.splitlines(keepends=True),
            proposed_text.splitlines(keepends=True),
            fromfile="transport_lattice_policy.v1.yaml (current)",
            tofile="transport_lattice_policy.v1.yaml (proposed)",
            lineterm="",
        )
    )
    diff_text = "".join(diff_lines) if diff_lines else "(no changes)"

    return {
        "lane_id": req.lane_id,
        "diff": diff_text,
        "applied_thresholds": applied,
        "ignored_thresholds": [k for k in req.thresholds if k not in applied],
        "note": "Read-only. This diff has not been applied. Apply manually after review.",
    }
