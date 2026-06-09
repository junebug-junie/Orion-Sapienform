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

from .service_logs import resolve_repo_root_details

router = APIRouter(prefix="/api/substrate-lattice", tags=["substrate-lattice"])


def _config_dir() -> Path:
    """Resolve config/substrate-lattice for dev, Docker (/app), and compose (/repo mount)."""
    return resolve_repo_root_details().repo_root / "config" / "substrate-lattice"

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


_engine_instance: Any = None


def _engine():
    global _engine_instance
    if _engine_instance is None:
        uri = os.getenv("POSTGRES_URI", "").strip()
        if not uri:
            raise HTTPException(status_code=503, detail="postgres_uri_not_configured")
        _engine_instance = create_engine(uri, pool_pre_ping=True)
    return _engine_instance


# Used by gate and simulate endpoints added in later tasks.
def _load_yaml(filename: str) -> dict[str, Any]:
    path = _config_dir() / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _first_json_with_ts(engine, table: str, json_col: str, ts_col: str = "generated_at") -> tuple[dict | None, str | None]:
    """Load the latest JSON payload and timestamp from a substrate table."""
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT {json_col}, {ts_col} FROM {table} ORDER BY {ts_col} DESC LIMIT 1")
        ).mappings().first()
    if not row:
        return None, None
    payload = row[json_col]
    ts = row[ts_col]
    if isinstance(payload, str):
        payload = json.loads(payload)
    if hasattr(ts, 'isoformat'):
        ts = ts.isoformat()
    elif ts is not None:
        ts = str(ts)
    return payload, ts


def _first_json(engine, table: str, col: str, order_col: str = "generated_at") -> dict | None:
    """Load the latest JSON payload from a substrate table. Table/col names are module-level constants."""
    payload, _ = _first_json_with_ts(engine, table, col, order_col)
    return payload


def _layer_meta(payload: dict | None, source_table: str, timestamp_str: str | None, freshness_sec: int) -> dict:
    """Return the standard layer wrapper with status/source_table/timestamp/age_sec/values."""
    if payload is None:
        return {
            "status": "missing",
            "source_table": source_table,
            "timestamp": None,
            "age_sec": None,
            "values": {}
        }
    age_sec = None
    status = "missing"
    if timestamp_str:
        try:
            ts = datetime.fromisoformat(timestamp_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_sec = (datetime.now(timezone.utc) - ts).total_seconds()
            status = "fresh" if age_sec <= freshness_sec else "stale"
        except Exception:
            status = "missing"
    return {
        "status": status,
        "source_table": source_table,
        "timestamp": timestamp_str,
        "age_sec": round(age_sec, 1) if age_sec is not None else None,
        "values": payload
    }


def _normalize_targets(targets_raw: list, bucket_name: str) -> list[dict]:
    """Normalize a list of attention targets — strings or dicts — to a consistent object shape."""
    result = []
    for t in targets_raw:
        if isinstance(t, str):
            result.append({
                "target_id": t,
                "bucket": bucket_name,
                "salience_score": None,
                "suggested_observation_mode": None,
                "dominant_channels": [],
                "reasons": [],
            })
        elif isinstance(t, dict):
            result.append({
                "target_id": t.get("target_id") or t.get("id") or "",
                "bucket": bucket_name,
                "salience_score": t.get("salience_score"),
                "suggested_observation_mode": t.get("suggested_observation_mode"),
                "dominant_channels": t.get("dominant_channels") or [],
                "reasons": t.get("reasons") or [],
            })
    return result


def _load_transport_proof_chain(freshness_threshold_sec: int = 60) -> dict[str, Any] | None:
    """Aggregate M3-L11 tables into a normalized proof chain dict. Returns None if no projection exists."""
    engine = _engine()

    # M3 transport projection
    proj_payload = None
    proj_ts = None
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT projection_json, updated_at FROM substrate_transport_bus_projection"
                " ORDER BY updated_at DESC LIMIT 1"
            )
        ).mappings().first()
    if row:
        proj_payload = row["projection_json"]
        if isinstance(proj_payload, str):
            proj_payload = json.loads(proj_payload)
        ts = row["updated_at"]
        proj_ts = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None

    if proj_payload is None:
        return None

    m3_values = dict(proj_payload)
    m3 = _layer_meta(m3_values, "substrate_transport_bus_projection", proj_ts, freshness_threshold_sec)

    # M3 latest reducer receipts (transport only)
    with engine.connect() as conn:
        receipt_rows = conn.execute(
            text(
                """
                SELECT receipt_json, created_at FROM substrate_reduction_receipts
                WHERE reducer_name LIKE '%transport%'
                ORDER BY created_at DESC
                LIMIT 5
                """
            )
        ).mappings().all()
    receipts = []
    latest_receipt_ts = None
    for r in receipt_rows:
        if latest_receipt_ts is None:
            ts = r["created_at"]
            latest_receipt_ts = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts) if ts else None
        payload = r["receipt_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        receipts.append(payload)

    m3_receipts_values = {"count": len(receipts), "receipts": receipts}
    m3_receipts = _layer_meta(m3_receipts_values, "substrate_reduction_receipts", latest_receipt_ts, freshness_threshold_sec)

    # M4 field: capability:transport vector
    field_raw, field_ts = _first_json_with_ts(engine, "substrate_field_state", "field_json", "generated_at")
    m4_values = None
    if field_raw:
        cap_vectors = field_raw.get("capability_vectors", {})
        field_vector = cap_vectors.get("capability:transport", {})
        m4_values = {
            "field_vector": field_vector,
            "has_transport_vector": bool(field_vector),
        }
    m4 = _layer_meta(m4_values, "substrate_field_state", field_ts, freshness_threshold_sec)

    # M5 attention
    attn_raw, attn_ts = _first_json_with_ts(engine, "substrate_attention_frames", "frame_json", "generated_at")
    m5_values = None
    if attn_raw:
        dominant_targets = _normalize_targets(attn_raw.get("dominant_targets", []), "dominant_targets")
        capability_targets = _normalize_targets(attn_raw.get("capability_targets", []), "capability_targets")
        suppressed_targets = _normalize_targets(attn_raw.get("suppressed_targets", []), "suppressed_targets")

        capability_transport_bucket = None
        for bucket_name, targets in [
            ("dominant_targets", dominant_targets),
            ("capability_targets", capability_targets),
            ("suppressed_targets", suppressed_targets),
        ]:
            for t in targets:
                if t.get("target_id") == "capability:transport":
                    capability_transport_bucket = bucket_name
                    break
            if capability_transport_bucket is not None:
                break

        m5_values = {
            "frame_id": attn_raw.get("frame_id"),
            "dominant_targets": dominant_targets,
            "capability_targets": capability_targets,
            "suppressed_targets": suppressed_targets,
            "capability_transport_bucket": capability_transport_bucket,
        }
    m5 = _layer_meta(m5_values, "substrate_attention_frames", attn_ts, freshness_threshold_sec)

    # L6 self-state: transport_integrity dimension
    ss_raw, ss_ts = _first_json_with_ts(engine, "substrate_self_state", "self_state_json", "generated_at")
    l6_values = None
    if ss_raw:
        dims = ss_raw.get("dimensions", {})
        l6_values = {
            "self_state_id": ss_raw.get("self_state_id"),
            "overall_condition": ss_raw.get("overall_condition"),
            "overall_intensity": ss_raw.get("overall_intensity"),
            "transport_integrity": dims.get("transport_integrity"),
        }
    l6 = _layer_meta(l6_values, "substrate_self_state", ss_ts, freshness_threshold_sec)

    # L7 proposals
    prop_raw, prop_ts = _first_json_with_ts(engine, "substrate_proposal_frames", "proposal_frame_json", "generated_at")
    l7_values = None
    if prop_raw:
        candidates = prop_raw.get("candidates", [])
        transport_candidates = [
            c for c in candidates if "transport" in c.get("target_id", "")
        ]
        l7_values = {
            "frame_id": prop_raw.get("frame_id"),
            "count": len(candidates),
            "transport_count": len(transport_candidates),
            "candidates": transport_candidates,
        }
    l7 = _layer_meta(l7_values, "substrate_proposal_frames", prop_ts, freshness_threshold_sec)

    # L8 policy decisions
    pol_raw, pol_ts = _first_json_with_ts(engine, "substrate_policy_decision_frames", "policy_decision_frame_json", "generated_at")
    l8_values = None
    if pol_raw:
        l8_values = {
            "frame_id": pol_raw.get("frame_id"),
            "approved_count": pol_raw.get("approved_count", 0),
            "rejected_count": pol_raw.get("rejected_count", 0),
            "policy_mode": pol_raw.get("policy_mode"),
        }
    l8 = _layer_meta(l8_values, "substrate_policy_decision_frames", pol_ts, freshness_threshold_sec)

    # L9 execution dispatch
    disp_raw, disp_ts = _first_json_with_ts(engine, "substrate_execution_dispatch_frames", "dispatch_frame_json", "generated_at")
    l9_values = None
    if disp_raw:
        l9_values = {
            "frame_id": disp_raw.get("frame_id"),
            "dispatch_mode": disp_raw.get("dispatch_mode"),
            "dispatch_count": disp_raw.get("dispatch_count", 0),
            "blocked_count": disp_raw.get("blocked_count", 0),
        }
    l9 = _layer_meta(l9_values, "substrate_execution_dispatch_frames", disp_ts, freshness_threshold_sec)

    # L10 feedback
    fb_raw, fb_ts = _first_json_with_ts(engine, "substrate_feedback_frames", "feedback_frame_json", "generated_at")
    l10_values = None
    if fb_raw:
        l10_values = {
            "frame_id": fb_raw.get("frame_id"),
            "outcome_status": fb_raw.get("outcome_status"),
            "feedback_kind": fb_raw.get("feedback_kind"),
        }
    l10 = _layer_meta(l10_values, "substrate_feedback_frames", fb_ts, freshness_threshold_sec)

    # L11 consolidation motifs
    consol_raw, consol_ts = _first_json_with_ts(engine, "substrate_consolidation_frames", "consolidation_frame_json", "generated_at")
    l11_values = None
    if consol_raw:
        obs = consol_raw.get("motif_observations", [])
        motifs = [
            {
                "motif_id": m.get("motif_id"),
                "label": m.get("label") or m.get("motif_label") or m.get("motif_id"),
                "strength": m.get("strength"),
                "recurrence_count": m.get("recurrence_count", 0),
                "timestamp": consol_ts,
                "reasons": m.get("reasons") or [],
                "evidence": m.get("evidence") or [],
            }
            for m in obs
        ]
        l11_values = {
            "frame_id": consol_raw.get("frame_id"),
            "motifs": motifs,
        }
    l11 = _layer_meta(l11_values, "substrate_consolidation_frames", consol_ts, freshness_threshold_sec)

    transport = {
        "m3": m3,
        "m3_receipts": m3_receipts,
        "m4": m4,
        "m5": m5,
        "l6": l6,
        "l7": l7,
        "l8": l8,
        "l9": l9,
        "l10": l10,
        "l11": l11,
    }

    chain: dict[str, Any] = {
        "transport": transport,
        "freshness_threshold_sec": freshness_threshold_sec,
    }
    chain["verdict"] = _compute_verdict(chain)
    return chain


def _compute_verdict(chain: dict[str, Any]) -> str:
    """Return a human-readable summary of the transport lane's freshness state."""
    transport = chain.get("transport", {})
    m3 = transport.get("m3", {})

    if not m3 or m3.get("status") == "missing":
        return "Transport lane: no projection data found."

    if m3.get("status") == "stale":
        age_sec = m3.get("age_sec") or 0
        age_h = age_sec / 3600
        return f"Transport lane stale: latest M3 projection is {age_h:.0f}h old."

    # Collect layer statuses
    layer_order = ["m3", "m3_receipts", "m4", "m5", "l6", "l7", "l8", "l9", "l10", "l11"]
    layer_labels = {
        "m3": "M3", "m3_receipts": "M3 receipts", "m4": "M4", "m5": "M5",
        "l6": "L6", "l7": "L7", "l8": "L8", "l9": "L9", "l10": "L10", "l11": "L11",
    }

    stale_layers = []
    highest_fresh_layer = None
    for key in layer_order:
        layer = transport.get(key, {})
        status = layer.get("status")
        if status == "fresh":
            highest_fresh_layer = layer_labels[key]
        elif status == "stale":
            stale_layers.append(layer_labels[key])

    if not stale_layers:
        return "Transport lane live and fresh through all layers."

    stale_str = ", ".join(stale_layers)
    if highest_fresh_layer:
        return f"Transport lane live and fresh through {highest_fresh_layer}; {stale_str} stale."
    return f"Transport lane partially stale: {stale_str} stale."


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
    "ignore", "no_op_motif", "watch", "summarize",
    "read_only", "propose_read_only", "dry_run", "request_operator"
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

    transport = chain.get("transport", {})
    m3 = transport.get("m3", {})
    m3_receipts = transport.get("m3_receipts", {})
    m5 = transport.get("m5", {})
    l9 = transport.get("l9", {})

    m3_status = m3.get("status", "missing")
    m3_values = m3.get("values", {}) if m3_status != "missing" else {}
    buses = m3_values.get("buses", {})

    # --- freshness gate ---
    freshness_max_age = (
        gate_policy.get("gates", {}).get("freshness", {}).get("max_age_sec", 30)
    )
    if m3_status == "fresh":
        age_sec = m3.get("age_sec") or 0
        freshness_state = "pass"
        freshness_reason = f"projection {age_sec:.1f}s old (max {freshness_max_age}s)"
    elif m3_status == "stale":
        age_sec = m3.get("age_sec") or 0
        freshness_state = "blocked"
        freshness_reason = f"projection {age_sec:.1f}s old — exceeds {freshness_max_age}s"
    else:
        freshness_state = "unknown"
        freshness_reason = "no projection data available"

    # --- evidence gate ---
    evidence_min = (
        gate_policy.get("gates", {}).get("evidence", {}).get("min_events", 1)
    )
    evidence_count = (m3_receipts.get("values") or {}).get("count", 0)
    evidence_state = "pass" if evidence_count >= evidence_min else "blocked"
    evidence_reason = f"{evidence_count} reducer receipt(s) found (min {evidence_min})"

    # --- lineage gate ---
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
    transport_p = float(m3_values.get("transport_pressure") or 0.0)
    observer_p = float(m3_values.get("observer_failure_pressure") or 0.0)
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
    if m3_status in ("stale", "missing"):
        contract_state = "unknown"
        contract_reason = f"contract state unknown: M3 projection is {m3_status}"
    else:
        contract_p = float(m3_values.get("contract_pressure") or 0.0)
        if contract_p == 0.0:
            contract_state = "quiet"
        elif contract_p >= contract_watch_at:
            contract_state = "watch"
        else:
            contract_state = "pass"
        contract_reason = f"contract_pressure={contract_p:.2f} (watch_at={contract_watch_at})"

    # --- attention gate ---
    m5_status = m5.get("status", "missing")
    if m5_status == "missing":
        attention_state = "unknown"
        attention_reason = "no attention frame data available"
    else:
        cap_bucket = (m5.get("values") or {}).get("capability_transport_bucket")
        if cap_bucket is not None:
            attention_state = "pass"
            attention_reason = f"capability:transport in {cap_bucket}"
        else:
            attention_state = "blocked"
            attention_reason = "capability:transport not found in any attention bucket"

    # --- action_ceiling gate ---
    l9_status = l9.get("status", "missing")
    if l9_status == "missing":
        dispatch_mode = "unknown"
    else:
        dispatch_mode = (l9.get("values") or {}).get("dispatch_mode") or "dry_run"
    _KNOWN_CEILING_STATES = {
        "dry_run", "read_only", "propose_read_only", "request_operator",
        "summarize", "watch", "ignore",
        "prepare_only", "dispatch_read_only",  # legacy runtime modes
    }
    action_ceiling_state = dispatch_mode if dispatch_mode in _KNOWN_CEILING_STATES else "unknown"
    action_ceiling_reason = f"dispatch_mode={dispatch_mode}"

    return [
        {"gate_id": "freshness", "state": freshness_state, "reason": freshness_reason},
        {"gate_id": "evidence", "state": evidence_state, "reason": evidence_reason},
        {"gate_id": "lineage", "state": lineage_state, "reason": lineage_reason},
        {"gate_id": "pressure", "state": pressure_state, "reason": pressure_reason},
        {"gate_id": "contract", "state": contract_state, "reason": contract_reason},
        {"gate_id": "attention", "state": attention_state, "reason": attention_reason},
        {"gate_id": "action_ceiling", "state": action_ceiling_state, "reason": action_ceiling_reason},
    ]


@router.get("/lanes")
async def lattice_lanes() -> list[dict[str, Any]]:
    return _LANES


@router.get("/transport/latest")
async def transport_latest() -> dict[str, Any]:
    freshness_sec = int(os.getenv("SUBSTRATE_LATTICE_FRESHNESS_THRESHOLD_SEC", "60"))
    chain = _load_transport_proof_chain(freshness_threshold_sec=freshness_sec)
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

    # Extract bus values for salience computation
    m3 = chain.get("transport", {}).get("m3", {})
    bus = m3.get("values", {}) if m3.get("status") != "missing" else {}

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
    policy_path = _config_dir() / "transport_lattice_policy.v1.yaml"
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
