# Substrate Lattice Hub Tab V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a read-only tuning console Hub tab that makes the live transport substrate lane legible and simulatable, with a producer-aware data model for future biometrics and cortex-exec lanes.

**Architecture:** New FastAPI route module (`substrate_lattice_routes.py`) aggregates existing DB tables into lattice-aware endpoints; a new static HTML page (`substrate-lattice.html`) + JS file (`substrate-lattice.js`) embedded via iframe in the existing Hub tab nav; four YAML config skeletons define the grammar producer registry, gate policy, transport lattice policy, and action ceiling policy.

**Tech Stack:** Python/FastAPI, SQLAlchemy (read-only Postgres via `POSTGRES_URI` env var), Pydantic, vanilla JS, Tailwind CDN, existing `orion/schemas/*` models. No new Python dependencies.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `docs/superpowers/pr-reports/transport-substrate-live-proof-v1.md` | Create | Phase 0 live proof record |
| `config/substrate-lattice/grammar_producer_registry.v1.yaml` | Create | Declares known producer lanes |
| `config/substrate-lattice/gate_policy.v1.yaml` | Create | Gate thresholds (freshness, evidence, confidence, lineage, action ceiling) |
| `config/substrate-lattice/transport_lattice_policy.v1.yaml` | Create | Per-channel watch/summarize/propose thresholds for transport |
| `config/substrate-lattice/action_ceiling_policy.v1.yaml` | Create | Maps ceiling labels to allowed action flags |
| `services/orion-hub/scripts/substrate_lattice_routes.py` | Create | FastAPI router `/api/substrate-lattice/*` (read-only + simulate + draft patch) |
| `services/orion-hub/tests/test_substrate_lattice_routes.py` | Create | Unit tests for every route in the above module |
| `services/orion-hub/tests/test_substrate_lattice_hub_tab.py` | Create | Contract tests: nav button, section panel, iframe IDs, app.js wiring |
| `services/orion-hub/static/substrate-lattice.html` | Create | Standalone UI page (producer rail, proof chain, gate overlay, tuning panel) |
| `services/orion-hub/static/js/substrate-lattice.js` | Create | Fetches data, renders all UI components, handles simulate + draft patch |
| `services/orion-hub/scripts/api_routes.py` | Modify | Register `substrate_lattice_router` |
| `services/orion-hub/templates/index.html` | Modify | Add nav tab button + section panel with iframe |
| `services/orion-hub/static/js/app.js` | Modify | Add hash routing for `#substrate-lattice` |

---

## Task 1: Phase 0 — Pin live proof document

**Files:**
- Create: `docs/superpowers/pr-reports/transport-substrate-live-proof-v1.md`

- [ ] **Step 1: Write the live proof document**

```markdown
# Transport Substrate Live Proof V1

**Date:** 2026-06-08
**Status:** Live, dry-run / read-only mode

## Milestones Green

| Layer | Table | Status |
|-------|-------|--------|
| M3 reducer receipts | `substrate_reduction_receipts` | green |
| M3 transport projection | `substrate_transport_bus_projection` | green |
| M4 field state | `substrate_field_state` | green |
| M5 attention frames | `substrate_attention_frames` | green |
| L6 self-state | `substrate_self_state` | green |
| L7 proposal frames | `substrate_proposal_frames` | green |
| L8 policy decision frames | `substrate_policy_decision_frames` | green |
| L9 execution dispatch frames | `substrate_execution_dispatch_frames` | green |
| L10 feedback frames | `substrate_feedback_frames` | green |
| L11 consolidation frames | `substrate_consolidation_frames` | green |

## Known Constraints

- All dispatch runs in `dry_run` mode. No mutations to Redis, SQL catalog, or compose env.
- All proposals require `read_only` policy gate. No autonomous execution.
- Transport proof confirmed via `scripts/smoke_orion_bus_transport_full_stack.sh` modes: `m3`, `m4`, `m5`, `full-observe`.

## Known Issues

- Orphan policy frame queue poison: a stale policy frame can block downstream frames in the queue. Fix is required before enabling non-dry-run dispatch. Tracked separately.

## Next Step

Build Substrate Lattice Hub Tab V1 to surface this proof chain in the UI and allow threshold simulation.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/pr-reports/transport-substrate-live-proof-v1.md
git commit -m "docs: pin transport substrate live proof V1 (M3-L11 green, dry-run)"
```

---

## Task 2: Phase 1 — Config skeleton

**Files:**
- Create: `config/substrate-lattice/grammar_producer_registry.v1.yaml`
- Create: `config/substrate-lattice/gate_policy.v1.yaml`
- Create: `config/substrate-lattice/transport_lattice_policy.v1.yaml`
- Create: `config/substrate-lattice/action_ceiling_policy.v1.yaml`

- [ ] **Step 1: Create `config/substrate-lattice/grammar_producer_registry.v1.yaml`**

```yaml
schema_version: grammar_producer_registry.v1

producers:
  - producer_id: orion-bus
    lane_id: transport
    source_service: orion-bus
    trace_prefixes:
      - "bus.transport:"
    field_capability_id: "capability:transport"
    attention_target_id: "capability:transport"
    self_state_dimension_id: transport_integrity
    trusted_channels:
      - bus_health
      - transport_pressure
      - contract_pressure
      - catalog_drift_pressure
      - observer_failure_pressure
      - delivery_confidence
    action_ceiling_default: dry_run
    status: live

  - producer_id: orion-biometrics
    lane_id: biometrics
    source_service: orion-biometrics
    status: planned

  - producer_id: orion-cortex-exec
    lane_id: execution
    source_service: orion-cortex-exec
    status: planned
```

- [ ] **Step 2: Create `config/substrate-lattice/gate_policy.v1.yaml`**

```yaml
schema_version: gate_policy.v1

gates:
  freshness:
    max_age_sec: 30

  evidence:
    min_events: 1

  confidence:
    min_confidence: 0.60

  lineage:
    require_source_event: true
    require_projection: true

  action_ceiling:
    default: dry_run
    allow_mutation: false
```

- [ ] **Step 3: Create `config/substrate-lattice/transport_lattice_policy.v1.yaml`**

```yaml
schema_version: transport_lattice_policy.v1

lane_id: transport

# channel_id → dimension mapping for salience weighting
dimension_weights:
  delivery_integrity: 0.35
  contract_integrity: 0.30
  observability_integrity: 0.20
  topology_integrity: 0.15

# channel → dimension, thresholds, action ceiling
channels:
  transport_pressure:
    dimension: delivery_integrity
    watch_at: 0.25
    summarize_at: 0.50
    propose_at: 0.75
    action_ceiling: read_only

  contract_pressure:
    dimension: contract_integrity
    watch_at: 0.50
    summarize_at: 0.75
    propose_at: null
    required_windows: 2
    action_ceiling: summarize

  catalog_drift_pressure:
    dimension: topology_integrity
    watch_at: 0.50
    summarize_at: null
    propose_at: null
    required_windows: 3
    action_ceiling: watch

  observer_failure_pressure:
    dimension: observability_integrity
    watch_at: 0.25
    summarize_at: 0.50
    propose_at: null
    action_ceiling: summarize

healthy_idle:
  bus_health_min: 0.90
  transport_pressure_max: 0.10
  action: no_op_motif
```

- [ ] **Step 4: Create `config/substrate-lattice/action_ceiling_policy.v1.yaml`**

```yaml
schema_version: action_ceiling_policy.v1

ceilings:
  ignore:
    may_emit_attention: false
    may_emit_proposal: false
    may_dispatch: false

  watch:
    may_emit_attention: true
    may_emit_proposal: false
    may_dispatch: false

  summarize:
    may_emit_attention: true
    may_emit_proposal: true
    proposal_mode: read_only
    may_dispatch: false

  read_only:
    may_emit_attention: true
    may_emit_proposal: true
    proposal_mode: read_only
    may_dispatch: dry_run_only

  propose_read_only:
    may_emit_attention: true
    may_emit_proposal: true
    proposal_mode: read_only
    may_dispatch: dry_run_only

  request_operator:
    may_emit_attention: true
    may_emit_proposal: true
    proposal_mode: operator_required
    may_dispatch: dry_run_only

  dry_run:
    may_emit_attention: true
    may_emit_proposal: true
    proposal_mode: read_only
    may_dispatch: dry_run_only
```

- [ ] **Step 5: Commit**

```bash
git add config/substrate-lattice/
git commit -m "config: add substrate-lattice config skeleton (producer registry, gate policy, transport lattice, action ceiling)"
```

---

## Task 3: Phase 2 — Backend routes: lanes + transport latest

**Files:**
- Create: `services/orion-hub/scripts/substrate_lattice_routes.py`
- Create: `services/orion-hub/tests/test_substrate_lattice_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`

### Step 1: Write the failing test for `/api/substrate-lattice/lanes`

- [ ] **Step 1: Write failing test**

Create `services/orion-hub/tests/test_substrate_lattice_routes.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
HUB_ROOT = Path(__file__).resolve().parents[1]


def _ensure_hub_scripts_import_path() -> None:
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    for p in (str(REPO_ROOT), str(HUB_ROOT)):
        try:
            sys.path.remove(p)
        except ValueError:
            pass
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HUB_ROOT))


_ensure_hub_scripts_import_path()

from scripts import substrate_lattice_routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://test:test@localhost/test")
    app = FastAPI()
    app.include_router(substrate_lattice_routes.router)
    return TestClient(app)


# ── /lanes ──────────────────────────────────────────────────────


def test_lanes_returns_known_lanes(client) -> None:
    resp = client.get("/api/substrate-lattice/lanes")
    assert resp.status_code == 200
    lanes = resp.json()
    assert isinstance(lanes, list)
    lane_ids = [lane["lane_id"] for lane in lanes]
    assert "transport" in lane_ids


def test_lanes_transport_lane_is_live(client) -> None:
    resp = client.get("/api/substrate-lattice/lanes")
    lanes = {lane["lane_id"]: lane for lane in resp.json()}
    transport = lanes["transport"]
    assert transport["producer_id"] == "orion-bus"
    assert transport["source_service"] == "orion-bus"
    assert transport["status"] == "live"


def test_lanes_has_no_post_routes() -> None:
    post_routes = [
        r for r in substrate_lattice_routes.router.routes
        if "POST" in getattr(r, "methods", set())
        and getattr(r, "path", "").endswith("/lanes")
    ]
    assert post_routes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "lanes" 2>&1 | head -30`

Expected: `ModuleNotFoundError` or `ImportError` — `substrate_lattice_routes` does not exist yet.

- [ ] **Step 3: Implement `substrate_lattice_routes.py` — lanes endpoint only**

Create `services/orion-hub/scripts/substrate_lattice_routes.py`:

```python
"""Read-only Hub API for substrate lattice tuning console."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

router = APIRouter(prefix="/api/substrate-lattice", tags=["substrate-lattice"])

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "config" / "substrate-lattice"

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


def _load_yaml(filename: str) -> dict[str, Any]:
    path = _CONFIG_DIR / filename
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@router.get("/lanes")
async def lattice_lanes() -> list[dict[str, Any]]:
    return _LANES
```

- [ ] **Step 4: Run test to verify lanes pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "lanes" 2>&1 | tail -20`

Expected: `3 passed`

- [ ] **Step 5: Add failing tests for `/transport/latest`**

Append to `services/orion-hub/tests/test_substrate_lattice_routes.py`:

```python
# ── /transport/latest ───────────────────────────────────────────


def _fake_engine_projection(projection_json: dict | None):
    """Fake engine that returns a projection row for substrate_transport_bus_projection."""
    fake = MagicMock()
    conn = MagicMock()
    fake.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake.connect.return_value.__exit__ = MagicMock(return_value=False)

    def execute(stmt, params=None):
        m = MagicMock()
        if projection_json is None:
            m.mappings.return_value.first.return_value = None
        else:
            m.mappings.return_value.first.return_value = {
                "projection_json": projection_json
            }
        return m

    conn.execute.side_effect = execute
    return fake


def _sample_projection() -> dict:
    return {
        "schema_version": "transport_bus.projection.v1",
        "updated_at": "2026-06-08T01:00:00+00:00",
        "projection_id": "active_transport_bus_projection",
        "buses": {
            "bus:athena": {
                "schema_version": "transport_bus.state.v1",
                "target_id": "bus:athena",
                "node_id": "node:athena",
                "sample_window_id": "window:abc",
                "source_trace_id": "bus.transport:abc",
                "redis_ping_ok": True,
                "streams_observed": 10,
                "total_stream_depth": 100,
                "max_stream_depth": 20,
                "uncataloged_stream_count": 0,
                "backpressure_count": 0,
                "observer_failure_count": 0,
                "bus_health": 1.0,
                "delivery_confidence": 1.0,
                "stream_depth_pressure": 0.0,
                "backpressure": 0.0,
                "catalog_drift_pressure": 0.5,
                "observer_failure_pressure": 0.0,
                "transport_pressure": 0.0,
                "contract_pressure": 1.0,
                "reliability_pressure": 0.0,
                "evidence_event_ids": ["evt:abc"],
                "observed_at": "2026-06-08T01:00:00+00:00",
            }
        },
    }


def test_transport_latest_returns_proof_chain(client) -> None:
    proj = _sample_projection()
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value={
            "projection": proj,
            "field_vector": {"pressure": 0.5},
            "attention": {"dominant_targets": ["capability:transport"]},
            "self_state": {"transport_integrity": {"score": 0.8, "confidence": 0.9}},
            "proposals": {"count": 2, "candidates": []},
            "policy": {"approved_count": 2},
            "dispatch": {"dispatch_mode": "dry_run", "dispatch_count": 0},
            "feedback": {"outcome_status": "dry_run_only"},
            "motifs": [],
        }
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 200
    body = resp.json()
    assert "projection" in body
    assert "attention" in body
    assert "self_state" in body
    assert "proposals" in body
    assert "dispatch" in body
    assert "feedback" in body
    assert "motifs" in body


def test_transport_latest_404_when_no_projection(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.get("/api/substrate-lattice/transport/latest")
    assert resp.status_code == 404
```

- [ ] **Step 6: Run `/transport/latest` test to verify it fails**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "transport_latest" 2>&1 | tail -20`

Expected: `AttributeError` — `_load_transport_proof_chain` not defined yet.

- [ ] **Step 7: Implement `_load_transport_proof_chain` and the `/transport/latest` route**

Add to `services/orion-hub/scripts/substrate_lattice_routes.py`:

```python
def _load_transport_proof_chain() -> dict[str, Any] | None:
    """Aggregate M3-L11 tables into a single proof chain dict. Returns None if no projection exists."""
    engine = _engine()

    def _first_json(table: str, col: str, order_col: str = "generated_at") -> dict | None:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    f"SELECT {col} FROM {table} ORDER BY {order_col} DESC LIMIT 1"
                )
            ).mappings().first()
        if not row:
            return None
        payload = row[col]
        return json.loads(payload) if isinstance(payload, str) else payload

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

    # Extract first bus state for top-level channel summary
    buses = proj_row.get("buses", {})
    first_bus: dict[str, Any] = next(iter(buses.values()), {}) if buses else {}

    # M3 latest reducer receipts (filter transport)
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
    field_raw = _first_json("substrate_field_state", "field_json")
    field_vector: dict[str, Any] = {}
    if field_raw:
        cap_vectors = field_raw.get("capability_vectors", {})
        field_vector = cap_vectors.get("capability:transport", {})

    # M5 attention
    attn_raw = _first_json("substrate_attention_frames", "frame_json")
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
    ss_raw = _first_json("substrate_self_state", "self_state_json")
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
    prop_raw = _first_json("substrate_proposal_frames", "proposal_frame_json")
    proposals: dict[str, Any] = {}
    if prop_raw:
        candidates = prop_raw.get("candidates", [])
        transport_candidates = [
            c for c in candidates
            if "transport" in c.get("target_id", "")
        ]
        proposals = {
            "frame_id": prop_raw.get("frame_id"),
            "generated_at": prop_raw.get("generated_at"),
            "count": len(candidates),
            "transport_count": len(transport_candidates),
            "candidates": transport_candidates,
        }

    # L8 policy decisions
    pol_raw = _first_json("substrate_policy_decision_frames", "policy_decision_frame_json")
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
    disp_raw = _first_json("substrate_execution_dispatch_frames", "dispatch_frame_json")
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
    fb_raw = _first_json("substrate_feedback_frames", "feedback_frame_json")
    feedback: dict[str, Any] = {}
    if fb_raw:
        feedback = {
            "frame_id": fb_raw.get("frame_id"),
            "generated_at": fb_raw.get("generated_at"),
            "outcome_status": fb_raw.get("outcome_status"),
            "feedback_kind": fb_raw.get("feedback_kind"),
        }

    # L11 consolidation motifs
    consol_raw = _first_json("substrate_consolidation_frames", "consolidation_frame_json")
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


@router.get("/transport/latest")
async def transport_latest() -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")
    return chain
```

- [ ] **Step 8: Run `/transport/latest` tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "latest" 2>&1 | tail -20`

Expected: `2 passed`

- [ ] **Step 9: Register the router in `api_routes.py`**

At line ~153 in `services/orion-hub/scripts/api_routes.py`, after the existing substrate router imports, add:

```python
from .substrate_lattice_routes import router as substrate_lattice_router
```

And after the existing `router.include_router(substrate_consolidation_router)` line, add:

```python
router.include_router(substrate_lattice_router)
```

- [ ] **Step 10: Verify the import compiles**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m compileall services/orion-hub/scripts/substrate_lattice_routes.py services/orion-hub/scripts/api_routes.py 2>&1`

Expected: `Compiling ... OK`

- [ ] **Step 11: Commit**

```bash
git add services/orion-hub/scripts/substrate_lattice_routes.py \
        services/orion-hub/scripts/api_routes.py \
        services/orion-hub/tests/test_substrate_lattice_routes.py
git commit -m "feat: add substrate lattice routes — lanes + transport/latest (M3-L11 proof chain)"
```

---

## Task 4: Phase 2 — Backend routes: gate overlay

**Files:**
- Modify: `services/orion-hub/scripts/substrate_lattice_routes.py`
- Modify: `services/orion-hub/tests/test_substrate_lattice_routes.py`

The gate overlay reads the current transport proof chain and evaluates each gate:

| Gate | Logic |
|------|-------|
| `freshness` | `updated_at` of projection is < `max_age_sec` (30s) old |
| `evidence` | projection has ≥ 1 `evidence_event_ids` in the first bus state |
| `lineage` | projection `source_trace_id` is non-empty |
| `pressure` | `transport_pressure` + `observer_failure_pressure` > 0.0 |
| `contract` | `contract_pressure` >= `watch_at` (0.50) |
| `action_ceiling` | derived from current `dispatch_mode` in dispatch frame |

- [ ] **Step 1: Add failing tests for `/transport/gates`**

Append to `services/orion-hub/tests/test_substrate_lattice_routes.py`:

```python
# ── /transport/gates ─────────────────────────────────────────────


def _sample_proof_chain_for_gates(
    bus_age_sec: float = 5.0,
    contract_pressure: float = 1.0,
    transport_pressure: float = 0.0,
    evidence_event_ids: list | None = None,
    dispatch_mode: str = "dry_run",
) -> dict:
    from datetime import datetime, timedelta, timezone

    observed_at = (
        datetime.now(timezone.utc) - timedelta(seconds=bus_age_sec)
    ).isoformat()
    return {
        "projection": {
            "updated_at": (
                datetime.now(timezone.utc) - timedelta(seconds=bus_age_sec)
            ).isoformat(),
        },
        "bus_summary": {
            "bus_health": 1.0,
            "transport_pressure": transport_pressure,
            "contract_pressure": contract_pressure,
            "catalog_drift_pressure": 0.0,
            "observer_failure_pressure": 0.0,
            "delivery_confidence": 1.0,
            "observed_at": observed_at,
        },
        "receipts": [{"receipt_id": "r1"}] if (evidence_event_ids or []) else [],
        "field_vector": {"pressure": 0.5},
        "attention": {
            "capability_targets": ["capability:transport"],
        },
        "self_state": {"transport_integrity": {"score": 0.8}},
        "proposals": {"count": 1, "transport_count": 1, "candidates": []},
        "policy": {"approved_count": 1},
        "dispatch": {"dispatch_mode": dispatch_mode, "dispatch_count": 0},
        "feedback": {"outcome_status": "dry_run_only"},
        "motifs": [],
    }


def test_gates_freshness_pass(client) -> None:
    chain = _sample_proof_chain_for_gates(bus_age_sec=5.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 200
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["freshness"]["state"] == "pass"


def test_gates_freshness_blocked_when_stale(client) -> None:
    chain = _sample_proof_chain_for_gates(bus_age_sec=60.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 200
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["freshness"]["state"] == "blocked"


def test_gates_contract_watch_when_high(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["contract"]["state"] == "watch"


def test_gates_pressure_quiet_when_zero(client) -> None:
    chain = _sample_proof_chain_for_gates(transport_pressure=0.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["pressure"]["state"] == "quiet"


def test_gates_action_ceiling_reflects_dispatch_mode(client) -> None:
    chain = _sample_proof_chain_for_gates(dispatch_mode="dry_run")
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    gates = {g["gate_id"]: g for g in resp.json()["gates"]}
    assert gates["action_ceiling"]["state"] == "dry_run"


def test_gates_404_when_no_chain(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.get("/api/substrate-lattice/transport/gates")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run gates tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "gates" 2>&1 | tail -20`

Expected: `AttributeError` or `404` — `/transport/gates` route not implemented yet.

- [ ] **Step 3: Implement `_compute_gates` and the `/transport/gates` route**

Add to `services/orion-hub/scripts/substrate_lattice_routes.py`:

```python
def _compute_gates(chain: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate transport gate overlay from current proof chain."""
    gate_policy = _load_yaml("gate_policy.v1.yaml")
    lattice_policy = _load_yaml("transport_lattice_policy.v1.yaml")

    bus = chain.get("bus_summary", {})
    dispatch = chain.get("dispatch", {})
    receipts = chain.get("receipts", [])
    proj = chain.get("projection", {})

    # --- freshness gate ---
    freshness_max_age = gate_policy.get("gates", {}).get("freshness", {}).get("max_age_sec", 30)
    observed_at_str = bus.get("observed_at") or proj.get("updated_at")
    freshness_state = "unknown"
    freshness_reason = "no observed_at available"
    if observed_at_str:
        try:
            from datetime import timezone
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
    evidence_min = gate_policy.get("gates", {}).get("evidence", {}).get("min_events", 1)
    evidence_state = "pass" if len(receipts) >= evidence_min else "blocked"
    evidence_reason = (
        f"{len(receipts)} reducer receipt(s) found (min {evidence_min})"
        if evidence_state == "pass"
        else f"only {len(receipts)} receipt(s) found (min {evidence_min})"
    )

    # --- lineage gate ---
    # Lineage: check that source_trace_id is present in the projection
    buses = (chain.get("projection") or {}).get("buses", {})
    first_bus = next(iter(buses.values()), {}) if buses else {}
    has_trace = bool(first_bus.get("source_trace_id"))
    lineage_state = "pass" if has_trace else "blocked"
    lineage_reason = (
        f"source_trace_id={first_bus.get('source_trace_id')!r}"
        if has_trace
        else "no source_trace_id on bus state"
    )

    # --- pressure gate ---
    transport_p = float(bus.get("transport_pressure") or 0.0)
    observer_p = float(bus.get("observer_failure_pressure") or 0.0)
    pressure_state = "quiet" if (transport_p + observer_p) == 0.0 else "watch"
    pressure_reason = (
        f"transport_pressure={transport_p:.2f} observer_failure_pressure={observer_p:.2f}"
    )

    # --- contract gate ---
    channels = lattice_policy.get("channels", {})
    contract_watch_at = (
        channels.get("contract_pressure", {}).get("watch_at", 0.50)
    )
    contract_p = float(bus.get("contract_pressure") or 0.0)
    if contract_p == 0.0:
        contract_state = "quiet"
    elif contract_p >= contract_watch_at:
        contract_state = "watch"
    else:
        contract_state = "pass"
    contract_reason = (
        f"contract_pressure={contract_p:.2f} (watch_at={contract_watch_at})"
    )

    # --- action_ceiling gate ---
    dispatch_mode = dispatch.get("dispatch_mode") or "dry_run"
    action_ceiling_state = dispatch_mode
    action_ceiling_reason = f"dispatch_mode={dispatch_mode}"

    return [
        {
            "gate_id": "freshness",
            "state": freshness_state,
            "reason": freshness_reason,
        },
        {
            "gate_id": "evidence",
            "state": evidence_state,
            "reason": evidence_reason,
        },
        {
            "gate_id": "lineage",
            "state": lineage_state,
            "reason": lineage_reason,
        },
        {
            "gate_id": "pressure",
            "state": pressure_state,
            "reason": pressure_reason,
        },
        {
            "gate_id": "contract",
            "state": contract_state,
            "reason": contract_reason,
        },
        {
            "gate_id": "action_ceiling",
            "state": action_ceiling_state,
            "reason": action_ceiling_reason,
        },
    ]


@router.get("/transport/gates")
async def transport_gates() -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")
    return {
        "lane_id": "transport",
        "gates": _compute_gates(chain),
    }
```

You also need to add `from datetime import datetime` at the top of the file (it was already added in the earlier step; verify it's present).

- [ ] **Step 4: Run gates tests to verify they pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "gates" 2>&1 | tail -20`

Expected: `6 passed`

- [ ] **Step 5: Run all lattice route tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v 2>&1 | tail -25`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/substrate_lattice_routes.py \
        services/orion-hub/tests/test_substrate_lattice_routes.py
git commit -m "feat: add substrate lattice gate overlay endpoint (/transport/gates)"
```

---

## Task 5: Phase 3 — Hub tab contract tests

**Files:**
- Create: `services/orion-hub/tests/test_substrate_lattice_hub_tab.py`

These tests assert the expected DOM IDs and app.js wiring exist *before* you add them, so they fail first.

- [ ] **Step 1: Write the failing contract tests**

Create `services/orion-hub/tests/test_substrate_lattice_hub_tab.py`:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

HUB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(REPO_ROOT), str(HUB_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

INDEX_HTML = HUB_ROOT / "templates" / "index.html"
APP_JS = HUB_ROOT / "static" / "js" / "app.js"
LATTICE_STATIC = HUB_ROOT / "static" / "substrate-lattice.html"
LATTICE_JS = HUB_ROOT / "static" / "js" / "substrate-lattice.js"


def test_index_has_substrate_lattice_tab_nav_button() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="substrateLatticeTabButton"' in html
    assert 'href="#substrate-lattice"' in html
    assert 'data-hash-target="#substrate-lattice"' in html
    assert ">Substrate Lattice<" in html


def test_index_has_substrate_lattice_section_and_frame() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert '<section id="substrate-lattice" data-panel="substrate-lattice"' in html
    assert 'id="substrateLatticeFrame"' in html
    assert 'src="/static/substrate-lattice.html?v={{HUB_UI_ASSET_VERSION}}"' in html


def test_app_js_wires_substrate_lattice_hash_and_tab() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert 'getElementById("substrateLatticeTabButton")' in js
    assert 'getElementById("substrate-lattice")' in js
    assert 'getElementById("substrateLatticeFrame")' in js
    assert 'setActiveTab("substrate-lattice")' in js
    assert (
        'history.replaceState(null, "", "#substrate-lattice")' in js
        or '#substrate-lattice' in js
    )


def test_lattice_static_page_has_root_ids() -> None:
    html = LATTICE_STATIC.read_text(encoding="utf-8")
    for needle in [
        'id="substrateLatticeRoot"',
        'id="producerLaneRail"',
        'id="transportProofChain"',
        'id="gateOverlay"',
        'id="latticeInspector"',
    ]:
        assert needle in html, f"Missing: {needle}"


def test_lattice_js_exists_and_has_fetch_calls() -> None:
    js = LATTICE_JS.read_text(encoding="utf-8")
    assert "/api/substrate-lattice/transport/latest" in js
    assert "/api/substrate-lattice/lanes" in js
    assert "/api/substrate-lattice/transport/gates" in js
```

- [ ] **Step 2: Run contract tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_hub_tab.py -v 2>&1 | tail -20`

Expected: all 5 tests fail — files don't exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add services/orion-hub/tests/test_substrate_lattice_hub_tab.py
git commit -m "test: add failing contract tests for substrate lattice hub tab"
```

---

## Task 6: Phase 3 — Add tab nav button and section panel to index.html

**Files:**
- Modify: `services/orion-hub/templates/index.html`

- [ ] **Step 1: Add nav tab button**

In `templates/index.html`, after the `pressureAnalyticsTabButton` `<a>` element (around line 92), add:

```html
          <a
            id="substrateLatticeTabButton"
            href="#substrate-lattice"
            data-hash-target="#substrate-lattice"
            class="px-3 py-1.5 text-xs font-semibold rounded-full bg-gray-800 text-gray-200 border border-gray-700 hover:bg-gray-700"
            role="button"
          >Substrate Lattice</a>
```

- [ ] **Step 2: Add section panel**

At the end of `templates/index.html`, before the closing `</div>` of `id="appPanels"` (after the pressure section, around line 2689), add:

```html
      <section id="substrate-lattice" data-panel="substrate-lattice" class="hidden w-full bg-gray-900 rounded-2xl shadow-lg p-5 flex flex-col gap-4 min-h-[56rem]">
        <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 class="text-xl font-bold text-white">Substrate Lattice</h2>
            <p class="text-xs text-gray-400 mt-0.5">Producer Lanes → Gates → Lattice → Attention → Proposals → Policy → Dispatch</p>
          </div>
        </div>
        <div class="rounded-xl border border-gray-800 bg-gray-950/40 overflow-hidden flex-1 min-h-[40rem]">
          <iframe
            id="substrateLatticeFrame"
            src="/static/substrate-lattice.html?v={{HUB_UI_ASSET_VERSION}}"
            class="w-full h-full border-0 min-h-[40rem]"
            loading="lazy"
            title="Substrate Lattice"
          ></iframe>
        </div>
      </section>
```

- [ ] **Step 3: Run the nav/section contract tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_hub_tab.py -v -k "nav_button or section" 2>&1 | tail -20`

Expected: `test_index_has_substrate_lattice_tab_nav_button` and `test_index_has_substrate_lattice_section_and_frame` pass.

---

## Task 7: Phase 3 — Wire app.js tab routing

**Files:**
- Modify: `services/orion-hub/static/js/app.js`

The existing `app.js` uses a `setActiveTab(tabKey)` function and handles hash routing. Follow the same pattern as the `pressure` tab.

- [ ] **Step 1: Find the pressure tab block in app.js**

Search `app.js` for `isPressure` or `pressureAnalyticsTabButton` to find the tab routing block. The tab routing pattern looks like:

```javascript
const isPressure = (h === "#pressure");
// ...
const pressurePanel = document.getElementById("pressure");
if (pressurePanel) pressurePanel.classList.toggle("hidden", !isPressure);
```

- [ ] **Step 2: Add substrate-lattice tab routing**

In the `setActiveTab` function block (after the pressure tab handling), add:

```javascript
  const isSubstrateLattice = (h === "#substrate-lattice");
  const substrateLatticePanelEl = document.getElementById("substrate-lattice");
  if (substrateLatticePanelEl) substrateLatticePanelEl.classList.toggle("hidden", !isSubstrateLattice);
  const substrateLatticeTabBtn = document.getElementById("substrateLatticeTabButton");
  if (substrateLatticeTabBtn) {
    substrateLatticeTabBtn.className = isSubstrateLattice
      ? "px-3 py-1.5 text-xs font-semibold rounded-full bg-indigo-600 text-white shadow border border-indigo-500"
      : "px-3 py-1.5 text-xs font-semibold rounded-full bg-gray-800 text-gray-200 border border-gray-700 hover:bg-gray-700";
  }
```

Also ensure the hash switch / click handler calls `setActiveTab("substrate-lattice")` when `#substrate-lattice` is the hash.

In the `hashchange` / `DOMContentLoaded` block, add to the existing `tabKeys` array (or equivalent):

```javascript
"substrate-lattice"
```

And in the `substrateLatticeTabButton` click handler (in the `data-hash-target` click delegation block — the existing pattern uses `data-hash-target` attribute on nav links so this should be automatic if the attribute is set).

Verify the existing click delegation handles `data-hash-target` automatically; if it does, no additional click handler is needed.

- [ ] **Step 3: Run app.js contract test**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_hub_tab.py -v -k "app_js" 2>&1 | tail -20`

Expected: `test_app_js_wires_substrate_lattice_hash_and_tab` passes.

---

## Task 8: Phase 3 — Create static HTML page and JS

**Files:**
- Create: `services/orion-hub/static/substrate-lattice.html`
- Create: `services/orion-hub/static/js/substrate-lattice.js`

- [ ] **Step 1: Create `static/substrate-lattice.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Substrate Lattice</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-950 text-gray-200 min-h-screen p-4">

<div id="substrateLatticeRoot" class="flex flex-col gap-4 max-w-full">

  <!-- Header row -->
  <div class="flex items-center justify-between">
    <h1 class="text-base font-bold text-white">Substrate Lattice</h1>
    <div class="flex items-center gap-2">
      <button id="latticeRefresh" class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-200 rounded px-3 py-1 border border-gray-700" type="button">Refresh</button>
      <span id="latticeLastUpdated" class="text-[10px] text-gray-500"></span>
    </div>
  </div>

  <!-- Error banner -->
  <div id="latticeError" class="hidden bg-red-950/40 border border-red-700 text-red-200 text-xs rounded p-3"></div>

  <!-- Three-column layout -->
  <div class="flex flex-col lg:flex-row gap-4 min-h-0">

    <!-- Left: Producer Lane Rail -->
    <div id="producerLaneRail" class="w-full lg:w-56 flex-shrink-0 flex flex-col gap-2">
      <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Producer Lanes</h2>
      <div id="producerLaneList" class="flex flex-col gap-2">
        <div class="text-xs text-gray-600 italic">Loading...</div>
      </div>
    </div>

    <!-- Center: Transport Proof Chain -->
    <div id="transportProofChain" class="flex-1 flex flex-col gap-2 min-w-0">
      <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Transport Proof Chain</h2>

      <!-- M3: Reducer / Projection -->
      <div id="m3ReducerCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-indigo-400 uppercase">M3 Reducer / Projection</span>
          <span id="m3Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="m3Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- M4: Field Vector -->
      <div id="m4FieldCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-indigo-400 uppercase">M4 Field · capability:transport</span>
          <span id="m4Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="m4Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- M5: Attention -->
      <div id="m5AttentionCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-indigo-400 uppercase">M5 Attention</span>
          <span id="m5Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="m5Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- L6: Self-State -->
      <div id="l6SelfStateCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-violet-400 uppercase">L6 Self-State · transport_integrity</span>
          <span id="l6Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="l6Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- L7: Proposals -->
      <div id="l7ProposalCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-violet-400 uppercase">L7 Proposals</span>
          <span id="l7Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="l7Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- L8: Policy -->
      <div id="l8PolicyCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-violet-400 uppercase">L8 Policy</span>
          <span id="l8Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="l8Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- L9: Dispatch -->
      <div id="l9DispatchCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-amber-400 uppercase">L9 Dispatch</span>
          <span id="l9Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="l9Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- L10: Feedback -->
      <div id="l10FeedbackCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-amber-400 uppercase">L10 Feedback</span>
          <span id="l10Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="l10Body" class="text-xs text-gray-400">—</div>
      </div>

      <!-- L11: Consolidation -->
      <div id="l11MotifCard" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-1">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-semibold text-amber-400 uppercase">L11 Motifs</span>
          <span id="l11Status" class="text-[10px] text-gray-500">—</span>
        </div>
        <div id="l11Body" class="text-xs text-gray-400">—</div>
      </div>
    </div>

    <!-- Right: Lattice Inspector -->
    <div id="latticeInspector" class="w-full lg:w-72 flex-shrink-0 flex flex-col gap-3">

      <!-- Gate Overlay -->
      <div id="gateOverlay" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-2">
        <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Gate Overlay</h2>
        <div id="gateList" class="flex flex-col gap-1 text-xs text-gray-400">Loading...</div>
      </div>

      <!-- Lattice Values -->
      <div id="latticeValues" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-2">
        <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Lattice Values</h2>
        <div id="latticeValueBody" class="flex flex-col gap-1 text-xs text-gray-400">Loading...</div>
      </div>

      <!-- Replay / Simulate -->
      <div id="replaySimulator" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-2">
        <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Simulate Thresholds</h2>
        <div class="flex flex-col gap-2">
          <label class="text-[10px] text-gray-500">contract_pressure watch_at</label>
          <input id="simContractWatchAt" type="number" min="0" max="1" step="0.05" value="0.50"
            class="bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1 w-full" />
          <label class="text-[10px] text-gray-500">transport_pressure watch_at</label>
          <input id="simTransportWatchAt" type="number" min="0" max="1" step="0.05" value="0.25"
            class="bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1 w-full" />
          <button id="simRunBtn" type="button"
            class="text-xs bg-indigo-700 hover:bg-indigo-600 text-white rounded px-3 py-1 border border-indigo-600 mt-1">
            Run Simulation
          </button>
        </div>
        <div id="simResult" class="hidden flex flex-col gap-1 mt-2 text-xs"></div>
      </div>

      <!-- Draft Policy Patch -->
      <div id="draftPatch" class="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col gap-2">
        <h2 class="text-xs font-semibold text-gray-400 uppercase tracking-wide">Draft Policy Patch</h2>
        <button id="draftPatchBtn" type="button"
          class="text-xs bg-gray-800 hover:bg-gray-700 text-gray-200 rounded px-3 py-1 border border-gray-700">
          Generate Patch from Simulation
        </button>
        <pre id="draftPatchOutput" class="hidden text-[10px] text-green-300 bg-gray-950 rounded p-2 overflow-auto max-h-48 whitespace-pre-wrap"></pre>
      </div>

    </div>
  </div>
</div>

<script src="/static/js/substrate-lattice.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `static/js/substrate-lattice.js`**

```javascript
/**
 * substrate-lattice.js
 * Substrate Lattice Hub Tab — read-only tuning console.
 *
 * Fetches from /api/substrate-lattice/* and renders:
 *   - Producer Lane Rail
 *   - Transport Proof Chain (M3 reducer, M4 field, M5 attention,
 *     L6 self-state, L7 proposals, L8 policy, L9 dispatch,
 *     L10 feedback, L11 motifs)
 *   - Gate Overlay
 *   - Lattice Values
 *   - Simulate Thresholds
 *   - Draft Policy Patch
 *
 * No mutations. No automatic actions. Simulation is in-memory only.
 */

"use strict";

const API_BASE = "";

// ── state ─────────────────────────────────────────────────────────────────────

let _lastChain = null;
let _lastGates = null;
let _lastSimThresholds = null;

// ── helpers ───────────────────────────────────────────────────────────────────

function _esc(v) {
  if (v === null || v === undefined) return "—";
  return String(v)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function _fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v.toFixed(3);
  return String(v);
}

function _ts(isoStr) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    const sec = Math.round((Date.now() - d.getTime()) / 1000);
    if (sec < 5) return "just now";
    if (sec < 60) return `${sec}s ago`;
    if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
    return `${Math.round(sec / 3600)}h ago`;
  } catch {
    return isoStr;
  }
}

function _gateColor(state) {
  if (state === "pass") return "text-emerald-400";
  if (state === "quiet") return "text-gray-400";
  if (state === "watch") return "text-amber-400";
  if (state === "blocked") return "text-red-400";
  if (state === "dry_run") return "text-indigo-300";
  return "text-gray-500";
}

function _showError(msg) {
  const el = document.getElementById("latticeError");
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("hidden");
}

function _clearError() {
  const el = document.getElementById("latticeError");
  if (el) el.classList.add("hidden");
}

// ── fetch helpers ─────────────────────────────────────────────────────────────

async function _get(path) {
  const resp = await fetch(API_BASE + path);
  if (!resp.ok) {
    const detail = await resp.text().catch(() => resp.statusText);
    throw new Error(`GET ${path} → ${resp.status}: ${detail}`);
  }
  return resp.json();
}

async function _post(path, body) {
  const resp = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const detail = await resp.text().catch(() => resp.statusText);
    throw new Error(`POST ${path} → ${resp.status}: ${detail}`);
  }
  return resp.json();
}

// ── render: producer lane rail ────────────────────────────────────────────────

function _renderProducerLanes(lanes) {
  const container = document.getElementById("producerLaneList");
  if (!container) return;
  container.innerHTML = lanes
    .map((lane) => {
      const isLive = lane.status === "live";
      const dot = isLive
        ? '<span class="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1"></span>'
        : '<span class="inline-block w-2 h-2 rounded-full bg-gray-600 mr-1"></span>';
      return `
        <div class="bg-gray-900 border border-gray-800 rounded-lg p-2 flex flex-col gap-0.5">
          <div class="text-xs font-semibold text-gray-200">${dot}${_esc(lane.lane_id)}</div>
          <div class="text-[10px] text-gray-500">${_esc(lane.producer_id)}</div>
          <div class="text-[10px] text-gray-600">${_esc(lane.status)}</div>
        </div>
      `;
    })
    .join("");
}

// ── render: proof chain ───────────────────────────────────────────────────────

function _setText(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function _renderProofChain(chain) {
  if (!chain) {
    _setText("m3Body", '<span class="text-red-400">No data</span>');
    return;
  }

  const bus = chain.bus_summary || {};

  // M3
  _setText("m3Status", _ts(bus.observed_at));
  _setText(
    "m3Body",
    `bus_health: <b>${_fmt(bus.bus_health)}</b> &nbsp;|&nbsp;
     transport_pressure: <b>${_fmt(bus.transport_pressure)}</b> &nbsp;|&nbsp;
     contract_pressure: <b>${_fmt(bus.contract_pressure)}</b><br>
     catalog_drift_pressure: <b>${_fmt(bus.catalog_drift_pressure)}</b> &nbsp;|&nbsp;
     observer_failure_pressure: <b>${_fmt(bus.observer_failure_pressure)}</b><br>
     delivery_confidence: <b>${_fmt(bus.delivery_confidence)}</b>`
  );

  // M4
  const fv = chain.field_vector || {};
  const fvKeys = Object.keys(fv);
  _setText("m4Status", fvKeys.length ? "present" : "—");
  _setText(
    "m4Body",
    fvKeys.length
      ? fvKeys.map((k) => `${_esc(k)}: <b>${_fmt(fv[k])}</b>`).join(" &nbsp;|&nbsp; ")
      : "capability:transport vector not present in field state"
  );

  // M5
  const attn = chain.attention || {};
  const capTargets = attn.capability_targets || [];
  const dominated = capTargets.includes("capability:transport");
  _setText("m5Status", _ts(attn.generated_at));
  _setText(
    "m5Body",
    `dominant_targets: ${_esc((attn.dominant_targets || []).join(", ") || "—")}<br>
     capability:transport in bucket: <b class="${dominated ? "text-emerald-400" : "text-gray-400"}">${dominated ? "yes" : "no"}</b>`
  );

  // L6
  const ss = chain.self_state || {};
  const ti = ss.transport_integrity || {};
  _setText("l6Status", _ts(ss.generated_at));
  _setText(
    "l6Body",
    `condition: <b>${_esc(ss.overall_condition)}</b> &nbsp;|&nbsp;
     transport_integrity score: <b>${_fmt(ti.score)}</b> confidence: <b>${_fmt(ti.confidence)}</b>`
  );

  // L7
  const props = chain.proposals || {};
  _setText("l7Status", _ts(props.generated_at));
  _setText(
    "l7Body",
    `total candidates: <b>${_fmt(props.count)}</b> &nbsp;|&nbsp; transport candidates: <b>${_fmt(props.transport_count)}</b>`
  );

  // L8
  const pol = chain.policy || {};
  _setText("l8Status", _ts(pol.generated_at));
  _setText(
    "l8Body",
    `approved: <b>${_fmt(pol.approved_count)}</b> rejected: <b>${_fmt(pol.rejected_count)}</b> mode: <b>${_esc(pol.policy_mode)}</b>`
  );

  // L9
  const disp = chain.dispatch || {};
  _setText("l9Status", _ts(disp.generated_at));
  _setText(
    "l9Body",
    `dispatch_mode: <b>${_esc(disp.dispatch_mode)}</b> dispatched: <b>${_fmt(disp.dispatch_count)}</b> blocked: <b>${_fmt(disp.blocked_count)}</b>`
  );

  // L10
  const fb = chain.feedback || {};
  _setText("l10Status", _ts(fb.generated_at));
  _setText(
    "l10Body",
    `outcome_status: <b>${_esc(fb.outcome_status)}</b> feedback_kind: <b>${_esc(fb.feedback_kind)}</b>`
  );

  // L11
  const motifs = chain.motifs || [];
  _setText("l11Status", `${motifs.length} motif(s)`);
  _setText(
    "l11Body",
    motifs.length
      ? motifs
          .map(
            (m) =>
              `<span class="bg-gray-800 rounded px-1">${_esc(m.label || m.motif_id)}</span> ×${m.recurrence_count || "?"}`
          )
          .join(" ")
      : "—"
  );
}

// ── render: lattice values ────────────────────────────────────────────────────

function _renderLatticeValues(chain) {
  const el = document.getElementById("latticeValueBody");
  if (!el) return;
  if (!chain) {
    el.innerHTML = '<span class="text-red-400">No data</span>';
    return;
  }
  const bus = chain.bus_summary || {};
  const dispatch = chain.dispatch || {};
  const props = chain.proposals || {};
  const pol = chain.policy || {};
  const fb = chain.feedback || {};
  const rows = [
    ["bus_health", bus.bus_health],
    ["transport_pressure", bus.transport_pressure],
    ["contract_pressure", bus.contract_pressure],
    ["catalog_drift_pressure", bus.catalog_drift_pressure],
    ["observer_failure_pressure", bus.observer_failure_pressure],
    ["delivery_confidence", bus.delivery_confidence],
    ["dispatch_mode", dispatch.dispatch_mode],
    ["proposal_count", props.count],
    ["transport_proposals", props.transport_count],
    ["approved_count", pol.approved_count],
    ["feedback_outcome", fb.outcome_status],
  ];
  el.innerHTML = rows
    .map(
      ([k, v]) =>
        `<div class="flex justify-between gap-2"><span class="text-gray-500">${k}</span><span class="font-mono">${_fmt(v)}</span></div>`
    )
    .join("");
}

// ── render: gate overlay ──────────────────────────────────────────────────────

function _renderGates(gateData) {
  const el = document.getElementById("gateList");
  if (!el) return;
  if (!gateData || !gateData.gates) {
    el.innerHTML = '<span class="text-red-400">No gate data</span>';
    return;
  }
  el.innerHTML = gateData.gates
    .map(
      (g) =>
        `<div class="flex justify-between gap-2">
           <span class="text-gray-500">${_esc(g.gate_id)}</span>
           <span class="${_gateColor(g.state)}" title="${_esc(g.reason)}">${_esc(g.state)}</span>
         </div>`
    )
    .join("");
}

// ── simulate ──────────────────────────────────────────────────────────────────

async function _runSimulate() {
  const contractWatchAt = parseFloat(document.getElementById("simContractWatchAt")?.value || "0.50");
  const transportWatchAt = parseFloat(document.getElementById("simTransportWatchAt")?.value || "0.25");
  _lastSimThresholds = {
    contract_pressure_watch_at: contractWatchAt,
    transport_pressure_watch_at: transportWatchAt,
  };

  try {
    const result = await _post("/api/substrate-lattice/transport/simulate", {
      lane_id: "transport",
      thresholds: _lastSimThresholds,
    });
    const el = document.getElementById("simResult");
    if (!el) return;
    el.classList.remove("hidden");
    const changed = result.changed;
    el.innerHTML = `
      <div class="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span class="text-gray-500">bucket</span>
        <span class="${changed ? "text-amber-300" : "text-gray-300"}">
          ${_esc(result.current?.bucket)} → ${_esc(result.simulated?.bucket)}
        </span>
        <span class="text-gray-500">salience</span>
        <span class="${changed ? "text-amber-300" : "text-gray-300"}">
          ${_fmt(result.current?.salience)} → ${_fmt(result.simulated?.salience)}
        </span>
        <span class="text-gray-500">action ceiling</span>
        <span class="${changed ? "text-amber-300" : "text-gray-300"}">
          ${_esc(result.current?.action_ceiling)} → ${_esc(result.simulated?.action_ceiling)}
        </span>
      </div>
      <div class="text-[10px] mt-1 ${changed ? "text-amber-400" : "text-gray-500"}">
        ${changed ? "⚠ outcome would change" : "✓ no change"}
      </div>
    `;
  } catch (err) {
    const el = document.getElementById("simResult");
    if (el) {
      el.classList.remove("hidden");
      el.innerHTML = `<span class="text-red-400">${_esc(err.message)}</span>`;
    }
  }
}

// ── draft policy patch ────────────────────────────────────────────────────────

async function _runDraftPatch() {
  if (!_lastSimThresholds) {
    const el = document.getElementById("draftPatchOutput");
    if (el) {
      el.classList.remove("hidden");
      el.textContent = "Run simulation first to set candidate thresholds.";
    }
    return;
  }
  try {
    const result = await _post("/api/substrate-lattice/transport/draft-policy-patch", {
      lane_id: "transport",
      thresholds: _lastSimThresholds,
    });
    const el = document.getElementById("draftPatchOutput");
    if (!el) return;
    el.classList.remove("hidden");
    el.textContent = result.diff || "(no changes)";
  } catch (err) {
    const el = document.getElementById("draftPatchOutput");
    if (el) {
      el.classList.remove("hidden");
      el.textContent = `Error: ${err.message}`;
    }
  }
}

// ── main load ─────────────────────────────────────────────────────────────────

async function _loadAll() {
  _clearError();
  try {
    const [lanes, chain, gates] = await Promise.all([
      _get("/api/substrate-lattice/lanes"),
      _get("/api/substrate-lattice/transport/latest").catch(() => null),
      _get("/api/substrate-lattice/transport/gates").catch(() => null),
    ]);
    _lastChain = chain;
    _lastGates = gates;

    _renderProducerLanes(lanes || []);
    _renderProofChain(chain);
    _renderLatticeValues(chain);
    _renderGates(gates);

    const ts = document.getElementById("latticeLastUpdated");
    if (ts) ts.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    _showError(`Load error: ${err.message}`);
  }
}

// ── event wiring ──────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  _loadAll();

  document.getElementById("latticeRefresh")?.addEventListener("click", _loadAll);
  document.getElementById("simRunBtn")?.addEventListener("click", _runSimulate);
  document.getElementById("draftPatchBtn")?.addEventListener("click", _runDraftPatch);
});
```

- [ ] **Step 3: Run the static page and JS contract tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_hub_tab.py -v 2>&1 | tail -20`

Expected: all 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/templates/index.html \
        services/orion-hub/static/js/app.js \
        services/orion-hub/static/substrate-lattice.html \
        services/orion-hub/static/js/substrate-lattice.js \
        services/orion-hub/tests/test_substrate_lattice_hub_tab.py
git commit -m "feat: add Substrate Lattice hub tab — nav, panel, static page, JS"
```

---

## Task 9: Phase 4 — Simulate endpoint

**Files:**
- Modify: `services/orion-hub/scripts/substrate_lattice_routes.py`
- Modify: `services/orion-hub/tests/test_substrate_lattice_routes.py`

The simulate endpoint recomputes salience/bucket/action ceiling with candidate thresholds. No DB writes.

**Salience formula (per channel):**
```
contribution = channel_value × dimension_weight  (if channel_value >= watch_at threshold)
contribution = 0                                  (if channel_value < watch_at)
```

**Attention bucket:**
- Any channel contributes → `capability_targets`
- No channel contributes → `suppressed_targets`

**Action ceiling:** Lowest ceiling (by rank) of all promoted channels.
Ceiling rank (most permissive → most restrictive): `no_op` < `watch` < `summarize` < `read_only` < `dry_run`.

- [ ] **Step 1: Add failing test for `/transport/simulate`**

Append to `services/orion-hub/tests/test_substrate_lattice_routes.py`:

```python
# ── /transport/simulate ──────────────────────────────────────────


def test_simulate_returns_comparison_when_thresholds_change(client) -> None:
    chain = _sample_proof_chain_for_gates(contract_pressure=1.0, transport_pressure=0.0)
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=chain
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={
                "lane_id": "transport",
                "thresholds": {
                    "contract_pressure_watch_at": 0.99,  # raise threshold above 1.0 → should suppress
                },
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "current" in body
    assert "simulated" in body
    assert "changed" in body
    assert body["simulated"]["bucket"] in ("capability_targets", "suppressed_targets")


def test_simulate_404_when_no_chain(client) -> None:
    with patch.object(
        substrate_lattice_routes, "_load_transport_proof_chain", return_value=None
    ):
        resp = client.post(
            "/api/substrate-lattice/transport/simulate",
            json={"lane_id": "transport", "thresholds": {}},
        )
    assert resp.status_code == 404


def test_simulate_no_db_writes(client) -> None:
    """Simulate must not produce any POST/write route on the lanes path."""
    write_routes = [
        r for r in substrate_lattice_routes.router.routes
        if "POST" in getattr(r, "methods", set())
        and "/lanes" in getattr(r, "path", "")
    ]
    assert write_routes == []
```

- [ ] **Step 2: Run simulate tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "simulate" 2>&1 | tail -20`

Expected: `404` or routing failure — `/transport/simulate` not yet defined.

- [ ] **Step 3: Implement simulate logic**

Add to `services/orion-hub/scripts/substrate_lattice_routes.py`:

```python
# Ceiling rank: higher index = more restrictive
_CEILING_RANK = ["ignore", "no_op_motif", "watch", "summarize", "read_only", "dry_run", "request_operator"]

# Channel definitions (mirrors transport_lattice_policy.v1.yaml)
_TRANSPORT_CHANNELS = {
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


def _compute_salience(
    bus_summary: dict[str, Any],
    threshold_overrides: dict[str, float],
) -> dict[str, Any]:
    """Compute attention bucket, salience, and action ceiling from bus values and thresholds."""
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
        # Most restrictive ceiling wins
        action_ceiling = max(
            promoted_ceilings, key=lambda c: _CEILING_RANK.index(c) if c in _CEILING_RANK else 0
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


class SimulateRequest(BaseModel):
    lane_id: str
    thresholds: dict[str, float] = Field(default_factory=dict)


@router.post("/transport/simulate")
async def transport_simulate(req: SimulateRequest) -> dict[str, Any]:
    chain = _load_transport_proof_chain()
    if chain is None:
        raise HTTPException(status_code=404, detail="transport_projection_not_found")

    bus = chain.get("bus_summary", {})

    # Load current policy from YAML for default thresholds
    lattice_policy = _load_yaml("transport_lattice_policy.v1.yaml")
    policy_channels = lattice_policy.get("channels", {})

    # Build current thresholds from YAML
    current_thresholds: dict[str, float] = {}
    for ch_id, ch_def in _TRANSPORT_CHANNELS.items():
        yaml_watch = (policy_channels.get(ch_id) or {}).get("watch_at", ch_def["watch_at"])
        current_thresholds[f"{ch_id}_watch_at"] = float(yaml_watch)

    # Merge candidate overrides onto current thresholds for simulation
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
```

You also need `from pydantic import BaseModel, Field` at the top of the file — add it to the existing imports.

- [ ] **Step 4: Run simulate tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "simulate" 2>&1 | tail -20`

Expected: `3 passed`

- [ ] **Step 5: Run all lattice route tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v 2>&1 | tail -25`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/substrate_lattice_routes.py \
        services/orion-hub/tests/test_substrate_lattice_routes.py
git commit -m "feat: add substrate lattice simulate endpoint (in-memory threshold comparison, no DB writes)"
```

---

## Task 10: Phase 5 — Draft policy patch endpoint

**Files:**
- Modify: `services/orion-hub/scripts/substrate_lattice_routes.py`
- Modify: `services/orion-hub/tests/test_substrate_lattice_routes.py`

The draft patch endpoint generates a YAML unified diff of what would change in `transport_lattice_policy.v1.yaml`. No file writes.

- [ ] **Step 1: Add failing test**

Append to `services/orion-hub/tests/test_substrate_lattice_routes.py`:

```python
# ── /transport/draft-policy-patch ───────────────────────────────


def test_draft_patch_returns_diff_text(client) -> None:
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={
            "lane_id": "transport",
            "thresholds": {"contract_pressure_watch_at": 0.75},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "diff" in body
    assert isinstance(body["diff"], str)


def test_draft_patch_diff_contains_changed_value(client) -> None:
    resp = client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={
            "lane_id": "transport",
            "thresholds": {"contract_pressure_watch_at": 0.75},
        },
    )
    body = resp.json()
    # The diff should mention the old watch_at value and the new one
    assert "contract_pressure" in body["diff"] or "watch_at" in body["diff"]


def test_draft_patch_does_not_write_files(client) -> None:
    """Ensure the route only returns text and does not modify the filesystem."""
    import os
    from pathlib import Path
    policy_path = (
        Path(__file__).resolve().parents[3]
        / "config" / "substrate-lattice" / "transport_lattice_policy.v1.yaml"
    )
    before_mtime = policy_path.stat().st_mtime if policy_path.exists() else None
    client.post(
        "/api/substrate-lattice/transport/draft-policy-patch",
        json={"lane_id": "transport", "thresholds": {"contract_pressure_watch_at": 0.99}},
    )
    after_mtime = policy_path.stat().st_mtime if policy_path.exists() else None
    assert before_mtime == after_mtime, "Policy YAML was modified — must not happen"
```

- [ ] **Step 2: Run draft patch tests to verify they fail**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "draft_patch" 2>&1 | tail -20`

Expected: routing failure — route not yet defined.

- [ ] **Step 3: Implement the draft-policy-patch endpoint**

Add to `services/orion-hub/scripts/substrate_lattice_routes.py`:

```python
import difflib


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

    # Build the proposed document by applying threshold overrides
    import copy
    proposed_doc = copy.deepcopy(current_doc)
    channels = proposed_doc.setdefault("channels", {})

    for key, value in req.thresholds.items():
        # key format: "<channel_id>_<field>" e.g. "contract_pressure_watch_at"
        # Split on last underscore-separated suffix: watch_at, summarize_at, propose_at
        for suffix in ("_watch_at", "_summarize_at", "_propose_at"):
            if key.endswith(suffix):
                ch_id = key[: -len(suffix)]
                field = suffix.lstrip("_")
                if ch_id in channels:
                    channels[ch_id][field] = value
                break

    proposed_text = yaml.dump(proposed_doc, default_flow_style=False, sort_keys=False)

    diff_lines = list(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
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
        "applied_thresholds": req.thresholds,
        "note": "Read-only. This diff has not been applied. Apply manually after review.",
    }
```

Also add `import difflib` and `import copy` to the top of the file.

- [ ] **Step 4: Run draft patch tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py -v -k "draft_patch" 2>&1 | tail -20`

Expected: `3 passed`

- [ ] **Step 5: Run all lattice tests**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_lattice_routes.py services/orion-hub/tests/test_substrate_lattice_hub_tab.py -v 2>&1 | tail -30`

Expected: all tests pass.

- [ ] **Step 6: Compile check**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m compileall services/orion-hub/scripts/substrate_lattice_routes.py 2>&1`

Expected: `OK`

- [ ] **Step 7: Verify existing orion-hub substrate tests still pass**

Run: `cd /mnt/scripts/Orion-Sapienform && PYTHONPATH=. ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_consolidation_debug_api.py services/orion-hub/tests/test_substrate_field_debug_api.py services/orion-hub/tests/test_substrate_attention_debug_api.py -v 2>&1 | tail -20`

Expected: all pass (no regression).

- [ ] **Step 8: Commit**

```bash
git add services/orion-hub/scripts/substrate_lattice_routes.py \
        services/orion-hub/tests/test_substrate_lattice_routes.py
git commit -m "feat: add substrate lattice draft-policy-patch endpoint (diff only, no file writes)"
```

---

## Acceptance Criteria Checklist

Before declaring V1 complete, verify each acceptance criterion:

```
[ ] 1. Hub shows transport lane as live in ProducerLaneRail.
[ ] 2. Hub shows M3 reducer receipts and transport projection (bus_health, pressures, delivery_confidence).
[ ] 3. Hub shows M4 field capability:transport vector.
[ ] 4. Hub shows M5 attention bucket and whether capability:transport is present.
[ ] 5. Hub shows L6 transport_integrity score and confidence.
[ ] 6. Hub shows L7 proposal count and transport candidate count.
[ ] 7. Hub shows L8 policy approved/rejected count.
[ ] 8. Hub shows L9 dry-run dispatch mode and dispatch count.
[ ] 9. Hub shows L10 feedback outcome status.
[  ] 10. Hub shows L11 motif labels and recurrence counts.
[ ] 11. Gate overlay shows freshness/evidence/lineage/pressure/contract/action_ceiling with reason text.
[ ] 12. Simulate: changing contract_pressure watch_at changes reported bucket/salience/action ceiling.
[ ] 13. Draft patch: shows YAML diff text with changed threshold values.
[ ] 14. No policy file is modified automatically (verified by test_draft_patch_does_not_write_files).
[ ] 15. No runtime actions are executed (all routes are GET or POST returning JSON only).
[ ] 16. Existing transport tests still pass:
        PYTHONPATH=. ./venv/bin/python -m pytest tests/test_transport_substrate_reducer.py tests/test_transport_substrate_pipeline.py -v
```

---

## Self-Review

### Spec Coverage

| Spec Requirement | Task |
|-----------------|------|
| Phase 0: live proof doc | Task 1 |
| Phase 1: config skeletons | Task 2 |
| Phase 2: read-only Hub backend routes (lanes, latest, gates) | Tasks 3-4 |
| Phase 3: Hub UI MVP (producer rail, proof chain M3-L11, gate overlay, lattice values) | Tasks 5-8 |
| Phase 4: replay/simulate (in-memory, no DB writes) | Task 9 |
| Phase 5: draft policy patch (diff text, no auto-apply) | Task 10 |
| Phase 6: planned lanes (biometrics, cortex-exec) shown as planned | Tasks 3, 8 |
| No policy file auto-apply | Verified: Task 10 test step 3 |
| No Redis/SQL/catalog/compose mutations | All routes are read-only GET + stateless POST |

### Placeholder Scan

No `TODO`, `TBD`, `implement later`, or `fill in details` present in any code block above. Every step shows complete code.

### Type Consistency

- `_load_transport_proof_chain` → returns `dict[str, Any] | None`
- `_compute_gates(chain: dict[str, Any])` → `list[dict[str, Any]]`
- `_compute_salience(bus_summary: dict, threshold_overrides: dict)` → `dict[str, Any]`
- `SimulateRequest.thresholds` → `dict[str, float]` (consistent with `_compute_salience` parameter)
- `DraftPatchRequest.thresholds` → `dict[str, float]` (consistent with key format `channel_id_watch_at`)
- `_TRANSPORT_CHANNELS` keys used in `_compute_salience` match `bus_summary` keys from `TransportBusStateV1`: `transport_pressure`, `contract_pressure`, `catalog_drift_pressure`, `observer_failure_pressure`

All consistent.
