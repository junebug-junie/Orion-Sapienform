# Orion Field Digester v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `orion-field-digester` v1 — a biometrics-only digestion service that consumes committed `ReductionReceiptV1` / `StateDeltaV1` facts and compiles them into inspectable lattice field state with node/capability vectors, then exposes Hub debug endpoints proving atlas GPU pressure diffuses into `llm_inference` capability pressure.

**Architecture:** New worker service polls `substrate_reduction_receipts` (not raw grammar events), dedupes by stable `delta_id`, applies four digestion rules (perturbation → decay → diffusion → suppression) over a YAML-defined node→capability lattice, persists `FieldStateV1` snapshots to Postgres, and serves read-only projections via Hub. Shared Pydantic schemas live in `orion/schemas/`; digestion logic stays in the service under `services/orion-field-digester/app/`.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy, FastAPI/uvicorn, PyYAML, pytest. No mind service, no bus publish in v1 (projections-only).

**Design source:** User spec “Build digestion next — orion-field-digester” (2026-05-24).

**Depends on:** `feat/biometrics-substrate-delta-seam-hardening` (stable delta IDs + node-scoped lineage). **Base branch for worktree:** `feat/biometrics-substrate-delta-seam-hardening`, not `main`.

**Non-goals:** Vision/mind/dream digestion, field threshold events on bus, attractor poetry, autonomous scheduling, self-healing, LLM field interpretation, new organs.

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-orion-field-digester-v1 \
  -b feat/orion-field-digester-v1 \
  feat/biometrics-substrate-delta-seam-hardening
cd .worktrees/feat-orion-field-digester-v1
git check-ignore -q .worktrees   # must succeed
```

**Rules:**
- All commits only in `.worktrees/feat-orion-field-digester-v1`.
- Never bleed files to main checkout **except** copying `.env` keys from `.env_example` into operator `.env` on the local machine (Hub + field-digester).
- PR title: `PR: Orion field digester v1 (biometrics digestion)`.
- When done: run `requesting-code-review` subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-24-orion-field-digester-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Receipt/delta schemas | Present: `orion/schemas/reduction_receipt.py`, `state_delta.py` — registered in `registry.py` |
| Stable delta IDs | Present: `orion/substrate/ids.py` — digester dedupes on `delta_id` |
| Biometrics closed loop | `orion-substrate-runtime` persists receipts; Hub `/api/substrate/biometrics-node/{node_id}/latest` |
| Node catalog | `config/biometrics/node_catalog.yaml` — atlas/athena/circe/prometheus |
| Pressure delta shape | `target_kind` ∈ `{node_biometrics, active_node_pressure}`; `after` is projection node state JSON |
| Field digester service | **None** — create `services/orion-field-digester/` |
| Bus channels v1 | **No publish** — projections-only per spec; registry updated, channels unchanged |
| Port | Use `8116` (`SUBSTRATE_RUNTIME` uses `8115`) |

### Locked node pressure → field channel map (v1 biometrics)

| Source (`StateDeltaV1`) | Field node channel | Intensity source |
|-------------------------|-------------------|------------------|
| `active_node_pressure` + `strain` in `after.active_pressures` | `gpu_pressure` (atlas/circe), `cpu_pressure` (athena/prometheus) | `after.pressure_score` |
| `active_node_pressure` + `availability` pressure | `availability` ↓ | `1.0 - min(1.0, after.pressure_score + 0.2)` |
| `node_biometrics` + `pressure_hints.gpu` | `gpu_pressure` | hint value |
| `node_biometrics` + `pressure_hints.strain` | `cpu_pressure` | hint value |
| `node_biometrics` + `availability_status=stale` | `staleness` | `0.5` |
| `node_biometrics` + `expected_online=false` | `expected_offline_suppression` | `1.0` |
| `active_node_pressure` + `operation=suppress` | `expected_offline_suppression` | `1.0` |

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/field_state.py` | `FieldStateV1`, `FieldEdgeV1` |
| `orion/schemas/registry.py` | Register `FieldStateV1`, `FieldEdgeV1` |
| `config/field/biometrics_lattice.yaml` | Nodes, capabilities, edges, channel maps |
| `services/orion-sql-db/manual_migration_field_digester_v1.sql` | DDL: field state, applied deltas, cursor |
| `services/orion-field-digester/app/settings.py` | Env: decay/diffusion rates, lattice path, poll interval |
| `services/orion-field-digester/app/main.py` | FastAPI lifespan + health |
| `services/orion-field-digester/app/worker.py` | Poll receipts → digest tick |
| `services/orion-field-digester/app/store.py` | Postgres: cursor, receipts, field snapshots, applied deltas |
| `services/orion-field-digester/app/graph/lattice.py` | Load YAML lattice |
| `services/orion-field-digester/app/graph/node_registry.py` | Known nodes + default zero vectors |
| `services/orion-field-digester/app/graph/edge_registry.py` | node→capability edges + weights |
| `services/orion-field-digester/app/tensor/channels.py` | Channel name constants + defaults |
| `services/orion-field-digester/app/tensor/field_state.py` | Mutable field tensor ops on `FieldStateV1` |
| `services/orion-field-digester/app/tensor/update_rules.py` | Orchestrate perturb/decay/diffuse/suppress |
| `services/orion-field-digester/app/ingest/receipts.py` | Fetch new receipts since cursor |
| `services/orion-field-digester/app/ingest/state_deltas.py` | Map delta → perturbation requests |
| `services/orion-field-digester/app/digestion/perturbation.py` | Inject energy into node vectors |
| `services/orion-field-digester/app/digestion/decay.py` | `pressure[t+1] = pressure[t] * decay_rate` |
| `services/orion-field-digester/app/digestion/diffusion.py` | Spread node channel → capability channel |
| `services/orion-field-digester/app/digestion/suppression.py` | Expected-offline availability guard |
| `services/orion-field-digester/app/projections/node_field_projection.py` | Slice node vector + edges |
| `services/orion-field-digester/app/projections/capability_field_projection.py` | Slice capability vector |
| `services/orion-field-digester/app/projections/substrate_field_projection.py` | Full field export |
| `services/orion-field-digester/app/emit/field_events.py` | v1 stub (no bus publish) |
| `services/orion-hub/scripts/substrate_field_routes.py` | `GET /api/substrate/field/*` debug routes |
| `services/orion-hub/scripts/api_routes.py` | `include_router` for field routes |
| `tests/test_field_state_schemas.py` | Schema roundtrip |
| `tests/test_field_delta_ingest.py` | Delta → perturbation mapping |
| `tests/test_field_digestion_rules.py` | Decay, diffusion, suppression unit tests |
| `tests/test_field_deterministic_replay.py` | Replay receipts → identical field state |
| `services/orion-hub/tests/test_substrate_field_debug_api.py` | Hub route tests |
| `scripts/smoke_field_digester_biometrics.sh` | End-to-end smoke |

---

# Phase 0 — Worktree + branch

### Task 0: Create isolated worktree

- [ ] **Step 1: Create worktree from delta-seam base**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-orion-field-digester-v1 \
  -b feat/orion-field-digester-v1 \
  feat/biometrics-substrate-delta-seam-hardening
cd .worktrees/feat-orion-field-digester-v1
```

Expected: new branch checked out in worktree; `git branch --show-current` → `feat/orion-field-digester-v1`

- [ ] **Step 2: Verify isolation**

```bash
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

---

# Phase 1 — Shared schemas + lattice config

### Task 1: FieldStateV1 Pydantic models

**Files:**
- Create: `orion/schemas/field_state.py`
- Modify: `orion/schemas/registry.py`
- Test: `tests/test_field_state_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_field_state_schemas.py
from datetime import datetime, timezone

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1


def test_field_state_v1_roundtrip() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    state = FieldStateV1(
        generated_at=now,
        tick_id="tick_abc123",
        node_vectors={
            "node:atlas": {
                "availability": 1.0,
                "gpu_pressure": 0.72,
                "memory_pressure": 0.31,
            }
        },
        capability_vectors={
            "capability:llm_inference": {
                "pressure": 0.61,
                "confidence": 0.78,
                "available_capacity": 0.39,
            }
        },
        edges=[
            FieldEdgeV1(
                source_id="node:atlas",
                target_id="capability:llm_inference",
                edge_type="node_capability",
                weight=0.85,
                channel_map={"gpu_pressure": "pressure"},
            )
        ],
        recent_perturbations=["state_delta:atlas_gpu_pressure_reinforced"],
    )
    payload = state.model_dump(mode="json")
    restored = FieldStateV1.model_validate(payload)
    assert restored.schema_version == "field.state.v1"
    assert restored.node_vectors["node:atlas"]["gpu_pressure"] == 0.72
    assert restored.edges[0].weight == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd .worktrees/feat-orion-field-digester-v1
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_field_state_schemas.py::test_field_state_v1_roundtrip -v
```

Expected: FAIL with `ModuleNotFoundError: orion.schemas.field_state`

- [ ] **Step 3: Write minimal implementation**

```python
# orion/schemas/field_state.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldEdgeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    edge_type: Literal["node_capability", "node_service", "service_organ", "capability_cognitive", "node_dependency"]
    weight: float = Field(ge=0.0, le=1.0)
    channel_map: dict[str, str] = Field(default_factory=dict)


class FieldStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["field.state.v1"] = "field.state.v1"
    generated_at: datetime
    tick_id: str
    node_vectors: dict[str, dict[str, float]] = Field(default_factory=dict)
    capability_vectors: dict[str, dict[str, float]] = Field(default_factory=dict)
    edges: list[FieldEdgeV1] = Field(default_factory=list)
    recent_perturbations: list[str] = Field(default_factory=list)
```

Add to `orion/schemas/registry.py` imports and `_REGISTRY`:

```python
from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

# inside _REGISTRY:
"FieldStateV1": FieldStateV1,
"FieldEdgeV1": FieldEdgeV1,
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_field_state_schemas.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/field_state.py orion/schemas/registry.py tests/test_field_state_schemas.py
git commit -m "feat: add FieldStateV1 schema for substrate field digestion"
```

---

### Task 2: Biometrics lattice YAML

**Files:**
- Create: `config/field/biometrics_lattice.yaml`

- [ ] **Step 1: Create lattice config**

```yaml
# config/field/biometrics_lattice.yaml
schema_version: field_lattice.v1

nodes:
  - node_id: atlas
  - node_id: athena
  - node_id: circe
  - node_id: prometheus

capabilities:
  - capability_id: llm_inference
  - capability_id: orchestration
  - capability_id: storage
  - capability_id: vision
  - capability_id: graph
  - capability_id: memory

node_channels:
  - availability
  - staleness
  - cpu_pressure
  - memory_pressure
  - gpu_pressure
  - thermal_pressure
  - disk_pressure
  - expected_offline_suppression

capability_channels:
  - pressure
  - confidence
  - available_capacity

edges:
  - source_id: node:atlas
    target_id: capability:llm_inference
    edge_type: node_capability
    weight: 0.85
    channel_map:
      gpu_pressure: pressure
      memory_pressure: pressure

  - source_id: node:athena
    target_id: capability:orchestration
    edge_type: node_capability
    weight: 0.90
    channel_map:
      cpu_pressure: pressure

  - source_id: node:athena
    target_id: capability:storage
    edge_type: node_capability
    weight: 0.75
    channel_map:
      disk_pressure: pressure
      memory_pressure: pressure

  - source_id: node:athena
    target_id: capability:graph
    edge_type: node_capability
    weight: 0.70
    channel_map:
      cpu_pressure: pressure

  - source_id: node:prometheus
    target_id: capability:memory
    edge_type: node_capability
    weight: 0.60
    channel_map:
      cpu_pressure: pressure

  - source_id: node:circe
    target_id: capability:llm_inference
    edge_type: node_capability
    weight: 0.50
    channel_map:
      gpu_pressure: pressure
```

- [ ] **Step 2: Commit**

```bash
git add config/field/biometrics_lattice.yaml
git commit -m "feat: add biometrics field lattice graph config"
```

---

# Phase 2 — Database DDL

### Task 3: Field digester Postgres tables

**Files:**
- Create: `services/orion-sql-db/manual_migration_field_digester_v1.sql`

- [ ] **Step 1: Write migration**

```sql
-- Field digester v1 (apply before orion-field-digester)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_field_digester_v1.sql

create table if not exists substrate_field_digest_cursor (
    cursor_name text primary key,
    last_receipt_created_at timestamptz,
    last_receipt_id text,
    updated_at timestamptz not null default now()
);

create table if not exists substrate_field_applied_deltas (
    delta_id text primary key,
    receipt_id text not null,
    applied_at timestamptz not null default now()
);

create index if not exists idx_substrate_field_applied_deltas_receipt
  on substrate_field_applied_deltas (receipt_id);

create table if not exists substrate_field_state (
    tick_id text primary key,
    generated_at timestamptz not null,
    field_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_field_state_generated
  on substrate_field_state (generated_at desc);
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-sql-db/manual_migration_field_digester_v1.sql
git commit -m "feat: add field digester v1 postgres tables"
```

---

# Phase 3 — Digestion core (TDD)

### Task 4: Channel constants + zero field bootstrap

**Files:**
- Create: `services/orion-field-digester/app/tensor/channels.py`
- Create: `services/orion-field-digester/app/graph/lattice.py`
- Create: `services/orion-field-digester/app/graph/node_registry.py`
- Create: `services/orion-field-digester/app/graph/edge_registry.py`
- Create: `services/orion-field-digester/app/tensor/field_state.py`
- Test: `tests/test_field_digestion_rules.py`

- [ ] **Step 1: Write failing bootstrap test**

```python
# tests/test_field_digestion_rules.py (first test only)
from datetime import datetime, timezone
from pathlib import Path

from services.orion_field_digester.app.graph.lattice import load_lattice
from services.orion_field_digester.app.tensor.field_state import empty_field_state


def test_empty_field_state_has_all_lattice_nodes() -> None:
    lattice_path = Path("config/field/biometrics_lattice.yaml")
    lattice = load_lattice(lattice_path)
    now = datetime(2026, 5, 24, tzinfo=timezone.utc)
    state = empty_field_state(lattice=lattice, now=now, tick_id="tick_test")
    assert "node:atlas" in state.node_vectors
    assert state.node_vectors["node:atlas"]["availability"] == 1.0
    assert "capability:llm_inference" in state.capability_vectors
    assert len(state.edges) == len(lattice.edges)
```

Note: use `PYTHONPATH=.:services/orion-field-digester` for imports, or package as `app.*` like substrate-runtime.

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH=.:services/orion-field-digester ./venv/bin/python -m pytest tests/test_field_digestion_rules.py::test_empty_field_state_has_all_lattice_nodes -v
```

- [ ] **Step 3: Implement lattice loader + empty state**

```python
# services/orion-field-digester/app/tensor/channels.py
NODE_CHANNELS = [
    "availability", "staleness", "cpu_pressure", "memory_pressure",
    "gpu_pressure", "thermal_pressure", "disk_pressure", "expected_offline_suppression",
]
CAPABILITY_CHANNELS = ["pressure", "confidence", "available_capacity"]

DEFAULT_NODE_VECTOR = {ch: 0.0 for ch in NODE_CHANNELS}
DEFAULT_NODE_VECTOR["availability"] = 1.0

DEFAULT_CAPABILITY_VECTOR = {
    "pressure": 0.0,
    "confidence": 1.0,
    "available_capacity": 1.0,
}
```

```python
# services/orion-field-digester/app/graph/lattice.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from orion.schemas.field_state import FieldEdgeV1


@dataclass(frozen=True)
class LatticeGraph:
    nodes: list[str]
    capabilities: list[str]
    edges: list[FieldEdgeV1]


def load_lattice(path: Path) -> LatticeGraph:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    nodes = [f"node:{n['node_id']}" for n in raw["nodes"]]
    capabilities = [f"capability:{c['capability_id']}" for c in raw["capabilities"]]
    edges = [FieldEdgeV1.model_validate(e) for e in raw["edges"]]
    return LatticeGraph(nodes=nodes, capabilities=capabilities, edges=edges)
```

```python
# services/orion-field-digester/app/tensor/field_state.py
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from uuid import uuid4

from orion.schemas.field_state import FieldStateV1

from ..graph.lattice import LatticeGraph
from .channels import DEFAULT_CAPABILITY_VECTOR, DEFAULT_NODE_VECTOR


def new_tick_id() -> str:
    return f"tick_{uuid4().hex[:12]}"


def empty_field_state(*, lattice: LatticeGraph, now: datetime, tick_id: str) -> FieldStateV1:
    node_vectors = {nid: deepcopy(DEFAULT_NODE_VECTOR) for nid in lattice.nodes}
    capability_vectors = {cid: deepcopy(DEFAULT_CAPABILITY_VECTOR) for cid in lattice.capabilities}
    return FieldStateV1(
        generated_at=now,
        tick_id=tick_id,
        node_vectors=node_vectors,
        capability_vectors=capability_vectors,
        edges=list(lattice.edges),
        recent_perturbations=[],
    )
```

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-field-digester/app/tensor/ services/orion-field-digester/app/graph/ tests/test_field_digestion_rules.py
git commit -m "feat: add field lattice loader and empty field bootstrap"
```

---

### Task 5: Perturbation rule

**Files:**
- Create: `services/orion-field-digester/app/digestion/perturbation.py`
- Create: `services/orion-field-digester/app/ingest/state_deltas.py`
- Test: `tests/test_field_delta_ingest.py`

- [ ] **Step 1: Write failing delta ingest test**

```python
# tests/test_field_delta_ingest.py
from datetime import datetime, timezone

from orion.schemas.state_delta import StateDeltaV1
from services.orion_field_digester.app.ingest.state_deltas import delta_to_perturbations


def test_active_node_pressure_strain_maps_to_gpu_pressure() -> None:
    delta = StateDeltaV1(
        delta_id="delta_test1",
        target_projection="active_node_pressure_projection",
        target_kind="active_node_pressure",
        target_id="atlas",
        operation="reinforce",
        before=None,
        after={
            "node_id": "atlas",
            "active_pressures": ["strain"],
            "pressure_score": 0.72,
            "availability_status": "online",
            "suppressed_pressures": [],
            "capability_impacts": [],
            "evidence_event_ids": ["gev_1"],
            "last_updated_at": "2026-05-24T12:00:00+00:00",
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="node_pressure_reducer",
    )
    perturbations = delta_to_perturbations(delta)
    assert len(perturbations) == 1
    assert perturbations[0].node_id == "node:atlas"
    assert perturbations[0].channel == "gpu_pressure"
    assert perturbations[0].intensity == 0.72
    assert perturbations[0].label == "delta_test1"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
PYTHONPATH=.:services/orion-field-digester ./venv/bin/python -m pytest tests/test_field_delta_ingest.py -v
```

- [ ] **Step 3: Implement perturbation mapping + apply**

```python
# services/orion-field-digester/app/ingest/state_deltas.py
from __future__ import annotations

from dataclasses import dataclass

from orion.schemas.state_delta import StateDeltaV1

GPU_NODES = {"atlas", "circe"}


@dataclass(frozen=True)
class Perturbation:
    node_id: str
    channel: str
    intensity: float
    label: str


def _node_key(raw: str) -> str:
    nid = raw.strip().lower()
    return nid if nid.startswith("node:") else f"node:{nid}"


def delta_to_perturbations(delta: StateDeltaV1) -> list[Perturbation]:
    if delta.operation == "noop":
        return []
    after = delta.after or {}
    node_id = _node_key(str(after.get("node_id") or delta.target_id))
    out: list[Perturbation] = []

    if delta.target_kind == "active_node_pressure":
        score = float(after.get("pressure_score", 0.0))
        pressures = list(after.get("active_pressures") or [])
        if "strain" in pressures:
            channel = "gpu_pressure" if node_id.replace("node:", "") in GPU_NODES else "cpu_pressure"
            out.append(Perturbation(node_id=node_id, channel=channel, intensity=score, label=delta.delta_id))
        if "availability" in pressures:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="availability",
                    intensity=max(0.0, 1.0 - min(1.0, score + 0.2)),
                    label=delta.delta_id,
                )
            )
        if delta.operation == "suppress":
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=1.0,
                    label=delta.delta_id,
                )
            )

    if delta.target_kind == "node_biometrics":
        hints = dict(after.get("pressure_hints") or {})
        if "gpu" in hints:
            out.append(Perturbation(node_id=node_id, channel="gpu_pressure", intensity=float(hints["gpu"]), label=delta.delta_id))
        if "strain" in hints:
            out.append(Perturbation(node_id=node_id, channel="cpu_pressure", intensity=float(hints["strain"]), label=delta.delta_id))
        status = str(after.get("availability_status") or "")
        if status == "stale":
            out.append(Perturbation(node_id=node_id, channel="staleness", intensity=0.5, label=delta.delta_id))
        if after.get("expected_online") is False:
            out.append(
                Perturbation(
                    node_id=node_id,
                    channel="expected_offline_suppression",
                    intensity=1.0,
                    label=delta.delta_id,
                )
            )
    return out
```

```python
# services/orion-field-digester/app/digestion/perturbation.py
from __future__ import annotations

from orion.schemas.field_state import FieldStateV1

from ..ingest.state_deltas import Perturbation


def apply_perturbations(state: FieldStateV1, perturbations: list[Perturbation]) -> FieldStateV1:
    for p in perturbations:
        node_vec = state.node_vectors.setdefault(p.node_id, {})
        if p.channel == "availability":
            node_vec[p.channel] = min(node_vec.get(p.channel, 1.0), p.intensity)
        else:
            node_vec[p.channel] = min(1.0, node_vec.get(p.channel, 0.0) + p.intensity)
        if p.label not in state.recent_perturbations:
            state.recent_perturbations.append(p.label)
    state.recent_perturbations = state.recent_perturbations[-20:]
    return state
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-field-digester/app/ingest/state_deltas.py services/orion-field-digester/app/digestion/perturbation.py tests/test_field_delta_ingest.py
git commit -m "feat: map committed state deltas to field perturbations"
```

---

### Task 6: Decay, diffusion, suppression rules

**Files:**
- Create: `services/orion-field-digester/app/digestion/decay.py`
- Create: `services/orion-field-digester/app/digestion/diffusion.py`
- Create: `services/orion-field-digester/app/digestion/suppression.py`
- Create: `services/orion-field-digester/app/tensor/update_rules.py`
- Test: extend `tests/test_field_digestion_rules.py`

- [ ] **Step 1: Write failing decay test**

```python
def test_decay_fades_pressure_channels() -> None:
    from services.orion_field_digester.app.digestion.decay import apply_decay

    state = FieldStateV1(
        generated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tick_id="tick_decay",
        node_vectors={"node:atlas": {"gpu_pressure": 0.8, "availability": 1.0}},
        capability_vectors={},
        edges=[],
    )
    apply_decay(state, decay_rate=0.5)
    assert state.node_vectors["node:atlas"]["gpu_pressure"] == 0.4
    assert state.node_vectors["node:atlas"]["availability"] == 1.0
```

- [ ] **Step 2: Write failing diffusion test**

```python
def test_diffusion_spreads_gpu_pressure_to_capability() -> None:
    from services.orion_field_digester.app.digestion.diffusion import apply_diffusion

    edge = FieldEdgeV1(
        source_id="node:atlas",
        target_id="capability:llm_inference",
        edge_type="node_capability",
        weight=0.85,
        channel_map={"gpu_pressure": "pressure"},
    )
    state = FieldStateV1(
        generated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tick_id="tick_diff",
        node_vectors={"node:atlas": {"gpu_pressure": 0.8}},
        capability_vectors={"capability:llm_inference": {"pressure": 0.0, "confidence": 1.0, "available_capacity": 1.0}},
        edges=[edge],
    )
    apply_diffusion(state, diffusion_rate=1.0)
    assert state.capability_vectors["capability:llm_inference"]["pressure"] == 0.68
```

- [ ] **Step 3: Write failing suppression test**

```python
def test_suppression_blocks_availability_panic_for_circe() -> None:
    from services.orion_field_digester.app.digestion.suppression import apply_suppression

    state = FieldStateV1(
        generated_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        tick_id="tick_sup",
        node_vectors={
            "node:circe": {
                "availability": 0.2,
                "expected_offline_suppression": 1.0,
                "staleness": 0.9,
            }
        },
        capability_vectors={},
        edges=[],
    )
    apply_suppression(state)
    vec = state.node_vectors["node:circe"]
    assert vec["availability"] >= 0.8
    assert vec["staleness"] == 0.0
```

- [ ] **Step 4: Run tests — expect FAIL**

- [ ] **Step 5: Implement rules**

```python
# services/orion-field-digester/app/digestion/decay.py
PRESSURE_CHANNELS = {
    "staleness", "cpu_pressure", "memory_pressure", "gpu_pressure",
    "thermal_pressure", "disk_pressure",
}


def apply_decay(state, *, decay_rate: float) -> None:
    for vec in state.node_vectors.values():
        for ch in PRESSURE_CHANNELS:
            if ch in vec:
                vec[ch] = vec[ch] * decay_rate
    for vec in state.capability_vectors.values():
        if "pressure" in vec:
            vec["pressure"] = vec["pressure"] * decay_rate
        if "available_capacity" in vec:
            vec["available_capacity"] = min(1.0, 1.0 - vec.get("pressure", 0.0))
```

```python
# services/orion-field-digester/app/digestion/diffusion.py
def apply_diffusion(state, *, diffusion_rate: float) -> None:
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
```

```python
# services/orion-field-digester/app/digestion/suppression.py
def apply_suppression(state) -> None:
    for node_id, vec in state.node_vectors.items():
        if vec.get("expected_offline_suppression", 0.0) >= 1.0:
            vec["availability"] = max(vec.get("availability", 1.0), 0.85)
            vec["staleness"] = 0.0
```

```python
# services/orion-field-digester/app/tensor/update_rules.py
def run_digestion_tick(state, *, perturbations, decay_rate, diffusion_rate):
    from ..digestion.decay import apply_decay
    from ..digestion.diffusion import apply_diffusion
    from ..digestion.perturbation import apply_perturbations
    from ..digestion.suppression import apply_suppression

    apply_perturbations(state, perturbations)
    apply_decay(state, decay_rate=decay_rate)
    apply_diffusion(state, diffusion_rate=diffusion_rate)
    apply_suppression(state)
    return state
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
PYTHONPATH=.:services/orion-field-digester ./venv/bin/python -m pytest tests/test_field_digestion_rules.py tests/test_field_delta_ingest.py -v
```

- [ ] **Step 7: Commit**

```bash
git add services/orion-field-digester/app/digestion/ services/orion-field-digester/app/tensor/update_rules.py tests/test_field_digestion_rules.py
git commit -m "feat: add field decay diffusion and suppression rules"
```

---

### Task 7: Deterministic replay test

**Files:**
- Create: `tests/test_field_deterministic_replay.py`

- [ ] **Step 1: Write replay test with fixture receipts**

```python
# tests/test_field_deterministic_replay.py
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.reduction_receipt import ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from services.orion_field_digester.app.graph.lattice import load_lattice
from services.orion_field_digester.app.ingest.state_deltas import delta_to_perturbations
from services.orion_field_digester.app.tensor.field_state import empty_field_state, new_tick_id
from services.orion_field_digester.app.tensor.update_rules import run_digestion_tick


def _replay(receipts: list[ReductionReceiptV1], *, decay_rate: float, diffusion_rate: float):
    lattice = load_lattice(Path("config/field/biometrics_lattice.yaml"))
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    state = empty_field_state(lattice=lattice, now=now, tick_id=new_tick_id())
    seen: set[str] = set()
    for receipt in receipts:
        perturbations = []
        for delta in receipt.state_deltas:
            if delta.delta_id in seen:
                continue
            seen.add(delta.delta_id)
            perturbations.extend(delta_to_perturbations(delta))
        run_digestion_tick(
            state,
            perturbations=perturbations,
            decay_rate=decay_rate,
            diffusion_rate=diffusion_rate,
        )
    return state


def test_replay_is_deterministic_for_same_receipts() -> None:
    delta = StateDeltaV1(
        delta_id="delta_replay_atlas",
        target_projection="active_node_pressure_projection",
        target_kind="active_node_pressure",
        target_id="atlas",
        operation="reinforce",
        after={
            "node_id": "atlas",
            "active_pressures": ["strain"],
            "pressure_score": 0.72,
            "availability_status": "online",
            "suppressed_pressures": [],
            "capability_impacts": [],
            "evidence_event_ids": [],
            "last_updated_at": "2026-05-24T12:00:00+00:00",
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="node_pressure_reducer",
    )
    receipt = ReductionReceiptV1(
        receipt_id="rcpt_replay",
        accepted_event_ids=["gev_1"],
        state_deltas=[delta],
        created_at=datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc),
    )
    a = _replay([receipt], decay_rate=0.92, diffusion_rate=1.0)
    b = _replay([receipt], decay_rate=0.92, diffusion_rate=1.0)
    assert a.model_dump(mode="json") == b.model_dump(mode="json")
    assert a.node_vectors["node:atlas"]["gpu_pressure"] > 0.0
    assert a.capability_vectors["capability:llm_inference"]["pressure"] > 0.0


def test_duplicate_delta_id_skipped_on_replay() -> None:
    # same delta twice in two receipts — second must not double-apply
    ...
```

(Finish `test_duplicate_delta_id_skipped_on_replay` with two receipts sharing same `delta_id`; assert gpu_pressure equals single-apply value.)

- [ ] **Step 2: Run test — expect PASS after Task 6**

- [ ] **Step 3: Commit**

```bash
git add tests/test_field_deterministic_replay.py
git commit -m "test: add deterministic field replay from committed deltas"
```

---

# Phase 4 — Service scaffold + worker

### Task 8: Service settings, store, worker

**Files:**
- Create: `services/orion-field-digester/app/settings.py`
- Create: `services/orion-field-digester/app/store.py`
- Create: `services/orion-field-digester/app/ingest/receipts.py`
- Create: `services/orion-field-digester/app/worker.py`
- Create: `services/orion-field-digester/app/main.py`
- Create: `services/orion-field-digester/requirements.txt`
- Create: `services/orion-field-digester/Dockerfile`
- Create: `services/orion-field-digester/docker-compose.yml`
- Create: `services/orion-field-digester/.env_example`
- Create: `services/orion-field-digester/README.md`

- [ ] **Step 1: Add settings**

```python
# services/orion-field-digester/app/settings.py
class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-field-digester", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    lattice_path: str = Field("config/field/biometrics_lattice.yaml", alias="LATTICE_PATH")
    receipt_poll_interval_sec: float = Field(2.0, alias="RECEIPT_POLL_INTERVAL_SEC")
    biometrics_field_decay_rate: float = Field(0.92, alias="BIOMETRICS_FIELD_DECAY_RATE")
    biometrics_field_diffusion_rate: float = Field(1.0, alias="BIOMETRICS_FIELD_DIFFUSION_RATE")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
```

- [ ] **Step 2: Implement store + receipt ingest**

`FieldDigesterStore` methods:
- `fetch_new_receipts(limit=50)` — ordered by `created_at, receipt_id` after cursor
- `is_delta_applied(delta_id)` / `mark_delta_applied(delta_id, receipt_id)`
- `load_latest_field()` / `save_field(state: FieldStateV1)`
- `advance_cursor(receipt_id, created_at)`

Cursor name: `field_digest_receipt_consumer`

- [ ] **Step 3: Implement worker tick**

```python
# worker pseudologic
def _tick(self):
    receipts = self._store.fetch_new_receipts()
    if not receipts:
        return None
    state = self._store.load_latest_field() or empty_field_state(...)
    perturbations = []
    for receipt in receipts:
        for delta in receipt.state_deltas:
            if self._store.is_delta_applied(delta.delta_id):
                continue
            perturbations.extend(delta_to_perturbations(delta))
            self._store.mark_delta_applied(delta.delta_id, receipt.receipt_id)
    state.generated_at = now
    state.tick_id = new_tick_id()
    run_digestion_tick(state, perturbations=perturbations, ...)
    self._store.save_field(state)
    self._store.advance_cursor(receipts[-1].receipt_id, receipts[-1].created_at)
```

- [ ] **Step 4: Wire main.py health endpoint** (mirror substrate-runtime)

- [ ] **Step 5: Add docker/requirements/README**

`requirements.txt` — same pins as substrate-runtime.

`Dockerfile` — copy `orion/`, `config/field/`, service dir; port `8116`.

`docker-compose.yml`:

```yaml
services:
  field-digester:
    build:
      context: ../..
      dockerfile: services/orion-field-digester/Dockerfile
    container_name: ${PROJECT}-field-digester
    ports:
      - "${FIELD_DIGESTER_PORT:-8116}:8116"
    environment:
      - POSTGRES_URI=${POSTGRES_URI}
      - LATTICE_PATH=/app/config/field/biometrics_lattice.yaml
      - BIOMETRICS_FIELD_DECAY_RATE=${BIOMETRICS_FIELD_DECAY_RATE:-0.92}
      - BIOMETRICS_FIELD_DIFFUSION_RATE=${BIOMETRICS_FIELD_DIFFUSION_RATE:-1.0}
      - RECEIPT_POLL_INTERVAL_SEC=${RECEIPT_POLL_INTERVAL_SEC:-2.0}
```

- [ ] **Step 6: Sync `.env_example` → local `.env`**

```bash
# in services/orion-field-digester/
cp .env_example .env
# merge any new keys into services/orion-field-digester/.env on operator machine
```

`.env_example`:

```
PROJECT=orion-athena
SERVICE_NAME=orion-field-digester
POSTGRES_URI=postgresql://orion:orion@postgres:5432/orion
LATTICE_PATH=/app/config/field/biometrics_lattice.yaml
BIOMETRICS_FIELD_DECAY_RATE=0.92
BIOMETRICS_FIELD_DIFFUSION_RATE=1.0
RECEIPT_POLL_INTERVAL_SEC=2.0
LOG_LEVEL=INFO
```

- [ ] **Step 7: Compile check**

```bash
PYTHONPATH=.:services/orion-field-digester python -m compileall services/orion-field-digester
```

- [ ] **Step 8: Commit**

```bash
git add services/orion-field-digester/
git commit -m "feat: scaffold orion-field-digester service and receipt worker"
```

---

### Task 9: Projection helpers + emit stub

**Files:**
- Create: `services/orion-field-digester/app/projections/node_field_projection.py`
- Create: `services/orion-field-digester/app/projections/capability_field_projection.py`
- Create: `services/orion-field-digester/app/projections/substrate_field_projection.py`
- Create: `services/orion-field-digester/app/emit/field_events.py`

- [ ] **Step 1: Implement node projection builder**

```python
def build_node_field_projection(state: FieldStateV1, node_id: str) -> dict:
    nid = node_id if node_id.startswith("node:") else f"node:{node_id.strip().lower()}"
    connected = []
    for edge in state.edges:
        if edge.source_id != nid:
            continue
        cap = edge.target_id.replace("capability:", "")
        connected.append({
            "capability_id": cap,
            "pressure": state.capability_vectors.get(edge.target_id, {}).get("pressure", 0.0),
            "edge_weight": edge.weight,
        })
    return {
        "node_id": nid.replace("node:", ""),
        "field_vector": dict(state.node_vectors.get(nid, {})),
        "connected_capabilities": connected,
        "recent_perturbations": list(state.recent_perturbations),
    }
```

- [ ] **Step 2: emit/field_events.py stub**

```python
"""v1: projections persisted to Postgres only; bus emit deferred."""
def publish_field_projection(*_args, **_kwargs) -> None:
    return None
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-field-digester/app/projections/ services/orion-field-digester/app/emit/
git commit -m "feat: add field projection builders (no bus emit in v1)"
```

---

# Phase 5 — Hub debug endpoints

### Task 10: Substrate field debug API

**Files:**
- Create: `services/orion-hub/scripts/substrate_field_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Test: `services/orion-hub/tests/test_substrate_field_debug_api.py`

- [ ] **Step 1: Write failing Hub test**

```python
# services/orion-hub/tests/test_substrate_field_debug_api.py
def test_field_node_atlas_returns_vector_and_capabilities(client):
    resp = client.get("/api/substrate/field/node/atlas")
    assert resp.status_code == 200
    body = resp.json()
    assert body["node_id"] == "atlas"
    assert "gpu_pressure" in body["field_vector"]
    assert isinstance(body["connected_capabilities"], list)
```

Use fake engine pattern from `test_substrate_biometrics_debug_api.py`.

- [ ] **Step 2: Implement routes**

```python
# services/orion-hub/scripts/substrate_field_routes.py
router = APIRouter(prefix="/api/substrate/field", tags=["substrate-field"])

@router.get("/latest")
async def field_latest(): ...

@router.get("/node/{node_id}")
async def field_node(node_id: str):
  # return spec example shape for atlas

@router.get("/capability/{capability_id}")
async def field_capability(capability_id: str): ...
```

Query `substrate_field_state ORDER BY generated_at DESC LIMIT 1`, reuse projection builder logic (inline or import from shared module under `orion/substrate/field/` if hub cannot import service — prefer small duplicate read helper in hub routes like biometrics routes).

- [ ] **Step 3: Register router in `api_routes.py`**

- [ ] **Step 4: Run Hub tests**

```bash
PYTHONPATH=.:services/orion-hub ./venv/bin/python -m pytest services/orion-hub/tests/test_substrate_field_debug_api.py -v
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/substrate_field_routes.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_substrate_field_debug_api.py
git commit -m "feat: add Hub debug endpoints for substrate field state"
```

---

# Phase 6 — Smoke + integration verification

### Task 11: Smoke script

**Files:**
- Create: `scripts/smoke_field_digester_biometrics.sh`

- [ ] **Step 1: Implement smoke**

```bash
#!/usr/bin/env bash
set -euo pipefail
HUB_PORT="${HUB_PORT:-8080}"
BASE="http://localhost:${HUB_PORT}"

echo "1. Biometrics substrate chain (prerequisite):"
curl -s "${BASE}/api/substrate/biometrics-node/atlas/latest" | jq '.active_node_pressure_projection'

echo "2. Latest field state:"
curl -s "${BASE}/api/substrate/field/latest" | jq .

echo "3. Atlas node field:"
curl -s "${BASE}/api/substrate/field/node/atlas" | jq .

echo "4. LLM inference capability field:"
curl -s "${BASE}/api/substrate/field/capability/llm_inference" | jq .
```

- [ ] **Step 2: chmod +x and commit**

```bash
chmod +x scripts/smoke_field_digester_biometrics.sh
git add scripts/smoke_field_digester_biometrics.sh
git commit -m "chore: add field digester biometrics smoke script"
```

---

### Task 12: Full test suite

- [ ] **Step 1: Field unit tests**

```bash
cd .worktrees/feat-orion-field-digester-v1
PYTHONPATH=.:services/orion-field-digester ./venv/bin/python -m pytest \
  tests/test_field_state_schemas.py \
  tests/test_field_delta_ingest.py \
  tests/test_field_digestion_rules.py \
  tests/test_field_deterministic_replay.py \
  orion/grammar/tests/test_biometrics_substrate_schemas.py -q
```

Expected: all PASS

- [ ] **Step 2: Hub API tests**

```bash
PYTHONPATH=.:services/orion-hub ./venv/bin/python -m pytest \
  services/orion-hub/tests/test_substrate_field_debug_api.py \
  services/orion-hub/tests/test_substrate_biometrics_debug_api.py -q
```

- [ ] **Step 3: Substrate regression**

```bash
PYTHONPATH=. ./venv/bin/python -m pytest \
  tests/test_node_pressure_reducer.py \
  tests/test_biometrics_pipeline.py -q
```

- [ ] **Step 4: Live smoke (operator)**

Apply `manual_migration_field_digester_v1.sql`, start postgres + substrate-runtime + field-digester + hub, emit atlas biometrics sample, wait for poll cycles, run `scripts/smoke_field_digester_biometrics.sh`.

Expected chain:

```text
atlas biometrics → reduction receipt → field digester perturbation
→ atlas node gpu_pressure > 0 → llm_inference pressure > 0
```

If stack unavailable: mark live smoke **UNVERIFIED** in PR report.

---

# Phase 7 — Code review, PR report, push

### Task 13: Code review subagent

- [ ] **Step 1: Invoke `requesting-code-review` skill** against full diff + acceptance criteria below

- [ ] **Step 2: Fix all blocking issues — commit fixes**

---

### Task 14: PR report + GitHub PR

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-24-orion-field-digester-v1-pr.md`

- [ ] **Step 1: Write PR report** (sections: Summary, Architecture diagram, Changes by area, Acceptance table, Test plan, Verification evidence, Operator notes)

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feat/orion-field-digester-v1
gh pr create --base feat/biometrics-substrate-delta-seam-hardening \
  --title "PR: Orion field digester v1 (biometrics digestion)" \
  --body "$(cat <<'EOF'
## Summary
- Adds orion-field-digester: consumes committed ReductionReceiptV1/StateDeltaV1 and compiles biometrics field state.
- Proves atlas GPU pressure diffuses into llm_inference capability vector.
- Hub debug endpoints: /api/substrate/field/latest, /node/{id}, /capability/{id}.

## Test plan
- [ ] Field unit + replay tests
- [ ] Hub field debug API tests
- [ ] smoke_field_digester_biometrics.sh with stack up

EOF
)"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Consumes receipts/deltas not debug traces | Task 8 worker polls `substrate_reduction_receipts` |
| Biometrics pressure deltas perturb node vectors | Task 5 perturbation |
| Node pressure diffuses to capability vectors | Task 6 diffusion |
| Expected-offline suppression | Task 6 suppression + Task 5 circe mapping |
| Field state decays over ticks | Task 6 decay |
| Inspectable via debug endpoint | Task 10 |
| Deterministic replay from committed deltas | Task 7, Task 8 delta_id dedupe |
| No mind service | No mind imports anywhere |
| nodes × capabilities × pressure channels × time | Task 2 lattice + FieldStateV1 |
| Git worktree isolation | Task 0 |
| .env_example + local .env sync | Task 8 |
| registry update | Task 1 |
| channels (if needed) | v1 projections-only — no channel entry |
| code review + PR | Tasks 13–14 |

**Placeholder scan:** None — all tasks include concrete paths, code, and commands.

**Type consistency:** `FieldStateV1`, `Perturbation`, `FieldEdgeV1` names match across Tasks 1–10.

---

## Core invariant (carry through implementation)

```text
raw grammar event = proposal
reduction receipt / state delta = committed fact
field digester metabolizes committed facts
projections make the field inspectable
organs participate later — not in v1
```
