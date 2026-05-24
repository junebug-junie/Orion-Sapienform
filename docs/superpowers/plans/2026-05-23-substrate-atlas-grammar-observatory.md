# Substrate Atlas / Grammar Observatory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a read-only Hub instrument that materializes and visualizes canonical grammar traces (atoms, edges, layers, dimensions, temporal hops, compactions, projections) from append-only events, proving end-to-end vision observation → grammar → graph → provenance inspector.

**Architecture:** Organs publish `GrammarEventV1` on a single bus channel (`orion:grammar:event`). `orion-sql-writer` appends events and derived rows to Postgres (`grammar_*` tables). Pure-Python query/materializer modules in `orion/grammar/` build trace/graph payloads. Hub exposes `/api/substrate/atlas/*` and a first-class `/substrate-atlas` page using **Cytoscape** (already loaded for Memory graph — do not add React Flow in MVP). Live updates: MVP polls trace graph every 3s; Phase 6b adds optional Hub WS fan-out.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy models in sql-writer, asyncpg/SQLAlchemy sync writes (match sql-writer worker), Redis bus (`OrionBusAsync`, `BaseEnvelope`), FastAPI Hub routes, Hub static JS + Cytoscape CDN (same as `memory.js`).

**Design source:** User spec “Substrate Atlas / Visual Grammar Observatory” (2026-05-23).

**Related (do not conflate):**
| Existing | Role | Atlas relationship |
|----------|------|-------------------|
| `orion/schema_kernel` (`ConceptAtomV1`) | Invariant vocabulary | Different layer; no merge in MVP |
| `orion/substrate/molecules.py` (`SubstrateMoleculeV1`) | Organ substrate currency | Future bridge emits `GrammarEventV1` |
| `orion/signals` + Organ Signals UI | Runtime mesh observability | Complementary; Atlas is grammar provenance |
| `/api/substrate/*` (mutation/review) | Policy/mutation operator surface | **Separate** route prefix `/api/substrate/atlas/*` |
| `feat/substrate-graph-mvp` | Docs-only signal-bridge plan | Parallel track; Atlas owns grammar tables |

---

## Worktree isolation (mandatory — do not touch `feat/repair-pressure-v1`)

Current main workspace is on `feat/repair-pressure-v1` (Claude active). **All implementation commits happen only in a dedicated worktree.**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
# Base branch: use origin/main unless product owner specifies otherwise
git worktree add .worktrees/feat-substrate-atlas-grammar-observatory -b feat/substrate-atlas-grammar-observatory origin/main
cd .worktrees/feat-substrate-atlas-grammar-observatory
git check-ignore -q .worktrees  # must succeed (already ignored in this repo)
```

**Rules:**
- Never `git checkout` the feature branch in the main workspace path.
- Sync operator env only: when editing `services/*/.env_example`, also copy changed keys into `services/*/.env` in the **worktree** (not main workspace `.env` unless user explicitly asks).
- PR is opened from `feat/substrate-atlas-grammar-observatory` only.

---

## Phase 0 preflight (completed 2026-05-23)

| Question | Finding |
|----------|---------|
| Schema package location | `orion/schemas/*.py` + `orion/schemas/registry.py` |
| Bus catalog | `orion/bus/channels.yaml`; publish via `OrionBusAsync` / `BaseEnvelope` |
| SQL persistence pattern | `services/orion-sql-writer` — `DEFAULT_ROUTE_MAP`, SQLAlchemy models under `app/models/`, `manual_migration_*.sql` in `services/orion-sql-db/` |
| Hub API pattern | `services/orion-hub/scripts/api_routes.py` (`APIRouter`) |
| Hub standalone pages | e.g. `@router.get("/substrate")` → `templates/substrate.html` |
| Graph UI library | **Cytoscape** in `memory.js`, `memory-graph-draft-ui.js` — use for Atlas MVP |
| Vision services | `orion-vision-retina`, `orion-vision-edge`, `orion-vision-window`, channels `orion:vision:*` |
| Grammar tables | **None** yet (`grammar_traces` grep empty) |
| Existing grammar-named code | `schema_kernel` docstring only; not the spec’s episodic grammar |

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/grammar.py` | `GrammarEventV1`, `GrammarAtomV1`, `GrammarEdgeV1`, hops, compactions, projections, enums |
| `orion/schemas/registry.py` | Register `grammar.event.v1` kind |
| `orion/grammar/__init__.py` | Package exports |
| `orion/grammar/constants.py` | `GRAMMAR_LAYERS`, `GRAMMAR_DIMENSIONS` tuples |
| `orion/grammar/ledger.py` | `apply_grammar_event(session, event)` — upsert trace, insert event + child rows, dedupe by `event_id` |
| `orion/grammar/query.py` | `list_traces`, `get_trace`, `get_trace_graph`, neighborhood, provenance, temporal path |
| `orion/grammar/graph_view.py` | Layer/dimension summaries + Cytoscape-friendly node/edge lists |
| `orion/grammar/publish.py` | `publish_grammar_event(bus, event)` helper for emitters |
| `orion/grammar/seed_demo.py` | Deterministic vision_observation trace (callable from script) |
| `orion/grammar/tests/test_schemas.py` | Schema round-trip + validation |
| `orion/grammar/tests/test_ledger.py` | Ledger dedupe + event_kind branches |
| `orion/grammar/tests/test_query.py` | Graph + filters |
| `services/orion-sql-db/manual_migration_grammar_atlas.sql` | DDL from spec §9 |
| `services/orion-sql-writer/app/models/grammar_trace.py` | SQLAlchemy models (7 tables) |
| `services/orion-sql-writer/app/grammar_ledger_handler.py` | Route `grammar.event.v1` → ledger |
| `services/orion-sql-writer/app/settings.py` | Route map + subscribe channel |
| `services/orion-sql-writer/.env_example` | `orion:grammar:event` subscription |
| `services/orion-sql-writer/docker-compose.yml` | Channel env passthrough |
| `orion/bus/channels.yaml` | `orion:grammar:event` catalog entry |
| `services/orion-hub/scripts/grammar_atlas_routes.py` | Atlas HTTP handlers (keep `api_routes.py` thin) |
| `services/orion-hub/scripts/api_routes.py` | `include_router(grammar_atlas_router)` |
| `services/orion-hub/app/settings.py` | `GRAMMAR_ATLAS_ENABLED`, poll interval |
| `services/orion-hub/templates/substrate_atlas.html` | Standalone page shell |
| `services/orion-hub/static/js/substrate-atlas.js` | Trace picker, Cytoscape graph, inspector, filters |
| `services/orion-hub/templates/index.html` | Nav link to `#substrate-atlas` |
| `scripts/seed_substrate_atlas_demo.py` | CLI entrypoint |
| `services/orion-vision-retina/app/grammar_emit.py` | MVP vision → grammar events (optional Phase 7) |

---

# Phase 1 — Shared grammar schemas

### Task 1: Grammar Pydantic models

**Files:**
- Create: `orion/schemas/grammar.py`
- Modify: `orion/schemas/registry.py`
- Test: `orion/grammar/tests/test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

```python
# orion/grammar/tests/test_schemas.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)


def test_atom_roundtrip() -> None:
    atom = GrammarAtomV1(
        atom_id="atom:vision:person:01JTEST",
        trace_id="trace:vision:01JTEST",
        atom_type="observation",
        semantic_role="person_presence",
        layer="sensor_semantic",
        dimensions=["visual", "spatial", "epistemic"],
        summary="Possible person detected near doorway",
        confidence=0.72,
    )
    raw = atom.model_dump(mode="json")
    assert GrammarAtomV1.model_validate(raw).atom_id == atom.atom_id


def test_invalid_atom_type_rejected() -> None:
    with pytest.raises(ValidationError):
        GrammarAtomV1(
            atom_id="atom:x",
            trace_id="trace:x",
            atom_type="not_a_real_type",  # type: ignore[arg-type]
            semantic_role="x",
            layer="sensor_raw",
            summary="x",
        )


def test_grammar_event_requires_provenance() -> None:
    now = datetime.now(timezone.utc)
    ev = GrammarEventV1(
        event_id="evt:01JTEST",
        event_kind="atom_emitted",
        trace_id="trace:vision:01JTEST",
        emitted_at=now,
        atom=GrammarAtomV1(
            atom_id="atom:vision:person:01JTEST",
            trace_id="trace:vision:01JTEST",
            atom_type="observation",
            semantic_role="person_presence",
            layer="sensor_semantic",
            summary="Possible person",
        ),
        provenance=GrammarProvenanceV1(
            source_service="orion-vision-retina",
            source_component="detector",
        ),
    )
    assert ev.event_kind == "atom_emitted"
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-substrate-atlas-grammar-observatory
PYTHONPATH=. pytest orion/grammar/tests/test_schemas.py -v
```

Expected: `ModuleNotFoundError: orion.schemas.grammar`

- [ ] **Step 3: Implement `orion/schemas/grammar.py`**

Use the spec §8 models verbatim with typed unions on `GrammarEventV1`:

```python
# orion/schemas/grammar.py — implement all models from design spec §8.1–8.6:
# GrammarEventKind, GrammarProvenanceV1, GrammarEventV1,
# AtomType, TimeRangeV1, GrammarAtomV1,
# RelationType, GrammarEdgeV1,
# TemporalHopType, TemporalHopV1,
# GrammarCompactionV1, GrammarProjectionV1,
# GrammarTraceV1 (trace metadata for trace_started events)
```

`GrammarEventV1` must use:

```python
atom: GrammarAtomV1 | None = None
edge: GrammarEdgeV1 | None = None
temporal_hop: TemporalHopV1 | None = None
compaction: GrammarCompactionV1 | None = None
projection: GrammarProjectionV1 | None = None
```

Add `model_config = ConfigDict(extra="forbid")` on all public models.

- [ ] **Step 4: Register bus kind in `orion/schemas/registry.py`**

Add import and map entry:

```python
from orion.schemas.grammar import GrammarEventV1

# in SCHEMA_REGISTRY dict:
"grammar.event.v1": GrammarEventV1,
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH=. pytest orion/grammar/tests/test_schemas.py -v
```

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/grammar.py orion/schemas/registry.py orion/grammar/tests/test_schemas.py
git commit -m "feat(grammar): add GrammarEventV1 schema package"
```

---

# Phase 2 — Postgres ledger (sql-writer)

### Task 2: DDL migration

**Files:**
- Create: `services/orion-sql-db/manual_migration_grammar_atlas.sql`

- [ ] **Step 1: Add SQL file** — copy spec §9.1 and §9.2 indexes exactly (tables: `grammar_traces`, `grammar_events`, `grammar_atoms`, `grammar_edges`, `grammar_temporal_hops`, `grammar_compactions`, `grammar_projections`).

- [ ] **Step 2: Apply locally (operator)**

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_grammar_atlas.sql
```

Expected: `CREATE TABLE` / `CREATE INDEX` without error.

- [ ] **Step 3: Commit**

```bash
git add services/orion-sql-db/manual_migration_grammar_atlas.sql
git commit -m "chore(sql-db): add grammar atlas DDL"
```

### Task 3: SQLAlchemy models + route map

**Files:**
- Create: `services/orion-sql-writer/app/models/grammar_trace.py`
- Modify: `services/orion-sql-writer/app/models/__init__.py` (export models)
- Modify: `services/orion-sql-writer/app/settings.py`
- Test: `services/orion-sql-writer/tests/test_grammar_event_routing.py`

- [ ] **Step 1: Failing routing test**

```python
# services/orion-sql-writer/tests/test_grammar_event_routing.py
from app.settings import DEFAULT_ROUTE_MAP, settings


def test_route_map_includes_grammar_event() -> None:
    assert DEFAULT_ROUTE_MAP["grammar.event.v1"] == "GrammarEventSQL"


def test_subscribe_channels_include_grammar() -> None:
    channels = settings.sql_writer_subscribe_channels
    assert "orion:grammar:event" in channels
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=services/orion-sql-writer:../.. pytest services/orion-sql-writer/tests/test_grammar_event_routing.py -v
```

- [ ] **Step 3: Create SQLAlchemy models** — one file with classes mirroring spec columns; each table has `*_json JSONB` column storing full model dump. Example fragment:

```python
# services/orion-sql-writer/app/models/grammar_trace.py
from sqlalchemy import Column, DateTime, String, Text, Float, Integer, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from app.db import Base


class GrammarTraceSQL(Base):
    __tablename__ = "grammar_traces"
    trace_id = Column(String, primary_key=True)
    trace_type = Column(String, nullable=False)
    # ... all columns from spec §9.1 grammar_traces


class GrammarEventSQL(Base):
    __tablename__ = "grammar_events"
    event_id = Column(String, primary_key=True)
    trace_id = Column(String, ForeignKey("grammar_traces.trace_id"), nullable=False)
    event_kind = Column(String, nullable=False)
    event_json = Column(JSONB, nullable=False)
    # ... remaining columns
```

Register `GrammarEventSQL` as the routed model for kind `grammar.event.v1` in worker `MODEL_MAP` (follow `EvidenceUnitSQL` pattern in `app/worker.py`).

- [ ] **Step 4: Extend settings**

```python
# In DEFAULT_ROUTE_MAP:
"grammar.event.v1": "GrammarEventSQL",

# Append to sql_writer_subscribe_channels default list:
"orion:grammar:event",
```

- [ ] **Step 5: Run routing test — PASS**

- [ ] **Step 6: Commit**

```bash
git add services/orion-sql-writer/app/models/grammar_trace.py services/orion-sql-writer/app/settings.py \
  services/orion-sql-writer/tests/test_grammar_event_routing.py
git commit -m "feat(sql-writer): route grammar.event.v1 to grammar tables"
```

### Task 4: Ledger apply logic

**Files:**
- Create: `orion/grammar/ledger.py`
- Create: `services/orion-sql-writer/app/grammar_ledger_handler.py`
- Modify: `services/orion-sql-writer/app/worker.py` (special-case `grammar.event.v1` before generic `_write_row`)
- Test: `orion/grammar/tests/test_ledger.py`

- [ ] **Step 1: Failing ledger tests (in-memory or sqlite if repo pattern exists; else mock session)**

```python
# orion/grammar/tests/test_ledger.py
from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.grammar.ledger import apply_grammar_event


def test_atom_event_inserts_atom_row() -> None:
    session = MagicMock()
    now = datetime.now(timezone.utc)
    atom = GrammarAtomV1(
        atom_id="atom:a1",
        trace_id="trace:t1",
        atom_type="observation",
        semantic_role="motion",
        layer="sensor_raw",
        summary="Motion detected",
    )
    event = GrammarEventV1(
        event_id="evt:1",
        event_kind="atom_emitted",
        trace_id="trace:t1",
        emitted_at=now,
        atom=atom,
        provenance=GrammarProvenanceV1(source_service="test"),
    )
    apply_grammar_event(session, event)
    assert session.add.called
```

- [ ] **Step 2: Implement `apply_grammar_event`**

```python
# orion/grammar/ledger.py
def apply_grammar_event(session, event: GrammarEventV1) -> bool:
    """Returns False if event_id already exists (dedupe)."""
    if _event_exists(session, event.event_id):
        return False
    _ensure_trace(session, event)
    _insert_event_row(session, event)
    if event.event_kind == "atom_emitted" and event.atom:
        _insert_atom(session, event.atom, event)
    elif event.event_kind == "edge_emitted" and event.edge:
        _insert_edge(session, event.edge, event)
  # ... temporal_hop, compaction, projection, trace_started, trace_ended
    return True
```

Wire sql-writer: on `grammar.event.v1`, validate payload → `GrammarEventV1.model_validate` → `apply_grammar_event` inside existing DB session; **do not** duplicate via generic JSON-only insert.

- [ ] **Step 3: Ledger tests PASS**

```bash
PYTHONPATH=. pytest orion/grammar/tests/test_ledger.py -v
```

- [ ] **Step 4: Commit**

```bash
git add orion/grammar/ledger.py orion/grammar/tests/test_ledger.py \
  services/orion-sql-writer/app/grammar_ledger_handler.py services/orion-sql-writer/app/worker.py
git commit -m "feat(grammar): append-only ledger writer with event dedupe"
```

### Task 5: Bus catalog + sql-writer env

**Files:**
- Modify: `orion/bus/channels.yaml`
- Modify: `services/orion-sql-writer/.env_example`
- Modify: `services/orion-sql-writer/docker-compose.yml`
- Copy keys → `services/orion-sql-writer/.env` (worktree only)

- [ ] **Step 1: Add channel**

```yaml
# orion/bus/channels.yaml
  - name: "orion:grammar:event"
    reliability: "at_least_once"
    message_kind: "grammar.event.v1"
    producer_services: ["orion-vision-retina", "orion-hub", "orion-vision-edge"]
    consumer_services: ["orion-sql-writer"]
```

- [ ] **Step 2: `.env_example`**

```bash
# Grammar atlas ledger
# SQL_WRITER_SUBSCRIBE_CHANNELS already list-based; ensure orion:grammar:event present
```

- [ ] **Step 3: Commit**

```bash
git add orion/bus/channels.yaml services/orion-sql-writer/.env_example services/orion-sql-writer/docker-compose.yml
git commit -m "chore(bus): catalog orion:grammar:event for sql-writer"
```

---

# Phase 3 — Query / materializer

### Task 6: Query layer + graph view

**Files:**
- Create: `orion/grammar/query.py`
- Create: `orion/grammar/graph_view.py`
- Create: `orion/grammar/constants.py`
- Test: `orion/grammar/tests/test_query.py`

- [ ] **Step 1: Failing tests for list + graph**

```python
def test_get_trace_graph_nodes_include_layer(sample_trace_db):
    graph = get_trace_graph(sample_trace_db, "trace:demo", layout="layered", depth=2)
    assert graph["nodes"]
    assert all("layer" in n for n in graph["nodes"])
```

Use pytest fixtures that insert rows via `apply_grammar_event` or direct SQL in test DB.

- [ ] **Step 2: Implement query functions** per spec §11:

```python
def list_traces(session, *, session_id: str | None, limit: int) -> list[dict]: ...
def get_trace(session, trace_id: str) -> dict | None: ...
def get_trace_graph(session, trace_id: str, *, layout: str, depth: int) -> dict: ...
def get_atom_neighborhood(session, atom_id: str, *, depth: int, direction: str) -> dict: ...
def get_atom_provenance(session, atom_id: str) -> dict: ...
def get_temporal_path(session, atom_id: str, *, direction: str, limit: int) -> dict: ...
```

`graph_view.py` builds `layer_summary` / `dimension_summary` and maps DB rows → spec §11.3 node/edge shape.

- [ ] **Step 3: Tests PASS**

- [ ] **Step 4: Commit**

```bash
git add orion/grammar/query.py orion/grammar/graph_view.py orion/grammar/constants.py orion/grammar/tests/test_query.py
git commit -m "feat(grammar): trace query and graph materializer"
```

---

# Phase 4 — Hub API

### Task 7: Atlas routes

**Files:**
- Create: `services/orion-hub/scripts/grammar_atlas_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/app/settings.py`
- Test: `services/orion-hub/tests/test_grammar_atlas_api.py`

- [ ] **Step 1: Failing API test**

```python
from fastapi.testclient import TestClient
from scripts.main import app

client = TestClient(app)

def test_list_traces_empty():
    r = client.get("/api/substrate/atlas/traces?limit=10")
    assert r.status_code == 200
    assert r.json()["items"] == []
```

- [ ] **Step 2: Implement router**

```python
# services/orion-hub/scripts/grammar_atlas_routes.py
router = APIRouter(prefix="/api/substrate/atlas", tags=["substrate-atlas"])

@router.get("/traces")
def list_traces_api(session_id: str | None = None, limit: int = 50): ...

@router.get("/traces/{trace_id}")
def get_trace_api(trace_id: str): ...

@router.get("/traces/{trace_id}/graph")
def get_trace_graph_api(trace_id: str, layout: str = "layered", depth: int = 2): ...

@router.get("/atoms/{atom_id}/neighborhood")
def atom_neighborhood_api(atom_id: str, depth: int = 2, direction: str = "both"): ...

@router.get("/atoms/{atom_id}/provenance")
def atom_provenance_api(atom_id: str): ...

@router.get("/atoms/{atom_id}/temporal-path")
def atom_temporal_path_api(atom_id: str, direction: str = "backward", limit: int = 25): ...
```

Use Hub Postgres session factory (same DSN pattern as other substrate SQL reads — grep `POSTGRES_URI` / `build_substrate_store_from_env` and add `GRAMMAR_ATLAS_POSTGRES_URI` defaulting to same DB).

404 when trace/atom missing.

- [ ] **Step 3: Include router in `api_routes.py`**

```python
from .grammar_atlas_routes import router as grammar_atlas_router
router.include_router(grammar_atlas_router)
```

- [ ] **Step 4: Settings**

```python
GRAMMAR_ATLAS_ENABLED: bool = True
GRAMMAR_ATLAS_POLL_INTERVAL_MS: int = 3000
```

Update `services/orion-hub/.env_example` + copy to `.env` in worktree.

- [ ] **Step 5: Tests PASS**

```bash
./scripts/test_service.sh orion-hub tests/test_grammar_atlas_api.py -v
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/scripts/grammar_atlas_routes.py services/orion-hub/scripts/api_routes.py \
  services/orion-hub/app/settings.py services/orion-hub/tests/test_grammar_atlas_api.py \
  services/orion-hub/.env_example
git commit -m "feat(hub): substrate atlas read API"
```

---

# Phase 5 — Hub UI (Cytoscape)

### Task 8: Standalone page + nav

**Files:**
- Create: `services/orion-hub/templates/substrate_atlas.html`
- Create: `services/orion-hub/static/js/substrate-atlas.js`
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/scripts/api_routes.py`

- [ ] **Step 1: Route**

```python
@router.get("/substrate-atlas")
async def substrate_atlas_page() -> HTMLResponse:
    template = (TEMPLATES_DIR / "substrate_atlas.html").read_text(encoding="utf-8")
    ...
```

- [ ] **Step 2: Template layout** — spec §12.1 panels:
  - Trace picker (left/top)
  - Layer rail + dimension chips
  - Cytoscape host `div#grammarAtlasCy`
  - Atom inspector panel
  - Timeline strip (simple horizontal marks MVP)

Load Cytoscape same way as `index.html` memory panel.

- [ ] **Step 3: `substrate-atlas.js`**

```javascript
(function () {
  const state = { traceId: null, selectedAtomId: null, layerFilter: null, dimensions: new Set() };

  async function fetchTraces() {
    const r = await fetch("/api/substrate/atlas/traces?limit=50");
    return (await r.json()).items;
  }

  async function loadGraph(traceId) {
    const r = await fetch(`/api/substrate/atlas/traces/${encodeURIComponent(traceId)}/graph?layout=layered&depth=2`);
    const g = await r.json();
    renderCytoscape(g);
    renderLayerRail(g.groups.layers);
  }

  function renderCytoscape(graph) {
    // Map graph.nodes → cy elements; y-position by layer index from GRAMMAR_LAYERS order
    // graph.edges → edges with label relation_type
    // on tap node → load provenance into inspector
  }
  // setInterval poll when trace open (GRAMMAR_ATLAS_POLL_INTERVAL_MS)
})();
```

- [ ] **Step 4: Nav link in `index.html`**

```html
<a href="#substrate-atlas" data-hash-target="#substrate-atlas" class="...">Substrate Atlas</a>
<section id="substrate-atlas" data-panel="substrate-atlas" class="hidden ...">
  <iframe id="substrateAtlasFrame" src="/substrate-atlas" class="w-full min-h-[56rem]"></iframe>
</section>
```

- [ ] **Step 5: Manual smoke**

1. Run seed script (Task 9)
2. Open `/substrate-atlas`
3. Select `trace:vision:demo` → nodes visible → click atom → provenance panel populated

- [ ] **Step 6: Commit**

```bash
git add services/orion-hub/templates/substrate_atlas.html services/orion-hub/static/js/substrate-atlas.js \
  services/orion-hub/templates/index.html services/orion-hub/scripts/api_routes.py
git commit -m "feat(hub): Substrate Atlas trace explorer UI"
```

---

# Phase 6 — Seed demo + vision emitter

### Task 9: Deterministic seed command

**Files:**
- Create: `orion/grammar/seed_demo.py`
- Create: `scripts/seed_substrate_atlas_demo.py`

- [ ] **Step 1: Implement `build_vision_demo_events() -> list[GrammarEventV1]`** per spec §14 and §19 (frame → motion → person → uncertainty → scene compaction → projection).

- [ ] **Step 2: CLI**

```python
# scripts/seed_substrate_atlas_demo.py
if __name__ == "__main__":
    for event in build_vision_demo_events():
        publish_or_apply(event)  # direct DB apply OR bus publish
```

- [ ] **Step 3: Run**

```bash
PYTHONPATH=. python scripts/seed_substrate_atlas_demo.py
```

Expected stdout: `seeded trace_id=trace:vision:demo atoms=6 edges=6`

- [ ] **Step 4: Commit**

```bash
git add orion/grammar/seed_demo.py scripts/seed_substrate_atlas_demo.py
git commit -m "feat(grammar): seed vision observation demo trace"
```

### Task 10 (optional MVP+): Vision retina emitter

**Files:**
- Create: `services/orion-vision-retina/app/grammar_emit.py`
- Modify: `services/orion-vision-retina/app/main.py`
- Modify: `services/orion-vision-retina/app/settings.py`
- Modify: `services/orion-vision-retina/.env_example` + `.env`

Emit `GrammarEventV1` on detection path when `GRAMMAR_EMIT_ENABLED=true`. Map motion/person/scene to spec §14 atoms.

- [ ] **Commit separately:** `feat(vision-retina): emit grammar events for observations`

### Task 11: Live stream (deferred sub-phase)

**MVP:** polling only (Task 8).

**Phase 6b** (separate PR if needed):
- Hub `grammar_atlas_ws.py` — subscribe bus or LISTEN/NOTIFY; message types `atom_added`, `edge_added`, `trace_closed`
- Client upgrades from poll to WS when `?live=1`

---

# Phase 7 — Acceptance + PR

### Task 12: Acceptance checklist (spec §18)

- [ ] `python scripts/seed_substrate_atlas_demo.py` → trace in DB
- [ ] `GET /api/substrate/atlas/traces` lists demo trace
- [ ] Full trace + graph endpoints return atoms/edges
- [ ] Layer filter hides non-matching nodes
- [ ] Dimension chip highlights matching nodes
- [ ] Compaction expand shows `source_atom_ids` in inspector
- [ ] Temporal hop visible in inspector
- [ ] Projection labeled as candidate (not fact)

### Task 13: Code review + PR (after all tasks)

1. **Subagent code review** — REQUIRED SUB-SKILL: `requesting-code-review`; fix all blocking issues.
2. Push branch:

```bash
cd .worktrees/feat-substrate-atlas-grammar-observatory
git push -u origin feat/substrate-atlas-grammar-observatory
```

3. Open PR with body from template below.

---

## Self-review (plan author)

**Spec coverage:** §8 schemas → Task 1; §9 DB → Task 2–4; §10 bus → Task 5; §11 API → Task 7; §12 UI → Task 8; §14 vision path → Task 9–10; §18 acceptance → Task 12; §19 seed → Task 9. §13 modes 4–7 explicitly deferred.

**Placeholder scan:** No TBD steps.

**Type consistency:** `TemporalHopV1` in schemas; `GrammarTemporalHop` not used. Event kind `atom_emitted` matches ledger branches.

**Naming collision:** `GrammarAtomV1` ≠ `ConceptAtomV1` — documented in Related table.

---

## PR description template (fill at ship time)

```markdown
## Summary
- Adds canonical grammar event schemas and append-only Postgres ledger (`grammar_*` tables) via sql-writer.
- Exposes Substrate Atlas read API under `/api/substrate/atlas/*` and Hub UI at `/substrate-atlas` (Cytoscape trace explorer).
- Includes deterministic vision observation seed script for UI/API development.

## Test plan
- [ ] `PYTHONPATH=. pytest orion/grammar/tests -q`
- [ ] `pytest services/orion-sql-writer/tests/test_grammar_event_routing.py -q`
- [ ] `./scripts/test_service.sh orion-hub tests/test_grammar_atlas_api.py -q`
- [ ] `python scripts/seed_substrate_atlas_demo.py` then manual `/substrate-atlas` smoke
- [ ] Apply `manual_migration_grammar_atlas.sql` on staging DB before deploy

## Notes
- Does not modify `feat/repair-pressure-v1` workspace.
- Coexists with schema_kernel / SubstrateMoleculeV1; no runtime merge yet.
```

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-23-substrate-atlas-grammar-observatory.md`.**

**Worktree is feasible** (`.worktrees/` exists, is gitignored, active branch `feat/repair-pressure-v1` untouched).

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks.
2. **Inline Execution** — execute in this session with `executing-plans`, batched checkpoints.

**Which approach?**
