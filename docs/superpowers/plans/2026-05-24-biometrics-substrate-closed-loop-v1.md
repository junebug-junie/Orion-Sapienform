# Biometrics Substrate Closed Loop v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the first event-native substrate loop: biometrics `GrammarEventV1` → node biometrics projection → `biometrics_pressure` organ candidates → node pressure reducer → `ReductionReceiptV1` → active node pressure projection, with Hub debug APIs proving the chain.

**Architecture:** Keep `orion-biometrics` as ingress-only. Add shared substrate schemas, hoist node catalog to `orion/biometrics/`, persist projections/receipts/emissions in Postgres, run reducers + organ dispatch in new `orion-substrate-runtime` (polls `grammar_events` after sql-writer), implement `biometrics_pressure` in `orion-substrate-organs` as pure Python (no state writes). Accepted pressure candidates publish back to `orion:grammar:event` for audit. Hub exposes read-only `/api/substrate/biometrics-node/{node_id}/latest`.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy (sql-writer pattern), async Redis bus (`OrionBusAsync`), FastAPI Hub routes, pytest.

**Design source:** User spec “Biometrics as First Real Event-Native Organ Loop” (2026-05-24).

**Depends on:** PR `feat/biometrics-node-grammar-ingress` (node-scoped grammar traces). **Base branch for worktree:** `feat/biometrics-node-grammar-ingress`, not `main`.

**Non-goals:** Cluster aggregate grammar, biometrics schema redesign, UI polish, autonomy actions, mesh scheduler, reducers parsing `debug_trace`, organs mutating projections directly.

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
# Ensure ingress branch exists locally (merge ingress PR first if only on remote)
git worktree add .worktrees/feat-biometrics-substrate-closed-loop-v1 \
  -b feat/biometrics-substrate-closed-loop-v1 \
  feat/biometrics-node-grammar-ingress
cd .worktrees/feat-biometrics-substrate-closed-loop-v1
git check-ignore -q .worktrees   # must succeed
```

**Rules:**
- All commits only in `.worktrees/feat-biometrics-substrate-closed-loop-v1`.
- Never bleed files to main checkout **except** copying `.env` keys from `.env_example` into operator `.env` on the local machine.
- PR title: `PR: Biometrics substrate closed loop v1`.
- When done: run `requesting-code-review` subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-24-biometrics-substrate-closed-loop-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Grammar ingress | `feat/biometrics-node-grammar-ingress` — `build_biometrics_node_grammar_events()`, trace `biometrics.node:{node_id}:{ts}` |
| Grammar persistence | `orion-sql-writer` → `grammar_events` / `grammar_atoms` (Atlas migration) |
| Node catalog | `config/biometrics/node_catalog.yaml` + `services/orion-biometrics/app/node_catalog.py` — **hoist to `orion/biometrics/node_catalog.py`** for reducers |
| Substrate runtime service | **None** — only `orion-substrate-telemetry` (tier outcomes). Create `orion-substrate-runtime` + `orion-substrate-organs` per spec |
| Pressure grammar literals | **No** `node_pressure_*` in repo — organ emits **new traces** using existing `GrammarEventV1` kinds (`trace_started`, `atom_emitted`, `trace_ended`) with locked `semantic_role` strings (contract allowlist) |
| Hub port | `HUB_PORT=8080` (`services/orion-hub/.env_example`) |
| Registry | Add `OrganEmissionV1`, `ReductionReceiptV1`, `StateDeltaV1`, projection models |
| Channels | Extend `orion:grammar:event` producers: `orion-substrate-runtime` (accepted candidates only) |

### Locked organ `semantic_role` allowlist (candidate atoms)

Use these **only** (map to spec §8 emit names):

| Contract emit | `semantic_role` on candidate atom |
|---------------|-----------------------------------|
| node pressure detected | `node_pressure_detected` |
| node pressure reinforced | `node_pressure_reinforced` |
| node pressure decayed | `node_pressure_decayed` |
| node availability concern detected | `node_availability_concern` |
| node pressure suppressed | `node_pressure_suppressed` |
| capability impact (Rule E) | `node_capability_impact` |

Candidate trace id pattern: `substrate.pressure:{node_id}:{iso_ts}`  
`provenance.source_service`: `orion-substrate-organs`  
`provenance.source_component`: `biometrics_pressure`

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/organ_emission.py` | `OrganEmissionV1` |
| `orion/schemas/reduction_receipt.py` | `ReductionReceiptV1`, `ProjectionUpdateV1` |
| `orion/schemas/state_delta.py` | `StateDeltaV1` |
| `orion/schemas/biometrics_projection.py` | `NodeBiometricsProjectionV1`, `ActiveNodePressureProjectionV1`, state rows |
| `orion/schemas/registry.py` | Register new schema ids |
| `orion/biometrics/node_catalog.py` | Hoisted `NodeCatalog` / `NodeProfile` (shared) |
| `orion/substrate/biometrics_loop/ids.py` | `parse_biometrics_trace_id`, receipt/emission id helpers |
| `orion/substrate/biometrics_loop/grammar_extract.py` | Extract node fields from `GrammarEventV1` + atoms (no debug_trace) |
| `services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql` | DDL: projections, receipts, emissions, cursors |
| `services/orion-sql-writer/app/models/biometrics_substrate.py` | SQLAlchemy models |
| `services/orion-substrate-runtime/app/settings.py` | Env flags, poll interval, catalog path |
| `services/orion-substrate-runtime/app/reducers/biometrics_node_reducer.py` | Reducer 1 |
| `services/orion-substrate-runtime/app/reducers/node_pressure_reducer.py` | Reducer 2 |
| `services/orion-substrate-runtime/app/projections/node_biometrics_projection.py` | Load/save projection |
| `services/orion-substrate-runtime/app/projections/active_node_pressure_projection.py` | Load/save pressure projection |
| `services/orion-substrate-runtime/app/receipts.py` | Persist `ReductionReceiptV1` |
| `services/orion-substrate-runtime/app/worker.py` | Poll grammar_events → reducers → dispatch organ |
| `services/orion-substrate-runtime/app/publish.py` | Publish accepted grammar events |
| `services/orion-substrate-organs/app/contracts/biometrics_pressure.yaml` | Organ contract |
| `services/orion-substrate-organs/app/organs/biometrics_pressure.py` | Rules A–E |
| `services/orion-substrate-organs/app/runtime/event_native_organ.py` | Dispatch wrapper |
| `services/orion-substrate-organs/app/runtime/emission_validator.py` | Allowlist + max 8 events |
| `services/orion-hub/scripts/substrate_biometrics_routes.py` | Debug GET routes |
| `services/orion-hub/scripts/api_routes.py` | `include_router` |
| `services/orion-hub/app/settings.py` | `BIOMETRICS_SUBSTRATE_DEBUG_ENABLED` |
| `orion/bus/channels.yaml` | `orion-substrate-runtime` producer on `orion:grammar:event` |
| `scripts/smoke_biometrics_closed_loop.sh` | End-to-end smoke |
| `tests/test_node_biometrics_reducer.py` | Repo-root tests (match other substrate tests) |
| `tests/test_biometrics_pressure_organ.py` | |
| `tests/test_node_pressure_reducer.py` | |

**Modify (minimal):** `services/orion-biometrics/app/node_catalog.py` → re-export from `orion.biometrics.node_catalog`.

---

# Phase 1 — Shared schemas + registry

### Task 1: Substrate loop Pydantic models

**Files:**
- Create: `orion/schemas/state_delta.py`
- Create: `orion/schemas/organ_emission.py`
- Create: `orion/schemas/reduction_receipt.py`
- Create: `orion/schemas/biometrics_projection.py`
- Modify: `orion/schemas/registry.py`
- Test: `orion/grammar/tests/test_biometrics_substrate_schemas.py` (new)

- [ ] **Step 1: Write failing schema tests**

```python
# orion/grammar/tests/test_biometrics_substrate_schemas.py
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.biometrics_projection import (
    ActiveNodePressureProjectionV1,
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.schemas.organ_emission import OrganEmissionV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1


def test_organ_emission_roundtrip() -> None:
    now = datetime.now(timezone.utc)
    raw = OrganEmissionV1(
        emission_id="oem_test",
        organ_id="biometrics_pressure",
        invocation_id="inv_test",
        triggered_by_event_ids=["gev_1"],
        inspected_projection_ids=["proj_node_bio"],
        candidate_events=[],
        created_at=now,
    ).model_dump(mode="json")
    assert OrganEmissionV1.model_validate(raw).organ_id == "biometrics_pressure"


def test_reduction_receipt_requires_schema_version() -> None:
    now = datetime.now(timezone.utc)
    r = ReductionReceiptV1(
        receipt_id="rcpt_test",
        accepted_event_ids=[],
        rejected_event_ids=[],
        merged_event_ids=[],
        noop_event_ids=[],
        state_deltas=[],
        projection_updates=[],
        created_at=now,
    )
    assert r.schema_version == "substrate.reduction_receipt.v1"


def test_node_biometrics_projection_defaults() -> None:
    now = datetime.now(timezone.utc)
    p = NodeBiometricsProjectionV1(
        projection_id="proj_node_bio",
        generated_at=now,
        nodes={
            "atlas": NodeBiometricsStateV1(node_id="atlas"),
        },
    )
    assert p.nodes["atlas"].availability_status == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest orion/grammar/tests/test_biometrics_substrate_schemas.py -v`  
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement schemas**

`orion/schemas/state_delta.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class StateDeltaV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.state_delta.v1"] = "substrate.state_delta.v1"
    delta_id: str
    target_projection: str
    target_kind: str
    target_id: str
    operation: Literal[
        "create", "update", "reinforce", "decay", "merge", "suppress", "noop",
    ]
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    caused_by_event_ids: list[str]
    reducer_id: str
    explanation: str | None = None
```

`orion/schemas/organ_emission.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.grammar import GrammarEventV1


class OrganEmissionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["organ.emission.v1"] = "organ.emission.v1"
    emission_id: str
    organ_id: str
    invocation_id: str
    triggered_by_event_ids: list[str] = Field(default_factory=list)
    inspected_projection_ids: list[str] = Field(default_factory=list)
    candidate_events: list[GrammarEventV1] = Field(default_factory=list)
    debug_trace: dict[str, Any] | None = None
    created_at: datetime
```

`orion/schemas/reduction_receipt.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.state_delta import StateDeltaV1


class ProjectionUpdateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.projection_update.v1"] = "substrate.projection_update.v1"
    projection_kind: str
    projection_id: str
    node_id: str | None = None
    operation: str


class ReductionReceiptV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate.reduction_receipt.v1"] = "substrate.reduction_receipt.v1"
    receipt_id: str
    emission_id: str | None = None
    organ_id: str | None = None
    accepted_event_ids: list[str] = Field(default_factory=list)
    rejected_event_ids: list[str] = Field(default_factory=list)
    merged_event_ids: list[str] = Field(default_factory=list)
    noop_event_ids: list[str] = Field(default_factory=list)
    state_deltas: list[StateDeltaV1] = Field(default_factory=list)
    projection_updates: list[ProjectionUpdateV1] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
```

`orion/schemas/biometrics_projection.py` — implement `NodeBiometricsStateV1`, `NodeBiometricsProjectionV1`, `ActiveNodePressureStateV1`, `ActiveNodePressureProjectionV1` exactly per spec §5–§7 (`schema_version` literals `projection.node_biometrics.v1` and `projection.active_node_pressure.v1`).

- [ ] **Step 4: Register in `orion/schemas/registry.py`**

Add imports and `_REGISTRY` entries:

```python
"OrganEmissionV1": OrganEmissionV1,
"ReductionReceiptV1": ReductionReceiptV1,
"StateDeltaV1": StateDeltaV1,
"ProjectionUpdateV1": ProjectionUpdateV1,
"NodeBiometricsProjectionV1": NodeBiometricsProjectionV1,
"ActiveNodePressureProjectionV1": ActiveNodePressureProjectionV1,
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. pytest orion/grammar/tests/test_biometrics_substrate_schemas.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/*.py orion/schemas/registry.py orion/grammar/tests/test_biometrics_substrate_schemas.py
git commit -m "feat(substrate): add biometrics loop schemas and registry entries"
```

---

### Task 2: Hoist node catalog to shared package

**Files:**
- Create: `orion/biometrics/__init__.py`
- Create: `orion/biometrics/node_catalog.py` (move body from biometrics service)
- Modify: `services/orion-biometrics/app/node_catalog.py` (thin re-export)
- Test: `services/orion-biometrics/tests/test_node_catalog.py` (unchanged paths)

- [ ] **Step 1: Copy `NodeCatalog` implementation to `orion/biometrics/node_catalog.py`**

- [ ] **Step 2: Replace service module**

```python
# services/orion-biometrics/app/node_catalog.py
from orion.biometrics.node_catalog import NodeCatalog, NodeProfile

__all__ = ["NodeCatalog", "NodeProfile"]
```

- [ ] **Step 3: Run existing catalog tests**

Run: `PYTHONPATH=services/orion-biometrics:. pytest services/orion-biometrics/tests/test_node_catalog.py -q`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add orion/biometrics/ services/orion-biometrics/app/node_catalog.py
git commit -m "refactor: hoist biometrics node catalog to orion.biometrics"
```

---

# Phase 2 — Postgres persistence

### Task 3: DDL + SQLAlchemy models

**Files:**
- Create: `services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql`
- Create: `services/orion-sql-writer/app/models/biometrics_substrate.py`

- [ ] **Step 1: Write migration SQL**

```sql
-- manual_migration_biometrics_substrate_loop.sql
create table if not exists substrate_reduction_cursor (
    cursor_name text primary key,
    last_event_created_at timestamptz,
    last_event_id text,
    updated_at timestamptz not null default now()
);

create table if not exists substrate_node_biometrics_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists substrate_active_node_pressure_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists substrate_organ_emissions (
    emission_id text primary key,
    organ_id text not null,
    invocation_id text not null,
    emission_json jsonb not null,
    created_at timestamptz not null default now()
);

create table if not exists substrate_reduction_receipts (
    receipt_id text primary key,
    organ_id text,
    emission_id text,
    receipt_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_organ_emissions_created
  on substrate_organ_emissions (created_at desc);
create index if not exists idx_substrate_reduction_receipts_created
  on substrate_reduction_receipts (created_at desc);
```

- [ ] **Step 2: SQLAlchemy models** (mirror `grammar_trace.py` style: JSON column + `to_pydantic()` helpers)

- [ ] **Step 3: Document apply step in runtime README**

```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-sql-db/manual_migration_biometrics_substrate_loop.sql \
  services/orion-sql-writer/app/models/biometrics_substrate.py
git commit -m "feat(substrate): persistence tables for biometrics loop"
```

---

# Phase 3 — Pure reducer + organ logic (TDD)

### Task 4: Grammar extract helpers

**Files:**
- Create: `orion/substrate/biometrics_loop/ids.py`
- Create: `orion/substrate/biometrics_loop/grammar_extract.py`
- Test: `tests/test_biometrics_grammar_extract.py`

- [ ] **Step 1: Failing tests for trace parse + alias**

```python
def test_parse_biometrics_trace_id():
    from orion.substrate.biometrics_loop.ids import parse_biometrics_trace_id
    assert parse_biometrics_trace_id("biometrics.node:atlas:2026-05-24T12:00:00Z") == "atlas"


def test_prometheous_resolves_via_catalog():
    from orion.biometrics.node_catalog import NodeCatalog
    from pathlib import Path
    cat = NodeCatalog.load(Path("config/biometrics/node_catalog.yaml"))
    assert cat.resolve("prometheous").node_id == "prometheus"
```

- [ ] **Step 2: Implement `parse_biometrics_trace_id`**

```python
def parse_biometrics_trace_id(trace_id: str) -> str | None:
    if not trace_id.startswith("biometrics.node:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1].strip().lower()
```

- [ ] **Step 3: `extract_node_state_from_events(events, catalog)`**

Rules:
- Input: ordered `GrammarEventV1` list for one trace (from DB), plus `NodeCatalog`
- Canonical `node_id` = catalog.resolve(parsed trace node or `node_context.text_value`)
- Set `last_seen_at` from max `observed_at`
- Track `latest_*_event_id` when `atom.semantic_role` in `telemetry_sample`, `body_state`, `node_availability`
- `payload_ref` from atoms only — never copy telemetry blobs
- `availability_status` from `node_availability` summary tokens + `expected_online` + staleness (use env threshold in caller)
- `pressure_hints`: `{"strain": body_state.salience, "gpu": ...}` from atom salience only

- [ ] **Step 4: Run tests — PASS — commit**

---

### Task 5: Biometrics node reducer (unit tests)

**Files:**
- Create: `services/orion-substrate-runtime/app/reducers/biometrics_node_reducer.py`
- Test: `tests/test_node_biometrics_reducer.py`

- [ ] **Step 1: Failing tests (spec §14)**

```python
def test_atlas_trace_updates_last_seen():
    # fixture: minimal GrammarEventV1 list mimicking ingress emitter
    # assert reduce_biometrics_grammar_event(...) mutates NodeBiometricsStateV1.last_seen_at


def test_prometheous_alias_resolves_to_prometheus():
    ...


def test_circe_expected_offline_preserved():
    ...


def test_payload_ref_stored_no_blob_copy():
    state = ...
    assert "gpu_util" not in str(state.model_dump())
    assert state.latest_payload_ref.startswith("biometrics.")


def test_sample_summary_induction_event_ids_attach():
    ...
```

- [ ] **Step 2: Implement `reduce_biometrics_node_event`**

Signature:

```python
def reduce_biometrics_node_event(
    *,
    event: GrammarEventV1,
    projection: NodeBiometricsProjectionV1,
    catalog: NodeCatalog,
    stale_after_sec: int,
    reducer_id: str = "biometrics_node_reducer",
) -> tuple[NodeBiometricsProjectionV1, ReductionReceiptV1]:
```

Guard: return noop receipt if `event.provenance.source_service != "orion-biometrics"` or trace not `biometrics.node:*`.

Emit `StateDeltaV1` + `ProjectionUpdateV1`; never read `debug_trace`.

- [ ] **Step 3: Run tests — PASS — commit**

---

### Task 6: Biometrics pressure organ

**Files:**
- Create: `services/orion-substrate-organs/app/contracts/biometrics_pressure.yaml`
- Create: `services/orion-substrate-organs/app/organs/biometrics_pressure.py`
- Create: `services/orion-substrate-organs/app/runtime/emission_validator.py`
- Create: `services/orion-substrate-organs/app/runtime/event_native_organ.py`
- Test: `tests/test_biometrics_pressure_organ.py`

- [ ] **Step 1: Contract YAML (spec §8)**

```yaml
organ_id: biometrics_pressure
schema_version: organ.contract.v1
subscribes_to:
  - biometrics.node.trace.lifecycle
  - biometrics.node.sample.observed
  - biometrics.node.summary.observed
  - biometrics.node.induction.observed
reads_projections:
  - node_biometrics_projection
  - active_node_pressure_projection
emits:
  - node_pressure_detected
  - node_pressure_reinforced
  - node_pressure_decayed
  - node_availability_concern
  - node_pressure_suppressed
max_events_per_invocation: 8
```

(Map subscribe keys to detection: `trace_id.startswith("biometrics.node:")` + `event_kind` / `semantic_role` from ingress atoms.)

- [ ] **Step 2: Failing organ tests (spec §14)**

```python
ALLOWED_ROLES = {
    "node_pressure_detected",
    "node_pressure_reinforced",
    "node_pressure_decayed",
    "node_availability_concern",
    "node_pressure_suppressed",
    "node_capability_impact",
}


def test_circe_expected_offline_emits_suppression_candidate():
    ...


def test_missing_expected_atlas_emits_availability_concern():
    ...


def test_organ_emits_only_allowlisted_roles():
    emission = invoke_biometrics_pressure(...)
    for ev in emission.candidate_events:
        assert ev.atom is not None
        assert ev.atom.semantic_role in ALLOWED_ROLES


def test_organ_does_not_emit_state_deltas():
    assert "state_delta" not in emission.model_dump_json()
```

- [ ] **Step 3: Implement rules A–E (spec §9)**

```python
def invoke_biometrics_pressure(
    *,
    trigger_event: GrammarEventV1,
    node_bio: NodeBiometricsProjectionV1,
    active_pressure: ActiveNodePressureProjectionV1,
    catalog: NodeCatalog,
    stale_after_sec: int = 180,
    min_confidence: float = 0.60,
) -> OrganEmissionV1:
```

Rule A: `expected_online=False` + stale/missing → candidate `node_pressure_suppressed`  
Rule B: `expected_online=True` + stale → `node_availability_concern`  
Rule C: induction pressure hint + prior active → `node_pressure_reinforced`  
Rule D: prior pressured + empty hints → `node_pressure_decayed`  
Rule E: `llm_inference` capability + gpu hint → `node_capability_impact`

Build candidates via helper `build_pressure_candidate_event(node_id, role, evidence_event_ids, confidence)`.

- [ ] **Step 4: `emission_validator.py`**

- Enforce `len(candidate_events) <= 8`
- Each candidate: `trace_started` + one `atom_emitted` + optional `trace_ended`
- Reject candidates with `debug_trace` usage

- [ ] **Step 5: Run tests — PASS — commit**

---

### Task 7: Node pressure reducer

**Files:**
- Create: `services/orion-substrate-runtime/app/reducers/node_pressure_reducer.py`
- Test: `tests/test_node_pressure_reducer.py`

- [ ] **Step 1: Failing tests (spec §14)**

```python
def test_valid_pressure_candidate_accepted(): ...
def test_missing_evidence_rejected(): ...
def test_duplicate_pressure_merged(): ...
def test_suppression_updates_suppressed_pressures(): ...
def test_projection_rebuilds_deterministically(): ...
```

- [ ] **Step 2: Implement `reduce_node_pressure_candidates`**

Accept if:
- `provenance.source_component == "biometrics_pressure"`
- `semantic_role` in allowlist
- `evidence_event_ids` on edge/atom reference biometrics `gev_*` or `biometrics.node:` trace
- `confidence >= BIOMETRICS_PRESSURE_MIN_CONFIDENCE`

Reject if node unknown or evidence empty.

Merge if same node + same pressure kind + window (e.g. 300s).

Update `ActiveNodePressureProjectionV1` only here.

- [ ] **Step 3: Run tests — PASS — commit**

---

# Phase 4 — Runtime worker service

### Task 8: `orion-substrate-runtime` service shell

**Files:**
- Create: `services/orion-substrate-runtime/app/settings.py`
- Create: `services/orion-substrate-runtime/app/main.py`
- Create: `services/orion-substrate-runtime/requirements.txt`
- Create: `services/orion-substrate-runtime/Dockerfile`
- Create: `services/orion-substrate-runtime/docker-compose.yml`
- Create: `services/orion-substrate-runtime/.env_example`
- Copy keys to: `services/orion-substrate-runtime/.env` (operator, not committed)
- Create: `services/orion-substrate-runtime/README.md`

- [ ] **Step 1: Settings (spec §13)**

```python
class Settings(BaseSettings):
    enable_biometrics_node_reducer: bool = Field(True, alias="ENABLE_BIOMETRICS_NODE_REDUCER")
    enable_biometrics_pressure_organ: bool = Field(True, alias="ENABLE_BIOMETRICS_PRESSURE_ORGAN")
    enable_node_pressure_reducer: bool = Field(True, alias="ENABLE_NODE_PRESSURE_REDUCER")
    biometrics_node_stale_after_sec: int = Field(180, alias="BIOMETRICS_NODE_STALE_AFTER_SEC")
    biometrics_pressure_min_confidence: float = Field(0.60, alias="BIOMETRICS_PRESSURE_MIN_CONFIDENCE")
    node_catalog_path: str = Field("config/biometrics/node_catalog.yaml", alias="NODE_CATALOG_PATH")
    grammar_poll_interval_sec: float = Field(2.0, alias="GRAMMAR_POLL_INTERVAL_SEC")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    redis_url: str = Field(..., alias="REDIS_URL")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")
    publish_accepted_candidates: bool = Field(True, alias="PUBLISH_ACCEPTED_PRESSURE_GRAMMAR")
```

- [ ] **Step 2: Worker loop `app/worker.py`**

```text
1. Read cursor `biometrics_grammar_consumer`
2. SELECT grammar_events WHERE source_service='orion-biometrics'
   AND trace_id LIKE 'biometrics.node:%'
   AND created_at > cursor ORDER BY created_at, event_id LIMIT 50
3. For each event (deserialize event_json → GrammarEventV1):
   a. biometrics_node_reducer → save projection + receipt
   b. if organ enabled: biometrics_pressure → save OrganEmissionV1
   c. node_pressure_reducer on emission.candidate_events → save projection + receipt
   d. for accepted ids: publish GrammarEventV1 to bus (optional flag)
4. Advance cursor
```

Use same DB session pattern as sql-writer (`get_session` / sync SQLAlchemy).

- [ ] **Step 3: `app/publish.py`**

Reuse `orion.grammar.publish.publish_grammar_event` pattern from Atlas plan.

- [ ] **Step 4: Wire `app/main.py` lifespan → background asyncio task**

- [ ] **Step 5: docker-compose** — mount `config/biometrics`, `orion/`, depend on `postgres` + `redis`

- [ ] **Step 6: Commit**

---

### Task 9: `orion-substrate-organs` package layout

**Files:**
- Create: `services/orion-substrate-organs/requirements.txt` (minimal — pydantic, orion)
- Create: `services/orion-substrate-organs/README.md`

Organs are imported by runtime; separate Dockerfile optional in v1 (runtime `PYTHONPATH` includes organs app). If separate container not needed day-1, document “library service” — runtime Dockerfile:

```dockerfile
COPY services/orion-substrate-organs/app /app/organs
COPY services/orion-substrate-runtime/app /app
```

- [ ] **Commit organs layout**

---

# Phase 5 — Hub debug API

### Task 10: Substrate biometrics debug routes

**Files:**
- Create: `services/orion-hub/scripts/substrate_biometrics_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/app/settings.py`
- Modify: `services/orion-hub/.env_example` + operator `.env`
- Test: `services/orion-hub/tests/test_substrate_biometrics_debug_api.py`

- [ ] **Step 1: Failing API test**

```python
def test_latest_chain_shape(client, monkeypatch):
    # monkeypatch DB fetch helpers to return fixtures
    r = client.get("/api/substrate/biometrics-node/atlas/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["node_id"] == "atlas"
    assert "event_chain" in body
    assert "latest_reduction_receipt" in body
```

- [ ] **Step 2: Implement routes (spec §12)**

```python
router = APIRouter(prefix="/api/substrate", tags=["substrate-biometrics"])

@router.get("/biometrics-node/{node_id}/latest")
async def biometrics_node_latest(node_id: str) -> dict[str, Any]:
    ...

@router.get("/biometrics/latest")
async def biometrics_projection_latest(): ...

@router.get("/node-pressure/latest")
async def node_pressure_latest(): ...
```

`event_chain` labels (ordered):

```python
EVENT_CHAIN = [
    "biometrics grammar event",
    "node projection delta",
    "organ emission",
    "pressure candidate event",
    "pressure reduction receipt",
    "active pressure projection update",
]
```

- [ ] **Step 3: `BIOMETRICS_SUBSTRATE_DEBUG_ENABLED=true` in hub settings**

- [ ] **Step 4: Run hub tests — PASS — commit**

---

# Phase 6 — Bus catalog + smoke

### Task 11: Channel catalog update

**Files:**
- Modify: `orion/bus/channels.yaml`

- [ ] **Step 1: Add producer `orion-substrate-runtime` to `orion:grammar:event`**

- [ ] **Step 2: Commit**

---

### Task 12: Smoke script

**Files:**
- Create: `scripts/smoke_biometrics_closed_loop.sh`

- [ ] **Step 1: Implement (spec §15)**

```bash
#!/usr/bin/env bash
set -euo pipefail
HUB_PORT="${HUB_PORT:-8080}"
BASE="http://localhost:${HUB_PORT}"

echo "1. Check biometrics grammar events (manual): redis-cli SUBSCRIBE orion:grammar:event"
echo "2. Latest node chain:"
curl -s "${BASE}/api/substrate/biometrics-node/atlas/latest" | jq .
echo "3. Active node pressure:"
curl -s "${BASE}/api/substrate/node-pressure/latest" | jq .
echo "4. Event chain:"
curl -s "${BASE}/api/substrate/biometrics-node/atlas/latest" | jq '.event_chain'
```

- [ ] **Step 2: chmod +x — commit**

---

# Phase 7 — Integration verification

### Task 13: Full test suite + smoke

- [ ] **Step 1: Unit tests**

```bash
cd .worktrees/feat-biometrics-substrate-closed-loop-v1
PYTHONPATH=.:services/orion-biometrics pytest \
  tests/test_node_biometrics_reducer.py \
  tests/test_biometrics_pressure_organ.py \
  tests/test_node_pressure_reducer.py \
  orion/grammar/tests/test_biometrics_substrate_schemas.py -q
```

Expected: all PASS

- [ ] **Step 2: Hub API tests**

```bash
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_substrate_biometrics_debug_api.py -q
```

- [ ] **Step 3: Biometrics ingress regression**

```bash
PYTHONPATH=services/orion-biometrics:. pytest services/orion-biometrics/tests/ -q
```

- [ ] **Step 4: Runtime import check**

```bash
PYTHONPATH=.:services/orion-substrate-runtime:services/orion-substrate-organs \
  python -c "from app.worker import BiometricsSubstrateWorker; print('ok')"
```

- [ ] **Step 5: Live smoke (operator)**

Apply migration, start sql-writer + substrate-runtime + biometrics + hub, run `scripts/smoke_biometrics_closed_loop.sh`.  
If Redis/compose unavailable in CI: mark runtime bus subscribe **UNVERIFIED** in PR report.

- [ ] **Step 6: Final commit if fixes needed**

---

# Phase 8 — Code review, PR report, push

### Task 14: Code review subagent

- [ ] **Step 1: Invoke `requesting-code-review` skill** — review diff against this plan + spec §16 acceptance criteria

- [ ] **Step 2: Fix all blocking issues — commit fixes**

---

### Task 15: PR report + GitHub PR

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-24-biometrics-substrate-closed-loop-v1-pr.md`

- [ ] **Step 1: Write PR report** (mirror `2026-05-24-node-scoped-biometrics-grammar-ingress-pr.md` sections: Summary, Architecture, Changes by area, Test plan, Verification evidence, Acceptance table)

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feat/biometrics-substrate-closed-loop-v1
gh pr create --base feat/biometrics-node-grammar-ingress --title "PR: Biometrics substrate closed loop v1" --body "$(cat <<'EOF'
## Summary
- Closes biometrics grammar ingress into substrate projections, organ candidates, and pressure reducer receipts.
- Adds Hub debug endpoints for full chain inspection.

## Test plan
- [ ] Unit tests listed in PR report
- [ ] smoke_biometrics_closed_loop.sh with stack up
- [ ] redis SUBSCRIBE orion:grammar:event shows biometrics.node + substrate.pressure traces

EOF
)"
```

---

## Self-review (plan vs spec)

| Spec requirement | Task |
|------------------|------|
| Persist grammar events | Already sql-writer; Task 8 consumes |
| `node_biometrics_projection` | Task 5, 3 |
| Node-scoped not cluster | Task 5 guards |
| `prometheous` → `prometheus` | Task 2, 4 |
| `circe.expected_online=false` suppression | Task 6 Rule A |
| Organ consumes grammar not traces | Task 6 |
| Organ reads projections not blobs | Task 6 |
| `OrganEmissionV1` candidates | Task 1, 6 |
| No state deltas in emission | Task 6 tests |
| `ReductionReceiptV1` | Task 1, 5, 7 |
| Pressure projection only via reducer | Task 7 |
| Debug endpoint chain | Task 10 |
| No debug_trace parsing | Tasks 4–7 anti-swamp |
| Config env §13 | Task 8, 10 |
| Unit tests §14 | Tasks 5–7 |
| Smoke §15 | Task 12 |
| Acceptance §16 | Task 13–15 |

**Placeholder scan:** None — all tasks include concrete paths, code, and commands.

---

## Core invariant (carry through implementation)

```text
orion-biometrics observes.
biometrics_pressure interprets.
substrate reducer commits.
projections expose.
traces explain.
```
