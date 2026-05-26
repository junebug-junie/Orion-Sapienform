# Orion Bus Transport Full Substrate Integration v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish `orion-bus` substrate onboarding from trace-only (PR #630) through Layer 11 scaffolding: `bus.transport` traces → `transport_bus` reducer → field perturbation → attention visibility, with gated self/proposal/policy/dispatch/feedback/consolidation (defaults off).

**Architecture:** Mirror `orion/substrate/execution_loop/` for transport: bounded grammar rollups → `TransportBusStateV1` projection → `StateDeltaV1(target_kind=transport_bus)` → existing `substrate_reduction_receipts`. Field digester consumes receipts only (never `grammar_events`). Layers 6–11 extend existing frame runtimes via YAML policy + feature flags (same pattern as execution/biometrics frames on `main`).

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy/psycopg2, PyYAML, pytest, Docker Compose, Postgres (`grammar_events`, `substrate_*` tables).

**Design source:** User monster spec “Orion Bus Full Substrate Integration — Transport Legibility Through Layers 3–11” (2026-05-25). **This document is the controlling implementation spec** when it conflicts with context docs.

**Depends on:** PR #630 merged on `main` (`orion-bus` bus-observer → `bus.transport:*` → `grammar_events`).

**Non-goals:** Packet logging, raw Redis payloads, bus restart/replay/purge, catalog writes, central emitter, field reading `grammar_events`, automatic policy mutation, habit execution.

---

## Required context

Before implementing, read the substrate trace context pack:

```text
docs/context-engineering/README.md
docs/context-engineering/00_substrate_trace_doctrine.md
docs/context-engineering/01_service_taxonomy.md
docs/context-engineering/03_substrate_trace_emitter_pattern.md
docs/context-engineering/04_layer_1_to_11_pipeline.md
docs/context-engineering/05_reducer_and_state_delta_pattern.md
docs/context-engineering/06_frame_runtime_pattern.md
docs/context-engineering/08_testing_and_live_proof.md
```

Then read the service-local context:

```text
services/orion-bus/AGENT_CONTEXT.md
services/orion-bus/SERVICE_PORTS.yaml
services/orion-bus/SUBSTRATE_TRACE_MAP.md
services/orion-bus/LAYER_PIPELINE_PLAN.md
```

The documents provide doctrine and conventions. **This prompt is the controlling implementation spec.** If any doc conflicts with this plan, follow this plan and document the conflict in the PR report.

---

## Worktree and branch hygiene (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-orion-bus-transport-substrate-full-stack-v1 \
  -b feat/orion-bus-transport-substrate-full-stack-v1 origin/main
cd .worktrees/feat-orion-bus-transport-substrate-full-stack-v1
git check-ignore -q .worktrees   # must succeed
```

**Rules:**

- All implementation commits happen **only** inside `.worktrees/feat-orion-bus-transport-substrate-full-stack-v1`.
- Do **not** copy changed files back to the main workspace checkout except syncing operator `.env` from `.env_example` on the local machine (`.env` is gitignored).
- When updating any `services/*/.env_example`, also append/sync keys into the matching `services/*/.env` locally (not committed).
- PR branch: `feat/orion-bus-transport-substrate-full-stack-v1` → `main`.
- When done: run **requesting-code-review** subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-25-orion-bus-transport-full-stack-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-25)

| Question | Finding |
|----------|---------|
| Layer 1–2 (traces) | **Done** — PR #630: `services/orion-bus/app/grammar_emit.py`, `bus_observer.py`, `trace_id=bus.transport:{node}:{window}` |
| Live trace roles | `bus_observer_tick_started`, `bus_health_observed`, `bus_stream_depth_observed`, `bus_configured_stream_uncataloged`, `bus_observer_tick_completed` |
| Reducer package | **Missing** — create `orion/substrate/transport_loop/` (template: `execution_loop/`) |
| Substrate runtime | `services/orion-substrate-runtime` — biometrics + execution ticks in `worker.py`; add isolated `transport_substrate_tick` |
| Field digester | Polls `substrate_reduction_receipts` only — extend `delta_to_perturbations` for `transport_bus`; add `ENABLE_TRANSPORT_FIELD_DIGESTION` |
| Field lattice | Canonical: `config/field/orion_field_topology.v1.yaml` (`biometrics_lattice.yaml` is alias) — add `capability:transport` |
| Frame runtimes | Layers 5–11 services exist on `main` — extend policies/settings/tests; **no new bus publish** for consolidation |
| `proposal_frame` `target_kind` | No `config` literal — use `system` + `target_id: orion/bus/channels.yaml` for catalog inspect |
| Delta IDs | Use `orion.substrate.ids.stable_delta_id` (repo convention), not ad-hoc sha1 helper |
| Depth critical threshold | Align with bus observer: `BUS_STREAM_DEPTH_CRITICAL` default `100000` |

### Live expected reducer output (athena, catalog drift only)

```json
{
  "bus_health": 1.0,
  "delivery_confidence": 1.0,
  "stream_depth_pressure": 0.0,
  "backpressure": 0.0,
  "catalog_drift_pressure": 1.0,
  "observer_failure_pressure": 0.0,
  "transport_pressure": 0.0,
  "contract_pressure": 1.0,
  "reliability_pressure": 0.0
}
```

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/transport_projection.py` | `TransportBusStateV1`, `TransportBusProjectionV1` |
| `orion/schemas/registry.py` | Register transport projection schemas |
| `orion/substrate/transport_loop/constants.py` | Cursor names, reducer id, source service |
| `orion/substrate/transport_loop/extract.py` | Parse bus transport atoms → rollup state |
| `orion/substrate/transport_loop/reducer.py` | Pressure rules → `StateDeltaV1` |
| `orion/substrate/transport_loop/pipeline.py` | Group by trace, persist projection + receipts |
| `orion/substrate/transport_loop/__init__.py` | Package exports |
| `services/orion-sql-db/manual_migration_transport_substrate_loop.sql` | Projection + cursor tables |
| `services/orion-substrate-runtime/app/store.py` | fetch/load/save transport projection + grammar cursor |
| `services/orion-substrate-runtime/app/worker.py` | `transport_substrate_tick` (isolated try/except) |
| `services/orion-substrate-runtime/app/settings.py` | `ENABLE_TRANSPORT_BUS_REDUCER`, depth critical |
| `services/orion-field-digester/app/ingest/state_deltas.py` | `transport_bus` perturbation mapping |
| `services/orion-field-digester/app/tensor/channels.py` | Transport node/capability channels |
| `services/orion-field-digester/app/settings.py` | `ENABLE_TRANSPORT_FIELD_DIGESTION` |
| `config/field/orion_field_topology.v1.yaml` | `capability:transport` + edges |
| `config/attention/field_attention_policy.v1.yaml` | Track `capability:transport` |
| `config/self_state/self_state_policy.v1.yaml` | `transport_integrity` dimension (gated) |
| `config/proposals/proposal_policy.v1.yaml` | Transport read-only templates |
| `config/policy/substrate_policy.v1.yaml` | Transport gate rules |
| `config/execution_dispatch/execution_dispatch_policy.v1.yaml` | Dry-run transport constraints |
| `config/feedback/feedback_policy.v1.yaml` | Transport outcome observations |
| `config/consolidation/consolidation_policy.v1.yaml` | Transport motif rules |
| `orion/proposals/templates.py` | Transport template copy |
| `orion/consolidation/motif.py` | Transport motif detectors (or dedicated module) |
| `scripts/smoke_orion_bus_transport_full_stack.sh` | m3/m4/m5/full-observe smoke |
| `tests/test_transport_*.py` | Per-layer unit tests (spec list) |
| `docs/superpowers/pr-reports/2026-05-25-orion-bus-transport-full-stack-v1-pr.md` | Post-implementation PR report |

**Do not modify:** `orion-bus` bus-core/bus-exporter images, bus-tap, bus-mirror packet paths.

---

## Effect gates (defaults — all false unless noted)

| Layer | Env var | Default |
|-------|---------|---------|
| 3 Reducer | `ENABLE_TRANSPORT_BUS_REDUCER` | `false` |
| 4 Field | `ENABLE_TRANSPORT_FIELD_DIGESTION` | `false` |
| 5 Attention | `ENABLE_TRANSPORT_ATTENTION_VISIBILITY` | `false` |
| 6 Self-state | `ENABLE_TRANSPORT_SELF_STATE_INFLUENCE` | `false` |
| 7 Proposal | `ENABLE_TRANSPORT_PROPOSALS` | `false` |
| 7 Proposal | `TRANSPORT_PROPOSAL_MODE` | `read_only` |
| 8 Policy | `ENABLE_TRANSPORT_POLICY_GATES` | `false` |
| 8 Policy | `TRANSPORT_POLICY_MAX_SCOPE` | `read_only` |
| 9 Dispatch | `ENABLE_TRANSPORT_DISPATCH_DRY_RUN` | `false` |
| 9 Dispatch | `TRANSPORT_DISPATCH_MODE` | `dry_run` |
| 10 Feedback | `ENABLE_TRANSPORT_FEEDBACK` | `false` |
| 11 Consolidation | `ENABLE_TRANSPORT_CONSOLIDATION` | `false` |
| Debug | `TRANSPORT_SUBSTRATE_MATURITY` | `trace_only` (logging only) |

---

# Phase 1 — Transport projection schemas + registry

### Task 1: Transport projection models

**Files:**
- Create: `orion/schemas/transport_projection.py`
- Modify: `orion/schemas/registry.py`
- Test: `tests/test_transport_projection_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transport_projection_schemas.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1

NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)


def test_transport_bus_state_defaults() -> None:
    state = TransportBusStateV1(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="20260525T233010Z",
        source_trace_id="bus.transport:athena:20260525T233010Z",
        bus_health=1.0,
        delivery_confidence=1.0,
        catalog_drift_pressure=1.0,
        contract_pressure=1.0,
    )
    assert state.schema_version == "transport_bus.state.v1"


def test_transport_bus_state_rejects_out_of_range_pressure() -> None:
    with pytest.raises(ValidationError):
        TransportBusStateV1(
            target_id="bus:athena",
            node_id="athena",
            sample_window_id="w",
            source_trace_id="t",
            transport_pressure=1.5,
        )


def test_transport_bus_projection_roundtrip() -> None:
    state = TransportBusStateV1(
        target_id="bus:athena",
        node_id="athena",
        sample_window_id="20260525T233010Z",
        source_trace_id="bus.transport:athena:20260525T233010Z",
    )
    proj = TransportBusProjectionV1(updated_at=NOW, buses={"bus:athena": state})
    raw = proj.model_dump(mode="json")
    assert TransportBusProjectionV1.model_validate(raw).buses["bus:athena"].node_id == "athena"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_transport_projection_schemas.py -v`  
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement schemas**

```python
# orion/schemas/transport_projection.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransportBusStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["transport_bus.state.v1"] = "transport_bus.state.v1"

    target_id: str
    node_id: str

    sample_window_id: str
    source_trace_id: str

    redis_ping_ok: bool | None = None

    streams_observed: int = 0
    total_stream_depth: int = 0
    max_stream_depth: int = 0

    uncataloged_stream_count: int = 0
    backpressure_count: int = 0
    observer_failure_count: int = 0

    bus_health: float = Field(ge=0.0, le=1.0, default=0.5)
    delivery_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    stream_depth_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    backpressure: float = Field(ge=0.0, le=1.0, default=0.0)
    catalog_drift_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    observer_failure_pressure: float = Field(ge=0.0, le=1.0, default=0.0)

    transport_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    contract_pressure: float = Field(ge=0.0, le=1.0, default=0.0)
    reliability_pressure: float = Field(ge=0.0, le=1.0, default=0.0)

    evidence_event_ids: list[str] = Field(default_factory=list)
    observed_at: datetime | None = None


class TransportBusProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["transport_bus.projection.v1"] = "transport_bus.projection.v1"

    projection_id: str = "active_transport_bus_projection"
    updated_at: datetime
    buses: dict[str, TransportBusStateV1] = Field(default_factory=dict)
```

Register in `orion/schemas/registry.py` (follow `ExecutionTrajectoryProjectionV1` pattern):

```python
from orion.schemas.transport_projection import TransportBusProjectionV1, TransportBusStateV1
# ... add to SCHEMA_REGISTRY / exports alongside execution_projection
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_transport_projection_schemas.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/transport_projection.py orion/schemas/registry.py tests/test_transport_projection_schemas.py
git commit -m "feat: add transport bus projection schemas"
```

---

# Phase 2 — Transport loop extract + reducer

### Task 2: Constants and trace parsing

**Files:**
- Create: `orion/substrate/transport_loop/constants.py`
- Create: `orion/substrate/transport_loop/__init__.py`

```python
# orion/substrate/transport_loop/constants.py
TRANSPORT_BUS_PROJECTION_ID = "active_transport_bus_projection"
TRANSPORT_GRAMMAR_CURSOR_NAME = "transport_grammar_reducer"
TRANSPORT_SOURCE_SERVICE = "orion-bus"
TRANSPORT_TRACE_PREFIX = "bus.transport:"
TRANSPORT_REDUCER_ID = "transport_bus_reducer"
DEFAULT_STREAM_DEPTH_CRITICAL = 100_000
```

### Task 3: Grammar extract from bus transport atoms

**Files:**
- Create: `orion/substrate/transport_loop/extract.py`
- Test: `tests/test_transport_substrate_reducer.py` (extract section first)

- [ ] **Step 1: Write failing extract tests**

```python
# tests/test_transport_substrate_reducer.py (extract section)
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.transport_loop.extract import (
    compute_transport_pressures,
    extract_transport_bus_state_from_events,
    parse_bus_transport_trace_id,
)

NOW = datetime(2026, 5, 25, 23, 30, 10, tzinfo=timezone.utc)
TRACE = "bus.transport:athena:20260525T233010Z"


def _prov() -> GrammarProvenanceV1:
    return GrammarProvenanceV1(
        source_service="orion-bus",
        source_component="bus_transport_grammar_emit",
        source_event_id="20260525T233010Z",
    )


def _atom(role: str, summary: str) -> GrammarAtomV1:
    return GrammarAtomV1(
        atom_id=f"{TRACE}:{role}",
        trace_id=TRACE,
        atom_type="observation",
        semantic_role=role,
        layer="transport",
        dimensions=["bus"],
        summary=summary,
        confidence=1.0,
        salience=0.5,
        source_event_id="20260525T233010Z",
        payload_ref=f"bus.transport:{role}",
    )


def _event(event_id: str, role: str, summary: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=TRACE,
        session_id="bus-session",
        correlation_id=TRACE,
        emitted_at=NOW,
        observed_at=NOW,
        provenance=_prov(),
        atom=_atom(role, summary),
    )


def test_parse_bus_transport_trace_id() -> None:
    assert parse_bus_transport_trace_id("bus.transport:athena:20260525T233010Z") == ("athena", "20260525T233010Z")


def test_extract_live_athena_rollup_pressures() -> None:
    events = [
        _event("gev_h", "bus_health_observed", "redis_ping_ok=true node_id=athena sample_window_id=20260525T233010Z"),
        _event("gev_d1", "bus_stream_depth_observed", "stream_key=orion:evt:gateway stream_length=0 sample_window_id=20260525T233010Z"),
        _event("gev_d2", "bus_stream_depth_observed", "stream_key=orion:bus:out stream_length=0 sample_window_id=20260525T233010Z"),
        _event("gev_u1", "bus_configured_stream_uncataloged", "stream_key=orion:evt:gateway sample_window_id=20260525T233010Z"),
        _event("gev_u2", "bus_configured_stream_uncataloged", "stream_key=orion:bus:out sample_window_id=20260525T233010Z"),
        _event("gev_done", "bus_observer_tick_completed", "streams_observed=2 sample_window_id=20260525T233010Z"),
    ]
    state = extract_transport_bus_state_from_events(events, now=NOW)
    pressures = compute_transport_pressures(state, stream_depth_critical=100_000)
    assert pressures["bus_health"] == 1.0
    assert pressures["catalog_drift_pressure"] == 1.0
    assert pressures["transport_pressure"] == 0.0
    assert pressures["contract_pressure"] == 1.0
    assert state.target_id == "bus:athena"
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_transport_substrate_reducer.py::test_extract_live_athena_rollup_pressures -v`

- [ ] **Step 3: Implement extract.py**

Mirror `execution_loop/grammar_extract.py`:

- `_KV_RE` for `key=value` tokens in `atom.summary`
- Ignore events: no atom, blank `semantic_role`, `trace_started`/`trace_ended`/`edge_emitted`
- Roles: `bus_health_observed`, `bus_stream_depth_observed`, `bus_backpressure_observed`, `bus_configured_stream_uncataloged`, `bus_observer_tick_failed`, `bus_observer_tick_completed`
- `parse_bus_transport_trace_id(trace_id) -> tuple[str, str] | None` from `bus.transport:{node}:{window}`
- `compute_transport_pressures(state, *, stream_depth_critical: int)` per spec formulas:

```python
def compute_transport_pressures(state: TransportBusStateV1, *, stream_depth_critical: int) -> dict[str, float]:
    bus_health = 1.0 if state.redis_ping_ok is True else (0.0 if state.redis_ping_ok is False else 0.5)
    observer_failure_pressure = 1.0 if state.observer_failure_count > 0 else 0.0
    denom = max(state.streams_observed, 1)
    stream_depth_pressure = min(state.max_stream_depth / max(stream_depth_critical, 1), 1.0)
    backpressure = min(state.backpressure_count / denom, 1.0)
    catalog_drift_pressure = min(state.uncataloged_stream_count / denom, 1.0)
    delivery_confidence = (
        1.0 if bus_health >= 1.0 and observer_failure_pressure == 0.0
        else 0.0 if observer_failure_pressure > 0.0
        else 0.5
    )
    transport_pressure = max(stream_depth_pressure, backpressure)
    contract_pressure = catalog_drift_pressure
    reliability_pressure = max(observer_failure_pressure, 1.0 - delivery_confidence)
    return {
        "bus_health": bus_health,
        "delivery_confidence": delivery_confidence,
        "stream_depth_pressure": stream_depth_pressure,
        "backpressure": backpressure,
        "catalog_drift_pressure": catalog_drift_pressure,
        "observer_failure_pressure": observer_failure_pressure,
        "transport_pressure": transport_pressure,
        "contract_pressure": contract_pressure,
        "reliability_pressure": reliability_pressure,
    }
```

- [ ] **Step 4: Run extract tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/transport_loop/ tests/test_transport_substrate_reducer.py
git commit -m "feat: extract transport bus state from grammar events"
```

### Task 4: Reducer emits transport_bus StateDeltaV1

**Files:**
- Create: `orion/substrate/transport_loop/reducer.py`
- Test: append reducer tests to `tests/test_transport_substrate_reducer.py`

- [ ] **Step 1: Write failing reducer test**

```python
def test_reducer_emits_transport_bus_delta_with_pressure_hints() -> None:
    from orion.schemas.transport_projection import TransportBusProjectionV1
    from orion.substrate.transport_loop.reducer import reduce_transport_trace_events

    events = [...]  # same fixture as test_extract_live_athena_rollup_pressures
    projection = TransportBusProjectionV1(updated_at=NOW)
    projection, receipt = reduce_transport_trace_events(events=events, projection=projection, now=NOW)
    assert receipt.state_deltas
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "transport_bus"
    assert delta.target_id == "bus:athena"
    hints = (delta.after or {}).get("pressure_hints") or {}
    assert hints["catalog_drift_pressure"] == 1.0
    assert hints["transport_pressure"] == 0.0
```

- [ ] **Step 2: Implement reducer** (mirror `execution_loop/reducer.py`)

- `after` payload includes full `TransportBusStateV1` fields + `pressure_hints`
- `delta_id=stable_delta_id(reducer_id=TRANSPORT_REDUCER_ID, target_projection=TRANSPORT_BUS_PROJECTION_ID, target_kind="transport_bus", target_id="bus:athena", operation=..., caused_by_event_ids=...)`
- Update `projection.buses["bus:athena"]`

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: transport bus reducer emits state deltas"
```

### Task 5: Pipeline groups traces

**Files:**
- Create: `orion/substrate/transport_loop/pipeline.py`
- Test: `tests/test_transport_substrate_pipeline.py`

- [ ] **Step 1–5:** Mirror `tests/test_execution_substrate_pipeline.py` — two traces → two receipts, projection updated.

```bash
git commit -m "feat: transport substrate pipeline by trace"
```

---

# Phase 3 — SQL migration + substrate runtime wiring

### Task 6: DDL

**Files:**
- Create: `services/orion-sql-db/manual_migration_transport_substrate_loop.sql`

```sql
-- Apply: docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -f ...
create table if not exists substrate_transport_bus_projection (
    projection_id text primary key,
    projection_json jsonb not null,
    updated_at timestamptz not null default now()
);

create table if not exists substrate_transport_bus_cursor (
    cursor_id text primary key,
    last_created_at timestamptz,
    last_event_id text,
    updated_at timestamptz not null default now()
);
```

Uses existing `substrate_reduction_receipts` + `substrate_reduction_cursor` pattern OR dedicated cursor table per spec (implement `substrate_transport_bus_cursor` as specified; grammar poll can use `substrate_reduction_cursor` with `cursor_name=transport_grammar_reducer` like execution).

**Recommended:** Use `substrate_reduction_cursor` with `TRANSPORT_GRAMMAR_CURSOR_NAME` (matches execution) — if spec table is required, write both: dedicated table for projection bookkeeping only.

- [ ] **Commit migration**

### Task 7: Substrate store fetch/save

**Files:**
- Modify: `services/orion-substrate-runtime/app/store.py`

Add (mirror `fetch_execution_grammar_events`):

```python
def fetch_transport_grammar_events(self, *, limit: int = 50) -> list[GrammarEventV1]:
    # WHERE source_service = 'orion-bus' AND trace_id LIKE 'bus.transport:%'
    # cursor_name = TRANSPORT_GRAMMAR_CURSOR_NAME
```

Add `load_transport_bus_projection`, `save_transport_bus_projection`, `advance_transport_cursor`.

### Task 8: Worker isolated tick + settings

**Files:**
- Modify: `services/orion-substrate-runtime/app/worker.py`
- Modify: `services/orion-substrate-runtime/app/settings.py`
- Modify: `services/orion-substrate-runtime/.env_example`
- Modify: `services/orion-substrate-runtime/docker-compose.yml`
- Modify: `services/orion-substrate-runtime/README.md`
- Sync: `services/orion-substrate-runtime/.env` locally

```python
# settings.py
enable_transport_bus_reducer: bool = Field(False, alias="ENABLE_TRANSPORT_BUS_REDUCER")
bus_stream_depth_critical: int = Field(100_000, alias="BUS_STREAM_DEPTH_CRITICAL")
transport_substrate_maturity: str = Field("trace_only", alias="TRANSPORT_SUBSTRATE_MATURITY")
```

```python
# worker.py — inside _poll_loop, after execution tick
if self._settings.enable_transport_bus_reducer:
    try:
        last_id = await asyncio.to_thread(self._transport_tick)
        if last_id:
            created_at = self._store.grammar_event_created_at(last_id)
            if created_at:
                self._store.advance_transport_cursor(event_id=last_id, created_at=created_at)
    except Exception:
        logger.exception("transport_substrate_tick_failed")
```

```python
def _transport_tick(self) -> str | None:
    events = self._store.fetch_transport_grammar_events(limit=50)
    if not events:
        return None
    from orion.substrate.transport_loop.pipeline import process_transport_grammar_events
    # load/save projection + save_receipt
    return events[-1].event_id
```

- [ ] **Run tests**

```bash
PYTHONPATH=. pytest tests/test_transport_substrate_reducer.py tests/test_transport_substrate_pipeline.py -q
```

- [ ] **Commit**

```bash
git commit -m "feat: wire transport bus reducer into substrate runtime"
```

---

# Phase 4 — Field digestion (Layer 4)

### Task 9: Lattice + channels

**Files:**
- Modify: `config/field/orion_field_topology.v1.yaml`
- Modify: `config/field/biometrics_lattice.yaml` (keep alias in sync — same capability block)
- Modify: `services/orion-field-digester/app/tensor/channels.py`

Add node channels:

```python
# channels.py — extend NODE_CHANNELS
"transport_pressure",
"contract_pressure",
"catalog_drift_pressure",
"delivery_confidence",
"bus_health",
"observer_failure_pressure",
```

Add capability channels on `capability:transport`:

```yaml
# orion_field_topology.v1.yaml
capabilities:
  - capability_id: transport
edges:
  - source_id: node:athena
    target_id: capability:transport
    edge_type: node_capability
    weight: 0.85
    channel_map:
      transport_pressure: pressure
      catalog_drift_pressure: contract_pressure
      observer_failure_pressure: reliability_pressure
      delivery_confidence: confidence
      bus_health: available_capacity
  # capability_dependency to orchestration: defer if edge_type unsupported — document in PR
```

### Task 10: state_deltas transport_bus branch

**Files:**
- Modify: `services/orion-field-digester/app/ingest/state_deltas.py`
- Modify: `services/orion-field-digester/app/settings.py`
- Modify: `services/orion-field-digester/.env_example`, `docker-compose.yml`, `README.md`
- Sync: `services/orion-field-digester/.env`

```python
# settings.py
enable_transport_field_digestion: bool = Field(False, alias="ENABLE_TRANSPORT_FIELD_DIGESTION")
```

```python
# state_deltas.py — after execution_run block
if delta.target_kind == "transport_bus":
    hints = dict((delta.after or {}).get("pressure_hints") or {})
    node_id = _node_key(str((delta.after or {}).get("node_id") or delta.target_id.replace("bus:", "")))
    mapping = [
        ("bus_health", "bus_health"),
        ("delivery_confidence", "delivery_confidence"),
        ("transport_pressure", "transport_pressure"),
        ("stream_depth_pressure", "transport_pressure"),
        ("backpressure", "transport_pressure"),
        ("catalog_drift_pressure", "contract_pressure"),
        ("observer_failure_pressure", "observer_failure_pressure"),
        ("reliability_pressure", "reliability_pressure"),
        ("contract_pressure", "contract_pressure"),
    ]
    for hint_key, channel in mapping:
        if hint_key in hints:
            out.append(Perturbation(node_id=node_id, channel=channel, intensity=float(hints[hint_key]), label=delta.delta_id))
```

Worker gate:

```python
# worker.py _tick — skip transport deltas when flag false
if delta.target_kind == "transport_bus" and not self._settings.enable_transport_field_digestion:
    continue
```

### Task 11: Field tests

**Files:**
- Create: `tests/test_field_transport_perturbations.py`

```python
def test_transport_bus_delta_maps_to_node_and_capability() -> None:
    # Build delta with pressure_hints catalog_drift=1.0
    # load lattice, run_digestion_tick
    # assert node:athena contract_pressure > 0
    # assert capability:transport exists with contract_pressure

def test_field_digester_does_not_import_grammar_events() -> None:
    import services.orion_field_digester.app.store as store_mod  # adjust import path
    src = Path(store_mod.__file__).read_text()
    assert "grammar_events" not in src
```

- [ ] **Run**

```bash
PYTHONPATH=. pytest tests/test_field_transport_perturbations.py tests/test_field_execution_perturbations.py -q
```

- [ ] **Commit**

---

# Phase 5 — Attention visibility (Layer 5)

### Task 12: Attention policy + settings + tests

**Files:**
- Modify: `config/attention/field_attention_policy.v1.yaml`
- Modify: `services/orion-attention-runtime/app/settings.py`
- Modify: `services/orion-attention-runtime/.env_example`, `docker-compose.yml`
- Create: `tests/test_attention_transport_visibility.py`
- Sync: `.env`

```yaml
# field_attention_policy.v1.yaml
tracked_targets:
  - capability:transport

capability_channel_weights:
  transport_pressure: 0.75
  contract_pressure: 0.90
  reliability_pressure: 1.00
  bus_health: -0.30

attention_reason_templates:
  - match_channel: contract_pressure
    reason: capability contract_pressure is elevated
    observation_mode: inspect
  - match_channel: transport_pressure
    reason: capability transport_pressure is elevated
    observation_mode: watch
```

Runtime: when `ENABLE_TRANSPORT_ATTENTION_VISIBILITY=false`, filter out items where `target_id == capability:transport` (and optionally node transport-only reasons).

Tests prove salience from `contract_pressure=1.0` and flag disables transport targets.

- [ ] **Commit**

---

# Phase 6 — Self-state (Layer 6, gated)

### Task 13: transport_integrity dimension

**Files:**
- Modify: `config/self_state/self_state_policy.v1.yaml`
- Modify: `services/orion-self-state-runtime/app/settings.py` + env/compose/README
- Create: `tests/test_self_state_transport_dimension.py`

```python
def transport_integrity_score(hints: dict[str, float]) -> float:
    return min(
        hints.get("bus_health", 0.5),
        hints.get("delivery_confidence", 0.5),
        1.0 - hints.get("transport_pressure", 0.0),
        1.0 - hints.get("reliability_pressure", 0.0),
        1.0 - hints.get("contract_pressure", 0.0) * 0.5,
    )
```

Labels: `transport_contract_drift` when `contract_pressure >= 0.7` and `bus_health >= 0.9`; `transport_degraded` when integrity < 0.4; etc.

Expected for live hints: `transport_integrity ≈ 0.5`, label `transport_contract_drift`.

- [ ] **Commit**

---

# Phase 7 — Proposal read-only (Layer 7, gated)

### Task 14: Transport proposal templates

**Files:**
- Modify: `config/proposals/proposal_policy.v1.yaml`
- Modify: `orion/proposals/templates.py`
- Modify: `services/orion-proposal-runtime/app/settings.py` + env/compose
- Create: `tests/test_proposal_transport_readonly_candidates.py`

Templates (read_only gate):

```yaml
inspect_transport_status:
  kind: inspect
  target_kind: capability
  target_id: capability:transport
inspect_bus_channel_catalog:
  kind: inspect
  target_kind: system
  target_id: orion/bus/channels.yaml
summarize_transport_contract_drift:
  kind: summarize
  target_kind: capability
  target_id: capability:transport
watch_transport_backpressure:
  kind: observe
  target_kind: capability
  target_id: capability:transport
```

Builder gate: only score templates when `ENABLE_TRANSPORT_PROPOSALS=true` and `TRANSPORT_PROPOSAL_MODE=read_only`. **Never** emit restart/purge/replay/change_catalog candidates.

- [ ] **Commit**

---

# Phase 8 — Policy gates (Layer 8, gated)

### Task 15: Transport policy rules

**Files:**
- Modify: `config/policy/substrate_policy.v1.yaml`
- Modify: `services/orion-policy-runtime/app/settings.py` + env/compose
- Create: `tests/test_policy_transport_gates.py`

Approve read-only inspect/summarize/watch; `restart_bus` → rejected; `replay_stream`/`change_catalog` → `requires_operator_review`.

- [ ] **Commit**

---

# Phase 9 — Dispatch dry-run (Layer 9, gated)

### Task 16: Dry-run envelopes

**Files:**
- Modify: `config/execution_dispatch/execution_dispatch_policy.v1.yaml`
- Modify: `services/orion-execution-dispatch-runtime/app/settings.py` + env/compose
- Create: `tests/test_execution_dispatch_transport_dry_run.py`

Every approved transport envelope includes constraints:

```json
{
  "dry_run": true,
  "read_only": true,
  "no_file_writes": true,
  "no_service_restarts": true,
  "no_external_side_effects": true,
  "no_operator_notifications": true,
  "no_stream_replay": true,
  "no_stream_purge": true,
  "no_catalog_write": true
}
```

`dispatch_attempted=false`, `dispatch_count=0`.

- [ ] **Commit**

---

# Phase 10 — Feedback (Layer 10, gated)

### Task 17: Transport feedback observations

**Files:**
- Modify: `config/feedback/feedback_policy.v1.yaml`
- Modify: `services/orion-feedback-runtime/app/settings.py` + env/compose
- Create: `tests/test_feedback_transport_outcomes.py`

Dry-run → `outcome_kind=dry_run`, `outcome_status=dry_run_only`, `score=0.5`. Blocked mutations → `blocked` observations (`transport_restart_blocked`, etc.).

- [ ] **Commit**

---

# Phase 11 — Consolidation motifs (Layer 11, gated)

### Task 18: Transport motif rules

**Files:**
- Modify: `config/consolidation/consolidation_policy.v1.yaml`
- Extend: `orion/consolidation/motif.py` (or `orion/consolidation/transport_motif.py`)
- Modify: `services/orion-consolidation-runtime/app/settings.py` + env/compose
- Create: `tests/test_consolidation_transport_motifs.py`

```yaml
motif_rules:
  transport_contract_drift_loop:
    kind: field_pattern
    label: transport_contract_drift_loop
    conditions:
      capability_target: capability:transport
      min_contract_pressure: 0.7
  transport_healthy_idle:
    kind: field_pattern
    label: transport_healthy_idle
    conditions:
      max_transport_pressure: 0.1
      min_bus_health: 0.9
  # transport_backpressure_loop, transport_observer_failure_loop,
  # transport_readonly_policy_loop, transport_dry_run_feedback_loop
```

Schema candidates: `promotion_status: candidate_only` only.

- [ ] **Commit**

---

# Phase 12 — Smoke script + verification

### Task 19: Smoke script

**Files:**
- Create: `scripts/smoke_orion_bus_transport_full_stack.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-}"
case "${MODE/--mode /}" in
  m3)  # grammar traces + transport_bus receipts + pressure_hints
  m4)  # capability:transport in substrate_field_state
  m5)  # attention items for capability:transport or node:athena
  full-observe)  # optional SQL when flags enabled
esac
```

### Task 20: Full test + compile sweep

- [ ] **Run required tests**

```bash
PYTHONPATH=. pytest \
  tests/test_transport_projection_schemas.py \
  tests/test_transport_substrate_reducer.py \
  tests/test_transport_substrate_pipeline.py \
  tests/test_field_transport_perturbations.py \
  tests/test_attention_transport_visibility.py \
  tests/test_self_state_transport_dimension.py \
  tests/test_proposal_transport_readonly_candidates.py \
  tests/test_policy_transport_gates.py \
  tests/test_execution_dispatch_transport_dry_run.py \
  tests/test_feedback_transport_outcomes.py \
  tests/test_consolidation_transport_motifs.py \
  -q
```

- [ ] **Run regressions**

```bash
PYTHONPATH=. pytest \
  tests/test_biometrics_pipeline.py \
  tests/test_node_pressure_reducer.py \
  tests/test_execution_substrate_reducer.py \
  tests/test_execution_substrate_pipeline.py \
  tests/test_field_*.py \
  -q
```

- [ ] **Compile**

```bash
PYTHONPATH=. python -m compileall \
  orion/substrate/transport_loop \
  orion/schemas/transport_projection.py \
  services/orion-substrate-runtime \
  services/orion-field-digester \
  services/orion-attention-runtime \
  services/orion-self-state-runtime \
  services/orion-proposal-runtime \
  services/orion-policy-runtime \
  services/orion-execution-dispatch-runtime \
  services/orion-feedback-runtime \
  services/orion-consolidation-runtime \
  -q
```

### Task 21: Live rollout (operator)

1. Apply `manual_migration_transport_substrate_loop.sql`
2. Enable Layer 3 only → verify `substrate_reduction_receipts` (`target_kind=transport_bus`)
3. Enable Layer 4 → verify `substrate_field_state` / `capability:transport`
4. Enable Layer 5 → verify `substrate_attention_frames`
5. Progressive M6–M11 with flags

Paste SQL evidence into PR report.

---

# Phase 13 — Code review + PR

### Task 22: Requesting code review subagent

- [ ] Dispatch **requesting-code-review** subagent on full diff vs `main`
- [ ] Fix all blocking issues; re-run Task 20 verification

### Task 23: PR report + push

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-25-orion-bus-transport-full-stack-v1-pr.md` (use spec template)

```bash
git push -u origin feat/orion-bus-transport-substrate-full-stack-v1
gh pr create --base main --title "PR: Orion Bus Transport Full Substrate Integration v1" --body "$(cat <<'EOF'
## Summary
- Layers 3–5: transport_bus reducer, field digestion, attention visibility (gated, default off)
- Layers 6–11: read-only/dry-run scaffolding for self/proposal/policy/dispatch/feedback/consolidation

## Test plan
- [ ] pytest transport + field + frame tests
- [ ] smoke --mode m3/m4/m5 after enabling flags
EOF
)"
```

---

## Self-review checklist

| Spec requirement | Task |
|------------------|------|
| TransportBusStateV1 / ProjectionV1 | Task 1 |
| Reducer pressure formulas + live expected output | Task 3–4 |
| No field read of grammar_events | Task 10–11 |
| All effect gates default false | Tasks 7–18 settings |
| M5 live proof path | Tasks 12, 20–21 |
| M6–M11 gated | Tasks 13–18 |
| Smoke script modes | Task 19 |
| PR report template | Task 23 |
| channels.yaml / registry | Only if new schema types need bus registration (projections are internal) |

**Placeholder scan:** No TBD steps — implementers fill exact motif detector code following `orion/consolidation/motif.py` patterns in Tasks 13–18.

**Type consistency:** `target_kind=transport_bus`, `target_id=bus:{node_id}`, `pressure_hints` keys match across reducer → field → attention → self-state.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-25-orion-bus-transport-full-stack-v1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task + review between tasks.

**2. Inline Execution** — Execute tasks in this session using executing-plans with checkpoints.

Which approach?
