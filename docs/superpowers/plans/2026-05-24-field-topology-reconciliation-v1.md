# Field Topology Reconciliation + Execution Digestion Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the substrate digestion path so persisted `FieldStateV1` is reconciled against current lattice topology every tick, and execution-run reduction merges partial trace batches monotonically — preparing layer 5 (`AttentionFrameV1`) without building attention.

**Architecture:** Promote `config/field/orion_field_topology.v1.yaml` as canonical lattice config (keep `biometrics_lattice.yaml` as alias). Add `reconcile_field_state_with_lattice()` in field-digester and call it in `worker._tick` before `run_digestion_tick`. Add `merge_execution_run_state()` in `orion/substrate/execution_loop/` and use it in `reduce_execution_trace_events`. Add read-only smoke script for live SQL inspection.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest, Postgres (`orion-athena-sql-db`), Docker Compose.

**Design source:** User spec “Field topology reconciliation + execution digestion hardening” (2026-05-24).

**Depends on:** `main` includes merged `feat/cortex-exec-substrate-digestion-v1` (#618) — execution_run deltas → field-digester path is live.

**Non-goals:** `AttentionFrameV1`, `SelfStateV1`, mind service, action/proposal loops, policy gates, new organs, LLM interpretation, bus field events, recall/vision changes.

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-field-topology-reconciliation-v1 \
  -b feat/field-topology-reconciliation-v1 \
  main
cd .worktrees/feat-field-topology-reconciliation-v1
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

**Rules:**
- All commits only in `.worktrees/feat-field-topology-reconciliation-v1`.
- Never bleed files to the main checkout **except** syncing `.env` keys from `.env_example` into operator-local `.env` for `orion-field-digester` (and only `.env`).
- PR title: `PR: Field topology reconciliation + execution digestion hardening`.
- When done: run `requesting-code-review` subagent, fix all findings, write `docs/superpowers/pr-reports/2026-05-24-field-topology-reconciliation-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Strain | Root cause in code today |
|--------|--------------------------|
| Stale lattice edges in persisted field | `worker._tick` loads `FieldStateV1` as-is; `edges` only set on `empty_field_state()` |
| Digester marks deltas applied without visible effect | Perturbations apply to stale channel_map edges (e.g. only `cpu_pressure→pressure`) |
| Execution reducer downgrades on partial batches | `reducer.py` line 104: `updated.runs[trace_id] = merged` replaces prior run |
| Lattice misnamed | `biometrics_lattice.yaml` holds biometrics + execution dynamics |
| Topology not versioned on snapshots | `FieldStateV1` has no topology metadata fields |

| Question | Finding |
|----------|---------|
| Digestion on `main`? | Yes — merge `783d7239` |
| `LatticeGraph` type | `services/orion-field-digester/app/graph/lattice.py` — nodes, capabilities, edges |
| Channel defaults | `services/orion-field-digester/app/tensor/channels.py` — mirrors YAML `node_channels` / `capability_channels` |
| Bus channels | No change — field-digester still Postgres-only per README |
| Schema registry | Update only if `FieldStateV1` gains optional topology fields |

### Target tick flow after PR

```text
load FieldStateV1 (or empty_field_state)
reconcile_field_state_with_lattice(state, lattice=self._lattice)
collect perturbations from new receipts
run_digestion_tick → commit
```

---

## File structure

| Path | Responsibility |
|------|----------------|
| `config/field/orion_field_topology.v1.yaml` | Canonical field topology (copy of current lattice) |
| `config/field/biometrics_lattice.yaml` | Compatibility alias (same content + comment header) |
| `services/orion-field-digester/app/tensor/reconcile.py` | `reconcile_field_state_with_lattice` |
| `services/orion-field-digester/app/worker.py` | Call reconcile before digestion tick |
| `services/orion-field-digester/app/settings.py` | Default `LATTICE_PATH` → canonical file |
| `services/orion-field-digester/.env_example` | Canonical path |
| `services/orion-field-digester/.env` | Sync `LATTICE_PATH` locally (not committed) |
| `services/orion-field-digester/docker-compose.yml` | Default env path |
| `services/orion-field-digester/README.md` | Canonical vs alias docs |
| `orion/substrate/execution_loop/merge.py` | `merge_execution_run_state`, status rank helper |
| `orion/substrate/execution_loop/reducer.py` | Use merge instead of replace |
| `orion/schemas/field_state.py` | Optional topology metadata on `FieldStateV1` |
| `orion/schemas/registry.py` | No change unless registry lists field topology (already has `FieldStateV1`) |
| `scripts/smoke_execution_field_digestion.sh` | Live SQL inspection + optional reset |
| `tests/test_field_topology_reconciliation.py` | Reconcile behavior |
| `tests/test_execution_substrate_reducer.py` | Monotonic merge tests |
| `tests/test_field_execution_perturbations.py` | Point `LATTICE` at canonical path |
| `tests/test_field_digestion_rules.py` | Accept canonical or alias path |
| `docs/superpowers/pr-reports/2026-05-24-field-topology-reconciliation-v1-pr.md` | PR report (post-implementation) |

---

# Phase 0 — Worktree + branch

### Task 0: Create isolated worktree

- [ ] **Step 1: Create worktree from `main`**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-field-topology-reconciliation-v1 \
  -b feat/field-topology-reconciliation-v1 \
  main
cd .worktrees/feat-field-topology-reconciliation-v1
```

Expected: `git branch --show-current` → `feat/field-topology-reconciliation-v1`

- [ ] **Step 2: Verify isolation**

```bash
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "chore: start field topology reconciliation worktree"
```

---

# Phase 1 — Promote lattice language to field topology

### Task 1: Canonical topology config + compatibility alias

**Files:**
- Create: `config/field/orion_field_topology.v1.yaml`
- Modify: `config/field/biometrics_lattice.yaml` (add alias comment only; keep byte-identical body)
- Modify: `services/orion-field-digester/app/settings.py`
- Modify: `services/orion-field-digester/.env_example`
- Modify: `services/orion-field-digester/.env` (local sync, not committed)
- Modify: `services/orion-field-digester/docker-compose.yml`
- Modify: `services/orion-field-digester/README.md`

- [ ] **Step 1: Copy canonical topology**

```bash
cd .worktrees/feat-field-topology-reconciliation-v1
cp config/field/biometrics_lattice.yaml config/field/orion_field_topology.v1.yaml
```

Edit top of `config/field/orion_field_topology.v1.yaml` — first lines:

```yaml
# Canonical Orion field topology (biometrics + execution dynamics).
schema_version: field_lattice.v1
```

Edit top of `config/field/biometrics_lattice.yaml` — prepend only:

```yaml
# Compatibility alias — canonical: config/field/orion_field_topology.v1.yaml
```

- [ ] **Step 2: Update field-digester defaults**

`services/orion-field-digester/app/settings.py`:

```python
    lattice_path: str = Field(
        "config/field/orion_field_topology.v1.yaml",
        alias="LATTICE_PATH",
    )
```

`services/orion-field-digester/docker-compose.yml`:

```yaml
      - LATTICE_PATH=${LATTICE_PATH:-/app/config/field/orion_field_topology.v1.yaml}
```

`services/orion-field-digester/.env_example`:

```bash
LATTICE_PATH=/app/config/field/orion_field_topology.v1.yaml
```

Sync local operator file (from worktree, copy to main checkout `.env` only if operator runs digester from main — allowed bleed):

```bash
# In worktree
cp services/orion-field-digester/.env_example services/orion-field-digester/.env
# Optional: also update /mnt/scripts/Orion-Sapienform/services/orion-field-digester/.env LATTICE_PATH only
```

- [ ] **Step 3: README note**

Add to `services/orion-field-digester/README.md` under Environment table:

```markdown
`biometrics_lattice.yaml` is retained as a compatibility alias; `orion_field_topology.v1.yaml` is the canonical config. Operators may keep `LATTICE_PATH` pointed at either file.
```

- [ ] **Step 4: Compatibility test**

Create `tests/test_field_topology_config.py`:

```python
from pathlib import Path

from app.graph.lattice import load_lattice

REPO = Path(__file__).resolve().parents[1]


def test_canonical_and_alias_lattice_load_same_edges() -> None:
    canonical = load_lattice(REPO / "config" / "field" / "orion_field_topology.v1.yaml")
    alias = load_lattice(REPO / "config" / "field" / "biometrics_lattice.yaml")
    assert len(canonical.edges) == len(alias.edges)
    athena_orch = [
        e for e in canonical.edges
        if e.source_id == "node:athena" and e.target_id == "capability:orchestration"
    ]
    assert len(athena_orch) == 1
    assert "execution_load" in athena_orch[0].channel_map
    assert athena_orch[0].channel_map["execution_load"] == "execution_pressure"
```

Run:

```bash
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_topology_config.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/field/orion_field_topology.v1.yaml config/field/biometrics_lattice.yaml \
  services/orion-field-digester/app/settings.py services/orion-field-digester/.env_example \
  services/orion-field-digester/docker-compose.yml services/orion-field-digester/README.md \
  tests/test_field_topology_config.py
git commit -m "feat(field): promote orion_field_topology.v1.yaml as canonical lattice config"
```

---

# Phase 2 — Field topology reconciliation

### Task 2: `reconcile_field_state_with_lattice`

**Files:**
- Create: `services/orion-field-digester/app/tensor/reconcile.py`
- Modify: `services/orion-field-digester/app/worker.py`
- Test: `tests/test_field_topology_reconciliation.py`

- [ ] **Step 1: Write failing reconciliation tests**

Create `tests/test_field_topology_reconciliation.py`:

```python
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

from app.graph.lattice import load_lattice
from app.tensor.reconcile import reconcile_field_state_with_lattice

REPO = Path(__file__).resolve().parents[1]
LATTICE_PATH = REPO / "config" / "field" / "orion_field_topology.v1.yaml"
FIXED_TS = datetime(2026, 5, 24, 15, 0, tzinfo=timezone.utc)


def _stale_athena_state() -> FieldStateV1:
    """Simulates live stack: pre-execution topology edge + missing execution channels."""
    return FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_stale",
        node_vectors={
            "node:athena": {
                "availability": 0.95,
                "cpu_pressure": 0.42,
                "memory_pressure": 0.1,
                "custom_probe": 0.77,
            }
        },
        capability_vectors={
            "capability:orchestration": {
                "pressure": 0.2,
                "confidence": 0.9,
                "available_capacity": 0.85,
            }
        },
        edges=[
            FieldEdgeV1(
                source_id="node:athena",
                target_id="capability:orchestration",
                edge_type="node_capability",
                weight=0.90,
                channel_map={"cpu_pressure": "pressure"},
            )
        ],
        recent_perturbations=["perturb_exec_recent"],
    )


def test_reconcile_adds_execution_load_channel() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    assert "execution_load" in reconciled.node_vectors["node:athena"]
    assert reconciled.node_vectors["node:athena"]["execution_load"] == 0.0
    assert reconciled.node_vectors["node:athena"]["cpu_pressure"] == 0.42


def test_reconcile_refreshes_athena_orchestration_edge_mappings() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    edge = next(
        e for e in reconciled.edges
        if e.source_id == "node:athena" and e.target_id == "capability:orchestration"
    )
    assert edge.channel_map.get("execution_load") == "execution_pressure"
    assert edge.channel_map.get("execution_friction") == "reliability_pressure"
    assert edge.channel_map.get("failure_pressure") == "reliability_pressure"
    assert edge.channel_map.get("cpu_pressure") == "pressure"


def test_reconcile_preserves_existing_values_and_unknown_channels() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    assert reconciled.node_vectors["node:athena"]["availability"] == 0.95
    assert reconciled.node_vectors["node:athena"]["custom_probe"] == 0.77
    assert reconciled.recent_perturbations == ["perturb_exec_recent"]


def test_reconcile_adds_capability_pressure_channels_with_defaults() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    cap = reconciled.capability_vectors["capability:orchestration"]
    assert cap["execution_pressure"] == 0.0
    assert cap["reasoning_pressure"] == 0.0
    assert cap["reliability_pressure"] == 0.0
    assert cap["pressure"] == 0.2
    assert cap["confidence"] == 0.9


def test_reconciled_state_validates() -> None:
    lattice = load_lattice(LATTICE_PATH)
    reconciled = reconcile_field_state_with_lattice(_stale_athena_state(), lattice=lattice)
    roundtrip = FieldStateV1.model_validate(reconciled.model_dump(mode="json"))
    assert roundtrip.tick_id == "tick_stale"
```

Run:

```bash
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_topology_reconciliation.py -v
```

Expected: FAIL — `ModuleNotFoundError: app.tensor.reconcile`

- [ ] **Step 2: Implement reconcile module**

Create `services/orion-field-digester/app/tensor/reconcile.py`:

```python
from __future__ import annotations

from copy import deepcopy

from orion.schemas.field_state import FieldEdgeV1, FieldStateV1

from app.graph.lattice import LatticeGraph
from app.tensor.channels import (
    CAPABILITY_CHANNELS,
    DEFAULT_CAPABILITY_VECTOR,
    DEFAULT_NODE_VECTOR,
    NODE_CHANNELS,
)


def _ensure_node_vector(
    node_vectors: dict[str, dict[str, float]],
    node_id: str,
) -> dict[str, float]:
    existing = deepcopy(node_vectors.get(node_id, {}))
    merged = deepcopy(DEFAULT_NODE_VECTOR)
    merged.update(existing)
    for channel in NODE_CHANNELS:
        if channel not in merged:
            merged[channel] = DEFAULT_NODE_VECTOR[channel]
    for key, val in existing.items():
        if key not in NODE_CHANNELS:
            merged[key] = val
    node_vectors[node_id] = merged
    return merged


def _ensure_capability_vector(
    capability_vectors: dict[str, dict[str, float]],
    capability_id: str,
) -> dict[str, float]:
    existing = deepcopy(capability_vectors.get(capability_id, {}))
    merged = deepcopy(DEFAULT_CAPABILITY_VECTOR)
    merged.update(existing)
    for channel in CAPABILITY_CHANNELS:
        if channel not in merged:
            merged[channel] = DEFAULT_CAPABILITY_VECTOR[channel]
    for key, val in existing.items():
        if key not in CAPABILITY_CHANNELS:
            merged[key] = val
    capability_vectors[capability_id] = merged
    return merged


def reconcile_field_state_with_lattice(
    state: FieldStateV1,
    *,
    lattice: LatticeGraph,
) -> FieldStateV1:
    updated = deepcopy(state)
    for node_id in lattice.nodes:
        _ensure_node_vector(updated.node_vectors, node_id)
    for capability_id in lattice.capabilities:
        _ensure_capability_vector(updated.capability_vectors, capability_id)
    updated.edges = [
        FieldEdgeV1.model_validate(edge.model_dump(mode="json"))
        for edge in lattice.edges
    ]
    return updated
```

- [ ] **Step 3: Run reconciliation tests**

```bash
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_topology_reconciliation.py -v
```

Expected: PASS (all 5 tests)

- [ ] **Step 4: Wire into worker**

`services/orion-field-digester/app/worker.py` — add import:

```python
from app.tensor.reconcile import reconcile_field_state_with_lattice
```

In `_tick`, after load/create state and **before** perturbation collection:

```python
        state = reconcile_field_state_with_lattice(state, lattice=self._lattice)
```

Full segment:

```python
        state = self._store.load_latest_field()
        if state is None:
            state = empty_field_state(
                lattice=self._lattice,
                now=now,
                tick_id=new_tick_id(),
            )
        state = reconcile_field_state_with_lattice(state, lattice=self._lattice)
```

- [ ] **Step 5: Regression — field tests**

```bash
PYTHONPATH=.:services/orion-field-digester pytest \
  tests/test_field_topology_reconciliation.py \
  tests/test_field_execution_perturbations.py \
  tests/test_field_*.py -q
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add services/orion-field-digester/app/tensor/reconcile.py \
  services/orion-field-digester/app/worker.py \
  tests/test_field_topology_reconciliation.py
git commit -m "feat(field-digester): reconcile persisted field state with lattice each tick"
```

---

# Phase 3 — Monotonic execution-run merge

### Task 3: `merge_execution_run_state` + reducer wiring

**Files:**
- Create: `orion/substrate/execution_loop/merge.py`
- Modify: `orion/substrate/execution_loop/reducer.py`
- Modify: `tests/test_execution_substrate_reducer.py`

- [ ] **Step 1: Write failing monotonic merge tests**

Append to `tests/test_execution_substrate_reducer.py`:

```python
from orion.schemas.execution_projection import ExecutionRunStateV1
from orion.substrate.execution_loop.merge import merge_execution_run_state


def _run_with_full_egress() -> ExecutionRunStateV1:
    events = [
        _exec_atom(
            "exec_plan_started",
            "Execution plan started for verb=chat_general; step_count=2; depth=none",
            event_id="gev_full_1",
        ),
        _exec_atom(
            "exec_step_started",
            "Step started: order=1, step=step_1, verb=chat_general, services=LLMGatewayService",
            event_id="gev_full_2",
        ),
        _exec_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            event_id="gev_full_3",
        ),
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
            event_id="gev_full_4",
        ),
    ]
    return extract_execution_state_from_events(events, now=FIXED_TS)


def _partial_batch_no_egress() -> list:
    return [
        _exec_atom(
            "exec_step_started",
            "Step started: order=1, step=step_2, verb=chat_general, services=LLMGatewayService",
            event_id="gev_partial_1",
        ),
    ]


def test_merge_does_not_downgrade_egress_confidence() -> None:
    full = _run_with_full_egress()
    assert full.pressure_hints["egress_confidence"] == 1.0
    partial = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    merged = merge_execution_run_state(full, partial)
    assert merged.pressure_hints["egress_confidence"] == 1.0


def test_merge_does_not_downgrade_status_or_flags() -> None:
    full = _run_with_full_egress()
    partial = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    merged = merge_execution_run_state(full, partial)
    assert merged.status == "success"
    assert merged.final_text_present is True
    assert merged.reasoning_present is True


def test_merge_unions_evidence_event_ids() -> None:
    full = _run_with_full_egress()
    partial = extract_execution_state_from_events(_partial_batch_no_egress(), now=FIXED_TS)
    merged = merge_execution_run_state(full, partial)
    assert "gev_full_4" in merged.evidence_event_ids
    assert "gev_partial_1" in merged.evidence_event_ids


def test_reducer_partial_batch_after_full_does_not_downgrade() -> None:
    full_events = [
        _exec_atom(
            "exec_plan_started",
            "Execution plan started for verb=chat_general; step_count=1; depth=none",
            event_id="gev_r1",
        ),
        _exec_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            event_id="gev_r2",
        ),
        _exec_atom(
            "exec_result_emitted",
            "Cortex exec result emitted to reply_to=True, status=success",
            event_id="gev_r3",
        ),
    ]
    proj = _empty_projection()
    proj, _ = reduce_execution_trace_events(events=full_events, projection=proj, now=FIXED_TS)
    proj, receipt2 = reduce_execution_trace_events(
        events=_partial_batch_no_egress(),
        projection=proj,
        now=FIXED_TS,
    )
    run = proj.runs[TRACE]
    assert run.pressure_hints["egress_confidence"] == 1.0
    assert run.status == "success"
    assert run.final_text_present is True
    assert run.reasoning_present is True
    delta = receipt2.state_deltas[0]
    assert delta.before["pressure_hints"]["egress_confidence"] == 1.0


def test_merge_failed_step_raises_failure_pressure() -> None:
    full = _run_with_full_egress()
    fail_events = [
        _exec_atom("exec_step_failed", "Step failed: step=step_9, error_kind=timeout", event_id="gev_fail"),
    ]
    incoming = extract_execution_state_from_events(fail_events, now=FIXED_TS)
    merged = merge_execution_run_state(full, incoming)
    assert merged.failed_step_count >= 1
    assert merged.pressure_hints["failure_pressure"] == 1.0


def test_stable_delta_id_same_evidence_set() -> None:
    events = [_exec_atom("exec_result_emitted", "Cortex exec result emitted", event_id="gev_stable")]
    _, r1 = reduce_execution_trace_events(events=events, projection=_empty_projection(), now=FIXED_TS)
    _, r2 = reduce_execution_trace_events(events=events, projection=_empty_projection(), now=FIXED_TS)
    assert r1.state_deltas[0].delta_id == r2.state_deltas[0].delta_id
```

Run:

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py::test_merge_does_not_downgrade_egress_confidence -v
```

Expected: FAIL — import error for `merge`

- [ ] **Step 2: Implement merge module**

Create `orion/substrate/execution_loop/merge.py`:

```python
from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from orion.schemas.execution_projection import ExecutionRunStateV1

from .grammar_extract import compute_pressure_hints

_STATUS_RANK = {
    "unknown": 0,
    "success": 1,
    "partial": 2,
    "failed": 3,
    "fail": 3,
    "error": 4,
}


def _status_rank(status: str) -> int:
    return _STATUS_RANK.get((status or "unknown").strip().lower(), 0)


def _pick_status(existing: str, incoming: str) -> str:
    if _status_rank(incoming) > _status_rank(existing):
        return incoming
    return existing


def _pick_thinking_source(existing: str, incoming: str) -> str:
    ex = (existing or "none").strip().lower()
    inc = (incoming or "none").strip().lower()
    if ex != "none":
        return existing
    if inc != "none":
        return incoming
    return existing or "none"


def merge_execution_run_state(
    existing: ExecutionRunStateV1 | None,
    incoming: ExecutionRunStateV1,
) -> ExecutionRunStateV1:
    if existing is None:
        return incoming

    merged = deepcopy(existing)
    merged.started_step_count = max(existing.started_step_count, incoming.started_step_count)
    merged.completed_step_count = max(existing.completed_step_count, incoming.completed_step_count)
    merged.failed_step_count = max(existing.failed_step_count, incoming.failed_step_count)
    merged.step_count = max(existing.step_count, incoming.step_count)
    merged.recall_observed = existing.recall_observed or incoming.recall_observed
    merged.final_text_present = existing.final_text_present or incoming.final_text_present
    merged.reasoning_present = existing.reasoning_present or incoming.reasoning_present
    merged.thinking_source = _pick_thinking_source(existing.thinking_source, incoming.thinking_source)
    merged.status = _pick_status(existing.status, incoming.status)
    if incoming.verb != "unknown":
        merged.verb = incoming.verb
    if incoming.mode != "unknown":
        merged.mode = incoming.mode
    if incoming.session_id:
        merged.session_id = incoming.session_id
    if incoming.turn_id:
        merged.turn_id = incoming.turn_id

    evidence = sorted(set(existing.evidence_event_ids) | set(incoming.evidence_event_ids))
    merged.evidence_event_ids = evidence
    merged.last_updated_at = incoming.last_updated_at

    egress_emitted = (
        existing.pressure_hints.get("egress_confidence", 0.0) >= 1.0
        or incoming.pressure_hints.get("egress_confidence", 0.0) >= 1.0
    )
    merged.pressure_hints = compute_pressure_hints(merged, egress_emitted=egress_emitted)
    return merged
```

- [ ] **Step 3: Update reducer**

`orion/substrate/execution_loop/reducer.py` — add import:

```python
from .merge import merge_execution_run_state
```

Replace lines 102-104:

```python
    existing = updated.runs.get(trace_id)
    operation = "create" if existing is None else "update"
    incoming = extract_execution_state_from_events(events, now=clock)
    merged = merge_execution_run_state(existing, incoming)
    updated.runs[trace_id] = merged
```

Rename prior `merged = extract_execution_state_from_events(...)` to `incoming = ...` (as above).

StateDelta `before` stays `existing`; `after` is `merged`.

- [ ] **Step 4: Run execution tests**

```bash
PYTHONPATH=. pytest \
  tests/test_execution_substrate_reducer.py \
  tests/test_execution_substrate_pipeline.py \
  tests/test_execution_projection_schemas.py -q
```

Expected: all PASS

- [ ] **Step 5: Biometrics regression**

```bash
PYTHONPATH=. pytest tests/test_biometrics_pipeline.py tests/test_node_pressure_reducer.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/substrate/execution_loop/merge.py \
  orion/substrate/execution_loop/reducer.py \
  tests/test_execution_substrate_reducer.py
git commit -m "fix(execution-loop): monotonically merge partial trace batches into run state"
```

---

# Phase 4 — Optional topology metadata on FieldStateV1

### Task 4: Self-describing field snapshots (low risk)

**Files:**
- Modify: `orion/schemas/field_state.py`
- Modify: `services/orion-field-digester/app/worker.py`
- Modify: `services/orion-field-digester/app/tensor/reconcile.py` (optional helper)
- Test: extend `tests/test_field_topology_reconciliation.py`

- [ ] **Step 1: Add optional fields**

`orion/schemas/field_state.py` in `FieldStateV1`:

```python
    topology_id: str | None = None
    topology_version: str | None = None
    topology_loaded_from: str | None = None
```

- [ ] **Step 2: Populate on reconcile in worker**

After reconcile in `worker._tick`:

```python
        state.topology_loaded_from = self._settings.lattice_path
        state.topology_id = "orion_field_topology"
        state.topology_version = "v1"
```

- [ ] **Step 3: Schema roundtrip test**

Add to `tests/test_field_state_schemas.py` or reconciliation test:

```python
def test_field_state_accepts_topology_metadata() -> None:
    state = FieldStateV1(
        generated_at=FIXED_TS,
        tick_id="tick_meta",
        topology_id="orion_field_topology",
        topology_version="v1",
        topology_loaded_from="config/field/orion_field_topology.v1.yaml",
    )
    assert FieldStateV1.model_validate(state.model_dump(mode="json")).topology_id == "orion_field_topology"
```

- [ ] **Step 4: Commit**

```bash
git add orion/schemas/field_state.py services/orion-field-digester/app/worker.py tests/
git commit -m "feat(field): optional topology metadata on FieldStateV1 snapshots"
```

**Note:** `orion/schemas/registry.py` already registers `FieldStateV1` — no registry edit unless codegen requires explicit field list (verify `tests/test_field_state_schemas.py` passes).

---

# Phase 5 — Smoke / inspection script

### Task 5: `scripts/smoke_execution_field_digestion.sh`

**Files:**
- Create: `scripts/smoke_execution_field_digestion.sh`

- [ ] **Step 1: Create script**

```bash
#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

run_sql() {
  docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "$1"
}

echo "=== Orion execution → field digestion smoke (read-only) ==="
echo "DB=$DB PGDATABASE=$PGDATABASE PGUSER=$PGUSER"
echo ""

echo "--- 1. Latest cortex-exec grammar events ---"
run_sql "
select created_at, event_id, trace_id,
       event_json::jsonb #>> '{atom,semantic_role}' as semantic_role
from grammar_events
where source_service = 'orion-cortex-exec'
  and trace_id like 'cortex.exec:%'
order by created_at desc
limit 10;
"

echo "--- 2. Latest execution_run receipts ---"
run_sql "
select r.created_at, r.receipt_id, d ->> 'delta_id' as delta_id,
       d ->> 'target_kind' as target_kind,
       d ->> 'target_id' as target_id,
       d #> '{after,pressure_hints}' as pressure_hints
from substrate_reduction_receipts r
cross join lateral jsonb_array_elements(
  coalesce(r.receipt_json::jsonb -> 'state_deltas', '[]'::jsonb)
) d
where d ->> 'target_kind' = 'execution_run'
order by r.created_at desc
limit 10;
"

echo "--- 3. Applied execution deltas ---"
run_sql "
with exec_deltas as (
  select r.receipt_id, d ->> 'delta_id' as delta_id
  from substrate_reduction_receipts r
  cross join lateral jsonb_array_elements(
    coalesce(r.receipt_json::jsonb -> 'state_deltas', '[]'::jsonb)
  ) d
  where d ->> 'target_kind' = 'execution_run'
)
select a.applied_at, a.delta_id, a.receipt_id
from substrate_field_applied_deltas a
join exec_deltas e on e.delta_id = a.delta_id
order by a.applied_at desc
limit 10;
"

echo "--- 4. Latest field vector ---"
run_sql "
select generated_at,
       field_json::jsonb -> 'node_vectors' -> 'node:athena' as athena_vector,
       field_json::jsonb -> 'capability_vectors' -> 'capability:orchestration' as orchestration_vector,
       field_json::jsonb -> 'recent_perturbations' as recent_perturbations
from substrate_field_state
order by generated_at desc
limit 1;
"

if [[ "${RESET_FIELD_STATE:-0}" == "1" ]]; then
  echo ""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "WARNING: RESET_FIELD_STATE=1 — destructive dev reset requested."
  echo "This deletes substrate_field_state, cursor, and applied execution deltas."
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  sleep 3
  run_sql "delete from substrate_field_applied_deltas where delta_id in (
    select d ->> 'delta_id'
    from substrate_reduction_receipts r
    cross join lateral jsonb_array_elements(
      coalesce(r.receipt_json::jsonb -> 'state_deltas', '[]'::jsonb)
    ) d
    where d ->> 'target_kind' = 'execution_run'
  );"
  run_sql "delete from substrate_field_state;"
  run_sql "delete from substrate_field_digester_cursor;"
  echo "Reset complete."
fi

echo ""
echo "Done. Optional destructive reset: RESET_FIELD_STATE=1 $0"
```

```bash
chmod +x scripts/smoke_execution_field_digestion.sh
```

- [ ] **Step 2: Commit**

```bash
git add scripts/smoke_execution_field_digestion.sh
git commit -m "chore(scripts): add execution field digestion smoke inspector"
```

---

# Phase 6 — Test path hygiene + compile

### Task 6: Point tests at canonical topology

- [ ] **Step 1: Update test lattice constants**

`tests/test_field_execution_perturbations.py`:

```python
LATTICE = REPO_ROOT / "config" / "field" / "orion_field_topology.v1.yaml"
```

`tests/test_field_digestion_rules.py` — use canonical path in `test_empty_field_state_has_all_lattice_nodes`.

- [ ] **Step 2: Full verification suite**

```bash
cd .worktrees/feat-field-topology-reconciliation-v1

PYTHONPATH=.:services/orion-field-digester pytest \
  tests/test_field_topology_reconciliation.py \
  tests/test_field_topology_config.py \
  tests/test_field_execution_perturbations.py \
  tests/test_field_*.py -q

PYTHONPATH=. pytest \
  tests/test_execution_substrate_reducer.py \
  tests/test_execution_substrate_pipeline.py \
  tests/test_execution_projection_schemas.py -q

PYTHONPATH=. pytest tests/test_biometrics_pipeline.py tests/test_node_pressure_reducer.py -q

PYTHONPATH=. python -m compileall \
  orion/substrate/execution_loop \
  services/orion-field-digester/app \
  services/orion-substrate-runtime/app
```

Expected: all PASS, compileall exit 0

- [ ] **Step 3: Commit**

```bash
git add tests/test_field_execution_perturbations.py tests/test_field_digestion_rules.py
git commit -m "test(field): use canonical orion_field_topology.v1.yaml in field tests"
```

---

# Phase 7 — Code review, PR report, push

### Task 7: Subagent code review + PR

- [ ] **Step 1: Run requesting-code-review subagent**

Dispatch subagent with prompt:

```text
Review branch feat/field-topology-reconciliation-v1 in worktree
/mnt/scripts/Orion-Sapienform/.worktrees/feat-field-topology-reconciliation-v1.

Focus: reconcile_field_state_with_lattice correctness, worker wiring before every tick,
merge_execution_run_state monotonic rules, no attention/self-state bleed, test coverage.

Return: APPROVED or list of required fixes with file:line.
```

- [ ] **Step 2: Fix all review findings**

Address each issue; re-run Phase 6 verification.

- [ ] **Step 3: Write PR report**

Create `docs/superpowers/pr-reports/2026-05-24-field-topology-reconciliation-v1-pr.md` including:

- Summary of stale-lattice bug (persisted edges vs config)
- Reconciliation behavior (edges refreshed, channels added, values preserved)
- Monotonic execution merge (partial batch no longer downgrades egress/status/flags)
- Test output (paste commands + counts)
- Manual smoke: `./scripts/smoke_execution_field_digestion.sh`
- Explicit note: prepares substrate for `AttentionFrameV1`; does not implement attention

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/field-topology-reconciliation-v1

gh pr create --title "PR: Field topology reconciliation + execution digestion hardening" --body "$(cat <<'EOF'
## Summary
- Reconcile persisted `FieldStateV1` against current lattice topology every field-digester tick
- Promote `orion_field_topology.v1.yaml` as canonical config (`biometrics_lattice.yaml` alias retained)
- Monotonically merge partial cortex-exec trace batches in execution reducer
- Add `scripts/smoke_execution_field_digestion.sh` for live chain inspection

## Stale-lattice bug
Persisted field snapshots kept old `node:athena → capability:orchestration` edge (`cpu_pressure` only) after config gained execution channel maps; required manual `substrate_field_state` reset.

## Reconciliation
Every tick: ensure nodes/capabilities/channels exist, refresh `edges` from lattice, preserve vector values and `recent_perturbations`.

## Monotonic execution merge
Partial batches no longer downgrade `egress_confidence`, `status`, `final_text_present`, or `reasoning_present`.

## Tests
See PR report in `docs/superpowers/pr-reports/2026-05-24-field-topology-reconciliation-v1-pr.md`.

## Test plan
- [x] `test_field_topology_reconciliation.py`
- [x] `test_execution_substrate_reducer.py` merge cases
- [x] Field + biometrics regressions
- [ ] Live: `./scripts/smoke_execution_field_digestion.sh`

## Non-goals
No AttentionFrameV1, SelfStateV1, mind, proposals, or bus field events.

EOF
)"
```

---

## Self-review (plan author checklist)

| Spec requirement | Task |
|------------------|------|
| Canonical `orion_field_topology.v1.yaml` + alias | Task 1 |
| Reconcile every tick before digestion | Task 2 |
| Preserve values, unknown channels, perturbations | Task 2 tests + reconcile impl |
| Refresh edges from lattice | Task 2 |
| Monotonic execution merge | Task 3 |
| Smoke script + optional reset | Task 5 |
| Optional topology metadata | Task 4 |
| No attention/self-state/mind | Non-goals header |
| Bus/registry | Preflight: no bus change; registry only if schema extended |
| `.env` sync | Task 1 Step 2 |
| Worktree isolation | Phase 0 + rules |

**Placeholder scan:** No TBD/TODO implementation gaps in task steps.

**Type consistency:** `LatticeGraph` used throughout (not `FieldLattice`). `merge_execution_run_state(existing, incoming)` matches reducer wiring.

---

## Acceptance criteria (from spec)

- [ ] Field-digester reconciles loaded state before every tick
- [ ] Stale edges auto-upgrade on next tick (no manual reset required for topology drift)
- [ ] Athena → orchestration edge includes execution_load / execution_friction / failure_pressure mappings
- [ ] Existing vector values preserved; new channels get safe defaults
- [ ] Execution reducer does not downgrade prior run state from partial batches
- [ ] Biometrics + execution digestion tests pass
- [ ] `scripts/smoke_execution_field_digestion.sh` works read-only
- [ ] No attention/self-state/proposal logic added
