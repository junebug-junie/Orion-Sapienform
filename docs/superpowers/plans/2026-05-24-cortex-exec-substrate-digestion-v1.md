# Cortex Exec Substrate Digestion v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge `orion-cortex-exec` grammar traces into committed `ReductionReceiptV1` / `StateDeltaV1(target_kind=execution_run)` and perturb `orion-field-digester` lattice state — proving `cortex.exec:* → receipt → field` without teaching field-digester to read raw grammar.

**Architecture:** Add `orion/substrate/execution_loop/` (grammar extract → trajectory projection → reducer, no organ). Extend `orion-substrate-runtime` with a second poll cursor for `source_service='orion-cortex-exec'`. Extend `orion-field-digester` `delta_to_perturbations` and `config/field/biometrics_lattice.yaml` with execution channels/edges. Biometrics loop stays unchanged.

**Tech Stack:** Python 3.12, Pydantic v2, SQLAlchemy, pytest, existing `stable_delta_id` / `stable_receipt_id` from `orion/substrate/ids.py`.

**Design source:** User spec “Cortex exec substrate reduction + field digestion” (2026-05-24).

**Depends on:** `main` includes merged `feat/cortex-exec-grammar-ingress` (#617), `orion-substrate-runtime` biometrics loop, and `orion-field-digester` v1.

**Non-goals:** Exec organ, mind service, raw-grammar field ingest, cortex-exec behavior changes, bus publish of field events, Hub execution debug routes (optional follow-up).

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-cortex-exec-substrate-digestion-v1 \
  -b feat/cortex-exec-substrate-digestion-v1 \
  main
cd .worktrees/feat-cortex-exec-substrate-digestion-v1
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

**Rules:**
- All commits only in `.worktrees/feat-cortex-exec-substrate-digestion-v1`.
- Never bleed files to main checkout **except** copying `.env` keys from `.env_example` into operator `.env` on the local machine (`orion-substrate-runtime`, `orion-field-digester` if lattice path changes).
- PR title: `PR: Cortex exec substrate digestion v1`.
- When done: run `requesting-code-review` subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-24-cortex-exec-substrate-digestion-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Cortex-exec grammar on `main` | Yes — `services/orion-cortex-exec/app/grammar_emit.py`, trace `cortex.exec:{node}:{correlation_id}` |
| Substrate runtime biometrics-only fetch | `services/orion-substrate-runtime/app/store.py` filters `orion-biometrics` + `biometrics.node:%` |
| Field digester consumes receipts | `services/orion-field-digester/app/ingest/receipts.py` polls `substrate_reduction_receipts` |
| Delta kinds today | `node_biometrics`, `active_node_pressure` only in `state_deltas.py` |
| Stable IDs | `orion/substrate/ids.py` — reuse for execution reducer |
| Bus channels | `orion-cortex-exec` already producer on `orion:grammar:event` — **no channel change required** |
| Schema registry | Add `ExecutionTrajectoryProjectionV1`, `ExecutionRunStateV1` to `orion/schemas/registry.py` |

### End-to-end proof target

```text
orion-cortex-exec (PUBLISH_CORTEX_EXEC_GRAMMAR=true)
  → grammar_events
  → orion-substrate-runtime execution tick
  → substrate_reduction_receipts (target_kind=execution_run)
  → orion-field-digester
  → substrate_field_state (execution_load on node:athena, execution_pressure on capability:orchestration)
```

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/execution_projection.py` | `ExecutionRunStateV1`, `ExecutionTrajectoryProjectionV1` |
| `orion/schemas/registry.py` | Register execution projection models |
| `orion/substrate/execution_loop/constants.py` | Cursor, projection id, source service, trace prefix |
| `orion/substrate/execution_loop/ids.py` | `parse_execution_trace_id` |
| `orion/substrate/execution_loop/grammar_extract.py` | `extract_execution_state_from_events` |
| `orion/substrate/execution_loop/projection.py` | Empty projection helper |
| `orion/substrate/execution_loop/reducer.py` | `reduce_execution_trace_events` |
| `orion/substrate/execution_loop/pipeline.py` | `process_execution_grammar_events` |
| `orion/substrate/execution_loop/__init__.py` | Package exports |
| `services/orion-sql-db/manual_migration_execution_substrate_loop.sql` | `substrate_execution_trajectory_projection` DDL |
| `services/orion-sql-writer/app/models/biometrics_substrate.py` | Add `SubstrateExecutionTrajectoryProjectionSQL` (optional parity) |
| `services/orion-substrate-runtime/app/store.py` | Execution fetch, projection load/save, execution cursor advance |
| `services/orion-substrate-runtime/app/worker.py` | Execution tick beside biometrics |
| `services/orion-substrate-runtime/app/settings.py` | `ENABLE_EXECUTION_TRAJECTORY_REDUCER` |
| `services/orion-substrate-runtime/.env_example` + local `.env` | New flag |
| `services/orion-substrate-runtime/docker-compose.yml` | Env passthrough |
| `services/orion-substrate-runtime/README.md` | Execution loop docs |
| `services/orion-field-digester/app/ingest/state_deltas.py` | `execution_run` perturbation mapping |
| `config/field/biometrics_lattice.yaml` | Execution node/capability channels + edges |
| `services/orion-field-digester/README.md` | Note lattice is no longer biometrics-only |
| `tests/test_execution_substrate_reducer.py` | Reducer + stable id tests |
| `tests/test_execution_substrate_pipeline.py` | Pipeline integration |
| `tests/test_field_execution_perturbations.py` | Field ingest + diffusion |

---

# Phase 0 — Worktree + branch

### Task 0: Create isolated worktree

- [ ] **Step 1: Create worktree from `main`**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-cortex-exec-substrate-digestion-v1 \
  -b feat/cortex-exec-substrate-digestion-v1 \
  main
cd .worktrees/feat-cortex-exec-substrate-digestion-v1
```

Expected: `git branch --show-current` → `feat/cortex-exec-substrate-digestion-v1`

- [ ] **Step 2: Verify isolation**

```bash
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "chore: start cortex-exec substrate digestion worktree"
```

---

# Phase 1 — Execution projection schemas

### Task 1: ExecutionTrajectoryProjectionV1 models

**Files:**
- Create: `orion/schemas/execution_projection.py`
- Modify: `orion/schemas/registry.py`
- Test: `tests/test_execution_projection_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_execution_projection_schemas.py
from datetime import datetime, timezone

from orion.schemas.execution_projection import (
    ExecutionRunStateV1,
    ExecutionTrajectoryProjectionV1,
)


def test_execution_projection_roundtrip() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    run = ExecutionRunStateV1(
        trace_id="cortex.exec:athena:corr-1",
        correlation_id="corr-1",
        node_id="athena",
        status="success",
        step_count=2,
        started_step_count=2,
        completed_step_count=2,
        failed_step_count=0,
        pressure_hints={"execution_load": 0.25},
        evidence_event_ids=["gev_1"],
        last_updated_at=now,
    )
    proj = ExecutionTrajectoryProjectionV1(
        projection_id="active_execution_trajectory",
        generated_at=now,
        runs={"cortex.exec:athena:corr-1": run},
    )
    data = proj.model_dump(mode="json")
    assert ExecutionTrajectoryProjectionV1.model_validate(data).runs["cortex.exec:athena:corr-1"].node_id == "athena"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd .worktrees/feat-cortex-exec-substrate-digestion-v1
PYTHONPATH=. pytest tests/test_execution_projection_schemas.py -v
```

Expected: FAIL `ModuleNotFoundError: orion.schemas.execution_projection`

- [ ] **Step 3: Write minimal implementation**

```python
# orion/schemas/execution_projection.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRunStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    correlation_id: str
    session_id: str | None = None
    turn_id: str | None = None
    node_id: str
    verb: str = "unknown"
    mode: str = "unknown"
    status: str = "unknown"
    step_count: int = 0
    started_step_count: int = 0
    completed_step_count: int = 0
    failed_step_count: int = 0
    recall_observed: bool = False
    final_text_present: bool = False
    reasoning_present: bool = False
    thinking_source: str = "none"
    pressure_hints: dict[str, float] = Field(default_factory=dict)
    evidence_event_ids: list[str] = Field(default_factory=list)
    last_updated_at: datetime


class ExecutionTrajectoryProjectionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["projection.execution_trajectory.v1"] = (
        "projection.execution_trajectory.v1"
    )
    projection_id: str
    generated_at: datetime
    runs: dict[str, ExecutionRunStateV1] = Field(default_factory=dict)
```

Register in `orion/schemas/registry.py` (near biometrics projection imports):

```python
from orion.schemas.execution_projection import (
    ExecutionRunStateV1,
    ExecutionTrajectoryProjectionV1,
)
# In SCHEMA_REGISTRY dict:
"ExecutionRunStateV1": ExecutionRunStateV1,
"ExecutionTrajectoryProjectionV1": ExecutionTrajectoryProjectionV1,
```

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. pytest tests/test_execution_projection_schemas.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/execution_projection.py orion/schemas/registry.py tests/test_execution_projection_schemas.py
git commit -m "feat(schemas): add execution trajectory projection models"
```

---

# Phase 2 — Execution loop package (extract + reducer)

### Task 2: constants + trace id parser

**Files:**
- Create: `orion/substrate/execution_loop/constants.py`
- Create: `orion/substrate/execution_loop/ids.py`
- Create: `orion/substrate/execution_loop/__init__.py`

- [ ] **Step 1: Write constants and ids**

```python
# orion/substrate/execution_loop/constants.py
EXECUTION_TRAJECTORY_PROJECTION_ID = "active_execution_trajectory"
EXECUTION_GRAMMAR_CURSOR_NAME = "execution_grammar_reducer"
EXECUTION_SOURCE_SERVICE = "orion-cortex-exec"
EXECUTION_TRACE_PREFIX = "cortex.exec:"
EXECUTION_REDUCER_ID = "execution_trajectory_reducer"
```

```python
# orion/substrate/execution_loop/ids.py
from __future__ import annotations


def parse_execution_trace_id(trace_id: str) -> tuple[str, str] | None:
    if not trace_id.startswith("cortex.exec:"):
        return None
    parts = trace_id.split(":", 2)
    if len(parts) < 3 or not parts[1].strip() or not parts[2].strip():
        return None
    return parts[1].strip().lower(), parts[2]
```

```python
# orion/substrate/execution_loop/__init__.py
from .pipeline import process_execution_grammar_events

__all__ = ["process_execution_grammar_events"]
```

- [ ] **Step 2: Commit**

```bash
git add orion/substrate/execution_loop/
git commit -m "feat(substrate): add execution loop package skeleton"
```

---

### Task 3: grammar_extract + pressure hints

**Files:**
- Create: `orion/substrate/execution_loop/grammar_extract.py`
- Test: `tests/test_execution_substrate_reducer.py` (first tests)

- [ ] **Step 1: Write failing tests for extract + hints**

```python
# tests/test_execution_substrate_reducer.py (excerpt — add at top)
from datetime import datetime, timezone

import pytest

from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.grammar_extract import (
    compute_pressure_hints,
    extract_execution_state_from_events,
)

FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
TRACE = "cortex.exec:athena:corr-abc"


def _exec_atom(role: str, summary: str, *, event_id: str = "gev_x") -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{TRACE}:{role}",
        trace_id=TRACE,
        atom_type="observation",
        semantic_role=role,
        layer="execution",
        summary=summary,
    )
    return GrammarEventV1(
        event_id=event_id,
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=atom,
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
        ),
        correlation_id="corr-abc",
        session_id="sess-1",
        turn_id="turn-1",
    )


def test_extract_builds_run_state_from_exec_atoms() -> None:
    events = [
        _exec_atom("exec_plan_started", "Execution plan started for verb=chat_general; step_count=2; depth=none", event_id="gev_1"),
        _exec_atom("exec_step_started", "Step started: order=1, step=step_1, verb=chat_general, services=LLMGatewayService", event_id="gev_2"),
        _exec_atom("exec_step_completed", "Step completed: step=step_1, status=success, latency_ms=120, result_services=LLMGatewayService", event_id="gev_3"),
        _exec_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning",
            event_id="gev_4",
        ),
        _exec_atom("exec_result_emitted", "Cortex exec result emitted to reply_to=True, status=success", event_id="gev_5"),
    ]
    run = extract_execution_state_from_events(events, now=FIXED_TS)
    assert run.trace_id == TRACE
    assert run.node_id == "athena"
    assert run.status == "success"
    assert run.started_step_count == 1
    assert run.completed_step_count == 1
    assert run.final_text_present is True
    assert run.reasoning_present is True
    assert run.thinking_source == "provider_reasoning"
    assert "gev_5" in run.evidence_event_ids


def test_pressure_hints_reasoning_and_egress() -> None:
    run = extract_execution_state_from_events(
        [_exec_atom("exec_result_assembled", "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=provider_reasoning")],
        now=FIXED_TS,
    )
    hints = compute_pressure_hints(run, egress_emitted=False)
    assert hints["reasoning_load"] == 0.35
    assert hints["egress_confidence"] == 0.25


def test_pressure_hints_failure_pressure() -> None:
    run = extract_execution_state_from_events(
        [_exec_atom("exec_step_failed", "Step failed: step=step_1, error_kind=timeout")],
        now=FIXED_TS,
    )
    hints = compute_pressure_hints(run, egress_emitted=True)
    assert hints["failure_pressure"] == 1.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py -v
```

- [ ] **Step 3: Implement grammar_extract.py**

```python
# orion/substrate/execution_loop/grammar_extract.py
from __future__ import annotations

import re
from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionRunStateV1
from orion.schemas.grammar import GrammarEventV1

from .constants import EXECUTION_SOURCE_SERVICE
from .ids import parse_execution_trace_id

_KV_RE = re.compile(r"(\w+)=([^,;\s]+)")


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def _parse_summary_kv(summary: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, val in _KV_RE.findall(summary or ""):
        out[key.lower()] = val.strip()
    return out


def _boolish(val: str | None) -> bool:
    return str(val or "").strip().lower() in {"true", "1", "yes", "on"}


def compute_pressure_hints(
    run: ExecutionRunStateV1,
    *,
    egress_emitted: bool,
) -> dict[str, float]:
    started = max(0, run.started_step_count)
    failed = max(0, run.failed_step_count)
    execution_load = min(1.0, started / 8.0)
    execution_friction = min(1.0, failed / max(1, started))
    reasoning_load = 0.35 if run.reasoning_present else 0.05
    status_fail = run.status.lower() in {"fail", "partial", "failed", "error"}
    failure_pressure = 1.0 if status_fail or failed > 0 else 0.0
    egress_confidence = 1.0 if egress_emitted else 0.25
    return {
        "execution_load": execution_load,
        "execution_friction": execution_friction,
        "reasoning_load": reasoning_load,
        "failure_pressure": failure_pressure,
        "egress_confidence": egress_confidence,
    }


def extract_execution_state_from_events(
    events: list[GrammarEventV1],
    *,
    now: datetime | None = None,
) -> ExecutionRunStateV1:
    clock = _utc_now(now)
    if not events:
        raise ValueError("events must not be empty")

    trace_id = events[0].trace_id
    parsed = parse_execution_trace_id(trace_id or "")
    node_id = parsed[0] if parsed else "unknown"
    correlation_id = parsed[1] if parsed else (events[0].correlation_id or "unknown")

    run = ExecutionRunStateV1(
        trace_id=trace_id or "",
        correlation_id=correlation_id,
        session_id=events[0].session_id,
        turn_id=events[0].turn_id,
        node_id=node_id,
        last_updated_at=clock,
    )

    egress_emitted = False
    for event in events:
        if event.provenance.source_service != EXECUTION_SOURCE_SERVICE:
            continue
        atom = event.atom
        if not atom:
            continue
        role = atom.semantic_role or ""
        kv = _parse_summary_kv(atom.summary or "")
        run.evidence_event_ids.append(event.event_id)

        if role == "exec_request_received":
            run.verb = kv.get("verb", run.verb)
            run.mode = kv.get("mode", run.mode)
        elif role == "exec_plan_started":
            run.step_count = int(kv.get("step_count", run.step_count) or run.step_count)
        elif role == "exec_recall_gate_observed":
            run.recall_observed = True
        elif role == "exec_step_started":
            run.started_step_count += 1
        elif role == "exec_step_completed":
            run.completed_step_count += 1
        elif role == "exec_step_failed":
            run.failed_step_count += 1
        elif role == "exec_result_assembled":
            run.status = kv.get("status", run.status)
            run.final_text_present = _boolish(kv.get("final_text_present"))
            run.reasoning_present = _boolish(kv.get("reasoning_present"))
            run.thinking_source = kv.get("thinking_source", run.thinking_source)
        elif role == "exec_result_emitted":
            egress_emitted = True

    run.pressure_hints = compute_pressure_hints(run, egress_emitted=egress_emitted)
    return run
```

- [ ] **Step 4: Run tests — expect PASS for extract tests**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py::test_extract_builds_run_state_from_exec_atoms tests/test_execution_substrate_reducer.py::test_pressure_hints_reasoning_and_egress tests/test_execution_substrate_reducer.py::test_pressure_hints_failure_pressure -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/execution_loop/grammar_extract.py tests/test_execution_substrate_reducer.py
git commit -m "feat(substrate): execution grammar extract and pressure hints"
```

---

### Task 4: reducer + stable ids

**Files:**
- Create: `orion/substrate/execution_loop/projection.py`
- Create: `orion/substrate/execution_loop/reducer.py`
- Modify: `tests/test_execution_substrate_reducer.py`

- [ ] **Step 1: Add failing reducer tests**

```python
# Append to tests/test_execution_substrate_reducer.py
from copy import deepcopy

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.reducer import reduce_execution_trace_events


def _empty_projection() -> ExecutionTrajectoryProjectionV1:
    return ExecutionTrajectoryProjectionV1(
        projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
        generated_at=FIXED_TS,
        runs={},
    )


def test_reducer_emits_execution_run_delta() -> None:
    events = [
        _exec_atom("exec_plan_started", "Execution plan started for verb=chat_general; step_count=1; depth=none"),
        _exec_atom("exec_result_emitted", "Cortex exec result emitted to reply_to=True, status=success"),
    ]
    proj, receipt = reduce_execution_trace_events(
        events=events,
        projection=_empty_projection(),
        now=FIXED_TS,
    )
    assert receipt.accepted_event_ids
    assert len(receipt.state_deltas) == 1
    delta = receipt.state_deltas[0]
    assert delta.target_kind == "execution_run"
    assert delta.target_id == TRACE
    assert delta.after["node_id"] == "athena"
    assert "execution_load" in delta.after["pressure_hints"]
    assert TRACE in proj.runs


def test_reducer_noops_non_cortex_exec() -> None:
    bio = GrammarEventV1(
        event_id="gev_bio",
        event_kind="atom_emitted",
        trace_id="biometrics.node:atlas:ts",
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id="a1",
            trace_id="biometrics.node:atlas:ts",
            atom_type="signal",
            semantic_role="body_state",
            layer="biometrics",
            summary="body",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-biometrics", source_component="x"),
    )
    proj, receipt = reduce_execution_trace_events(events=[bio], projection=_empty_projection(), now=FIXED_TS)
    assert proj.runs == {}
    assert receipt.noop_event_ids == ["gev_bio"]


def test_stable_delta_id_on_replay() -> None:
    events = [_exec_atom("exec_result_emitted", "Cortex exec result emitted to reply_to=True, status=success")]
    _, r1 = reduce_execution_trace_events(events=events, projection=_empty_projection(), now=FIXED_TS)
    _, r2 = reduce_execution_trace_events(events=events, projection=_empty_projection(), now=FIXED_TS)
    assert r1.receipt_id == r2.receipt_id
    assert r1.state_deltas[0].delta_id == r2.state_deltas[0].delta_id
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py -v -k "reducer"
```

- [ ] **Step 3: Implement projection.py + reducer.py**

```python
# orion/substrate/execution_loop/projection.py
from __future__ import annotations

from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1

from .constants import EXECUTION_TRAJECTORY_PROJECTION_ID


def empty_execution_projection(*, now: datetime | None = None) -> ExecutionTrajectoryProjectionV1:
    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return ExecutionTrajectoryProjectionV1(
        projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
        generated_at=clock,
        runs={},
    )
```

```python
# orion/substrate/execution_loop/reducer.py
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarEventV1
from orion.schemas.reduction_receipt import ProjectionUpdateV1, ReductionReceiptV1
from orion.schemas.state_delta import StateDeltaV1
from orion.substrate.ids import stable_delta_id, stable_receipt_id

from .constants import (
    EXECUTION_REDUCER_ID,
    EXECUTION_SOURCE_SERVICE,
    EXECUTION_TRAJECTORY_PROJECTION_ID,
)
from .grammar_extract import extract_execution_state_from_events
from .ids import parse_execution_trace_id


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def reduce_execution_trace_events(
    *,
    events: list[GrammarEventV1],
    projection: ExecutionTrajectoryProjectionV1,
    now: datetime | None = None,
    reducer_id: str = EXECUTION_REDUCER_ID,
) -> tuple[ExecutionTrajectoryProjectionV1, ReductionReceiptV1]:
    clock = _utc_now(now)
    if not events:
        receipt = ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=[],
            ),
            noop_event_ids=[],
            created_at=clock,
        )
        return projection, receipt

    trace_id = events[0].trace_id or ""
    if not parse_execution_trace_id(trace_id):
        noop_ids = [e.event_id for e in events]
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=noop_ids,
            ),
            noop_event_ids=noop_ids,
            created_at=clock,
        )

    if any(e.provenance.source_service != EXECUTION_SOURCE_SERVICE for e in events):
        noop_ids = [e.event_id for e in events]
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=noop_ids,
            ),
            noop_event_ids=noop_ids,
            created_at=clock,
        )

    updated = deepcopy(projection)
    updated.generated_at = clock
    if updated.projection_id != EXECUTION_TRAJECTORY_PROJECTION_ID:
        updated.projection_id = EXECUTION_TRAJECTORY_PROJECTION_ID

    warnings: list[str] = []
    try:
        merged = extract_execution_state_from_events(events, now=clock)
    except ValueError as exc:
        warnings.append(str(exc))
        noop_ids = [e.event_id for e in events]
        return projection, ReductionReceiptV1(
            receipt_id=stable_receipt_id(
                reducer_id=reducer_id,
                accepted_event_ids=[],
                rejected_event_ids=[],
                merged_event_ids=[],
                noop_event_ids=noop_ids,
            ),
            noop_event_ids=noop_ids,
            warnings=warnings,
            created_at=clock,
        )

    existing = updated.runs.get(trace_id)
    operation = "create" if existing is None else "update"
    updated.runs[trace_id] = merged

    event_ids = [e.event_id for e in events if e.atom]
    receipt = ReductionReceiptV1(
        receipt_id=stable_receipt_id(
            reducer_id=reducer_id,
            accepted_event_ids=event_ids,
            rejected_event_ids=[],
            merged_event_ids=[],
            noop_event_ids=[],
        ),
        accepted_event_ids=event_ids,
        state_deltas=[
            StateDeltaV1(
                delta_id=stable_delta_id(
                    reducer_id=reducer_id,
                    target_projection=EXECUTION_TRAJECTORY_PROJECTION_ID,
                    target_kind="execution_run",
                    target_id=trace_id,
                    operation=operation,
                    caused_by_event_ids=event_ids,
                ),
                target_projection=EXECUTION_TRAJECTORY_PROJECTION_ID,
                target_kind="execution_run",
                target_id=trace_id,
                operation=operation,
                before=existing.model_dump(mode="json") if existing else None,
                after=merged.model_dump(mode="json"),
                caused_by_event_ids=event_ids,
                reducer_id=reducer_id,
            )
        ],
        projection_updates=[
            ProjectionUpdateV1(
                projection_kind="execution_trajectory",
                projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
                node_id=merged.node_id,
                operation=operation,
            )
        ],
        warnings=warnings,
        created_at=clock,
    )
    return updated, receipt
```

- [ ] **Step 4: Run full reducer test file**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/execution_loop/projection.py orion/substrate/execution_loop/reducer.py tests/test_execution_substrate_reducer.py
git commit -m "feat(substrate): execution trajectory reducer with stable ids"
```

---

### Task 5: pipeline

**Files:**
- Create: `orion/substrate/execution_loop/pipeline.py`
- Test: `tests/test_execution_substrate_pipeline.py`

- [ ] **Step 1: Write failing pipeline test**

```python
# tests/test_execution_substrate_pipeline.py
from datetime import datetime, timezone

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarAtomV1, GrammarEventV1, GrammarProvenanceV1
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.pipeline import process_execution_grammar_events

FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
TRACE = "cortex.exec:athena:corr-pipe"


def _event(role: str, eid: str) -> GrammarEventV1:
    return GrammarEventV1(
        event_id=eid,
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        observed_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{TRACE}:{role}",
            trace_id=TRACE,
            atom_type="observation",
            semantic_role=role,
            layer="execution",
            summary=f"stub {role}",
        ),
        provenance=GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
        ),
        correlation_id="corr-pipe",
    )


def test_pipeline_groups_by_trace_and_persists_receipts() -> None:
    state = {
        "projection": ExecutionTrajectoryProjectionV1(
            projection_id=EXECUTION_TRAJECTORY_PROJECTION_ID,
            generated_at=FIXED_TS,
            runs={},
        ),
        "receipts": [],
    }

    stats = process_execution_grammar_events(
        events=[
            _event("exec_plan_started", "gev_1"),
            _event("exec_result_emitted", "gev_2"),
        ],
        load_projection=lambda: state["projection"],
        save_projection=lambda p: state.update(projection=p),
        save_receipt=lambda r: state["receipts"].append(r),
        now=FIXED_TS,
    )

    assert stats["events"] == 2
    assert stats["receipts"] == 1
    assert TRACE in state["projection"].runs
    assert state["receipts"][0].state_deltas[0].target_kind == "execution_run"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_pipeline.py -v
```

- [ ] **Step 3: Implement pipeline.py**

```python
# orion/substrate/execution_loop/pipeline.py
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable

from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.schemas.grammar import GrammarEventV1

from .projection import empty_execution_projection
from .reducer import reduce_execution_trace_events


ExecutionProjectionLoader = Callable[[], ExecutionTrajectoryProjectionV1]
ExecutionProjectionSaver = Callable[[ExecutionTrajectoryProjectionV1], None]
ReceiptSaver = Callable[[Any], None]


def process_execution_grammar_events(
    *,
    events: list[GrammarEventV1],
    load_projection: ExecutionProjectionLoader,
    save_projection: ExecutionProjectionSaver,
    save_receipt: ReceiptSaver,
    now: datetime | None = None,
) -> dict[str, int]:
    clock = now or datetime.now(timezone.utc)
    stats = {"events": 0, "receipts": 0, "traces": 0}

    by_trace: dict[str, list[GrammarEventV1]] = defaultdict(list)
    for event in events:
        stats["events"] += 1
        by_trace[event.trace_id or ""].append(event)

    projection = load_projection()
    for trace_id, trace_events in by_trace.items():
        if not trace_id:
            continue
        stats["traces"] += 1
        projection, receipt = reduce_execution_trace_events(
            events=trace_events,
            projection=projection,
            now=clock,
        )
        save_receipt(receipt)
        stats["receipts"] += 1

    save_projection(projection)
    return stats
```

Update `__init__.py` export if needed.

- [ ] **Step 4: Run pipeline test**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_pipeline.py -q
```

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/execution_loop/pipeline.py tests/test_execution_substrate_pipeline.py
git commit -m "feat(substrate): execution grammar pipeline grouped by trace"
```

---

# Phase 3 — Substrate runtime worker + SQL

### Task 6: SQL migration

**Files:**
- Create: `services/orion-sql-db/manual_migration_execution_substrate_loop.sql`

- [ ] **Step 1: Add migration file**

```sql
-- Execution substrate trajectory projection (apply before enabling execution reducer)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql

create table if not exists substrate_execution_trajectory_projection (
    projection_id text primary key,
    generated_at timestamptz not null,
    projection_json jsonb not null,
    created_at timestamptz not null default now()
);
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-sql-db/manual_migration_execution_substrate_loop.sql
git commit -m "chore(sql): execution trajectory projection table"
```

---

### Task 7: Extend BiometricsSubstrateStore + worker

**Files:**
- Modify: `services/orion-substrate-runtime/app/store.py`
- Modify: `services/orion-substrate-runtime/app/worker.py`
- Modify: `services/orion-substrate-runtime/app/settings.py`
- Modify: `services/orion-substrate-runtime/.env_example`
- Modify: `services/orion-substrate-runtime/docker-compose.yml`
- Modify: `services/orion-substrate-runtime/README.md`
- Copy: `services/orion-substrate-runtime/.env` (local only, not committed)

- [ ] **Step 1: Extend store — add imports and execution cursor constant usage**

At top of `store.py`, add:

```python
from orion.schemas.execution_projection import ExecutionTrajectoryProjectionV1
from orion.substrate.execution_loop.constants import EXECUTION_GRAMMAR_CURSOR_NAME
```

Add methods (mirror biometrics fetch pattern; use `EXECUTION_GRAMMAR_CURSOR_NAME`):

```python
def fetch_execution_grammar_events(self, *, limit: int = 50) -> list[GrammarEventV1]:
    # Same cursor table; cursor_name = EXECUTION_GRAMMAR_CURSOR_NAME
    # WHERE source_service = 'orion-cortex-exec' AND trace_id LIKE 'cortex.exec:%'

def advance_execution_cursor(self, *, event_id: str, created_at: datetime) -> None:
    # Same INSERT/ON CONFLICT as advance_cursor but cursor_name = EXECUTION_GRAMMAR_CURSOR_NAME

def load_execution_trajectory(self, projection_id: str) -> ExecutionTrajectoryProjectionV1 | None:
    return self._load_projection(
        "substrate_execution_trajectory_projection",
        projection_id,
        ExecutionTrajectoryProjectionV1,
    )

def save_execution_trajectory(self, projection: ExecutionTrajectoryProjectionV1) -> None:
    self._save_projection("substrate_execution_trajectory_projection", projection)
```

Refactor existing `advance_cursor` to remain biometrics-only (unchanged behavior).

- [ ] **Step 2: Extend settings**

```python
# services/orion-substrate-runtime/app/settings.py
enable_execution_trajectory_reducer: bool = Field(
    False,
    alias="ENABLE_EXECUTION_TRAJECTORY_REDUCER",
)
```

- [ ] **Step 3: Extend worker — execution tick**

In `worker.py`:

```python
from orion.substrate.execution_loop.constants import EXECUTION_TRAJECTORY_PROJECTION_ID
from orion.substrate.execution_loop.pipeline import process_execution_grammar_events
from orion.substrate.execution_loop.projection import empty_execution_projection
```

In `_poll_loop`, after biometrics tick:

```python
if self._settings.enable_execution_trajectory_reducer:
    last_exec_id = await asyncio.to_thread(self._execution_tick)
    if last_exec_id:
        created_at = self._store.grammar_event_created_at(last_exec_id)
        if created_at:
            self._store.advance_execution_cursor(
                event_id=last_exec_id,
                created_at=created_at,
            )
```

Add `_execution_tick` mirroring `_tick` but calling `fetch_execution_grammar_events` + `process_execution_grammar_events`.

- [ ] **Step 4: Update .env_example and docker-compose**

```bash
# services/orion-substrate-runtime/.env_example (append)
ENABLE_EXECUTION_TRAJECTORY_REDUCER=false
```

```yaml
# docker-compose.yml environment section
- ENABLE_EXECUTION_TRAJECTORY_REDUCER=${ENABLE_EXECUTION_TRAJECTORY_REDUCER:-false}
```

Sync to local `.env` (not committed):

```bash
grep -q ENABLE_EXECUTION_TRAJECTORY_REDUCER services/orion-substrate-runtime/.env || \
  echo "ENABLE_EXECUTION_TRAJECTORY_REDUCER=false" >> services/orion-substrate-runtime/.env
```

- [ ] **Step 5: README note**

Document execution loop, migration apply command, and flag default `false` for safe rollout.

- [ ] **Step 6: Run regression tests**

```bash
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py tests/test_execution_substrate_pipeline.py tests/test_biometrics_pipeline.py tests/test_node_pressure_reducer.py -q
```

- [ ] **Step 7: Commit**

```bash
git add services/orion-substrate-runtime/
git commit -m "feat(substrate-runtime): poll and reduce cortex-exec grammar events"
```

---

# Phase 4 — Field digester execution perturbations

### Task 8: state_deltas + lattice

**Files:**
- Modify: `services/orion-field-digester/app/ingest/state_deltas.py`
- Modify: `config/field/biometrics_lattice.yaml`
- Modify: `services/orion-field-digester/README.md`
- Test: `tests/test_field_execution_perturbations.py`

- [ ] **Step 1: Write failing field test**

```python
# tests/test_field_execution_perturbations.py
from datetime import datetime, timezone
from pathlib import Path

from orion.schemas.state_delta import StateDeltaV1

from app.ingest.state_deltas import delta_to_perturbations
from app.graph.lattice import load_lattice
from app.tensor.update_rules import apply_digestion_tick

REPO_ROOT = Path(__file__).resolve().parents[1]
LATTICE = REPO_ROOT / "config" / "field" / "biometrics_lattice.yaml"
FIXED_TS = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_execution_run_delta_maps_to_node_channels() -> None:
    delta = StateDeltaV1(
        delta_id="delta_exec_1",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id="cortex.exec:athena:corr-1",
        operation="update",
        after={
            "node_id": "athena",
            "pressure_hints": {
                "execution_load": 0.45,
                "execution_friction": 0.05,
                "reasoning_load": 0.35,
                "failure_pressure": 0.0,
            },
        },
        caused_by_event_ids=["gev_1"],
        reducer_id="execution_trajectory_reducer",
    )
    perturbations = delta_to_perturbations(delta)
    channels = {p.channel: p.intensity for p in perturbations}
    assert channels["execution_load"] == 0.45
    assert channels["reasoning_load"] == 0.35
    assert perturbations[0].node_id == "node:athena"


def test_execution_perturbations_diffuse_to_orchestration_capability() -> None:
    lattice = load_lattice(LATTICE)
    delta = StateDeltaV1(
        delta_id="delta_exec_2",
        target_projection="active_execution_trajectory",
        target_kind="execution_run",
        target_id="cortex.exec:athena:corr-2",
        operation="update",
        after={
            "node_id": "athena",
            "pressure_hints": {"execution_load": 0.8, "execution_friction": 0.1, "failure_pressure": 0.2},
        },
        caused_by_event_ids=["gev_2"],
        reducer_id="execution_trajectory_reducer",
    )
    from app.tensor.field_state import empty_field_state

    field = empty_field_state(lattice=lattice, generated_at=FIXED_TS)
    field = apply_digestion_tick(field, perturbations=delta_to_perturbations(delta), lattice=lattice)
    cap = field.capability_vectors.get("orchestration") or {}
    assert cap.get("execution_pressure", 0.0) > 0.0
```

- [ ] **Step 2: Run — expect FAIL**

```bash
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_execution_perturbations.py -v
```

- [ ] **Step 3: Extend state_deltas.py**

After `node_biometrics` block in `delta_to_perturbations`, add:

```python
    if delta.target_kind == "execution_run":
        hints = dict(after.get("pressure_hints") or {})
        node_key = _node_key(str(after.get("node_id") or delta.target_id))
        for channel, key in (
            ("execution_load", "execution_load"),
            ("execution_friction", "execution_friction"),
            ("reasoning_load", "reasoning_load"),
            ("failure_pressure", "failure_pressure"),
        ):
            if key in hints:
                out.append(
                    Perturbation(
                        node_id=node_key,
                        channel=channel,
                        intensity=float(hints[key]),
                        label=delta.delta_id,
                    )
                )
```

- [ ] **Step 4: Extend biometrics_lattice.yaml**

Add to `node_channels`:

```yaml
  - execution_load
  - execution_friction
  - reasoning_load
  - failure_pressure
```

Add to `capability_channels`:

```yaml
  - execution_pressure
  - reasoning_pressure
  - reliability_pressure
```

Add edges (after existing athena orchestration edge — extend channel_map):

```yaml
  - source_id: node:athena
    target_id: capability:orchestration
    edge_type: node_capability
    weight: 0.85
    channel_map:
      cpu_pressure: pressure
      execution_load: execution_pressure
      execution_friction: reliability_pressure
      failure_pressure: reliability_pressure

  - source_id: node:atlas
    target_id: capability:llm_inference
    edge_type: node_capability
    weight: 0.80
    channel_map:
      gpu_pressure: pressure
      reasoning_load: reasoning_pressure
```

**Note:** If duplicate `node:athena → orchestration` edge exists, merge channel_maps into one edge instead of duplicating.

- [ ] **Step 5: Run field tests**

```bash
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_execution_perturbations.py -q
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_*.py -q
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-field-digester/app/ingest/state_deltas.py config/field/biometrics_lattice.yaml services/orion-field-digester/README.md tests/test_field_execution_perturbations.py
git commit -m "feat(field-digester): ingest execution_run deltas into lattice"
```

---

# Phase 5 — Full verification + PR

### Task 9: Full test suite

- [ ] **Step 1: Run all required tests**

```bash
cd .worktrees/feat-cortex-exec-substrate-digestion-v1
PYTHONPATH=. pytest tests/test_execution_substrate_reducer.py -q
PYTHONPATH=. pytest tests/test_execution_substrate_pipeline.py -q
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_execution_perturbations.py -q
PYTHONPATH=. pytest tests/test_node_pressure_reducer.py tests/test_biometrics_pipeline.py -q
PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_*.py -q
```

Expected: all PASS

- [ ] **Step 2: Commit any fixes**

```bash
git status
# commit if needed
```

---

### Task 10: Code review subagent + PR

- [ ] **Step 1: Dispatch `requesting-code-review` subagent**

Review scope:
- Execution loop does not read raw grammar in field-digester
- Biometrics path unchanged
- Stable ids on replay
- No secrets in commits
- Lattice YAML valid

- [ ] **Step 2: Fix all review findings**

- [ ] **Step 3: Write PR report**

Create `docs/superpowers/pr-reports/2026-05-24-cortex-exec-substrate-digestion-v1-pr.md` with:
- Summary (bridge diagram)
- Files changed table
- Test commands + results
- Live verification checklist (`PUBLISH_CORTEX_EXEC_GRAMMAR=true`, `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true`, migration applied)
- Rollout notes (flags default false)

- [ ] **Step 4: Push and open PR**

```bash
git push -u origin feat/cortex-exec-substrate-digestion-v1
gh pr create --base main --title "PR: Cortex exec substrate digestion v1" --body "$(cat <<'EOF'
## Summary
- Adds execution substrate reducer: cortex-exec grammar → `StateDeltaV1(target_kind=execution_run)` receipts
- Extends field-digester to perturb execution_load/reasoning_load channels and diffuse to orchestration/llm_inference capabilities
- Biometrics digestion unchanged; no organ, no mind service

## Test plan
- [ ] `PYTHONPATH=. pytest tests/test_execution_substrate_*.py -q`
- [ ] `PYTHONPATH=.:services/orion-field-digester pytest tests/test_field_execution_perturbations.py -q`
- [ ] Biometrics regression: `tests/test_biometrics_pipeline.py`, `tests/test_field_*.py`
- [ ] Apply `manual_migration_execution_substrate_loop.sql`
- [ ] Live: exec chat → grammar_events → receipt → field node/capability pressure in Hub

EOF
)"
```

---

## Self-review checklist

| Spec requirement | Task |
|------------------|------|
| execution_loop package | Tasks 2–5 |
| ExecutionTrajectoryProjectionV1 | Task 1 |
| grammar extract safe summaries only | Task 3 |
| deterministic pressure hints | Task 3 |
| reducer StateDelta execution_run | Task 4 |
| substrate-runtime separate cursor | Task 7 |
| SQL migration | Task 6 |
| field digester execution_run mapping | Task 8 |
| lattice channels/edges | Task 8 |
| 10 test assertions | Tasks 3–5, 8 |
| biometrics unchanged | Task 9 regression |
| worktree isolation | Phase 0 |
| code review + PR md | Task 10 |

**Placeholder scan:** No TBD/TODO/similar-to tasks.

**Type consistency:** `execution_run`, `active_execution_trajectory`, `EXECUTION_TRAJECTORY_PROJECTION_ID` aligned across reducer, pipeline, field ingest.

---

## Live verification notes (post-merge)

1. `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_execution_substrate_loop.sql`
2. Set `PUBLISH_CORTEX_EXEC_GRAMMAR=true` on cortex-exec; `ENABLE_EXECUTION_TRAJECTORY_REDUCER=true` on substrate-runtime
3. Run one plan execution; confirm `substrate_reduction_receipts` contains `target_kind":"execution_run"`
4. Confirm field-digester advances; `GET /api/substrate/field/node/athena` shows `execution_load` > 0
5. Confirm `GET /api/substrate/field/capability/orchestration` shows `execution_pressure` > 0
