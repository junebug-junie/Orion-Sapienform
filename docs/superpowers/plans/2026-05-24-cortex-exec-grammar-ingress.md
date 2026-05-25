# Cortex Exec Grammar Ingress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `orion-cortex-exec` emit valid `GrammarEventV1` execution traces on `orion:grammar:event` (shadow observability only) without changing plan execution behavior.

**Architecture:** Add a pure `grammar_emit.py` builder (mirroring `services/orion-biometrics/app/grammar_emit.py`) plus a fail-open `grammar_publish.py` wrapper around `orion.grammar.publish.publish_grammar_event`. Instrument `PlanRunner.run_plan()` for plan/step/result lifecycle and `main.handle()` for intake/egress atoms. Use closed `GrammarEventKind` values only; encode execution semantics in `GrammarAtomV1.semantic_role`. Legacy `legacy.plan` runs through `run_plan()` — no duplicate traces.

**Tech Stack:** Python 3.12, Pydantic v2 (`orion/schemas/grammar.py`, `orion/schemas/cortex/schemas.py`), Redis bus (`OrionBusAsync`, `BaseEnvelope`), `orion.grammar.publish`, pytest 8.3.x.

**Design source:** User handoff “Cortex Exec Grammar Ingress” (2026-05-24).

**Non-goals:** Field digester, reducers, projections, new `GrammarEventKind` / `RelationType`, fatal grammar publish, `executor.py` service internals, raw prompts/LLM blobs in atoms.

---

## Worktree isolation (mandatory)

All implementation commits happen **only** in a dedicated worktree. Do not checkout the feature branch in the main workspace.

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-cortex-exec-grammar-ingress \
  -b feat/cortex-exec-grammar-ingress origin/main
cd .worktrees/feat-cortex-exec-grammar-ingress
git check-ignore -q .worktrees   # must succeed
```

**Rules:**
- Never bleed changed files back to the main checkout except copying `.env` keys locally (see below).
- When editing `services/orion-cortex-exec/.env_example`, also copy new keys into `services/orion-cortex-exec/.env` in the **worktree** (gitignored; operator sync).
- PR and push from `feat/cortex-exec-grammar-ingress` only.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Grammar schemas | `orion/schemas/grammar.py` — closed `GrammarEventKind`, `AtomType`, `RelationType` |
| Shared publisher | `orion/grammar/publish.py` → `publish_grammar_event(bus, event, source_name=...)` |
| Reference emitter | `services/orion-biometrics/app/grammar_emit.py` |
| Primary seam | `PlanRunner.run_plan()` in `services/orion-cortex-exec/app/router.py` (~L820–L1322) |
| Exec intake | `handle()` in `services/orion-cortex-exec/app/main.py` (~L396–L575) |
| Legacy plan path | `LegacyPlanVerb.execute()` → `router.run_plan()` (`verb_adapters.py` ~L234) — **not** `handle()` |
| Grammar on bus today | `orion:grammar:event` producers: vision, hub, biometrics, substrate-runtime — **add `orion-cortex-exec`** |
| Registry | `GrammarEventV1` already in `orion/schemas/registry.py` — **no registry change** |
| Existing cortex grammar | None (`grep` clean) |

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-cortex-exec/app/grammar_emit.py` | Pure builder: `CortexExecGrammarCollector`, `build_cortex_exec_grammar_events()` |
| `services/orion-cortex-exec/app/grammar_publish.py` | Fail-open `publish_cortex_exec_grammar_trace()` |
| `services/orion-cortex-exec/app/settings.py` | `publish_cortex_exec_grammar`, `grammar_event_channel` |
| `services/orion-cortex-exec/app/router.py` | Collector lifecycle inside `run_plan()` |
| `services/orion-cortex-exec/app/main.py` | Intake/egress + validation-failure grammar in `handle()` |
| `services/orion-cortex-exec/.env_example` | New env keys |
| `services/orion-cortex-exec/.env` | Operator sync (not committed) |
| `services/orion-cortex-exec/docker-compose.yml` | Env passthrough |
| `services/orion-cortex-exec/README.md` | Grammar channel docs |
| `orion/bus/channels.yaml` | Add `orion-cortex-exec` producer on `orion:grammar:event` |
| `services/orion-cortex-exec/tests/test_exec_grammar_emit.py` | Builder/schema tests |
| `services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py` | Publish failure non-fatal |
| `scripts/smoke_cortex_exec_grammar.sh` | Test + bus tap instructions |

**Do not modify:** `orion/schemas/registry.py`, `executor.py` (unless import fix only), field digester, substrate reducer.

---

## Trace and id conventions

```text
trace_id = cortex.exec:{node_name}:{correlation_id}
atom_id  = {trace_id}:{semantic_role}   # stable per role per trace
edge_id  = {trace_id}:edge:{from_role}:{relation_type}:{to_role}
event_id = gev_{sha1(trace_id|event_kind|body_key)[:16]}  # same as biometrics
```

- `session_id` / `turn_id`: from `ctx` (`session_id`, `turn_id` or `message_id` as turn fallback).
- `correlation_id` on `GrammarEventV1`: string form of bus correlation id.
- `payload_ref` only — never embed prompts, `final_text`, or full `StepExecutionResult.result` dicts.

---

# Phase 1 — Settings and bus catalog

### Task 1: Settings + env + compose

**Files:**
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/.env_example`
- Modify: `services/orion-cortex-exec/.env` (worktree only, not committed)
- Modify: `services/orion-cortex-exec/docker-compose.yml`

- [ ] **Step 1: Add settings fields** (after `diagnostic_mode` block ~L84)

```python
    publish_cortex_exec_grammar: bool = Field(False, alias="PUBLISH_CORTEX_EXEC_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")
```

- [ ] **Step 2: Update `.env_example`**

```bash
# Substrate grammar ingress (shadow observability; default off)
PUBLISH_CORTEX_EXEC_GRAMMAR=false
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
```

- [ ] **Step 3: Sync `.env` in worktree** (same keys; set `PUBLISH_CORTEX_EXEC_GRAMMAR=true` locally if testing live bus)

- [ ] **Step 4: Pass through in `docker-compose.yml`** under `orion-cortex-exec` service `environment:`

```yaml
      - PUBLISH_CORTEX_EXEC_GRAMMAR=${PUBLISH_CORTEX_EXEC_GRAMMAR:-false}
      - GRAMMAR_EVENT_CHANNEL=${GRAMMAR_EVENT_CHANNEL:-orion:grammar:event}
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/settings.py \
  services/orion-cortex-exec/.env_example \
  services/orion-cortex-exec/docker-compose.yml
git commit -m "feat(cortex-exec): add grammar publish settings"
```

### Task 2: Bus channel catalog

**Files:**
- Modify: `orion/bus/channels.yaml` (entry `orion:grammar:event` ~L1789)

- [ ] **Step 1: Add producer**

```yaml
    producer_services:
      - orion-vision-retina
      - orion-hub
      - orion-vision-edge
      - orion-vision-window
      - orion-biometrics
      - orion-substrate-runtime
      - orion-cortex-exec
```

- [ ] **Step 2: Commit**

```bash
git add orion/bus/channels.yaml
git commit -m "chore(bus): register orion-cortex-exec as grammar event producer"
```

---

# Phase 2 — Pure grammar builder (TDD)

### Task 3: `grammar_emit.py` skeleton + collector

**Files:**
- Create: `services/orion-cortex-exec/app/grammar_emit.py`
- Create: `services/orion-cortex-exec/tests/test_exec_grammar_emit.py`

- [ ] **Step 1: Write failing tests** (`test_exec_grammar_emit.py`)

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args
from uuid import UUID

import pytest

from app.grammar_emit import (
    CortexExecGrammarCollector,
    build_cortex_exec_grammar_events,
    cortex_exec_trace_id,
)
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest, PlanStep
from orion.schemas.cortex.types import StepExecutionResult
from orion.schemas.grammar import AtomType, GrammarEventKind, RelationType

FIXED_OBS = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
CORR = "corr-abc-123"
NODE = "athena"


def _minimal_plan(*, steps: int = 2) -> PlanExecutionRequest:
    plan_steps = [
        PlanStep(
            step_name=f"step_{i}",
            verb_name="chat_general" if i == 1 else "noop",
            order=i,
            services=["LLMGatewayService"] if i == 1 else [],
        )
        for i in range(1, steps + 1)
    ]
    return PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="chat_general", steps=plan_steps),
        args={"request_id": "req-1", "extra": {"mode": "brain"}},
    )


def test_trace_id_stable_for_node_and_correlation() -> None:
    assert cortex_exec_trace_id(NODE, CORR) == f"cortex.exec:{NODE}:{CORR}"


def test_builds_valid_grammar_events_with_required_semantic_roles() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
        session_id="sess-1",
        turn_id="turn-9",
    )
    req = _minimal_plan()
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=2)
    collector.record_recall_gate_observed(
        run_recall=False,
        profile="assist.light.v1",
        reason="gating_disabled",
    )
    collector.record_step_started(
        order=1,
        step_name="step_1",
        verb_name="chat_general",
        services=["LLMGatewayService"],
    )
    collector.record_step_completed(
        order=1,
        step_name="step_1",
        latency_ms=120,
        result_service_keys=["LLMGatewayService"],
    )
    collector.record_step_started(
        order=2,
        step_name="step_2",
        verb_name="noop",
        services=[],
    )
    collector.record_step_completed(
        order=2,
        step_name="step_2",
        latency_ms=5,
        result_service_keys=[],
    )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="success")

    events = build_cortex_exec_grammar_events(collector)
    assert events
    kinds = {e.event_kind for e in events}
    assert kinds <= set(get_args(GrammarEventKind))
    assert "trace_started" in kinds
    assert "trace_ended" in kinds
    roles = {e.atom.semantic_role for e in events if e.atom}
    required = {
        "exec_request_received",
        "exec_plan_started",
        "exec_recall_gate_observed",
        "exec_step_started",
        "exec_step_completed",
        "exec_result_assembled",
        "exec_result_emitted",
    }
    assert required <= roles
    assert all(e.trace_id == cortex_exec_trace_id(NODE, CORR) for e in events)
    assert events[0].session_id == "sess-1"
    assert events[0].turn_id == "turn-9"


def test_step_failure_emits_exec_step_failed() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=1)
    collector.record_recall_gate_observed(run_recall=False, profile=None, reason="skipped")
    collector.record_step_started(
        order=1, step_name="step_1", verb_name="chat_general", services=["LLMGatewayService"]
    )
    collector.record_step_failed(
        order=1, step_name="step_1", error_kind="timeout"
    )
    collector.record_result_assembled(
        status="fail",
        final_text_present=False,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="fail")
    roles = {e.atom.semantic_role for e in build_cortex_exec_grammar_events(collector) if e.atom}
    assert "exec_step_failed" in roles
    assert "exec_step_completed" not in roles


def test_no_raw_prompt_or_llm_blobs_in_atoms() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=1)
    collector.record_recall_gate_observed(run_recall=False, profile=None, reason="skipped")
    collector.record_step_started(
        order=1, step_name="step_1", verb_name="chat_general", services=["LLMGatewayService"]
    )
    collector.record_step_completed(
        order=1,
        step_name="step_1",
        latency_ms=1,
        result_service_keys=["LLMGatewayService"],
    )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=True,
        thinking_source="provider_reasoning",
    )
    collector.record_result_emitted(reply_present=True, status="success")
    for event in build_cortex_exec_grammar_events(collector):
        if event.atom:
            assert event.atom.text_value is None
            summary = event.atom.summary or ""
            assert "hello user" not in summary.lower()
            assert len(summary) < 500
            assert event.atom.payload_ref
            assert "prompt" not in (event.atom.payload_ref or "")


def test_edges_use_allowed_relation_types_only() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=2)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=2)
    collector.record_recall_gate_observed(run_recall=False, profile="p", reason="r")
    for i in (1, 2):
        collector.record_step_started(
            order=i,
            step_name=f"step_{i}",
            verb_name="chat_general",
            services=["LLMGatewayService"],
        )
        collector.record_step_completed(
            order=i,
            step_name=f"step_{i}",
            latency_ms=10,
            result_service_keys=["LLMGatewayService"],
        )
    collector.record_result_assembled(
        status="success",
        final_text_present=True,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=True, status="success")
    allowed = set(get_args(RelationType))
    for event in build_cortex_exec_grammar_events(collector):
        if event.edge:
            assert event.edge.relation_type in allowed


def test_atom_types_are_allowed_literals() -> None:
    collector = CortexExecGrammarCollector(
        node_name=NODE,
        correlation_id=CORR,
        code_version="0.2.0",
        observed_at=FIXED_OBS,
    )
    req = _minimal_plan(steps=1)
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=1)
    collector.record_recall_gate_observed(run_recall=True, profile="assist.light.v1", reason="run")
    collector.record_step_started(
        order=1, step_name="s1", verb_name="v", services=["RecallService"]
    )
    collector.record_step_completed(
        order=1, step_name="s1", latency_ms=3, result_service_keys=["RecallService"]
    )
    collector.record_result_assembled(
        status="success",
        final_text_present=False,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=False, status="success")
    allowed = set(get_args(AtomType))
    for event in build_cortex_exec_grammar_events(collector):
        if event.atom:
            assert event.atom.atom_type in allowed
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd .worktrees/feat-cortex-exec-grammar-ingress
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
```

Expected: `ModuleNotFoundError: app.grammar_emit`

- [ ] **Step 3: Implement `grammar_emit.py`** (full module)

```python
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from orion.schemas.cortex.schemas import PlanExecutionRequest
from orion.schemas.grammar import (
    GrammarAtomV1,
    GrammarEdgeV1,
    GrammarEventV1,
    GrammarProvenanceV1,
)


def cortex_exec_trace_id(node_name: str, correlation_id: str) -> str:
    return f"cortex.exec:{node_name}:{correlation_id}"


def _hash_id(*parts: object, prefix: str) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _short_error_kind(error: str | None) -> str:
    if not error:
        return "unknown"
    token = re.split(r"[:|\s]+", str(error).strip(), maxsplit=1)[0]
    return (token or "unknown")[:64]


@dataclass
class CortexExecGrammarCollector:
    node_name: str
    correlation_id: str
    code_version: str | None
    observed_at: datetime
    session_id: str | None = None
    turn_id: str | None = None
    _atoms: dict[str, GrammarAtomV1] = field(default_factory=dict)
    _edge_specs: list[tuple[str, str, str]] = field(default_factory=list)  # from_atom_id, to_atom_id, relation_type
    _last_completed_atom_id: str | None = None
    _last_started_atom_id: str | None = None

    @property
    def trace_id(self) -> str:
        return cortex_exec_trace_id(self.node_name, self.correlation_id)

    def _provenance(self, payload_ref: str) -> GrammarProvenanceV1:
        return GrammarProvenanceV1(
            source_service="orion-cortex-exec",
            source_component="cortex_exec_grammar_emit",
            source_event_id=f"{self.correlation_id}:{payload_ref}",
            source_trace_id=self.trace_id,
            source_payload_ref=payload_ref,
            code_version=self.code_version,
        )

    def _atom_id(self, role: str) -> str:
        return f"{self.trace_id}:{role}"

    def _put_atom(self, atom: GrammarAtomV1) -> None:
        self._atoms[atom.semantic_role] = atom

    def record_request_received(self, *, req: PlanExecutionRequest, mode: str) -> None:
        verb = req.plan.verb_name or "unknown"
        n = len(req.plan.steps or [])
        ref = f"cortex.exec.request:{self.correlation_id}"
        self._put_atom(
            GrammarAtomV1(
                atom_id=self._atom_id("exec_request_received"),
                trace_id=self.trace_id,
                atom_type="observation",
                semantic_role="exec_request_received",
                layer="intake",
                dimensions=["execution", "request", "cortex"],
                summary=f"Cortex exec received plan request for verb={verb}, mode={mode}, steps={n}",
                confidence=1.0,
                salience=1.0,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            )
        )

    def record_plan_started(
        self, *, req: PlanExecutionRequest, depth: int | None, step_count: int
    ) -> None:
        verb = req.plan.verb_name or "unknown"
        ref = f"cortex.exec.plan:{self.correlation_id}"
        self._put_atom(
            GrammarAtomV1(
                atom_id=self._atom_id("exec_plan_started"),
                trace_id=self.trace_id,
                atom_type="action_candidate",
                semantic_role="exec_plan_started",
                layer="plan",
                dimensions=["execution", "plan", "agency"],
                summary=(
                    f"Execution plan started for verb={verb}; "
                    f"step_count={step_count}; depth={depth if depth is not None else 'none'}"
                ),
                confidence=1.0,
                salience=0.9,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            )
        )
        self._edge_specs.append(
            (self._atoms["exec_request_received"].atom_id, self._atoms["exec_plan_started"].atom_id, "contains")
        )

    def record_recall_gate_observed(
        self, *, run_recall: bool, profile: str | None, reason: str
    ) -> None:
        ref = f"cortex.exec.recall_gate:{self.correlation_id}"
        self._put_atom(
            GrammarAtomV1(
                atom_id=self._atom_id("exec_recall_gate_observed"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="exec_recall_gate_observed",
                layer="memory_gate",
                dimensions=["execution", "recall", "memory"],
                summary=(
                    f"Recall policy resolved: run={run_recall}, "
                    f"profile={profile or 'none'}, reason={reason}"
                ),
                confidence=0.95,
                salience=0.7,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            )
        )
        self._edge_specs.append(
            (
                self._atoms["exec_plan_started"].atom_id,
                self._atoms["exec_recall_gate_observed"].atom_id,
                "contains",
            )
        )

    def record_step_started(
        self,
        *,
        order: int,
        step_name: str,
        verb_name: str,
        services: list[str],
    ) -> None:
        key = f"exec_step_started:{order}:{step_name}"
        ref = f"cortex.exec.step:{self.correlation_id}:{order}:{step_name}"
        svc_list = ",".join(services) if services else "none"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="action_candidate",
            semantic_role="exec_step_started",
            layer="step",
            dimensions=["execution", "step", "service"],
            summary=(
                f"Step started: order={order}, step={step_name}, "
                f"verb={verb_name}, services={svc_list}"
            ),
            confidence=1.0,
            salience=0.85,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._atoms[key] = atom
        if self._last_completed_atom_id:
            self._edge_specs.append(
                (self._last_completed_atom_id, atom.atom_id, "temporal_successor")
            )
            self._edge_specs.append((atom.atom_id, self._last_completed_atom_id, "derived_from"))
        else:
            self._edge_specs.append(
                (
                    self._atoms["exec_recall_gate_observed"].atom_id,
                    atom.atom_id,
                    "contains",
                )
            )
        self._last_started_atom_id = atom.atom_id

    def record_step_completed(
        self,
        *,
        order: int,
        step_name: str,
        latency_ms: int | None,
        result_service_keys: list[str],
    ) -> None:
        started_key = f"exec_step_started:{order}:{step_name}"
        key = f"exec_step_completed:{order}:{step_name}"
        ref = f"cortex.exec.step_result:{self.correlation_id}:{order}:{step_name}"
        keys = ",".join(sorted(result_service_keys)) if result_service_keys else "none"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="reasoning_step",
            semantic_role="exec_step_completed",
            layer="step",
            dimensions=["execution", "step", "result"],
            summary=(
                f"Step completed: step={step_name}, status=success, "
                f"latency_ms={latency_ms or 0}, result_services={keys}"
            ),
            confidence=0.95,
            salience=0.8,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._atoms[key] = atom
        started = self._atoms.get(started_key)
        if started:
            self._edge_specs.append((started.atom_id, atom.atom_id, "derived_from"))
        self._last_completed_atom_id = atom.atom_id

    def record_step_failed(self, *, order: int, step_name: str, error_kind: str) -> None:
        started_key = f"exec_step_started:{order}:{step_name}"
        key = f"exec_step_failed:{order}:{step_name}"
        ref = f"cortex.exec.step_result:{self.correlation_id}:{order}:{step_name}"
        atom = GrammarAtomV1(
            atom_id=self._atom_id(key),
            trace_id=self.trace_id,
            atom_type="uncertainty_marker",
            semantic_role="exec_step_failed",
            layer="step",
            dimensions=["execution", "failure", "step"],
            summary=f"Step failed: step={step_name}, error_kind={error_kind}",
            confidence=0.9,
            salience=0.9,
            source_event_id=f"{self.correlation_id}:{order}",
            payload_ref=ref,
        )
        self._atoms[key] = atom
        started = self._atoms.get(started_key)
        if started:
            self._edge_specs.append((started.atom_id, atom.atom_id, "derived_from"))
        self._last_completed_atom_id = atom.atom_id

    def record_result_assembled(
        self,
        *,
        status: str,
        final_text_present: bool,
        reasoning_present: bool,
        thinking_source: str,
    ) -> None:
        ref = f"cortex.exec.result:{self.correlation_id}"
        self._put_atom(
            GrammarAtomV1(
                atom_id=self._atom_id("exec_result_assembled"),
                trace_id=self.trace_id,
                atom_type="spoken_output",
                semantic_role="exec_result_assembled",
                layer="result",
                dimensions=["execution", "speech", "reasoning"],
                summary=(
                    f"Final result assembled: status={status}, "
                    f"final_text_present={final_text_present}, "
                    f"reasoning_present={reasoning_present}, "
                    f"thinking_source={thinking_source}"
                ),
                confidence=0.95,
                salience=0.95,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            )
        )
        if self._last_completed_atom_id:
            self._edge_specs.append(
                (self._last_completed_atom_id, self._atoms["exec_result_assembled"].atom_id, "derived_from")
            )

    def record_result_emitted(self, *, reply_present: bool, status: str) -> None:
        ref = f"cortex.exec.egress:{self.correlation_id}"
        self._put_atom(
            GrammarAtomV1(
                atom_id=self._atom_id("exec_result_emitted"),
                trace_id=self.trace_id,
                atom_type="signal",
                semantic_role="exec_result_emitted",
                layer="egress",
                dimensions=["execution", "result", "bus"],
                summary=(
                    f"Cortex exec result emitted to reply_to={reply_present}, status={status}"
                ),
                confidence=1.0,
                salience=0.8,
                source_event_id=self.correlation_id,
                payload_ref=ref,
            )
        )
        self._edge_specs.append(
            (
                self._atoms["exec_result_assembled"].atom_id,
                self._atoms["exec_result_emitted"].atom_id,
                "rendered_as",
            )
        )


def _event(
    *,
    event_kind: str,
    trace_id: str,
    emitted_at: datetime,
    observed_at: datetime,
    provenance: GrammarProvenanceV1,
    atom: GrammarAtomV1 | None = None,
    edge: GrammarEdgeV1 | None = None,
    parent_event_id: str | None = None,
    root_event_id: str | None = None,
    layer: str | None = None,
    dimensions: list[str] | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
    correlation_id: str | None = None,
) -> GrammarEventV1:
    body_key = atom.atom_id if atom else edge.edge_id if edge else uuid4().hex
    return GrammarEventV1(
        event_id=_hash_id(trace_id, event_kind, body_key, prefix="gev"),
        event_kind=event_kind,  # type: ignore[arg-type]
        trace_id=trace_id,
        parent_event_id=parent_event_id,
        root_event_id=root_event_id,
        session_id=session_id,
        turn_id=turn_id,
        correlation_id=correlation_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        layer=layer,
        dimensions=dimensions or [],
        atom=atom,
        edge=edge,
        provenance=provenance,
    )


def build_cortex_exec_grammar_events(
    collector: CortexExecGrammarCollector,
) -> list[GrammarEventV1]:
    observed_at = collector.observed_at
    emitted_at = datetime.now(timezone.utc)
    trace_id = collector.trace_id
    provenance = collector._provenance(f"cortex.exec.trace:{collector.correlation_id}")

    root = _event(
        event_kind="trace_started",
        trace_id=trace_id,
        emitted_at=emitted_at,
        observed_at=observed_at,
        provenance=provenance,
        layer="execution",
        dimensions=["execution", "cortex", "plan"],
        session_id=collector.session_id,
        turn_id=collector.turn_id,
        correlation_id=collector.correlation_id,
    )
    root_id = root.event_id
    events: list[GrammarEventV1] = [root]

    for atom in collector._atoms.values():
        events.append(
            _event(
                event_kind="atom_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
                provenance=provenance,
                atom=atom,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=atom.layer,
                dimensions=atom.dimensions,
                session_id=collector.session_id,
                turn_id=collector.turn_id,
                correlation_id=collector.correlation_id,
            )
        )

    for from_atom_id, to_atom_id, relation_type in collector._edge_specs:
        edge = GrammarEdgeV1(
            edge_id=f"{trace_id}:edge:{from_atom_id}:{relation_type}:{to_atom_id}",
            trace_id=trace_id,
            from_atom_id=from_atom_id,
            to_atom_id=to_atom_id,
            relation_type=relation_type,  # type: ignore[arg-type]
            confidence=0.9,
            salience=0.7,
            layer_from=None,
            layer_to=None,
            evidence_event_ids=[collector.correlation_id],
        )
        events.append(
            _event(
                event_kind="edge_emitted",
                trace_id=trace_id,
                emitted_at=emitted_at,
                observed_at=observed_at,
                provenance=provenance,
                edge=edge,
                parent_event_id=root_id,
                root_event_id=root_id,
                layer=edge.layer_to,
                dimensions=["execution", "plan", "step"],
                session_id=collector.session_id,
                turn_id=collector.turn_id,
                correlation_id=collector.correlation_id,
            )
        )

    events.append(
        _event(
            event_kind="trace_ended",
            trace_id=trace_id,
            emitted_at=datetime.now(timezone.utc),
            observed_at=observed_at,
            provenance=provenance,
            parent_event_id=root_id,
            root_event_id=root_id,
            layer="execution",
            dimensions=["execution", "cortex", "plan"],
            session_id=collector.session_id,
            turn_id=collector.turn_id,
            correlation_id=collector.correlation_id,
        )
    )
    return events
```

- [ ] **Step 4: Add test `test_two_steps_emit_temporal_successor_edge`** — assert ≥1 edge with `relation_type == "temporal_successor"` when two steps complete.

- [ ] **Step 5: Run tests — expect PASS**

```bash
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-cortex-exec/app/grammar_emit.py \
  services/orion-cortex-exec/tests/test_exec_grammar_emit.py
git commit -m "feat(cortex-exec): add execution grammar event builder"
```

### Task 4: Fail-open publisher tests

**Files:**
- Create: `services/orion-cortex-exec/app/grammar_publish.py`
- Create: `services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py`

- [ ] **Step 1: Write failing test**

```python
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.grammar_emit import CortexExecGrammarCollector, build_cortex_exec_grammar_events
from app.grammar_publish import publish_cortex_exec_grammar_trace
from orion.schemas.cortex.schemas import ExecutionPlan, PlanExecutionRequest


@pytest.mark.asyncio
async def test_publish_failure_is_non_fatal() -> None:
    bus = AsyncMock()
    bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))
    collector = CortexExecGrammarCollector(
        node_name="athena",
        correlation_id="c1",
        code_version="0.2.0",
        observed_at=datetime.now(timezone.utc),
    )
    req = PlanExecutionRequest(
        plan=ExecutionPlan(verb_name="chat_quick", steps=[]),
        args={"request_id": "r1"},
    )
    collector.record_request_received(req=req, mode="brain")
    collector.record_plan_started(req=req, depth=None, step_count=0)
    collector.record_recall_gate_observed(run_recall=False, profile=None, reason="skip")
    collector.record_result_assembled(
        status="success",
        final_text_present=False,
        reasoning_present=False,
        thinking_source="none",
    )
    collector.record_result_emitted(reply_present=False, status="success")
    events = build_cortex_exec_grammar_events(collector)
    # must not raise
    await publish_cortex_exec_grammar_trace(
        bus,
        events,
        correlation_id="c1",
        channel="orion:grammar:event",
        source_name="orion-cortex-exec",
    )
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `grammar_publish.py`**

```python
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from orion.grammar.publish import publish_grammar_event
from orion.schemas.grammar import GrammarEventV1

logger = logging.getLogger("orion.cortex.exec.grammar_publish")


async def publish_cortex_exec_grammar_trace(
    bus: Any,
    events: list[GrammarEventV1],
    *,
    correlation_id: str,
    channel: str,
    source_name: str = "orion-cortex-exec",
    enabled: bool = True,
) -> None:
    if not enabled or not events:
        return
    try:
        corr_uuid = UUID(str(correlation_id))
    except (ValueError, TypeError):
        corr_uuid = None
    for event in events:
        try:
            await publish_grammar_event(
                bus,
                event,
                source_name=source_name,
                correlation_id=corr_uuid,
            )
        except Exception:
            logger.warning(
                "cortex_exec_grammar_publish_failed corr=%s event_kind=%s",
                correlation_id,
                event.event_kind,
                exc_info=True,
            )
```

**Implementer note:** If `publish_grammar_event` always uses `GRAMMAR_EVENT_CHANNEL` constant, either extend `orion/grammar/publish.py` with optional `channel:` parameter (minimal, shared) or publish via `bus.publish(channel, envelope)` matching biometrics `_publish` pattern while keeping `kind="grammar.event.v1"`. Prefer extending `publish_grammar_event` with `channel: str | None = None` defaulting to `GRAMMAR_EVENT_CHANNEL` so settings override works.

- [ ] **Step 4: Run — expect PASS**

```bash
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/grammar_publish.py \
  services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py
git commit -m "feat(cortex-exec): fail-open grammar trace publisher"
```

---

# Phase 3 — Wire `run_plan()` (primary seam)

### Task 5: Collector helpers + `run_plan` instrumentation

**Files:**
- Create: `services/orion-cortex-exec/app/grammar_session.py` (optional thin helpers — or keep helpers in `grammar_emit.py`)
- Modify: `services/orion-cortex-exec/app/router.py`

- [ ] **Step 1: Add helper to create collector from ctx**

```python
# In grammar_emit.py or grammar_session.py
def new_cortex_exec_collector(
    *,
    correlation_id: str,
    ctx: dict[str, Any],
    code_version: str | None,
) -> CortexExecGrammarCollector:
    from .settings import settings

    session_id = str(ctx.get("session_id") or ctx.get("sessionId") or "") or None
    turn_id = str(ctx.get("turn_id") or ctx.get("message_id") or ctx.get("messageId") or "") or None
    return CortexExecGrammarCollector(
        node_name=settings.node_name,
        correlation_id=correlation_id,
        code_version=code_version,
        observed_at=datetime.now(timezone.utc),
        session_id=session_id,
        turn_id=turn_id,
    )


async def flush_cortex_exec_grammar(
    bus: Any,
    collector: CortexExecGrammarCollector | None,
    *,
    correlation_id: str,
) -> None:
    from .grammar_publish import publish_cortex_exec_grammar_trace
    from .settings import settings

    if collector is None or not settings.publish_cortex_exec_grammar:
        return
    events = build_cortex_exec_grammar_events(collector)
    await publish_cortex_exec_grammar_trace(
        bus,
        events,
        correlation_id=correlation_id,
        channel=settings.grammar_event_channel,
        source_name=settings.service_name,
        enabled=settings.publish_cortex_exec_grammar,
    )
```

- [ ] **Step 2: Instrument `PlanRunner.run_plan()`** — add at top after `plan` resolved:

```python
from .grammar_emit import new_cortex_exec_collector, flush_cortex_exec_grammar
# ...
grammar_collector = ctx.get("_cortex_exec_grammar_collector")
if grammar_collector is None:
    grammar_collector = new_cortex_exec_collector(
        correlation_id=correlation_id,
        ctx=ctx,
        code_version=settings.service_version,
    )
    ctx["_cortex_exec_grammar_collector"] = grammar_collector
if not ctx.get("_cortex_exec_grammar_request_recorded"):
    grammar_collector.record_request_received(req=req, mode=str(start_mode))
    ctx["_cortex_exec_grammar_request_recorded"] = True
grammar_collector.record_plan_started(
    req=req,
    depth=depth,
    step_count=len(plan.steps),
)
grammar_collector.record_recall_gate_observed(
    run_recall=bool(recall_policy["run_recall"]),
    profile=str(selected_profile) if selected_profile else None,
    reason=str(recall_policy.get("reason") or recall_policy.get("recall_gating_reason") or "unknown"),
)
```

- [ ] **Step 3: Before `call_step_services` loop** (~L1092), inside `for step in sorted(...)`:

```python
grammar_collector.record_step_started(
    order=step.order,
    step_name=step.step_name,
    verb_name=step.verb_name or plan.verb_name or "unknown",
    services=list(step.services or []),
)
```

- [ ] **Step 4: After `step_res` returned**

```python
if step_res.status == "success":
    keys = sorted(step_res.result.keys()) if isinstance(step_res.result, dict) else []
    grammar_collector.record_step_completed(
        order=step.order,
        step_name=step.step_name,
        latency_ms=step_res.latency_ms,
        result_service_keys=keys,
    )
else:
    grammar_collector.record_step_failed(
        order=step.order,
        step_name=step.step_name,
        error_kind=_short_error_kind(step_res.error),
    )
```

Import `_short_error_kind` from `grammar_emit` or duplicate minimal helper in router.

- [ ] **Step 5: Before each early `return PlanExecutionResult`** (inactive verb, recall empty, supervisor path exception): call `record_result_assembled` + `flush_cortex_exec_grammar` in `finally`-style helper — **supervisor path** (`Supervisor.execute`) may skip step loop; still emit plan + recall + assembled with `status` from early return.

- [ ] **Step 6: Before final `return PlanExecutionResult`** (~L1304):

```python
grammar_collector.record_result_assembled(
    status=overall_status,
    final_text_present=bool((final_text or "").strip()),
    reasoning_present=bool(reasoning_content or reasoning_trace or metacog_traces),
    thinking_source=str(thinking_source or "none"),
)
await flush_cortex_exec_grammar(bus, grammar_collector, correlation_id=correlation_id)
```

- [ ] **Step 7: Run full cortex-exec tests**

```bash
PYTHONPATH=. pytest services/orion-cortex-exec/tests/ -q
```

Expected: all existing tests still pass (grammar off by default).

- [ ] **Step 8: Commit**

```bash
git add services/orion-cortex-exec/app/router.py services/orion-cortex-exec/app/grammar_emit.py
git commit -m "feat(cortex-exec): emit grammar trace from PlanRunner.run_plan"
```

---

# Phase 4 — Wire `main.handle()` (intake / egress / validation)

### Task 6: `handle()` instrumentation

**Files:**
- Modify: `services/orion-cortex-exec/app/main.py`

- [ ] **Step 1: On validation failure** (~L404), before return — optional minimal trace when `settings.publish_cortex_exec_grammar`:

Build collector with only `exec_request_received` (if payload partially known) or skip; prefer logging-only on validation failure unless `plan_v`/`step_n` were extracted. Minimal approach:

```python
if settings.publish_cortex_exec_grammar:
    try:
        from .grammar_emit import new_cortex_exec_collector, build_cortex_exec_grammar_events
        from .grammar_publish import publish_cortex_exec_grammar_trace
        c = new_cortex_exec_collector(correlation_id=corr_id, ctx={}, code_version=settings.service_version)
        # record validation_failed atom via collector method (add record_validation_failed)
        await publish_cortex_exec_grammar_trace(...)
    except Exception:
        logger.warning("cortex_exec_grammar_validation_publish_failed", exc_info=True)
```

Add `record_validation_failed` atom (`semantic_role="exec_request_invalid"`, `atom_type="uncertainty_marker"`) — optional but recommended for spec coverage.

- [ ] **Step 2: Before `router.run_plan`** — create collector in `ctx`:

```python
from .grammar_emit import new_cortex_exec_collector
ctx["_cortex_exec_grammar_collector"] = new_cortex_exec_collector(
    correlation_id=corr_id,
    ctx=ctx,
    code_version=settings.service_version,
)
ctx["_cortex_exec_grammar_request_recorded"] = False  # run_plan will record
```

- [ ] **Step 3: After successful reply publish** (~L549), record egress:

```python
collector = ctx.get("_cortex_exec_grammar_collector")
if collector is not None:
    collector.record_result_emitted(reply_present=bool(env.reply_to), status=str(res.status))
    from .grammar_emit import build_cortex_exec_grammar_events
    from .grammar_publish import publish_cortex_exec_grammar_trace
    if settings.publish_cortex_exec_grammar:
        try:
            await publish_cortex_exec_grammar_trace(
                svc.bus,
                build_cortex_exec_grammar_events(collector),
                correlation_id=corr_id,
                channel=settings.grammar_event_channel,
                source_name=settings.service_name,
            )
        except Exception:
            logger.warning("cortex_exec_grammar_egress_publish_failed", exc_info=True)
```

**Double-publish guard:** `run_plan` already calls `flush_cortex_exec_grammar` without `exec_result_emitted`. Either:
- Remove flush from `run_plan`; only flush from `main.handle` after `record_result_emitted`, **or**
- `run_plan` flushes without `exec_result_emitted`; `main.handle` publishes **delta** only for egress atom (complex).

**Recommended:** `run_plan` does **not** flush; only records atoms through `record_result_assembled`. Single flush in `main.handle` after `record_result_emitted`. For `legacy.plan` (no `handle`), add flush at end of `LegacyPlanVerb.execute` after `run_plan` with `record_result_emitted(reply_present=True, ...)`.

- [ ] **Step 4: Legacy plan flush** in `verb_adapters.py` after `run_plan`:

```python
collector = ctx_payload.get("_cortex_exec_grammar_collector")
if collector is not None:
    collector.record_result_emitted(reply_present=True, status=str(result.status))
    await flush_cortex_exec_grammar(bus, collector, correlation_id=correlation_id)
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. pytest services/orion-cortex-exec/tests/ -q
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-cortex-exec/app/main.py services/orion-cortex-exec/app/verb_adapters.py
git commit -m "feat(cortex-exec): wire grammar intake and egress in main and legacy.plan"
```

---

# Phase 5 — Docs and smoke

### Task 7: README + smoke script

**Files:**
- Modify: `services/orion-cortex-exec/README.md`
- Create: `scripts/smoke_cortex_exec_grammar.sh`

- [ ] **Step 1: README section** (after Consumed Channels table)

```markdown
### Grammar substrate (shadow observability)

| Channel | Env Var | Kind | Description |
| :--- | :--- | :--- | :--- |
| `orion:grammar:event` | `GRAMMAR_EVENT_CHANNEL` | `grammar.event.v1` | Execution trajectory trace (one per plan run). |

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PUBLISH_CORTEX_EXEC_GRAMMAR` | `false` | Enable grammar publish after plan execution. |
| `GRAMMAR_EVENT_CHANNEL` | `orion:grammar:event` | Grammar bus channel. |

Trace id format: `cortex.exec:{NODE_NAME}:{correlation_id}`. Execution semantics use `GrammarAtomV1.semantic_role` (e.g. `exec_step_started`), not custom `event_kind` values.
```

- [ ] **Step 2: Smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "1. Run unit tests"
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q

echo "2. Optional live bus tap"
echo "In another shell:"
echo 'redis-cli -u "${ORION_BUS_URL:-redis://127.0.0.1:6379/0}" SUBSCRIBE orion:grammar:event'

echo "3. Trigger a normal brain/chat path through existing harness"
echo "python scripts/bus_harness.py brain 'hello from cortex exec grammar smoke'"
```

```bash
chmod +x scripts/smoke_cortex_exec_grammar.sh
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-cortex-exec/README.md scripts/smoke_cortex_exec_grammar.sh
git commit -m "docs(cortex-exec): document grammar ingress and add smoke script"
```

---

# Phase 6 — Verification, code review, PR

### Task 8: Full verification

- [ ] **Step 1: Run required tests**

```bash
cd .worktrees/feat-cortex-exec-grammar-ingress
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q
PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q
PYTHONPATH=. pytest services/orion-cortex-exec/tests/ -q
```

- [ ] **Step 2: Run smoke script**

```bash
./scripts/smoke_cortex_exec_grammar.sh
```

- [ ] **Step 3: Record live bus status** — mark **unverified** in PR report unless Redis tap was run during a live exec.

### Task 9: Code review subagent + fixes

**REQUIRED SUB-SKILL:** `superpowers:requesting-code-review` then fix all reported issues.

- [ ] **Step 1: Dispatch code-reviewer subagent** with:

```text
Review feat/cortex-exec-grammar-ingress in worktree .worktrees/feat-cortex-exec-grammar-ingress.
Focus: no execution behavior change, closed GrammarEventKind, semantic_role only for exec semantics,
fail-open publish, no double trace for handle vs legacy.plan, no raw blobs in atoms.
```

- [ ] **Step 2: Fix every valid issue** and re-run pytest.

- [ ] **Step 3: Commit fixes**

```bash
git commit -am "fix(cortex-exec): address grammar ingress code review"
```

### Task 10: PR report and GitHub push

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-24-cortex-exec-grammar-ingress-pr.md`

- [ ] **Step 1: Write PR report** including:

  - Files changed (list)
  - Emitted semantic roles (table)
  - Example trace id: `cortex.exec:athena:<correlation_id>`
  - Test command output summary
  - Live bus: verified or explicitly unverified

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feat/cortex-exec-grammar-ingress
gh pr create --base main --title "feat(cortex-exec): substrate grammar ingress for execution traces" --body "$(cat <<'EOF'
## Summary
- Emit valid `GrammarEventV1` traces on `orion:grammar:event` for plan/step/result lifecycle (shadow observability).
- Fail-open publish behind `PUBLISH_CORTEX_EXEC_GRAMMAR`; no execution behavior changes.

## Test plan
- [x] `pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py`
- [x] `pytest services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py`
- [x] `pytest services/orion-cortex-exec/tests/`
- [ ] Live `redis-cli SUBSCRIBE orion:grammar:event` (optional)

EOF
)"
```

---

## Self-review checklist

| Spec requirement | Task |
|------------------|------|
| Closed `GrammarEventKind` only | Task 3 tests + emit |
| `semantic_role` for exec semantics | Task 3 `CortexExecGrammarCollector` |
| Reuse `orion/grammar/publish.py` | Task 4 |
| Biometrics-shaped trace | Task 3 |
| `PlanRunner.run_plan` seam | Task 5 |
| `main.handle` intake/egress | Task 6 |
| `legacy.plan` no double emit | Task 6 Step 4 + flush guard |
| Fail-open publish | Task 4 |
| Settings + channels + README | Tasks 1–2, 7 |
| No executor.py internals | Non-goals |
| PR report + push | Task 10 |

**Placeholder scan:** Clean — edge specs use `atom_id` tuples throughout.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-cortex-exec-grammar-ingress.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute in this session with `executing-plans`, batched checkpoints

Which approach?
