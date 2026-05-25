# Cortex Exec Grammar Ingress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `orion-cortex-exec` emit execution-shaped `GrammarEventV1` shadow traces on `orion:grammar:event` without changing execution semantics or adding field-digester/reducer behavior.

**Architecture:** Mirror the biometrics ingress pattern: pure `grammar_emit.py` builder, fail-open `grammar_publish.py` wrapper over `orion.grammar.publish.publish_grammar_event`, instrumentation at `PlanRunner.run_plan()` plus `main.handle()` / `legacy.plan` egress flush. Use closed `GrammarEventKind` only; encode execution meaning in `GrammarAtomV1.semantic_role`, bounded `summary` key=value hints, and `GrammarProvenanceV1`.

**Tech Stack:** Python 3.12, Pydantic v2 (`orion/schemas/grammar.py`, `orion/schemas/cortex/schemas.py`), Redis bus (`OrionBusAsync`, `BaseEnvelope`), `orion.grammar.publish`, pytest 8.3.x.

**Design source:** User handoff “Cortex Exec Grammar Ingress” (2026-05-24).

**Non-goals:** Field digester, substrate reducer changes, new `GrammarEventKind` / `RelationType`, fatal grammar publish, `executor.py` service internals, raw prompts/LLM blobs in atoms.

---

## Implementation status (2026-05-24)

| Item | Status |
|------|--------|
| Landed on `main` | Yes — `ed5b9ad9` (`feat/cortex-exec-grammar-ingress`) |
| Unit tests | `8 passed` — `test_exec_grammar_emit.py` + `test_exec_grammar_publish_fail_open.py` |
| Channel catalog | `orion-cortex-exec` listed under `orion:grammar:event` in `orion/bus/channels.yaml` |
| Live bus tap | **Unverified** (operator step) |

**If you are implementing from an older `main`:** use the worktree section below and execute Phases 1–6. **If you are on current `main`:** skip to Phase 7 (gap closure + verification) unless spec deltas below are required.

---

## Spec reconciliation (handoff vs landed)

The handoff spec and the landed patch align on platform constraints; a few naming choices differ **on purpose** because downstream substrate code already depends on them.

| Handoff | Landed choice | Rationale |
|---------|---------------|-----------|
| `semantic_role` = `cortex_exec.step.started` | `exec_step_started` | `orion/substrate/execution_loop/grammar_extract.py` keys off `exec_*` roles |
| `trace_id` = `cortex.exec:<run_id>:<phase>:<suffix>` | `cortex.exec:{NODE_NAME}:{correlation_id}` | `parse_execution_trace_id()` expects `cortex.exec:node:corr` |
| `CORTEX_EXEC_GRAMMAR_EVENTS_ENABLED` | `PUBLISH_CORTEX_EXEC_GRAMMAR` | Matches biometrics `PUBLISH_BIOMETRICS_GRAMMAR` lineage |
| `source_node` / `source_kind` top-level fields | `GrammarProvenanceV1.source_service` + `dimensions` + summary KV | `GrammarEventV1` has no extra payload dict — `extra=forbid` |
| Per-step live publish | Batch flush at plan end | Shadow-only; one trace per run; fewer bus messages |
| `cortex_exec.step.timeout` distinct role | `exec_step_failed` + `error_kind=timeout` in summary | Same closed kinds; grep `error_kind=timeout` |

**Do not rename** `exec_*` semantic roles or trace id shape without updating `grammar_extract.py`, `execution_loop` tests, and substrate-runtime fetch filters in the same PR.

### Semantic role map (grep-friendly)

| Handoff role | Landed `semantic_role` |
|--------------|------------------------|
| `cortex_exec.request.accepted` | `exec_request_received` |
| `cortex_exec.plan.started` | `exec_plan_started` |
| `cortex_exec.step.started` | `exec_step_started` |
| `cortex_exec.step.completed` | `exec_step_completed` |
| `cortex_exec.step.failed` | `exec_step_failed` |
| `cortex_exec.step.timeout` | `exec_step_failed` (summary contains `error_kind=timeout`) |
| `cortex_exec.run.completed` | `exec_result_assembled` (`status=success`) |
| `cortex_exec.run.failed` | `exec_result_assembled` (`status=fail`) |
| `cortex_exec.reply.published` | `exec_result_emitted` |

Additional landed roles: `exec_recall_gate_observed`, `exec_request_invalid` (validation failure path).

### Bounded hints (where they live)

| Hint | Location |
|------|----------|
| `run_id` / correlation | `GrammarEventV1.correlation_id`, `source_event_id` on atoms |
| `verb`, `mode`, `step_count` | `exec_request_received` / `exec_plan_started` **summary** (`verb=`, `mode=`, `step_count=`) |
| `step_name`, `order`, services | `exec_step_*` summaries |
| `latency_ms`, `status` | step/result summaries |
| `error_kind` (bounded) | `short_error_kind()` — first token, max 64 chars |
| `node` | `trace_id` segment + `settings.node_name` via collector |
| Capability surface | `dimensions` lists (`execution`, `recall`, `service`, …) |

Never put full prompts, model responses, secrets, stack traces, or raw `StepExecutionResult.result` dicts on the bus.

---

## Worktree isolation (mandatory for greenfield implementation)

All implementation commits happen **only** in a dedicated worktree when branching from pre-ingress `main`.

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-cortex-exec-grammar-ingress \
  -b feat/cortex-exec-grammar-ingress origin/main
cd .worktrees/feat-cortex-exec-grammar-ingress
git check-ignore -q .worktrees   # must succeed
```

**Rules:**
- Never bleed changed files back to the main checkout except copying `.env` keys locally.
- When editing `services/orion-cortex-exec/.env_example`, also copy new keys into `services/orion-cortex-exec/.env` in the **worktree** (gitignored).
- PR and push from `feat/cortex-exec-grammar-ingress` only.

---

## Preflight findings

| Question | Finding |
|----------|---------|
| Grammar schemas | `orion/schemas/grammar.py` — closed `GrammarEventKind`, `AtomType`, `RelationType` |
| Shared publisher | `orion/grammar/publish.py` → `publish_grammar_event(..., channel=...)` |
| Reference emitter | `services/orion-biometrics/app/grammar_emit.py` |
| Primary seam | `PlanRunner.run_plan()` in `services/orion-cortex-exec/app/router.py` |
| Exec intake | `handle()` in `services/orion-cortex-exec/app/main.py` |
| Legacy plan path | `LegacyPlanVerb.execute()` → `run_plan()` — flush in `verb_adapters.py` |
| Grammar on bus | `orion:grammar:event` — producer includes `orion-cortex-exec` |
| Registry | `GrammarEventV1` in `orion/schemas/registry.py` — no change |
| Downstream digestion | `orion-substrate-runtime` execution loop reads `source_service=orion-cortex-exec` |

---

## File structure

| Path | Responsibility |
|------|----------------|
| `services/orion-cortex-exec/app/grammar_emit.py` | `CortexExecGrammarCollector`, `build_cortex_exec_grammar_events()` |
| `services/orion-cortex-exec/app/grammar_publish.py` | Fail-open `publish_cortex_exec_grammar_trace`, `flush_cortex_exec_grammar` |
| `services/orion-cortex-exec/app/settings.py` | `publish_cortex_exec_grammar`, `grammar_event_channel` |
| `services/orion-cortex-exec/app/router.py` | Collector lifecycle in `run_plan()` |
| `services/orion-cortex-exec/app/main.py` | Validation trace, egress flush in `finally` |
| `services/orion-cortex-exec/app/verb_adapters.py` | `legacy.plan` flush |
| `services/orion-cortex-exec/.env_example` | Env keys |
| `services/orion-cortex-exec/docker-compose.yml` | Env passthrough |
| `services/orion-cortex-exec/README.md` | Grammar channel docs |
| `orion/bus/channels.yaml` | `orion-cortex-exec` producer on `orion:grammar:event` |
| `services/orion-cortex-exec/tests/test_exec_grammar_emit.py` | Builder/schema tests |
| `services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py` | Publish fail-open |
| `scripts/smoke_cortex_exec_grammar.sh` | Test + bus tap instructions |

**Do not modify:** `executor.py`, field digester, substrate reducer (separate PR chain).

---

## Trace and id conventions

```text
trace_id = cortex.exec:{node_name}:{correlation_id}
atom_id  = {trace_id}:{stable_key}   # e.g. exec_step_started:1:step_1
event_id = gev_{sha1(trace_id|event_kind|body_key)[:16]}
envelope = BaseEnvelope(kind="grammar.event.v1", schema_id via payload GrammarEventV1)
channel  = orion:grammar:event
```

- `session_id` / `turn_id`: from `ctx` (`session_id`, `turn_id` or `message_id`).
- `provenance.source_service`: always `orion-cortex-exec`.
- `payload_ref` only — never embed prompts or full results.

---

# Phase 1 — Settings and bus catalog

### Task 1: Settings + env + compose

**Files:**
- Modify: `services/orion-cortex-exec/app/settings.py`
- Modify: `services/orion-cortex-exec/.env_example`
- Modify: `services/orion-cortex-exec/docker-compose.yml`

- [x] **Step 1: Add settings fields**

```python
    publish_cortex_exec_grammar: bool = Field(False, alias="PUBLISH_CORTEX_EXEC_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")
```

- [x] **Step 2: Update `.env_example`**

```bash
# Substrate grammar ingress (shadow observability; default off)
PUBLISH_CORTEX_EXEC_GRAMMAR=false
GRAMMAR_EVENT_CHANNEL=orion:grammar:event
```

- [x] **Step 3: Pass through in `docker-compose.yml`**

```yaml
      PUBLISH_CORTEX_EXEC_GRAMMAR: ${PUBLISH_CORTEX_EXEC_GRAMMAR:-false}
      GRAMMAR_EVENT_CHANNEL: ${GRAMMAR_EVENT_CHANNEL:-orion:grammar:event}
```

- [x] **Step 4: Commit** — `feat(cortex-exec): add grammar publish settings`

### Task 2: Bus channel catalog

**Files:**
- Modify: `orion/bus/channels.yaml` (`orion:grammar:event` producers)

- [x] **Step 1: Add `orion-cortex-exec` to `producer_services`**
- [x] **Step 2: Commit** — `chore(bus): register orion-cortex-exec as grammar event producer`

---

# Phase 2 — Pure grammar builder (TDD)

### Task 3: `grammar_emit.py` + tests

**Files:**
- Create: `services/orion-cortex-exec/app/grammar_emit.py`
- Create: `services/orion-cortex-exec/tests/test_exec_grammar_emit.py`

- [x] **Step 1: Write tests** — roles, closed kinds, no blobs, edges, temporal_successor
- [x] **Step 2: Run tests** — `PYTHONPATH=. pytest services/orion-cortex-exec/tests/test_exec_grammar_emit.py -q` → 7 passed
- [x] **Step 3: Implement collector + `build_cortex_exec_grammar_events`**
- [x] **Step 4: Commit** — `feat(cortex-exec): add execution grammar event builder`

### Task 4: Fail-open publisher

**Files:**
- Create: `services/orion-cortex-exec/app/grammar_publish.py`
- Create: `services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py`

- [x] **Step 1: Test publish failure does not raise**
- [x] **Step 2: Implement `publish_cortex_exec_grammar_trace` with per-event try/except**
- [x] **Step 3: Run** — `pytest .../test_exec_grammar_publish_fail_open.py -q` → 1 passed
- [x] **Step 4: Commit** — `feat(cortex-exec): fail-open grammar trace publisher`

---

# Phase 3 — Wire `run_plan()` (primary seam)

### Task 5: `router.py` instrumentation

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py`
- Helpers in: `services/orion-cortex-exec/app/grammar_emit.py` (`begin_plan_grammar`, `record_assembled_grammar`, `get_or_create_collector`)

- [x] **Step 1: `begin_plan_grammar` after recall policy** — request, plan, recall gate
- [x] **Step 2: `record_step_started` / `record_step_completed` / `record_step_failed` around step loop**
- [x] **Step 3: Pre-recall step (order=0) grammar when recall runs outside plan steps**
- [x] **Step 4: `record_assembled_grammar` on early returns (inactive verb, recall empty, failures)**
- [x] **Step 5: Do not flush in `run_plan`** — flush deferred to `main` / `legacy.plan`
- [x] **Step 6: Run targeted tests** — grammar tests pass; full suite may have pre-existing collection issues
- [x] **Step 7: Commit** — `feat(cortex-exec): emit grammar trace from PlanRunner.run_plan`

---

# Phase 4 — Wire `main.handle()` and legacy plan

### Task 6: Intake / egress / validation

**Files:**
- Modify: `services/orion-cortex-exec/app/main.py`
- Modify: `services/orion-cortex-exec/app/verb_adapters.py`

- [x] **Step 1: Validation failure** — `record_validation_failed` + publish when enabled
- [x] **Step 2: `finally` block** — `record_result_emitted` + `flush_cortex_exec_grammar` (independent of cognition success)
- [x] **Step 3: `legacy.plan`** — flush after `run_plan` in `verb_adapters.py`
- [x] **Step 4: Commit** — `feat(cortex-exec): wire grammar intake and egress in main and legacy.plan`

---

# Phase 5 — Docs and smoke

### Task 7: README + smoke script

**Files:**
- Modify: `services/orion-cortex-exec/README.md`
- Create: `scripts/smoke_cortex_exec_grammar.sh`

- [x] **Step 1: README grammar section**
- [x] **Step 2: Smoke script** — unit tests + redis SUBSCRIBE instructions
- [x] **Step 3: Commit** — `docs(cortex-exec): document grammar ingress and add smoke script`

---

# Phase 6 — Verification and PR

### Task 8: Full verification

- [x] **Step 1: Grammar unit tests**

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. .venv/bin/python -m pytest \
  services/orion-cortex-exec/tests/test_exec_grammar_emit.py \
  services/orion-cortex-exec/tests/test_exec_grammar_publish_fail_open.py -q
```

Expected: `8 passed`

- [x] **Step 2: Smoke script**

```bash
./scripts/smoke_cortex_exec_grammar.sh
```

- [ ] **Step 3: Live bus tap** (operator)

```bash
# Terminal A
redis-cli -u "${ORION_BUS_URL:-redis://127.0.0.1:6379/0}" SUBSCRIBE orion:grammar:event

# Terminal B — enable in services/orion-cortex-exec/.env
PUBLISH_CORTEX_EXEC_GRAMMAR=true

# Trigger one plan execution (harness or hub chat), then confirm:
# - envelope.kind == grammar.event.v1
# - payload validates as GrammarEventV1
# - provenance.source_service == orion-cortex-exec
# - trace_id starts with cortex.exec:
```

### Task 9: PR report

**Files:**
- `docs/superpowers/pr-reports/2026-05-24-cortex-exec-grammar-ingress-pr.md`

- [x] **Step 1: PR report written**
- [ ] **Step 2: Live bus marked verified** in PR report after Step 8.3

---

# Phase 7 — Gap closure (optional, post-ingress)

Only implement if product explicitly requires handoff deltas **and** substrate is updated in the same change train.

### Task 10: Supervisor path grammar

**Files:**
- Modify: `services/orion-cortex-exec/app/supervisor.py`
- Modify: `services/orion-cortex-exec/app/router.py` (supervisor return path)

**Gap:** `Supervisor.execute()` bypasses the step loop; grammar today may only reflect plan start + assembled result.

- [ ] **Step 1: Write failing test** — `tests/test_supervisor_grammar_emit.py`

```python
@pytest.mark.asyncio
async def test_supervisor_run_records_plan_and_result_atoms(monkeypatch):
    # Mock supervisor to return PlanExecutionResult status=success
    # Assert ctx["_cortex_exec_grammar_collector"] has exec_plan_started and exec_result_assembled
    ...
```

- [ ] **Step 2: Run test** — expect FAIL until supervisor wires collector

- [ ] **Step 3: In `Supervisor.execute`**, reuse `get_or_create_collector` and call `record_assembled_grammar` before return; do not flush (still `main`/`legacy` responsibility)

- [ ] **Step 4: Run test** — PASS

- [ ] **Step 5: Commit** — `feat(cortex-exec): grammar atoms for supervised execution path`

### Task 11: Richer bounded hints in summaries

**Files:**
- Modify: `services/orion-cortex-exec/app/grammar_emit.py`

- [ ] **Step 1: Extend `record_step_failed` signature**

```python
def record_step_failed(
    self,
    *,
    order: int,
    step_name: str,
    error_kind: str,
    retry_count: int | None = None,
    timeout_ms: int | None = None,
) -> None:
    extra = []
    if retry_count is not None:
        extra.append(f"retry_count={retry_count}")
    if timeout_ms is not None:
        extra.append(f"timeout_ms={timeout_ms}")
    suffix = f", {', '.join(extra)}" if extra else ""
    summary = f"Step failed: step={step_name}, error_kind={error_kind}{suffix}"
```

- [ ] **Step 2: Pass `retry_count` / `timeout_ms` from router when `step_res` exposes them** (read `StepExecutionResult` fields; omit if absent)

- [ ] **Step 3: Test** — assert `retry_count=` appears in summary for synthetic failure

- [ ] **Step 4: Commit** — `feat(cortex-exec): bounded retry/timeout hints in grammar summaries`

### Task 12: Env alias (optional ergonomics)

**Files:**
- Modify: `services/orion-cortex-exec/app/settings.py`

- [ ] **Step 1: Accept handoff alias without breaking existing deploys**

```python
    publish_cortex_exec_grammar: bool = Field(
        False,
        validation_alias=AliasChoices("PUBLISH_CORTEX_EXEC_GRAMMAR", "CORTEX_EXEC_GRAMMAR_EVENTS_ENABLED"),
    )
```

- [ ] **Step 2: Document both names in `.env_example` and README**
- [ ] **Step 3: Commit** — `feat(cortex-exec): alias CORTEX_EXEC_GRAMMAR_EVENTS_ENABLED`

---

## Self-review checklist

| Spec requirement | Covered by |
|------------------|------------|
| Shadow-only `GrammarEventV1` on `orion:grammar:event` | Phases 2–4 |
| No execution semantic change | Collector sidecar only; publish fail-open |
| No field digester / reducer in this patch | Non-goals |
| Closed `GrammarEventKind` only | `trace_started`, `atom_emitted`, `edge_emitted`, `trace_ended` |
| `semantic_role` for execution meaning | `exec_*` roles (substrate contract) |
| Node-scoped trace | `cortex.exec:{node}:{correlation_id}` |
| `source_service=orion-cortex-exec` | `GrammarProvenanceV1` |
| Bounded payloads | Tests in `test_exec_grammar_emit.py` |
| Fail-open publish | `test_exec_grammar_publish_fail_open.py` |
| Channel catalog | Task 2 |
| Settings / compose lineage | Task 1 |
| Tests pass | Task 8 — 8 passed on 2026-05-24 |
| Live bus visible | Task 8 Step 3 — operator |

**Placeholder scan:** Clean.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-05-24-cortex-exec-grammar-ingress.md`.

**On current `main`:** Phases 1–5 and most of Phase 6 are complete. Run Task 8 Step 3 for live bus acceptance, then optionally Phase 7.

**Two execution options for remaining work:**

1. **Subagent-Driven (recommended)** — dispatch per Phase 7 task with code review between tasks  
2. **Inline Execution** — use `executing-plans` in this session with checkpoints

Which approach?
