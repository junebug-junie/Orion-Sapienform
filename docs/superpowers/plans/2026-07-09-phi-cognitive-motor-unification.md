# φ Cognitive Motor Unification — harness grammar + seed-v3 encoder input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unified Orion turns (harness-governor) and classic cortex-exec runs both feed the same `ExecutionTrajectoryProjectionV1`, and spark emits `seed-v3` inner features with an honest 11-dim encoder trainable subset so strict variance gates can pass.

**Architecture:** Harness emits cortex-exec-compatible lifecycle grammar (`exec_request_received` … `exec_result_emitted`) with `trace_id=cortex.exec:{node}:{correlation_id}` and `source_service=orion-harness-governor`. Substrate reducer widens source filter to both motors. Spark keeps full saturated felt dims in corpus for audit but moves `reliability_pressure` to infra and excludes flat felt dims from encoder/fit input. Encoder stays off until operator promotes on seed-v3 corpus.

**Tech Stack:** Python 3, Pydantic v2, pytest, Redis bus (`orion:grammar:event`), FastAPI substrate projection HTTP, numpy fit script.

**Spec:** `docs/superpowers/specs/2026-07-09-phi-cognitive-motor-unification-design.md`

---

## Scope and anti-stub rules

| Capability | This plan | Out of scope |
|---|---|---|
| Harness lifecycle grammar emit | ✅ | — |
| Substrate reducer widen to harness | ✅ | — |
| seed-v3 feature assembly | ✅ | — |
| fit_phi_encoder versioned input | ✅ | — |
| Enable `ORION_PHI_ENCODER_ENABLED=true` | — | operator after corpus accrual |
| New bus channel / dual projections | — | spec non-goal |
| Massage self-state scores for variance | — | spec non-goal |

**Repo rules:**
- Branch/worktree: `git worktree add ../Orion-Sapienform-phi-motor-unify -b feat/phi-cognitive-motor-unification`
- `ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0`
- After `.env_example` edits: `python scripts/sync_local_env_from_example.py`
- No keyword triggers for `reasoning_present`; deterministic rules only

---

## File structure (created / modified)

**Created:**
- `orion/harness/grammar_emit.py`
- `orion/harness/tests/test_harness_grammar_emit.py`
- `orion/harness/tests/test_harness_grammar_lifecycle_reducer.py`
- `services/orion-spark-introspector/tests/test_inner_state_seed_v3.py`

**Modified:**
- `orion/substrate/execution_loop/constants.py`
- `orion/substrate/execution_loop/grammar_extract.py`
- `orion/substrate/execution_loop/reducer.py`
- `orion/substrate/execution_loop/ids.py` (shared `cortex_exec_trace_id`)
- `orion/harness/runner.py`
- `services/orion-harness-governor/app/bus_listener.py`
- `services/orion-spark-introspector/app/inner_state.py`
- `services/orion-spark-introspector/app/settings.py`
- `services/orion-spark-introspector/.env_example`
- `services/orion-spark-introspector/docker-compose.yml`
- `scripts/fit_phi_encoder.py`
- `tests/test_execution_substrate_reducer.py`
- `tests/test_phi_encoder_fit_script.py`
- `tests/test_inner_state_trajectory_features.py`

---

## Task 1: Shared trace ID + substrate reducer widening

**Files:**
- Modify: `orion/substrate/execution_loop/ids.py`
- Modify: `orion/substrate/execution_loop/constants.py`
- Modify: `orion/substrate/execution_loop/grammar_extract.py`
- Modify: `orion/substrate/execution_loop/reducer.py`
- Modify: `tests/test_execution_substrate_reducer.py`
- Modify: `services/orion-cortex-exec/app/grammar_emit.py` (import shared trace helper)

- [ ] **Step 1: Write the failing harness reducer test**

Add to `tests/test_execution_substrate_reducer.py`:

```python
def _harness_atom(role: str, summary: str, *, event_id: str = "gev_h") -> GrammarEventV1:
    atom = GrammarAtomV1(
        atom_id=f"{TRACE}:{role}",
        trace_id=TRACE,
        atom_type="observation",
        semantic_role=role,
        layer="harness",
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
            source_service="orion-harness-governor",
            source_component="harness_grammar_emit",
        ),
        correlation_id="corr-abc",
    )


def test_extract_accepts_harness_governor_lifecycle() -> None:
    events = [
        _harness_atom(
            "exec_request_received",
            "Harness exec received request for verb=orion_unified, mode=orion, steps=0",
            event_id="h1",
        ),
        _harness_atom("exec_step_started", "Step started: order=1, step=fcc, verb=orion_unified, services=none", event_id="h2"),
        _harness_atom(
            "exec_result_assembled",
            "Final result assembled: status=success, final_text_present=True, reasoning_present=True, thinking_source=harness_fcc",
            event_id="h3",
        ),
        _harness_atom(
            "exec_result_emitted",
            "Harness exec result emitted to reply_to=True, status=success",
            event_id="h4",
        ),
    ]
    run = extract_execution_state_from_events(events, now=FIXED_TS)
    assert run.verb == "orion_unified"
    assert run.mode == "orion"
    assert run.started_step_count == 1
    assert run.reasoning_present is True
    assert run.thinking_source == "harness_fcc"


def test_reducer_noops_harness_fcc_step_role() -> None:
    bad = GrammarEventV1(
        event_id="noop1",
        event_kind="atom_emitted",
        trace_id=TRACE,
        emitted_at=FIXED_TS,
        atom=GrammarAtomV1(
            atom_id=f"{TRACE}:harness_fcc_step",
            trace_id=TRACE,
            semantic_role="harness_fcc_step",
            summary="Harness step 0: tool=none, ok",
        ),
        provenance=GrammarProvenanceV1(source_service="orion-harness-governor"),
    )
    proj, receipt = reduce_execution_trace_events(
        events=[bad], projection=_empty_projection(), now=FIXED_TS,
    )
    assert receipt.noop_event_ids == ["noop1"]
    assert proj.runs == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_execution_substrate_reducer.py::test_extract_accepts_harness_governor_lifecycle -v`

Expected: FAIL — harness events skipped or reducer noops batch

- [ ] **Step 3: Implement widening**

`constants.py`:

```python
EXECUTION_SOURCE_SERVICES = frozenset({
    "orion-cortex-exec",
    "orion-harness-governor",
})
# Keep EXECUTION_SOURCE_SERVICE for backward compat imports; prefer EXECUTION_SOURCE_SERVICES.
EXECUTION_SOURCE_SERVICE = "orion-cortex-exec"
```

`ids.py` — add:

```python
def cortex_exec_trace_id(node_name: str, correlation_id: str) -> str:
    return f"cortex.exec:{node_name}:{correlation_id}"
```

`grammar_extract.py` — replace single-service check:

```python
from .constants import EXECUTION_SOURCE_SERVICES

# inside loop:
if event.provenance.source_service not in EXECUTION_SOURCE_SERVICES:
    continue
if role == "harness_fcc_step":
    continue
```

`reducer.py` — replace batch guard:

```python
from .constants import EXECUTION_SOURCE_SERVICES

if any(e.provenance.source_service not in EXECUTION_SOURCE_SERVICES for e in events):
    ...
```

`services/orion-cortex-exec/app/grammar_emit.py` — import `cortex_exec_trace_id` from `orion.substrate.execution_loop.ids` and delete local duplicate.

- [ ] **Step 4: Run substrate tests**

Run: `pytest tests/test_execution_substrate_reducer.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/execution_loop/ tests/test_execution_substrate_reducer.py services/orion-cortex-exec/app/grammar_emit.py
git commit -m "feat(substrate): accept harness-governor execution grammar lifecycle"
```

---

## Task 2: Harness `grammar_emit.py` collector

**Files:**
- Create: `orion/harness/grammar_emit.py`
- Create: `orion/harness/tests/test_harness_grammar_emit.py`

- [ ] **Step 1: Write the failing unit test**

```python
from datetime import datetime, timezone

from orion.harness.grammar_emit import (
    HarnessGrammarCollector,
    build_harness_grammar_events,
    compute_harness_reasoning_present,
    publish_harness_lifecycle_grammar,
)
from orion.substrate.execution_loop.ids import cortex_exec_trace_id

NODE = "athena"
CORR = "corr-harness-1"
FIXED = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)


def test_trace_id_matches_cortex_exec_shape() -> None:
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    assert c.trace_id == cortex_exec_trace_id(NODE, CORR)


def test_compute_reasoning_present_rules() -> None:
    assert compute_harness_reasoning_present(step_count=2, reflection_ran=False, quick_lane_skipped_5b=True, grammar_receipt_count=0) is True
    assert compute_harness_reasoning_present(step_count=0, reflection_ran=True, quick_lane_skipped_5b=False, grammar_receipt_count=0) is True
    assert compute_harness_reasoning_present(step_count=0, reflection_ran=True, quick_lane_skipped_5b=True, grammar_receipt_count=0) is False
    assert compute_harness_reasoning_present(step_count=0, reflection_ran=False, quick_lane_skipped_5b=False, grammar_receipt_count=3) is True


def test_build_events_include_lifecycle_roles() -> None:
    c = HarnessGrammarCollector(node_name=NODE, correlation_id=CORR, observed_at=FIXED)
    c.record_request_received()
    c.record_plan_started(step_count=0)
    c.record_step_started(order=1, summary="fcc step")
    c.record_step_completed(order=1)
    c.record_result_assembled(
        status="success",
        final_text_present=True,
        step_count=1,
        grammar_receipt_count=1,
        reflection_ran=False,
        quick_lane_skipped_5b=True,
    )
    events = build_harness_grammar_events(c)
    roles = {e.atom.semantic_role for e in events if e.atom}
    assert {
        "exec_request_received",
        "exec_plan_started",
        "exec_step_started",
        "exec_step_completed",
        "exec_result_assembled",
    } <= roles
    assembled = next(e for e in events if e.atom and e.atom.semantic_role == "exec_result_assembled")
    assert "reasoning_present=True" in assembled.atom.summary
    assert "thinking_source=harness_fcc" in assembled.atom.summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/harness/tests/test_harness_grammar_emit.py -v`

Expected: FAIL — module not found

- [ ] **Step 3: Implement `grammar_emit.py`**

Key exports:

```python
def compute_harness_reasoning_present(
    *,
    step_count: int,
    reflection_ran: bool,
    quick_lane_skipped_5b: bool,
    grammar_receipt_count: int,
) -> bool:
    if step_count > 0:
        return True
    if reflection_ran and not quick_lane_skipped_5b:
        return True
    return grammar_receipt_count > 0


def compute_harness_thinking_source(*, step_count: int, reflection_ran: bool, quick_lane_skipped_5b: bool) -> str:
    if step_count > 0:
        return "harness_fcc"
    if reflection_ran and not quick_lane_skipped_5b:
        return "finalize_reflect"
    return "none"
```

`HarnessGrammarCollector` mirrors cortex lifecycle record methods. Summary strings must use `key=value` tokens parsed by `grammar_extract._parse_summary_kv`:

- `exec_request_received`: `verb=orion_unified, mode=orion, steps=0`
- `exec_plan_started`: `step_count={n}`
- `exec_result_assembled`: same shape as cortex-exec (`status=…, final_text_present=…, reasoning_present=…, thinking_source=…`)
- `exec_result_emitted`: `reply_to={bool}, status={status}` (harness wording ok; reducer only checks role)

`build_harness_grammar_events(collector) -> list[GrammarEventV1]` — same event envelope pattern as `build_cortex_exec_grammar_events`.

`async def publish_harness_lifecycle_grammar(bus, *, channel, events, publish_fn=None) -> None` — fail-open wrapper around `orion.grammar.publish.publish_grammar_event`; log warning on exception, never raise.

- [ ] **Step 4: Run harness grammar tests**

Run: `pytest orion/harness/tests/test_harness_grammar_emit.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/harness/grammar_emit.py orion/harness/tests/test_harness_grammar_emit.py
git commit -m "feat(harness): lifecycle grammar collector for execution trajectory"
```

---

## Task 3: Wire harness motor + finalize publish points

**Files:**
- Modify: `orion/harness/runner.py`
- Modify: `services/orion-harness-governor/app/bus_listener.py`
- Modify: `orion/harness/tests/test_harness_runner.py`

- [ ] **Step 1: Write failing runner integration test**

Extend `orion/harness/tests/test_harness_runner.py` with a test that stubs `publish_harness_lifecycle_grammar` and asserts lifecycle publish called on motor start/end with `step_count > 0` → `reasoning_present=True` in assembled summary.

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest orion/harness/tests/test_harness_runner.py -k lifecycle -v`

- [ ] **Step 3: Wire `HarnessRunner.run()`**

At run start (after prompt build):
- Create `HarnessGrammarCollector(node_name=settings.node_name or env, correlation_id=request.correlation_id)`
- `record_request_received()`, `record_plan_started(step_count=0)`
- If `recall_debug` passed in (thread via new optional param from bus_listener): `record_recall_gate_observed()`

Per fcc `step` event:
- `record_step_started(order=step_count+1, …)` before existing `publish_harness_step_grammar`
- On success path: `record_step_completed`; on error path in motor: `record_step_failed`

At motor end (both success and empty-draft paths):
- `record_result_assembled` with `reflection_ran=False`, `quick_lane_skipped_5b=True` (finalize unknown here)
- `await publish_harness_lifecycle_grammar(self.bus, channel=self.grammar_channel, events=build_harness_grammar_events(collector))`

Add `node_name: str` constructor param defaulting from `HARNESS_NODE_NAME` or `settings.node_name`.

Keep existing `publish_harness_step_grammar` calls unchanged.

- [ ] **Step 4: Wire finalize in `bus_listener.handle_harness_run_request`**

After successful `run_harness_finalize_chain` when `run.final_text` is set:
- Build collector (or extend motor collector via return value) with finalize fields:
  - `reflection_ran = chain.reflection is not None`
  - `quick_lane_skipped_5b = chain.quick_lane_skipped_5b`
- `record_result_assembled` with full `compute_harness_reasoning_present(...)`
- `record_result_emitted(reply_present=True, status="success")`
- Publish lifecycle events (fail-open)

Pass `recall_debug` into `motor_runner.run(..., recall_debug=recall_debug)`.

On refused/empty-draft paths: still emit `exec_result_assembled` with `status=failed|refused` so trajectory reflects motor outcome.

- [ ] **Step 5: Run harness tests**

Run: `pytest orion/harness/tests/ services/orion-harness-governor/tests/test_harness_governor_rpc.py -q`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/harness/runner.py services/orion-harness-governor/app/bus_listener.py orion/harness/tests/
git commit -m "feat(harness): publish lifecycle grammar on motor and finalize"
```

---

## Task 4: Spark `seed-v3` feature contract

**Files:**
- Modify: `services/orion-spark-introspector/app/inner_state.py`
- Create: `services/orion-spark-introspector/tests/test_inner_state_seed_v3.py`
- Modify: `tests/test_inner_state_trajectory_features.py`

- [ ] **Step 1: Write failing seed-v3 tests**

`services/orion-spark-introspector/tests/test_inner_state_seed_v3.py`:

```python
def test_seed_v3_moves_reliability_to_infra_only(sample_self_state):
    payload, _, _ = build_inner_state_features(
        sample_self_state, scaler, features_version="seed-v3", grammar_degraded=False,
    )
    feature_names = {f.name for f in payload.features}
    infra_names = {f.name for f in payload.infra}
    assert "reliability_pressure" not in feature_names
    assert "reliability_pressure" in infra_names


def test_seed_v3_trainable_encoder_dims():
    names = encoder_trainable_feature_names("seed-v3")
    assert names == [
        "coherence", "agency_readiness", "execution_pressure", "reasoning_pressure",
        "continuity_pressure", "social_pressure", "overall_intensity",
        "recall_gate_fired", "reasoning_present", "exec_step_fail_rate", "execution_friction",
    ]
    assert len(names) == 11


def test_seed_v3_still_emits_saturated_felt_for_audit(sample_self_state):
    payload, _, _ = build_inner_state_features(
        sample_self_state, scaler, features_version="seed-v3", grammar_degraded=False,
    )
    assert "field_intensity" in {f.name for f in payload.features}
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_seed_v3.py -v`

- [ ] **Step 3: Implement seed-v3 in `inner_state.py`**

Add:

```python
ENCODER_EXCLUDED_FELT: frozenset[str] = frozenset({
    "field_intensity",
    "resource_pressure",
    "introspection_pressure",
})
INFRA_ONLY_FELT: frozenset[str] = frozenset({"reliability_pressure"})


def encoder_trainable_feature_names(features_version: str) -> list[str]:
    if features_version == "seed-v3":
        felt = [k for k in FELT_DIMENSIONS if k not in ENCODER_EXCLUDED_FELT and k not in INFRA_ONLY_FELT]
        return felt + ["overall_intensity"] + list(COGNITIVE_FEATURE_NAMES)
    felt = [k for k in FELT_DIMENSIONS if k not in DROPPED_DIMENSIONS]
    return felt + ["overall_intensity"] + list(COGNITIVE_FEATURE_NAMES)
```

In `build_inner_state_features`:
- For `seed-v3`, skip appending `reliability_pressure` to `features[]`; instead add to `infra[]` from `self_state.dimensions.reliability_pressure` (same raw read as felt).
- Change cognitive include guard: `features_version.startswith("seed-v2") or features_version == "seed-v3" or trajectory_projection is not None`
- `honest_headline`: read `reliability_pressure` from `raw_map` if present else from infra raw when seed-v3 (keep headline behavior stable).

- [ ] **Step 4: Run spark + trajectory tests**

Run: `pytest services/orion-spark-introspector/tests/test_inner_state_seed_v3.py tests/test_inner_state_trajectory_features.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/app/inner_state.py services/orion-spark-introspector/tests/test_inner_state_seed_v3.py tests/test_inner_state_trajectory_features.py
git commit -m "feat(spark): seed-v3 encoder trainable subset and infra reliability"
```

---

## Task 5: `fit_phi_encoder.py` versioned input + variance gate

**Files:**
- Modify: `scripts/fit_phi_encoder.py`
- Modify: `tests/test_phi_encoder_fit_script.py`

- [ ] **Step 1: Write failing fit-script test**

```python
def test_input_features_seed_v3_excludes_flat_dims():
    from scripts.fit_phi_encoder import input_features_for_version
    names = input_features_for_version("seed-v3")
    assert "field_intensity" not in names
    assert "resource_pressure" not in names
    assert "introspection_pressure" not in names
    assert "reliability_pressure" not in names
    assert len(names) == 11


def test_variance_gate_seed_v3_needs_nine_of_eleven():
    # synthetic 11-dim matrix with 9 varying cols passes at 0.8 fraction
    ...
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_phi_encoder_fit_script.py -k seed_v3 -v`

- [ ] **Step 3: Implement versioned feature selection**

Replace flat `input_features(legacy_corpus=...)` with:

```python
def input_features_for_version(features_version: str, *, legacy_corpus: bool = False) -> list[str]:
    if legacy_corpus:
        return list(FELT_DIMENSIONS) + ["overall_intensity"]
    return _inner_state.encoder_trainable_feature_names(features_version)


def resolve_features_version(row: InnerStateFeaturesV1, *, legacy_flag: bool) -> str:
    if legacy_flag:
        return "seed-v1"
    return row.features_version or DEFAULT_FEATURES_VERSION
```

When loading JSONL:
- Group/filter rows by `features_version`
- Default train command uses `--features-version seed-v3` (new argparse flag; default `seed-v3` for new trains, keep `seed-v2` available for legacy replay)
- Variance gate: `required = ceil(len(feature_names) * variance_fraction)` — with 11 dims @ 0.8 → 9

Update manifest `features_version` field to match corpus slice.

- [ ] **Step 4: Run fit script tests**

Run: `pytest tests/test_phi_encoder_fit_script.py tests/test_phi_encoder_mlp.py -q`

Expected: PASS (update any hardcoded 15-dim assumptions)

- [ ] **Step 5: Commit**

```bash
git add scripts/fit_phi_encoder.py tests/test_phi_encoder_fit_script.py
git commit -m "feat(phi): seed-v3 trainable dims and versioned variance gate"
```

---

## Task 6: Config defaults + acceptance smokes

**Files:**
- Modify: `services/orion-spark-introspector/app/settings.py`
- Modify: `services/orion-spark-introspector/.env_example`
- Modify: `services/orion-spark-introspector/docker-compose.yml`
- Modify: `services/orion-spark-introspector/tests/test_compose_seed_v2_telemetry_mount.py` (assert seed-v3 default or param documented)
- Modify: `services/orion-spark-introspector/README.md` (short seed-v3 note)

- [ ] **Step 1: Update defaults**

`settings.py` / `.env_example` / compose:

```text
INNER_FEATURES_VERSION=seed-v3
ORION_PHI_ENCODER_ENABLED=false
```

Run: `python scripts/sync_local_env_from_example.py`

- [ ] **Step 2: Unified-turn projection smoke (manual / script)**

After redeploy harness-governor + substrate-runtime + spark:

```bash
# Trigger Hub unified turn (mode=orion), then:
curl -fsS http://localhost:<substrate-port>/projections/execution_trajectory | jq '.projection.runs | to_entries[] | select(.value.verb=="orion_unified") | {id:.key, reasoning:.value.reasoning_present, steps:.value.started_step_count}'
```

Expected within 120s: at least one run with `reasoning_present=true` when fcc `step_count > 0`.

- [ ] **Step 3: Regression — cortex-exec still works**

Run existing: `pytest tests/test_execution_substrate_reducer.py services/orion-substrate-runtime/tests/test_execution_trajectory_endpoint.py -q`

- [ ] **Step 4: Agent gate**

```bash
make agent-check SERVICE=orion-spark-introspector
pytest tests/test_execution_substrate_reducer.py orion/harness/tests/test_harness_grammar_emit.py tests/test_phi_encoder_fit_script.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-introspector/ scripts/sync_local_env_from_example.py
git commit -m "chore(spark): default INNER_FEATURES_VERSION to seed-v3"
```

---

## Operator follow-up (post-merge, not blocking PR)

1. Restart `orion-harness-governor`, `orion-substrate-runtime`, `orion-spark-introspector`.
2. Accrue seed-v3 corpus on shared `/mnt/telemetry/phi/corpus/inner_state.jsonl` with mixed unified + classic traffic (≥4h, ≥500 ok rows).
3. Strict train: `python scripts/fit_phi_encoder.py --corpus ... --out ... --features-version seed-v3`
4. Promote only when variance ≥9/11 and existing promote gates pass; then flip `ORION_PHI_ENCODER_ENABLED=true`.

---

## Acceptance mapping (spec §)

| Check | Task |
|---|---|
| Unified turn smoke | Task 6 step 2 |
| cortex-exec regression | Task 1 + Task 6 step 3 |
| Reducer harness fixture | Task 1 |
| Spark seed-v3 infra/reliability | Task 4 |
| Variance gate ≥9/11 | Task 5 + operator follow-up |
| grammar_truth healthy | Task 6 agent-check |
| Fail-open grammar | Task 2 `publish_harness_lifecycle_grammar` + Task 3 wiring |
