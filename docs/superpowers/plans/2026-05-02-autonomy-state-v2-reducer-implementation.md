# AutonomyStateV2 Reducer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the env-gated `AutonomyStateV2` deterministic reducer, upgrade path from graph `AutonomyStateV1`, cortex stance wiring, router metadata previews, Hub forwarding whitelist, tests, and operator doc exactly as specified in [`docs/superpowers/specs/2026-05-01-autonomy-state-v2-reducer-design.md`](../specs/2026-05-01-autonomy-state-v2-reducer-design.md).

**Architecture:** Pure-Pydantic models in `orion/autonomy/models.py`; state transition logic in new `orion/autonomy/reducer.py` (no I/O, no LLM); `summarize_autonomy_state` widened for V2-only summary hazards and attention-based proposal headlines; cortex `build_chat_stance_inputs` runs the reducer behind `AUTONOMY_STATE_V2_REDUCER_ENABLED`; router forwards compact previews to Hub via existing metadata path.

**Tech stack:** Python 3, Pydantic v2, existing `pytest` layout (`python3 -m pytest`). Service scope: shared `orion/` package + `services/orion-cortex-exec` + `services/orion-hub/scripts/autonomy_payloads.py`.

**Spec vs. chat_stance ordering:** The design doc shows the reducer after `_load_autonomy_state` and uses `ctx.get("chat_reasoning_summary")` for reasoning evidence. Today `chat_reasoning_summary` is written to `ctx` only after reasoning compilation, which currently runs *after* autonomy load. **Resolve by moving `_compile_reasoning_summary(ctx)` to run immediately before `_load_autonomy_state(ctx)`** in `build_chat_stance_inputs`, then set `ctx["chat_reasoning_summary"] = reasoning["summary"]` right after compile so reducer evidence matches the spec. Re-run the single `social["hazards"]` line that merges reasoning hazards after both are available (keep it after reasoning + autonomy as today).

**Note:** If exploratory edits exist under `orion/autonomy/` from an aborted run, reset those files to `main` before starting Task 1 so TDD stays clean.

---

## File map

| Path | Role |
|------|------|
| `orion/autonomy/models.py` | New evidence/attention/impulse/outcome/delta models; `AutonomyStateV2`; `upgrade_autonomy_state_v1_to_v2` |
| `orion/autonomy/reducer.py` | `AutonomyReducerInputV1`, `AutonomyReducerResultV1`, `reduce_autonomy_state` |
| `orion/autonomy/summary.py` | Union type + V2-only hazards and proposal headlines |
| `services/orion-cortex-exec/app/chat_stance.py` | Reasoning-before-autonomy order; `hashlib`; `_build_autonomy_reducer_evidence`; `_run_autonomy_reducer`; gated ctx + `inputs["autonomy"]` keys |
| `services/orion-cortex-exec/app/router.py` | `autonomy_state_v2_preview` + `autonomy_state_delta` in `_autonomy_payload_from_ctx` |
| `services/orion-hub/scripts/autonomy_payloads.py` | Whitelist two new keys |
| `orion/autonomy/tests/test_autonomy_state_v2_upgrade.py` | Upgrade + evidence ID tests |
| `orion/autonomy/tests/test_autonomy_reducer.py` | Reducer behavior tests |
| `orion/autonomy/tests/test_autonomy_summary_v2.py` | Summary V2-only hazards |
| `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py` | Integration: flag on/off, ctx keys, router preview |
| `services/orion-hub/tests/` or script check | If no hub test dir, add a tiny unit test next to hub scripts or document manual `extract_autonomy_payload` assertion in operator doc |
| `docs/autonomy_state_v2_reducer.md` | Operator-facing doc |
| `services/orion-cortex-exec/.env_example` | Document `AUTONOMY_STATE_V2_REDUCER_ENABLED` (optional but matches other feature flags) |

---

### Task 1: Schema and upgrade (`orion/autonomy/models.py`)

**Files:**
- Modify: `orion/autonomy/models.py`
- Test: `orion/autonomy/tests/test_autonomy_state_v2_upgrade.py`

- [ ] **Step 1: Failing test — upgrade preserves V1 fields and sets V2 defaults**

Create `orion/autonomy/tests/test_autonomy_state_v2_upgrade.py`:

```python
from __future__ import annotations

from datetime import datetime

from orion.autonomy.models import AutonomyGoalHeadlineV1, AutonomyStateV1, AutonomyStateV2, upgrade_autonomy_state_v1_to_v2


def test_upgrade_preserves_v1_fields_and_sets_v2_schema() -> None:
    v1 = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        latest_identity_snapshot_id="snap-1",
        latest_drive_audit_id="audit-1",
        latest_goal_ids=["g1"],
        dominant_drive="coherence",
        active_drives=["coherence"],
        drive_pressures={"coherence": 0.4},
        tension_kinds=["tension.x"],
        goal_headlines=[],
        source="graph",
        generated_at=datetime(2026, 1, 1, 12, 0, 0),
    )
    v2 = upgrade_autonomy_state_v1_to_v2(v1)
    assert isinstance(v2, AutonomyStateV2)
    assert v2.schema_version == "autonomy.state.v2"
    assert v2.subject == v1.subject
    assert v2.drive_pressures == v1.drive_pressures
    assert v2.confidence == 0.55
    assert "no_action_outcome_history" in v2.unknowns
    assert "evidence_from_graph_only" in v2.unknowns
```

Run: `cd /mnt/scripts/Orion-Sapienform && python3 -m pytest orion/autonomy/tests/test_autonomy_state_v2_upgrade.py::test_upgrade_preserves_v1_fields_and_sets_v2_schema -v`  
Expected: **FAIL** (import or missing symbols).

- [ ] **Step 2: Implement models + `upgrade_autonomy_state_v1_to_v2`**

Append to `orion/autonomy/models.py` (keep all existing V1 models unchanged):

1. `import hashlib` at top.
2. Add Pydantic models exactly as in spec §1: `AutonomyEvidenceRefV1`, `AttentionItemV1`, `CandidateImpulseV1`, `InhibitedImpulseV1`, `ActionOutcomeRefV1`, `AutonomyStateDeltaV1`.
3. `class AutonomyStateV2(AutonomyStateV1):` with extra fields: `schema_version`, `evidence_refs`, `freshness`, `confidence`, `unknowns`, `attention_items`, `candidate_impulses`, `inhibited_impulses`, `last_action_outcomes`, `previous_state_ref`.
4. Implement `upgrade_autonomy_state_v1_to_v2` per spec §1: stable evidence IDs `identity_snapshot:{id}`, `drive_audit:{id}`, `goal_ref:{goal_id}`; `confidence=0.55`; unknowns including `no_action_outcome_history`, `evidence_from_graph_only`, and `no_identity_snapshot` / `no_drive_audit` when IDs missing; attention item when `dominant_drive` or `tension_kinds` non-empty with `item_id = hashlib.sha256(f"{subject}:{seed_kind}:{dominant_drive or ''}".encode()).hexdigest()[:16]` using a fixed literal for the middle segment (e.g. `attention_seed`) as in the approved design conversation.

Run: same pytest path. Expected: **PASS**.

- [ ] **Step 3: Tests — evidence IDs, dedup, attention when dominant**

Add tests in the same file:

- `test_upgrade_evidence_ids_stable` — assert evidence_id strings match formulas for snapshot/audit/goals.
- `test_upgrade_merge_no_duplicate_evidence` — upgrade same `v1` twice, merge evidence ref lists by `evidence_id` in test helper simulating reducer merge; expect single entry per id.
- `test_upgrade_unknowns_without_snapshot_audit` — `None` IDs add `no_identity_snapshot` and `no_drive_audit`.
- `test_upgrade_attention_when_dominant_or_tensions` — empty dominant and empty tensions → no attention requirement beyond spec; with dominant → at least one attention item.

Run: `python3 -m pytest orion/autonomy/tests/test_autonomy_state_v2_upgrade.py -v`  
Expected: **PASS**.

- [ ] **Step 4: Commit**

```bash
git add orion/autonomy/models.py orion/autonomy/tests/test_autonomy_state_v2_upgrade.py
git commit -m "feat(autonomy): add AutonomyStateV2 schema and V1 upgrade helper"
```

---

### Task 2: Reducer core (`orion/autonomy/reducer.py`)

**Files:**
- Create: `orion/autonomy/reducer.py`
- Modify: none yet
- Test: `orion/autonomy/tests/test_autonomy_reducer.py`

- [ ] **Step 1: Failing test — cold baseline**

```python
from datetime import datetime

from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state


def test_reducer_cold_start_orion_binding() -> None:
    fixed = datetime(2026, 5, 2, 12, 0, 0)
    r = reduce_autonomy_state(
        AutonomyReducerInputV1(subject="orion", previous_state=None, evidence=[], action_outcomes=[], now=fixed)
    )
    assert r.state.confidence == 0.25
    assert "no_previous_state" in r.state.unknowns
    assert r.state.entity_id == "self:orion"
```

Run: `python3 -m pytest orion/autonomy/tests/test_autonomy_reducer.py::test_reducer_cold_start_orion_binding -v`  
Expected: **FAIL** (module missing).

- [ ] **Step 2: Implement `AutonomyReducerInputV1`, `AutonomyReducerResultV1`, `reduce_autonomy_state` skeleton**

- Import `SUBJECT_BINDINGS`, `SubjectBinding` from `orion.autonomy.repository`.
- `previous_state is None` → minimal `AutonomyStateV2` per spec (confidence 0.25, `unknowns=["no_previous_state"]`, `source="reducer"`, bindings for subject).
- `previous_state` is `AutonomyStateV1` → `upgrade_autonomy_state_v1_to_v2`.
- `previous_state` is `AutonomyStateV2` → `model_copy(deep=True)`.
- Deep-copy snapshot at start of turn for delta baseline.
- Merge `evidence` and `action_outcomes` with dedupe and bounds (20 / 8 / 8 / 8 / 12 / 12 per spec).
- Implement drive heuristics, tension table, impulses, inhibitions, confidence adjustments, unknowns rebuild, freshness keys, delta fields per spec §2.

Use `datetime.utcnow()` only when `inp.now is None` (tests inject `now`).

Run cold-start test: **PASS**.

- [ ] **Step 3: Tests from spec §7B (representative)**

Add tests (names should match intent):

- `test_reducer_user_message_and_infra_no_drive_pressure` — new evidence with `source=user_message` and `kind=infra_health` does not increase `coherence` vs baseline when those are the only deltas (use explicit prior pressures).
- `test_reducer_capability_timeout_unavailable` — evidence summary containing `GraphDB timeout unavailable` increases capability and sets `tension.capability_gap.v1`.
- `test_reducer_polarity_blind_no_contradiction_still_raises_coherence` — evidence text `"no contradiction"` still matches `contradiction` substring; name the test to document limitation.
- `test_reducer_all_proxy_inhibition_and_unknown` — only `proxy_telemetry` evidence, confidence rules, inhibition `proxy_signal_not_canonical_state`, `proxy_only_evidence` unknown.
- `test_reducer_high_surprise_outcome_reduces_confidence` — pass `action_outcomes` with `surprise=0.9`.
- `test_reducer_determinism_fixed_now` — two calls identical `model_dump(mode="json")`.
- `test_reducer_evidence_trim_twenty` — feed 25 new evidence items with unique ids, assert `len(state.evidence_refs) <= 20`.

Run: `python3 -m pytest orion/autonomy/tests/test_autonomy_reducer.py -v`  
Expected: **PASS**.

- [ ] **Step 4: Commit**

```bash
git add orion/autonomy/reducer.py orion/autonomy/tests/test_autonomy_reducer.py
git commit -m "feat(autonomy): add deterministic autonomy state v2 reducer"
```

---

### Task 3: Summary widening (`orion/autonomy/summary.py`)

**Files:**
- Modify: `orion/autonomy/summary.py`
- Test: `orion/autonomy/tests/test_autonomy_summary_v2.py`

- [ ] **Step 1: Failing tests for V2-only hazards and proposal headlines**

```python
from datetime import datetime

from orion.autonomy.models import AutonomyStateV2
from orion.autonomy.summary import summarize_autonomy_state


def test_v2_low_confidence_hazard() -> None:
    s = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="graph",
        generated_at=datetime.utcnow(),
        drive_pressures={"coherence": 0.1},
        tension_kinds=[],
        goal_headlines=[],
        confidence=0.35,
        unknowns=[],
        inhibited_impulses=[],
        attention_items=[],
    )
    out = summarize_autonomy_state(s)
    assert "avoid overconfident inner-state claims" in out.response_hazards


def test_v2_proxy_inhibition_hazard() -> None:
    from orion.autonomy.models import InhibitedImpulseV1

    s = AutonomyStateV2(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        source="graph",
        generated_at=datetime.utcnow(),
        drive_pressures={},
        tension_kinds=[],
        goal_headlines=[],
        confidence=0.9,
        unknowns=[],
        inhibited_impulses=[
            InhibitedImpulseV1(
                impulse_id="i1",
                kind="k",
                summary="s",
                inhibition_reason="proxy_signal_not_canonical_state",
            )
        ],
        attention_items=[],
    )
    out = summarize_autonomy_state(s)
    assert "do not treat proxy telemetry as canonical state" in out.response_hazards
```

Run: `python3 -m pytest orion/autonomy/tests/test_autonomy_summary_v2.py -v`  
Expected: **FAIL** until summary updated.

- [ ] **Step 2: Implement**

- Change signature to `def summarize_autonomy_state(state: AutonomyStateV1 | AutonomyStateV2 | None)`.
- Import `AutonomyStateV2`.
- After computing `proposal_headlines` from goals, if `isinstance(state, AutonomyStateV2)` and not `goal_headlines` and `state.attention_items`, replace with top 3 attention summaries (bounded).
- Before return, if V2: append the three conditional hazards per spec §3 (unknowns non-empty hazard; proxy inhibition hazard only when matching inhibition present).

Run tests + regression: `python3 -m pytest orion/autonomy/tests/test_autonomy_summary_v2.py tests/test_autonomy_summary.py -v`  
Expected: **PASS**.

- [ ] **Step 3: Commit**

```bash
git add orion/autonomy/summary.py orion/autonomy/tests/test_autonomy_summary_v2.py
git commit -m "feat(autonomy): summarize AutonomyStateV2 hazards and proposals"
```

---

### Task 4: Cortex chat stance integration

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py`
- Test: `services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py`

- [ ] **Step 1: Integration test with monkeypatch**

- Monkeypatch `_load_autonomy_state` to return a dict with `state` a minimal `AutonomyStateV1`, `summary` from `summarize_autonomy_state`, `debug` with `orion: {availability: "available"}`, other keys as `_load_autonomy_state` would.
- Set `AUTONOMY_STATE_V2_REDUCER_ENABLED=true`, call `build_chat_stance_inputs(ctx)`.
- Assert `ctx["chat_autonomy_state_v2"]` and `ctx["chat_autonomy_state_delta"]` are dicts with expected top-level keys (`schema_version`, `confidence`, etc.).
- Assert `inputs["autonomy"]["state_v2"]` and `["delta"]` present.
- Flag unset → keys absent (second test).

Use `monkeypatch.setenv` / `delenv` and reload or patch `os.getenv` locally if imports cached.

Run: `python3 -m pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py -v`  
Expected: **FAIL** until wired.

- [ ] **Step 2: Implement**

1. `import hashlib` at module top if not present.
2. Add `_build_autonomy_reducer_evidence(ctx: dict, autonomy: dict) -> list[AutonomyEvidenceRefV1]` exactly per spec §4 (user turn, infra from `autonomy["debug"]`, reasoning fallback from `ctx.get("chat_reasoning_summary")`, social_bridge hazards from `ctx.get("chat_social_bridge_summary")`).
3. Add `_run_autonomy_reducer(ctx, autonomy)` calling `reduce_autonomy_state(AutonomyReducerInputV1(subject=state.subject if state else "orion", previous_state=autonomy["state"], evidence=..., action_outcomes=[]))`.
4. In `build_chat_stance_inputs`:
   - Move `reasoning = _compile_reasoning_summary(ctx)` **above** `autonomy = _load_autonomy_state(ctx)`.
   - Immediately after reasoning compile: `ctx["chat_reasoning_summary"] = reasoning["summary"]` so evidence builder sees it (duplicate assignment later is fine).
   - After `autonomy = _load_autonomy_state(ctx)` and building `inputs` dict shell, if `os.getenv("AUTONOMY_STATE_V2_REDUCER_ENABLED", "").strip().lower() == "true":` try/except per spec, populate ctx + `inputs["autonomy"]["state_v2"]` / `["delta"]`.
5. Keep existing `summarize_autonomy_state` on V1 path unchanged; do not replace `inputs["autonomy"]["state"]` with V2.

Run: `python3 -m pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py -v`  
Expected: **PASS**.

- [ ] **Step 3: Reducer exception path**

Test that if `reduce_autonomy_state` raises, `build_chat_stance_inputs` still returns and `chat_autonomy_state_v2` absent (monkeypatch reducer to raise).

- [ ] **Step 4: Commit**

```bash
git add services/orion-cortex-exec/app/chat_stance.py services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py
git commit -m "feat(cortex-exec): gate AutonomyStateV2 reducer in chat stance"
```

---

### Task 5: Router metadata

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py`
- Test: extend `services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py` or add assertions in `test_chat_stance_autonomy_v2.py` via `_autonomy_payload_from_ctx` import

- [ ] **Step 1: Test `_autonomy_payload_from_ctx` includes preview + delta**

```python
from app.router import _autonomy_payload_from_ctx


def test_autonomy_payload_includes_v2_preview_and_delta() -> None:
    ctx = {
        "chat_autonomy_state_v2": {
            "schema_version": "autonomy.state.v2",
            "dominant_drive": "coherence",
            "active_drives": ["coherence", "continuity", "relational"],
            "confidence": 0.5,
            "unknowns": ["a", "b", "c", "d", "e", "f"],
            "attention_items": [{"summary": "x"}, {"summary": "y"}],
            "inhibited_impulses": [{"inhibition_reason": "proxy_signal_not_canonical_state"}],
        },
        "chat_autonomy_state_delta": {"subject": "orion", "confidence_delta": 0.1},
    }
    md = _autonomy_payload_from_ctx(ctx)
    assert md["autonomy_state_v2_preview"]["dominant_drive"] == "coherence"
    assert len(md["autonomy_state_v2_preview"]["active_drives"]) <= 3
    assert md["autonomy_state_delta"]["confidence_delta"] == 0.1
```

Run from cortex-exec service root with `PYTHONPATH` set per repo convention (often `pytest` from service dir). Example:

`cd /mnt/scripts/Orion-Sapienform/services/orion-cortex-exec && python3 -m pytest tests/test_router_autonomy_payload_export.py -k v2 -v`  
(adjust test file placement to match project convention)

Expected: **FAIL** until router updated.

- [ ] **Step 2: Implement** the block from spec §5 verbatim after existing payload fields in `_autonomy_payload_from_ctx`.

- [ ] **Step 3: Run full router autonomy tests**

`python3 -m pytest services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py -v`  
Expected: **PASS** (all prior tests unchanged).

- [ ] **Step 4: Commit**

```bash
git add services/orion-cortex-exec/app/router.py services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py
git commit -m "feat(cortex-exec): export autonomy v2 preview and delta in router metadata"
```

---

### Task 6: Hub whitelist

**Files:**
- Modify: `services/orion-hub/scripts/autonomy_payloads.py`
- Test: small test under `services/orion-hub/tests/` if the repo has pytest for hub; else add `test_autonomy_payloads_v2_forwarding.py` colocated or document manual check

- [ ] **Step 1: Assert forwarding**

```python
from types import SimpleNamespace

from autonomy_payloads import extract_autonomy_payload


def test_extract_autonomy_payload_forwards_v2_keys() -> None:
    result = SimpleNamespace(
        metadata={
            "autonomy_summary": {"stance_hint": "x"},
            "autonomy_state_v2_preview": {"dominant_drive": "coherence"},
            "autonomy_state_delta": {"subject": "orion"},
        }
    )
    payload = extract_autonomy_payload(result)
    assert payload["autonomy_state_v2_preview"]["dominant_drive"] == "coherence"
    assert payload["autonomy_state_delta"]["subject"] == "orion"
```

Run from hub scripts context with correct `PYTHONPATH`.

- [ ] **Step 2: Add keys** to the `for key in (` tuple in `extract_autonomy_payload`.

- [ ] **Step 3: Commit**

```bash
git add services/orion-hub/scripts/autonomy_payloads.py
# plus test file if added
git commit -m "feat(hub): forward autonomy v2 preview and delta in autonomy payload"
```

---

### Task 7: Operator doc + env example

**Files:**
- Create: `docs/autonomy_state_v2_reducer.md`
- Modify: `services/orion-cortex-exec/.env_example`

- [ ] **Step 1: Write** `docs/autonomy_state_v2_reducer.md` with causal diagram (ASCII), env var name, explicit “not sentience / not durable / phi is proxy”, known limitations from spec §Known limitations.

- [ ] **Step 2: Add** `AUTONOMY_STATE_V2_REDUCER_ENABLED=` with short comment to `.env_example`.

- [ ] **Step 3: Commit**

```bash
git add docs/autonomy_state_v2_reducer.md services/orion-cortex-exec/.env_example
git commit -m "docs(autonomy): operator notes for autonomy state v2 reducer"
```

---

### Task 8: Final verification

- [ ] **Run targeted suites**

```bash
cd /mnt/scripts/Orion-Sapienform
python3 -m pytest orion/autonomy/tests/ tests/test_autonomy_summary.py -q
python3 -m pytest services/orion-cortex-exec/tests/test_chat_stance_autonomy_v2.py services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py -q
```

Expected: all **PASS**.

- [ ] **Optional:** `python3 -m compileall orion/autonomy services/orion-cortex-exec/app/chat_stance.py services/orion-cortex-exec/app/router.py`

---

## Self-review (plan vs spec)

| Spec section | Covered by |
|--------------|------------|
| §1 Schema + upgrade | Task 1 |
| §2 Reducer behavior + bounds + delta semantics | Task 2 |
| §3 Summary V2 | Task 3 |
| §4 Chat stance + evidence | Task 4 (+ ordering note for reasoning) |
| §5 Router | Task 5 |
| §6 Hub | Task 6 |
| §7 Tests A–E | Tasks 1–6 distribute cases; Task 8 confirms |
| §8 Operator doc | Task 7 |
| Env gate default off | Task 4, 7 |

**Gaps addressed in plan:** Reasoning evidence timing vs `_load_autonomy_state` order (explicit swap + early `ctx` assignment). **No placeholders** in task steps beyond optional hub test location choice.

---

**Plan complete and saved to** `docs/superpowers/plans/2026-05-02-autonomy-state-v2-reducer-implementation.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — Dispatch a fresh subagent per task; review between tasks. **Required sub-skill:** superpowers:subagent-driven-development.

2. **Inline execution** — Run tasks in this session with checkpoints. **Required sub-skill:** superpowers:executing-plans.

Which approach do you want?
