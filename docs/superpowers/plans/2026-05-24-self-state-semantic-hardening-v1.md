# Self-State Semantic Hardening v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden Layer 6 self-state semantics so stabilizing channels never appear in `unresolved_pressures`, dimension evidence is dimension-specific, and attention saturation contributes only to `field_intensity`—without adding Layer 7 proposals, bus publish, or runtime wiring changes.

**Architecture:** Extend `self_state_policy.v1.yaml` with explicit `pressure_channels` and `context_channels` lists; teach `SelfStatePolicyV1` to load them; gate unresolved-pressure extraction in `builder.py` with `evidence_for_dimension()`; add `stabilized_but_loaded` summary label. No schema version bump—`SelfStateV1` shape unchanged.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest. Runtime `orion-self-state-runtime` reloads policy from mounted `config/self_state/` on restart (no new env vars).

**Design source:** User spec “PR: Self-State Semantic Hardening v1” (2026-05-24).

**Depends on:** `feat/self-state-v1` merged to `main` (verified: merge commit `be3ffed1`).

**Non-goals:** `ProposalFrameV1`, action candidates, policy gates, cortex-exec steering, bus publish, LLM interpretation, mind service, operator notifications, new organs, recall/vision changes, SQL migrations.

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-self-state-semantic-hardening-v1 \
  -b feat/self-state-semantic-hardening-v1 \
  main
cd .worktrees/feat-self-state-semantic-hardening-v1
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

**Rules:**
- All commits only in `.worktrees/feat-self-state-semantic-hardening-v1`.
- Never bleed files to the main checkout **except** copying `.env_example` → `.env` when `.env_example` changes (this PR: **no** `.env_example` changes expected).
- PR title: `PR: Self-State Semantic Hardening v1`.
- When done: run `requesting-code-review` subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-24-self-state-semantic-hardening-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Layer 6 on `main` | `orion/self_state/`, `config/self_state/self_state_policy.v1.yaml`, `services/orion-self-state-runtime`, Hub `GET /api/substrate/self-state/latest` |
| Bug location | `orion/self_state/builder.py:166-173` — any mapped channel ≥ threshold becomes unresolved, including stabilizers |
| Evidence bug | `builder.py:182-185` — reuses global `dominant_field_channels` top-3 for every dimension |
| `PRESSURE_CHANNELS` in scoring | `orion/self_state/scoring.py:7-22` — frozenset for field collection; keep aligned with YAML `pressure_channels` but **unresolved list is policy-gated in builder only** |
| Bus channels | **No publish** — `orion/bus/channels` unchanged |
| Schema registry | **No new schemas** — `orion/schemas/registry.py` unchanged |
| Service wiring | Policy path already in runtime; **no** `settings.py` / `docker-compose.yml` / `requirements.txt` / `README.md` edits unless smoke doc tweak |
| Live symptom | `availability→coherence`, `confidence→coherence`, `available_capacity→coherence` in `unresolved_pressures` |

### Before / after (live-shaped fixture)

**Before (current builder with live-like channels):**

```text
unresolved_pressures:
  availability→coherence
  confidence→coherence
  available_capacity→coherence
  execution_load→execution_pressure
  ...
```

**After (expected):**

```text
stabilizing_factors:
  availability=1.00
  confidence=1.00
  available_capacity=1.00

unresolved_pressures:
  cpu_pressure→resource_pressure
  execution_load→execution_pressure
  pressure→resource_pressure
  execution_pressure→execution_pressure

summary_labels includes: execution_loaded, stabilized_but_loaded (when coherence≥0.8 and execution_pressure≥0.7)
```

---

## File structure

| Path | Responsibility |
|------|----------------|
| `config/self_state/self_state_policy.v1.yaml` | Add `pressure_channels`, `context_channels` |
| `orion/self_state/policy.py` | `pressure_channels`, `context_channels` on `SelfStatePolicyV1` |
| `orion/self_state/builder.py` | Gate unresolved pressures; `evidence_for_dimension()`; `stabilized_but_loaded` label; sort/dedupe lists |
| `tests/test_self_state_policy_loader.py` | Assert new policy lists |
| `tests/test_self_state_builder_hardening.py` | **Create** — semantic hardening assertions |
| `docs/superpowers/pr-reports/2026-05-24-self-state-semantic-hardening-v1-pr.md` | PR report (post-implementation) |

**Explicitly out of scope (no edits):**

- `orion/schemas/self_state.py`, `orion/schemas/registry.py`
- `orion/bus/channels/*`
- `services/orion-self-state-runtime/*` (behavior changes via shared `orion/self_state` only)
- Hub routes (read-only passthrough of persisted JSON)

---

# Phase 0 — Worktree + branch

### Task 0: Create isolated worktree

- [ ] **Step 1: Create worktree from main**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-self-state-semantic-hardening-v1 \
  -b feat/self-state-semantic-hardening-v1 \
  main
cd .worktrees/feat-self-state-semantic-hardening-v1
```

Expected: `git branch --show-current` → `feat/self-state-semantic-hardening-v1`

- [ ] **Step 2: Verify isolation**

```bash
git check-ignore -q .worktrees && echo "worktree gitignored ok"
pwd
```

Expected: path ends with `.worktrees/feat-self-state-semantic-hardening-v1`

---

# Phase 1 — Policy channel roles

### Task 1: Extend `self_state_policy.v1.yaml`

**Files:**
- Modify: `config/self_state/self_state_policy.v1.yaml`

- [ ] **Step 1: Add channel role lists after `stabilizing_channels`**

Append to YAML (keep existing `stabilizing_channels` map unchanged):

```yaml
pressure_channels:
  - execution_load
  - execution_friction
  - failure_pressure
  - reasoning_load
  - cpu_pressure
  - gpu_pressure
  - memory_pressure
  - disk_pressure
  - thermal_pressure
  - staleness
  - pressure
  - execution_pressure
  - reasoning_pressure
  - reliability_pressure

context_channels:
  - recent_perturbation_count
  - overall_salience
```

- [ ] **Step 2: Commit**

```bash
git add config/self_state/self_state_policy.v1.yaml
git commit -m "feat(self-state): declare pressure and context channel roles in policy"
```

---

### Task 2: Extend `SelfStatePolicyV1`

**Files:**
- Modify: `orion/self_state/policy.py`
- Test: `tests/test_self_state_policy_loader.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_self_state_policy_loader.py`:

```python
def test_policy_channel_role_lists() -> None:
    policy = load_self_state_policy(POLICY)
    assert "execution_load" in policy.pressure_channels
    assert "availability" not in policy.pressure_channels
    assert "recent_perturbation_count" in policy.context_channels
    assert "availability" in policy.stabilizing_channels
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-self-state-semantic-hardening-v1
PYTHONPATH=. pytest tests/test_self_state_policy_loader.py::test_policy_channel_role_lists -v
```

Expected: FAIL — `SelfStatePolicyV1` has no attribute `pressure_channels`

- [ ] **Step 3: Add fields to policy model**

In `orion/self_state/policy.py`, inside `SelfStatePolicyV1`:

```python
    pressure_channels: list[str] = Field(default_factory=list)
    context_channels: list[str] = Field(default_factory=list)
```

Place after `stabilizing_channels` (order matches YAML mental model).

- [ ] **Step 4: Run test to verify it passes**

```bash
PYTHONPATH=. pytest tests/test_self_state_policy_loader.py -v
```

Expected: PASS (both loader tests)

- [ ] **Step 5: Commit**

```bash
git add orion/self_state/policy.py tests/test_self_state_policy_loader.py
git commit -m "feat(self-state): load pressure and context channel lists from policy"
```

---

# Phase 2 — Builder semantic hardening

### Task 3: Gate unresolved pressures + sort/dedupe

**Files:**
- Modify: `orion/self_state/builder.py`

- [ ] **Step 1: Replace unresolved/stabilizing loop**

In `build_self_state()`, replace lines ~166-173:

```python
    pressure_channel_set = set(policy.pressure_channels)

    unresolved: list[str] = []
    stabilizing: list[str] = []
    for ch, v in merged_channels.items():
        if ch in policy.stabilizing_channels and v >= 0.3:
            stabilizing.append(f"{ch}={v:.2f}")
        dim = policy.channel_dimension_map.get(ch)
        if (
            dim
            and ch in pressure_channel_set
            and v >= policy.unresolved_pressure_threshold
        ):
            unresolved.append(f"{ch}→{dim}")

    unresolved = sorted(set(unresolved))
    stabilizing = sorted(set(stabilizing))
```

- [ ] **Step 2: Commit**

```bash
git add orion/self_state/builder.py
git commit -m "fix(self-state): only pressure channels may appear in unresolved_pressures"
```

---

### Task 4: Dimension-specific evidence

**Files:**
- Modify: `orion/self_state/builder.py`
- Test: `tests/test_self_state_builder_hardening.py`

- [ ] **Step 1: Write failing hardening test file (evidence + pressures)**

Create `tests/test_self_state_builder_hardening.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
ATTENTION_POLICY = load_attention_policy(
    REPO / "config" / "attention" / "field_attention_policy.v1.yaml"
)
SELF_POLICY = load_self_state_policy(
    REPO / "config" / "self_state" / "self_state_policy.v1.yaml"
)
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _live_shaped_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_semantic_hardening",
        node_vectors={
            "node:athena": {
                "availability": 1.0,
                "confidence": 1.0,
                "available_capacity": 1.0,
                "execution_load": 1.0,
                "cpu_pressure": 0.92,
                "pressure": 1.0,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=[f"state_delta:{i}" for i in range(20)],
    )


def _built_state():
    field = _live_shaped_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    return build_self_state(
        field=field, attention=attention, policy=SELF_POLICY, now=NOW
    )


def test_stabilizers_not_in_unresolved_pressures() -> None:
    state = _built_state()
    assert "availability→coherence" not in state.unresolved_pressures
    assert "confidence→coherence" not in state.unresolved_pressures
    assert "available_capacity→coherence" not in state.unresolved_pressures


def test_stabilizers_in_stabilizing_factors() -> None:
    state = _built_state()
    assert "availability=1.00" in state.stabilizing_factors
    assert "confidence=1.00" in state.stabilizing_factors
    assert "available_capacity=1.00" in state.stabilizing_factors


def test_pressure_channels_in_unresolved_pressures() -> None:
    state = _built_state()
    assert "execution_load→execution_pressure" in state.unresolved_pressures
    assert "execution_pressure→execution_pressure" in state.unresolved_pressures
    assert "pressure→resource_pressure" in state.unresolved_pressures


def test_dimension_evidence_is_dimension_specific() -> None:
    state = _built_state()
    execution_evidence = state.dimensions["execution_pressure"].dominant_evidence
    coherence_evidence = state.dimensions["coherence"].dominant_evidence
    assert any(
        "execution_load" in x or "execution_pressure" in x for x in execution_evidence
    )
    assert any(
        "availability" in x or "confidence" in x for x in coherence_evidence
    )
    assert execution_evidence != coherence_evidence


def test_context_channels_not_unresolved() -> None:
    state = _built_state()
    assert not any(
        "recent_perturbation_count" in x for x in state.unresolved_pressures
    )
    assert not any("overall_salience" in x for x in state.unresolved_pressures)
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=. pytest tests/test_self_state_builder_hardening.py -v
```

Expected: FAIL on evidence tests (and possibly pressure list if Task 3 not done)

- [ ] **Step 3: Add `evidence_for_dimension` helper**

In `orion/self_state/builder.py`, before `build_self_state`:

```python
def evidence_for_dimension(
    *,
    dim_id: str,
    merged_channels: dict[str, float],
    policy: SelfStatePolicyV1,
    limit: int = 3,
) -> list[str]:
    pairs: list[tuple[str, float]] = []
    for channel, value in merged_channels.items():
        if policy.channel_dimension_map.get(channel) == dim_id:
            pairs.append((channel, value))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return [f"{ch}={v:.2f}" for ch, v in pairs[:limit]]
```

- [ ] **Step 4: Use helper when building dimensions**

Replace the dimension loop body (`dominant_evidence=...`) with:

```python
        dimensions[dim_id] = SelfStateDimensionV1(
            dimension_id=dim_id,
            score=clamp01(score),
            confidence=overall_confidence,
            dominant_evidence=evidence_for_dimension(
                dim_id=dim_id,
                merged_channels=merged_channels,
                policy=policy,
            ),
            reasons=[f"{dim_id} from field+attention channel synthesis"],
        )
```

Remove use of `dominant_field_channels` for per-dimension evidence (keep `dominant_field_channels` on `SelfStateV1` top-level field unchanged).

- [ ] **Step 5: Run hardening tests**

```bash
PYTHONPATH=. pytest tests/test_self_state_builder_hardening.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orion/self_state/builder.py tests/test_self_state_builder_hardening.py
git commit -m "fix(self-state): dimension-specific dominant evidence"
```

---

### Task 5: `stabilized_but_loaded` summary label

**Files:**
- Modify: `orion/self_state/builder.py`
- Test: `tests/test_self_state_builder_hardening.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_self_state_builder_hardening.py`:

```python
def test_stabilized_but_loaded_label() -> None:
    state = _built_state()
    assert state.dimensions["coherence"].score >= 0.8
    assert state.dimensions["execution_pressure"].score >= 0.7
    assert "stabilized_but_loaded" in state.summary_labels
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH=. pytest tests/test_self_state_builder_hardening.py::test_stabilized_but_loaded_label -v
```

Expected: FAIL — label missing

- [ ] **Step 3: Extend `_emit_summary_labels`**

In `_emit_summary_labels`, after existing label checks, before `return sorted(set(labels))`:

```python
    if (
        dimension_scores.get("coherence", 0.0) >= 0.8
        and dimension_scores.get("execution_pressure", 0.0) >= 0.7
    ):
        labels.append("stabilized_but_loaded")
```

- [ ] **Step 4: Run test to verify pass**

```bash
PYTHONPATH=. pytest tests/test_self_state_builder_hardening.py::test_stabilized_but_loaded_label -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/self_state/builder.py tests/test_self_state_builder_hardening.py
git commit -m "feat(self-state): add stabilized_but_loaded summary label"
```

---

# Phase 3 — Regression verification

### Task 6: Full self-state + attention regression

- [ ] **Step 1: Run required test suites**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-self-state-semantic-hardening-v1

PYTHONPATH=. pytest tests/test_self_state_*.py -q

PYTHONPATH=. pytest \
  tests/test_attention_frame_schemas.py \
  tests/test_attention_frame_builder.py \
  tests/test_attention_field_scoring.py \
  tests/test_attention_policy_loader.py \
  -q

PYTHONPATH=. python -m compileall \
  orion/self_state \
  orion/schemas/self_state.py \
  services/orion-self-state-runtime \
  -q
```

Expected: all tests PASS, compileall silent

- [ ] **Step 2: Optional live smoke (stack running)**

```bash
./scripts/smoke_self_state_v1.sh

curl -s http://localhost:8080/api/substrate/self-state/latest \
  | jq '.unresolved_pressures, .stabilizing_factors, .dimensions.execution_pressure, .dimensions.coherence'
```

Expected after runtime restart (picks up new `orion/self_state` via image rebuild or volume mount):

- `unresolved_pressures` lacks `availability→coherence`, `confidence→coherence`, `available_capacity→coherence`
- `stabilizing_factors` includes `availability=…`, `confidence=…`, `available_capacity=…`
- `overall_condition` still `loaded` (or similar) with `execution_pressure > 0`
- `summary_labels` includes `execution_loaded`

- [ ] **Step 3: Commit only if smoke script or docs updated**

If no file changes: skip commit.

---

# Phase 4 — Code review + PR

### Task 7: Code review subagent

- [ ] **Step 1: Dispatch `requesting-code-review` subagent**

Prompt the subagent with:
- Branch `feat/self-state-semantic-hardening-v1`
- Spec: semantic hardening only (this plan + user PR spec)
- Files changed: policy YAML, `policy.py`, `builder.py`, tests
- Require: no Layer 7 logic, no bus/registry churn, stabilizer gating correct

- [ ] **Step 2: Fix all actionable findings**

Re-run Task 6 commands after fixes.

- [ ] **Step 3: Commit review fixes**

```bash
git add -A
git commit -m "fix(self-state): address code review findings"
```

(omit commit if no changes)

---

### Task 8: PR report + push

**Files:**
- Create: `docs/superpowers/pr-reports/2026-05-24-self-state-semantic-hardening-v1-pr.md`

- [ ] **Step 1: Write PR report**

Include sections:
- Before/after `unresolved_pressures` / `stabilizing_factors` (live or fixture)
- Tests run with pass counts
- Explicit note: **semantic hardening only**; Layer 7 deferred
- Non-goals checklist from spec

- [ ] **Step 2: Push and open PR**

```bash
git push -u origin feat/self-state-semantic-hardening-v1

gh pr create --base main --title "PR: Self-State Semantic Hardening v1" --body "$(cat <<'EOF'
## Summary
- Gate `unresolved_pressures` to explicit `pressure_channels` only; stabilizers stay in `stabilizing_factors`.
- Per-dimension `dominant_evidence` from channels mapped to that dimension.
- Add `stabilized_but_loaded` when coherent but execution-loaded.

## Test plan
- [x] `pytest tests/test_self_state_*.py`
- [x] Attention regression suite (unchanged contracts)
- [ ] Optional: `./scripts/smoke_self_state_v1.sh` + Hub curl after runtime restart

## Layer scope
Layer 6 semantic hardening only. Layer 7 proposals/actions remain deferred.

EOF
)"
```

- [ ] **Step 3: Commit PR report in worktree**

```bash
git add docs/superpowers/pr-reports/2026-05-24-self-state-semantic-hardening-v1-pr.md
git commit -m "docs: self-state semantic hardening v1 PR report"
git push
```

---

## Self-review checklist

| Spec requirement | Task |
|------------------|------|
| Stabilizers not in unresolved | Task 3 + hardening tests |
| Stabilizers in stabilizing_factors | Task 3 + hardening tests |
| Pressure channels in unresolved | Task 1–3 + hardening tests |
| Dimension-specific evidence | Task 4 |
| Context channels not unresolved | Task 1 + 3 + hardening tests |
| Attention saturation → field_intensity only | Already in `field_intensity_score`; context_channels not in pressure list |
| `stabilized_but_loaded` label | Task 5 |
| No Layer 7 / bus / LLM | Preflight + Phase 4 review |
| Required pytest commands | Task 6 |

**Placeholder scan:** None — all steps include concrete code and commands.

---

## Acceptance criteria

- [ ] Stabilizing channels absent from `unresolved_pressures`
- [ ] Stabilizing channels present in `stabilizing_factors` when ≥ 0.3
- [ ] Pressure channels present in `unresolved_pressures` when ≥ threshold
- [ ] Each dimension’s `dominant_evidence` lists channels mapped to that dimension only
- [ ] `recent_perturbation_count` / `overall_salience` never in `unresolved_pressures`
- [ ] Existing `tests/test_self_state_*.py` pass
- [ ] No proposal/action/policy/cortex changes
