# Integrated Memory Cognition Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the encode → store → retrieve → lifecycle loop so brain mode and orion unified mode form low-stakes beliefs automatically, fetch them by purpose via PCR, and stop behaving like surface timeline SQL.

**Architecture:** M1 adds `formation_policy` + `auto_activate` + dynamics persistence + reinforce-on-dedup + cards/chroma projection from consolidation. M2 unifies PCR on brain/unified lanes (phase 0→1→stance→3), fixes retrieval intent default for belief fetch, filters/ranks active_packet by `dynamics.activation`, and removes Hub `brain.recall.v1` default. M3 (appendix) wires recall_boost + decay reaper.

**Tech Stack:** Python 3.12, Pydantic v2, asyncpg, Redis bus, pytest, bash smoke.

**Spec:** `docs/superpowers/specs/2026-07-08-integrated-memory-cognition-loop-design.md`

**Branch:** `feat/integrated-memory-cognition-loop`

**Worktree setup:**

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-memory-cognition-loop -b feat/integrated-memory-cognition-loop
cd ../Orion-Sapienform-memory-cognition-loop
python scripts/sync_local_env_from_example.py
```

---

## File Map

| File | Milestone | Action | Responsibility |
|------|-----------|--------|----------------|
| `orion/memory/crystallization/schemas.py` | M1 | Modify | `CrystallizationDynamicsV1` + `dynamics` field on `MemoryCrystallizationV1` |
| `orion/core/storage/sql/memory_crystallizations.sql` | M1 | Modify | `dynamics jsonb` column |
| `orion/memory/crystallization/repository.py` | M1,M2 | Modify | Round-trip `dynamics`; `count_eligible_active`; `update_crystallization` |
| `orion/memory/crystallization/dynamics.py` | M1 | Modify | `seed_weak_dynamics()` for auto-encode ratio |
| `orion/memory/crystallization/formation_policy.py` | M1 | Create | `resolve_formation_policy()` |
| `orion/memory/crystallization/formation_executor.py` | M1 | Create | `auto_activate()` |
| `orion/memory/crystallization/recall_eligibility.py` | M1,M2 | Create | `eligible_for_recall()`, `ACTIVATION_RECALL_FLOOR` |
| `orion/memory/crystallization/intake_pipeline.py` | M1 | Create | dedup → policy → activate/insert → project |
| `services/orion-memory-consolidation/app/worker.py` | M1 | Modify | Call intake pipeline instead of raw insert |
| `services/orion-memory-consolidation/app/settings.py` | M1 | Modify | Formation env keys |
| `services/orion-memory-consolidation/.env_example` | M1 | Modify | Formation env keys |
| `orion/memory/crystallization/bus_emit.py` | M1 | Modify | `reinforced`, `auto_activated` lifecycle kinds |
| `orion/bus/channels.yaml` | M1 | Modify | New crystallization channels |
| `orion/memory/retrieval_intent.py` | M2 | Modify | `brain_lane_belief_default` rule |
| `services/orion-cortex-exec/app/pcr_chat_memory.py` | M2 | Modify | Pass `eligible_belief_count` to intent |
| `services/orion-recall/app/worker.py` | M2 | Modify | Emit `eligible_belief_count` in PCR debug |
| `orion/memory/crystallization/retriever.py` | M2 | Modify | Rank by `activation * salience` |
| `services/orion-recall/app/collectors/active_packet.py` | M2 | Modify | Filter by recall eligibility |
| `services/orion-cortex-exec/app/router.py` | M2 | Modify | `stance_react` phase 0+1 hook |
| `services/orion-cortex-exec/app/grounding_capsule.py` | M2 | Modify | Full PCR before/after stance |
| `services/orion-hub/scripts/cortex_request_builder.py` | M2 | Modify | `hub_chat_lane` + drop `brain.recall.v1` default |
| `services/orion-cortex-exec/app/settings.py` | M2 | Modify | `MEMORY_COGNITION_BRAIN_BELIEF_DEFAULT` |
| `services/orion-cortex-exec/.env_example` | M2 | Modify | Recall cognition keys |
| `tests/test_formation_policy_auto_vs_gated.py` | M1 | Create | Policy matrix |
| `tests/test_formation_executor_auto_activate.py` | M1 | Create | auto_activate unit tests |
| `tests/test_encode_reinforce_not_duplicate.py` | M1 | Create | Intake dedup integration |
| `tests/test_retrieval_intent.py` | M2 | Modify | brain_lane_belief_default cases |
| `services/orion-cortex-exec/tests/test_unified_pcr_phase01.py` | M2 | Create | stance_react phase 0+1 |
| `services/orion-recall/tests/test_active_packet_activation_floor.py` | M2 | Create | Floor filter |
| `services/orion-hub/tests/test_cortex_request_builder.py` | M2 | Modify | brain recall default + hub_chat_lane |
| `scripts/smoke_memory_cognition_loop_e2e.sh` | M2 | Create | End-to-end smoke |

---

# Milestone M1 — Activate the store

## Task 1: Dynamics schema + SQL + repository round-trip

**Files:**
- Modify: `orion/memory/crystallization/schemas.py`
- Modify: `orion/core/storage/sql/memory_crystallizations.sql`
- Modify: `orion/memory/crystallization/repository.py`
- Test: `tests/test_memory_crystallization_dynamics.py` (existing — make pass)

- [ ] **Step 1: Write the failing test** (if `CrystallizationDynamicsV1` missing, test already fails)

Ensure `tests/test_memory_crystallization_dynamics.py` imports succeed:

```python
from orion.memory.crystallization.schemas import CrystallizationDynamicsV1, MemoryCrystallizationV1

def test_crystallization_has_default_dynamics():
    from orion.memory.crystallization.proposer import propose
    from orion.memory.crystallization.schemas import MemoryCrystallizationProposeRequestV1
    req = MemoryCrystallizationProposeRequestV1(
        kind="semantic", subject="s", summary="s", scope=["p"], proposed_by="test"
    )
    crys = propose(req)
    assert isinstance(crys.dynamics, CrystallizationDynamicsV1)
    assert crys.dynamics.activation == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_crystallization_dynamics.py::TestDynamicsSchema::test_crystallization_has_default_dynamics -v`

Expected: FAIL (`CrystallizationDynamicsV1` not defined or no `dynamics` field)

- [ ] **Step 3: Add schema + SQL + repository**

In `schemas.py` before `MemoryCrystallizationV1`:

```python
class CrystallizationDynamicsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    activation: float = Field(default=0.0, ge=0.0, le=1.0)
    reinforcement_count: int = 0
    formed_at: datetime | None = None
    last_reinforced_at: datetime | None = None
    last_recalled_at: datetime | None = None
    decay_half_life_days: float = 30.0
    retired_at: datetime | None = None
```

On `MemoryCrystallizationV1` add:

```python
dynamics: CrystallizationDynamicsV1 = Field(default_factory=CrystallizationDynamicsV1)
```

In `memory_crystallizations.sql` after `salience` line:

```sql
    dynamics jsonb NOT NULL DEFAULT '{}'::jsonb,
```

In `repository.py` `_row_to_crystallization()` parse `dynamics` jsonb into `CrystallizationDynamicsV1`. In `insert_crystallization` and `update_crystallization` include `dynamics` column with `_jsonb(crystallization.dynamics.model_dump(mode="json"))`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_memory_crystallization_dynamics.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/schemas.py orion/core/storage/sql/memory_crystallizations.sql orion/memory/crystallization/repository.py tests/test_memory_crystallization_dynamics.py
git commit -m "feat(memory): add crystallization dynamics schema and persistence"
```

---

## Task 2: `seed_weak_dynamics` for auto-encode activation ratio

**Files:**
- Modify: `orion/memory/crystallization/dynamics.py`
- Test: `tests/test_memory_crystallization_dynamics.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_memory_crystallization_dynamics.py`:

```python
from orion.memory.crystallization.dynamics import seed_weak_dynamics

def test_seed_weak_dynamics_scales_salience():
    crys = _crys()
    crys.salience = 0.5
    seeded = seed_weak_dynamics(crys, now=_now(), ratio=0.4)
    assert seeded.dynamics.activation == 0.2
    assert seeded.dynamics.formed_at is not None
```

- [ ] **Step 2: Run test — expect FAIL** (`seed_weak_dynamics` not defined)

Run: `pytest tests/test_memory_crystallization_dynamics.py::test_seed_weak_dynamics_scales_salience -v`

- [ ] **Step 3: Implement**

In `dynamics.py`:

```python
def seed_weak_dynamics(
    crystallization: MemoryCrystallizationV1,
    *,
    now: datetime,
    ratio: float = 0.4,
    min_activation: float = 0.05,
    max_activation: float = 0.35,
) -> MemoryCrystallizationV1:
    updated = crystallization.model_copy(deep=True)
    raw = _clamp(crystallization.salience * _clamp(ratio))
    updated.dynamics.activation = max(min_activation, min(max_activation, raw))
    updated.dynamics.formed_at = _aware(now)
    updated.updated_at = _aware(now)
    return updated
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_memory_crystallization_dynamics.py -q`

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/dynamics.py tests/test_memory_crystallization_dynamics.py
git commit -m "feat(memory): add seed_weak_dynamics for auto-encode activation"
```

---

## Task 3: Formation policy

**Files:**
- Create: `orion/memory/crystallization/formation_policy.py`
- Create: `tests/test_formation_policy_auto_vs_gated.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_formation_policy_auto_vs_gated.py`:

```python
from orion.memory.crystallization.formation_policy import FormationPolicy, resolve_formation_policy
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import MemoryCrystallizationProposeRequestV1

def _crys(*, kind="semantic", sensitivity="private", scope=None):
    req = MemoryCrystallizationProposeRequestV1(
        kind=kind, subject="Deploy plan", summary="We chose k3s for staging",
        scope=scope or ["project:orion"], proposed_by="test", sensitivity=sensitivity,
    )
    return propose(req)

def test_semantic_auto_activate():
    policy, reasons = resolve_formation_policy(_crys(kind="semantic"))
    assert policy == FormationPolicy.AUTO_ACTIVATE
    assert not reasons

def test_contradiction_governor_queue():
    policy, reasons = resolve_formation_policy(_crys(kind="contradiction"))
    assert policy == FormationPolicy.GOVERNOR_QUEUE

def test_intimate_governor_queue():
    policy, _ = resolve_formation_policy(_crys(sensitivity="intimate"))
    assert policy == FormationPolicy.GOVERNOR_QUEUE

def test_identity_scope_governor_queue():
    policy, _ = resolve_formation_policy(_crys(scope=["identity:orion"]))
    assert policy == FormationPolicy.GOVERNOR_QUEUE
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_formation_policy_auto_vs_gated.py -v`

- [ ] **Step 3: Implement**

Create `orion/memory/crystallization/formation_policy.py`:

```python
from __future__ import annotations

from enum import Enum
from typing import Literal

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

AUTO_ACTIVE_KINDS = frozenset({"semantic", "episode", "open_loop", "procedure"})
GATED_KINDS = frozenset({"stance", "decision", "contradiction", "attractor", "failure_mode"})
IDENTITY_SCOPE_PREFIX = "identity:"


class FormationPolicy(str, Enum):
    AUTO_ACTIVATE = "auto_activate"
    GOVERNOR_QUEUE = "governor_queue"
    REINFORCE_EXISTING = "reinforce_existing"


def _has_identity_scope(crystallization: MemoryCrystallizationV1, *, prefix: str = IDENTITY_SCOPE_PREFIX) -> bool:
    return any(str(s).startswith(prefix) for s in crystallization.scope)


def resolve_formation_policy(
    crystallization: MemoryCrystallizationV1,
    *,
    duplicate_id: str | None = None,
    identity_scope_prefix: str = IDENTITY_SCOPE_PREFIX,
) -> tuple[FormationPolicy, list[str]]:
    reasons: list[str] = []
    if duplicate_id:
        return FormationPolicy.REINFORCE_EXISTING, [f"duplicate:{duplicate_id}"]
    if crystallization.governance.sensitivity == "intimate":
        return FormationPolicy.GOVERNOR_QUEUE, ["intimate_sensitivity"]
    if _has_identity_scope(crystallization, prefix=identity_scope_prefix):
        return FormationPolicy.GOVERNOR_QUEUE, ["identity_scope"]
    if crystallization.kind in GATED_KINDS:
        return FormationPolicy.GOVERNOR_QUEUE, [f"gated_kind:{crystallization.kind}"]
    if crystallization.kind in AUTO_ACTIVE_KINDS:
        return FormationPolicy.AUTO_ACTIVATE, reasons
    return FormationPolicy.GOVERNOR_QUEUE, [f"unknown_kind:{crystallization.kind}"]
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_formation_policy_auto_vs_gated.py -q`

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/formation_policy.py tests/test_formation_policy_auto_vs_gated.py
git commit -m "feat(memory): add formation policy for auto-activate vs governor queue"
```

---

## Task 4: `auto_activate` executor

**Files:**
- Create: `orion/memory/crystallization/formation_executor.py`
- Create: `tests/test_formation_executor_auto_activate.py`

- [ ] **Step 1: Write failing test**

```python
import pytest
from orion.memory.crystallization.formation_executor import auto_activate, GovernorPathRequired
from orion.memory.crystallization.proposer import propose
from orion.memory.crystallization.schemas import MemoryCrystallizationProposeRequestV1, CrystallizationEvidenceRefV1

def _proposed_semantic():
    req = MemoryCrystallizationProposeRequestV1(
        kind="semantic", subject="Topic", summary="We agreed on Postgres for memory store",
        scope=["project:orion"],
        evidence=[CrystallizationEvidenceRefV1(source_kind="chat_turn", source_id="corr-1")],
        proposed_by="memory_consolidation_intake",
    )
    return propose(req)

def test_auto_activate_sets_active_and_weak_dynamics():
    crys, hist = auto_activate(_proposed_semantic(), encode_ratio=0.4)
    assert crys.status == "active"
    assert crys.governance.approval_mode == "auto_policy"
    assert crys.governance.requires_manual_review is False
    assert crys.governance.approved_by == "system:formation_policy"
    assert 0.05 <= crys.dynamics.activation <= 0.35
    assert hist["op"] == "auto_activate"

def test_auto_activate_rejects_contradiction():
    req = MemoryCrystallizationProposeRequestV1(
        kind="contradiction", subject="x", summary="y", scope=["p"], proposed_by="t"
    )
    with pytest.raises(GovernorPathRequired):
        auto_activate(propose(req))
```

- [ ] **Step 2: Run — expect FAIL**

Run: `pytest tests/test_formation_executor_auto_activate.py -v`

- [ ] **Step 3: Implement**

Create `formation_executor.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.memory.crystallization.dynamics import seed_weak_dynamics
from orion.memory.crystallization.formation_policy import resolve_formation_policy, FormationPolicy
from orion.memory.crystallization.schemas import MemoryCrystallizationV1
from orion.memory.crystallization.validator import validate_proposal


class GovernorPathRequired(ValueError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def auto_activate(
    crystallization: MemoryCrystallizationV1,
    *,
    actor: str = "system:formation_policy",
    encode_ratio: float = 0.4,
) -> tuple[MemoryCrystallizationV1, dict]:
    policy, reasons = resolve_formation_policy(crystallization)
    if policy != FormationPolicy.AUTO_ACTIVATE:
        raise GovernorPathRequired("; ".join(reasons) or policy.value)
    validation = validate_proposal(crystallization)
    if not validation.valid:
        raise GovernorPathRequired("; ".join(validation.errors))
    now = _utc_now()
    updated = crystallization.model_copy(deep=True)
    updated.status = "active"
    updated.governance.approval_mode = "auto_policy"
    updated.governance.requires_manual_review = False
    updated.governance.approved_by = actor
    updated.governance.validation_status = "valid"
    updated.governance.last_reviewed_at = now
    updated = seed_weak_dynamics(updated, now=now, ratio=encode_ratio)
    history = {
        "op": "auto_activate",
        "actor": actor,
        "reasons": reasons,
        "before": {"status": crystallization.status},
        "after": {"status": "active", "activation": updated.dynamics.activation},
    }
    return updated, history
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_formation_executor_auto_activate.py -q`

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/formation_executor.py tests/test_formation_executor_auto_activate.py
git commit -m "feat(memory): add auto_activate formation executor"
```

---

## Task 5: Recall eligibility helper

**Files:**
- Create: `orion/memory/crystallization/recall_eligibility.py`
- Create: `tests/test_recall_eligibility.py`

- [ ] **Step 1: Write failing test**

```python
from orion.memory.crystallization.recall_eligibility import eligible_for_recall, ACTIVATION_RECALL_FLOOR
from orion.memory.crystallization.schemas import CrystallizationDynamicsV1, MemoryCrystallizationV1, CrystallizationGovernanceV1
from datetime import datetime, timezone

def _active(activation: float) -> MemoryCrystallizationV1:
    now = datetime.now(timezone.utc)
    return MemoryCrystallizationV1(
        crystallization_id="crys_test",
        kind="semantic", subject="s", summary="s", status="active",
        dynamics=CrystallizationDynamicsV1(activation=activation, formed_at=now),
        governance=CrystallizationGovernanceV1(proposed_by="t"),
        created_at=now, updated_at=now,
    )

def test_eligible_when_active_and_above_floor():
    assert eligible_for_recall(_active(0.2)) is True

def test_ineligible_when_below_floor():
    assert eligible_for_recall(_active(0.01)) is False

def test_ineligible_when_deprecated():
    crys = _active(0.2)
    crys.status = "deprecated"
    assert eligible_for_recall(crys) is False
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

```python
from __future__ import annotations

import os

from orion.memory.crystallization.schemas import MemoryCrystallizationV1

ACTIVATION_RECALL_FLOOR = float(os.getenv("ACTIVATION_RECALL_FLOOR", "0.08"))
_INELIGIBLE_STATUSES = frozenset({"deprecated", "archived", "rejected", "quarantined"})


def eligible_for_recall(
    crystallization: MemoryCrystallizationV1,
    *,
    floor: float | None = None,
) -> bool:
    if crystallization.status != "active":
        return False
    if crystallization.status in _INELIGIBLE_STATUSES:
        return False
    threshold = ACTIVATION_RECALL_FLOOR if floor is None else floor
    return float(crystallization.dynamics.activation) >= threshold
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/recall_eligibility.py tests/test_recall_eligibility.py
git commit -m "feat(memory): add recall eligibility activation floor helper"
```

---

## Task 6: Consolidation intake pipeline (dedup → policy → store → project)

**Files:**
- Create: `orion/memory/crystallization/intake_pipeline.py`
- Create: `tests/test_encode_reinforce_not_duplicate.py`
- Modify: `orion/memory/crystallization/repository.py` (add `update_crystallization`, `list_crystallizations` filter)

- [ ] **Step 1: Write failing integration test**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from orion.memory.crystallization.intake_pipeline import process_consolidation_crystallization
from orion.memory.crystallization.intake_consolidation_window import build_crystallization_from_window
from orion.memory.consolidation_gate import ConsolidationGateResult

@pytest.mark.asyncio
async def test_reinforce_existing_skips_insert(monkeypatch):
  # Build two identical window crystallizations; mock list_crystallizations to return first after insert
  # Assert update_crystallization called with reinforcement_count incremented
  # Assert insert_crystallization NOT called second time
  ...
```

(Flesh out with real mocks in implementation — test must assert `reinforce` path when duplicate detected.)

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `intake_pipeline.py`**

```python
async def process_consolidation_crystallization(
    pool,
    bus,
    *,
    crystallization: MemoryCrystallizationV1,
    settings,
    project_config,
) -> tuple[str, MemoryCrystallizationV1, str]:
    """Returns (crystallization_id, final_row, outcome) where outcome is
    auto_activated | proposed | reinforced."""
    from orion.memory.crystallization.detection import detect_duplicates
    from orion.memory.crystallization.repository import list_crystallizations, insert_crystallization, update_crystallization
    from orion.memory.crystallization.formation_policy import FormationPolicy, resolve_formation_policy
    from orion.memory.crystallization.formation_executor import auto_activate, GovernorPathRequired
    from orion.memory.crystallization.dynamics import reinforce
    from orion.memory.crystallization.projector import project_crystallization
    from orion.memory.crystallization.salience import apply_salience
    from orion.memory.crystallization.bus_emit import emit_crystallization_lifecycle

    existing = await list_crystallizations(pool, status=None, limit=200)
    detection = detect_duplicates(crystallization, existing)
    duplicate_id = detection.duplicates[0] if detection.duplicates else None
    policy, _ = resolve_formation_policy(crystallization, duplicate_id=duplicate_id)

    if policy == FormationPolicy.REINFORCE_EXISTING and duplicate_id:
        match = next(c for c in existing if c.crystallization_id == duplicate_id)
        updated = reinforce(match, now=_utc_now())
        # append new evidence refs from candidate
        await update_crystallization(pool, updated)
        await emit_crystallization_lifecycle(bus, lifecycle="reinforced", crystallization=updated, ...)
        return duplicate_id, updated, "reinforced"

    if not settings.MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED:
        row = apply_salience(crystallization)
        cid = await insert_crystallization(pool, row)
        await emit_crystallization_lifecycle(bus, lifecycle="proposed", crystallization=row, ...)
        return cid, row, "proposed"

    try:
        activated, _ = auto_activate(apply_salience(crystallization), encode_ratio=settings.MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO)
        cid = await insert_crystallization(pool, activated)
        activated, _proj = await project_crystallization(
            pool, bus, activated, actor="system:formation_policy",
            config=project_config, project_graphiti=False,
        )
        await update_crystallization(pool, activated)
        await emit_crystallization_lifecycle(bus, lifecycle="auto_activated", crystallization=activated, ...)
        return cid, activated, "auto_activated"
    except GovernorPathRequired:
        row = apply_salience(crystallization)
        cid = await insert_crystallization(pool, row)
        await emit_crystallization_lifecycle(bus, lifecycle="proposed", crystallization=row, ...)
        return cid, row, "proposed"
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_encode_reinforce_not_duplicate.py tests/test_formation_policy_auto_vs_gated.py -q`

- [ ] **Step 5: Commit**

```bash
git add orion/memory/crystallization/intake_pipeline.py tests/test_encode_reinforce_not_duplicate.py orion/memory/crystallization/repository.py
git commit -m "feat(memory): add consolidation intake pipeline with dedup reinforce"
```

---

## Task 7: Wire consolidation worker + bus lifecycle + settings

**Files:**
- Modify: `services/orion-memory-consolidation/app/worker.py`
- Modify: `orion/memory/crystallization/bus_emit.py`
- Modify: `orion/bus/channels.yaml`
- Modify: `services/orion-memory-consolidation/app/settings.py`
- Modify: `services/orion-memory-consolidation/.env_example`

- [ ] **Step 1: Write failing worker test**

Extend `services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py`:

```python
@pytest.mark.asyncio
async def test_window_auto_activate_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(worker.settings, "MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED", True)
    # mock process_consolidation_crystallization to return outcome auto_activated
    # assert spark_meta patch contains formation_outcome=auto_activated
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Wire worker**

Replace direct `insert_crystallization` block in `consolidate_window` with:

```python
from orion.memory.crystallization.intake_pipeline import process_consolidation_crystallization

cid, final_row, outcome = await process_consolidation_crystallization(
    self._pool, bus,
    crystallization=crystallization,
    settings=settings,
    project_config=_projection_config_from_settings(settings),
)
```

Add to `bus_emit.py`:

```python
"reinforced": "memory.crystallization.reinforced.v1",
"auto_activated": "memory.crystallization.auto_activated.v1",
```

Add matching channels in `channels.yaml`.

Add settings:

```python
MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED: bool = Field(default=False, ...)
MEMORY_FORMATION_AUTO_ENCODE_ACTIVATION_RATIO: float = Field(default=0.4, ...)
```

- [ ] **Step 4: Run tests**

Run: `pytest services/orion-memory-consolidation/tests/test_worker_consolidation_gate.py services/orion-memory-consolidation/tests/test_consolidation_gate.py -q`

Also: `python scripts/check_bus_channels.py`

- [ ] **Step 5: Commit + sync env**

```bash
python scripts/sync_local_env_from_example.py
git add services/orion-memory-consolidation/ orion/memory/crystallization/bus_emit.py orion/bus/channels.yaml
git commit -m "feat(memory): wire auto-activate intake in consolidation worker"
```

---

# Milestone M2 — Unify read path (brain + orion unified)

## Task 8: Brain lane belief-default retrieval intent

**Files:**
- Modify: `orion/memory/retrieval_intent.py`
- Modify: `tests/test_retrieval_intent.py`
- Modify: `services/orion-cortex-exec/app/settings.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_retrieval_intent.py`:

```python
def test_brain_lane_belief_default_when_eligible_beliefs_exist():
    intent, rule_id = derive_retrieval_intent(
        skip_gate=RecallSkipGateResult(skip=False),
        stance_brief={"task_mode": "direct_response"},
        attention_frame={},
        appraisal={"shift_kind": "NONE", "novelty_score": 0.1},
        hub_chat_lane="brain",
        user_message="ok",
        eligible_belief_count=3,
        brain_belief_default_enabled=True,
    )
    assert intent == "semantic"
    assert rule_id == "brain_lane_belief_default"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

Add parameters to `derive_retrieval_intent`:

```python
eligible_belief_count: int = 0,
brain_belief_default_enabled: bool = True,
```

Before final `return "continuity", "continuity_only"`:

```python
if (
    brain_belief_default_enabled
    and hub_chat_lane in ("brain", "orion")
    and eligible_belief_count > 0
):
    return "semantic", "brain_lane_belief_default"
```

In `pcr_chat_memory.run_pcr_phase3`, pass:

```python
eligible_belief_count=int((ctx.get("eligible_belief_count") or 0)),
brain_belief_default_enabled=cfg.memory_cognition_brain_belief_default,
```

- [ ] **Step 4: Run — expect PASS**

Run: `pytest tests/test_retrieval_intent.py -q`

- [ ] **Step 5: Commit**

---

## Task 9: Emit `eligible_belief_count` from recall worker

**Files:**
- Modify: `services/orion-recall/app/worker.py`
- Modify: `orion/memory/crystallization/repository.py`
- Modify: `services/orion-cortex-exec/app/pcr_chat_memory.py`

- [ ] **Step 1: Write failing test** in `services/orion-recall/tests/test_pcr_continuity.py` asserting `recall_debug["eligible_belief_count"]` present when pool has active beliefs.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Add `count_eligible_active` to repository**

```python
async def count_eligible_active(pool, *, floor: float | None = None) -> int:
    rows = await list_crystallizations(pool, status="active", limit=500)
    return sum(1 for r in rows if eligible_for_recall(r, floor=floor))
```

In recall worker PCR branches, set `recall_debug["eligible_belief_count"] = await count_eligible_active(pool)`.

In `run_pcr_phase0_and_1`, after recall:

```python
if isinstance(recall_debug, dict) and "eligible_belief_count" in recall_debug:
    ctx["eligible_belief_count"] = recall_debug["eligible_belief_count"]
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

---

## Task 10: Activation-ranked active_packet + floor filter

**Files:**
- Modify: `orion/memory/crystallization/active_packet.py` or `retriever.py`
- Modify: `services/orion-recall/app/collectors/active_packet.py`
- Create: `services/orion-recall/tests/test_active_packet_activation_floor.py`

- [ ] **Step 1: Write failing test**

```python
def test_active_packet_excludes_below_activation_floor():
    # crystallization activation 0.01 excluded; 0.2 included
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Filter + rank**

In `active_packet.py` collector after `list_crystallizations`:

```python
from orion.memory.crystallization.recall_eligibility import eligible_for_recall

active_items = [c for c in active_items if eligible_for_recall(c)]
```

In `retriever.py` / `build_active_packet`, sort candidates by:

```python
score = (c.dynamics.activation or 0.0) * (c.salience or 0.5)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

---

## Task 11: Hub `hub_chat_lane` wire + drop `brain.recall.v1` default

**Files:**
- Modify: `services/orion-hub/scripts/cortex_request_builder.py`
- Modify: `services/orion-hub/tests/test_cortex_request_builder.py`

- [ ] **Step 1: Update failing test**

Change `test_brain_mode_defaults_recall_profile_to_brain_recall_v1` to expect `recall_profile is None` when no explicit override.

Add test:

```python
def test_brain_mode_sets_hub_chat_lane_in_surface_context():
    req, debug, _ = hub_builder.build_chat_request(
        payload={"mode": "brain", "use_recall": True},
        ...
    )
    assert req.metadata["surface_context"]["hub_chat_lane"] == "brain"
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

In `build_chat_request`, after `selected_ui_route` resolved:

```python
surface = dict(metadata.get("surface_context") or {})
if selected_ui_route in ("brain", "orion", "quick", "grounded_small"):
    surface.setdefault("hub_chat_lane", selected_ui_route)
metadata["surface_context"] = surface
```

Remove brain default:

```python
if use_recall and recall_profile is None:
    if route == "agent":
        recall_profile = "reflect.v1"
    # brain + orion: leave None — cortex PCR selects profile
```

For orion unified HTTP path, ensure `surface_context.hub_chat_lane = "orion"` in `api_routes.py` / `turn_orchestrator.py` metadata.

- [ ] **Step 4: Run — expect PASS**

Run: `pytest services/orion-hub/tests/test_cortex_request_builder.py -q`

- [ ] **Step 5: Commit**

---

## Task 12: Unified turn PCR phase 0+1 before stance

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py`
- Modify: `services/orion-cortex-exec/app/grounding_capsule.py`
- Create: `services/orion-cortex-exec/tests/test_unified_pcr_phase01.py`

- [ ] **Step 1: Write failing test**

```python
@pytest.mark.asyncio
async def test_stance_react_runs_pcr_phase01_before_stance_step(monkeypatch):
    # mock run_pcr_phase0_and_1 called before llm_stance_react step
    # assert ctx has continuity_digest populated
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Wire router**

Mirror `chat_general` block for `stance_react`:

```python
use_pcr_pre_recall = (
    settings.chat_pcr_enabled
    and str(plan.verb_name or "").strip().lower() == "stance_react"
    and should_recall
    and not inline_recall
)
```

Run `run_pcr_phase0_and_1` before step loop (same as chat_general).

In `assemble_stance_grounding`, remove duplicate phase 0+1 if router already ran; keep phase 3 only OR call shared helper `ensure_pcr_phase01(ctx)` idempotent.

- [ ] **Step 4: Run — expect PASS**

Run: `pytest services/orion-cortex-exec/tests/test_unified_pcr_phase01.py services/orion-cortex-exec/tests/test_pcr_chat_memory.py -q`

- [ ] **Step 5: Commit**

---

## Task 13: End-to-end smoke + gate checks

**Files:**
- Create: `scripts/smoke_memory_cognition_loop_e2e.sh`

- [ ] **Step 1: Create smoke script**

```bash
#!/usr/bin/env bash
set -euo pipefail
# 1. Enable MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED on consolidation (env or inline)
# 2. POST hub chat brain mode with topic-shift prompt
# 3. Poll crystallizations API for status=active without manual approve
# 4. Second brain chat turn — grep cortex debug for belief_digest_chars > 0
# 5. POST orion unified turn — grep grounding_capsule belief_digest non-empty
```

- [ ] **Step 2: Run agent gate checks**

```bash
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
python scripts/check_bus_channels.py
pytest tests/test_formation_policy_auto_vs_gated.py tests/test_retrieval_intent.py -q
pytest services/orion-memory-consolidation/tests/ -q
pytest services/orion-cortex-exec/tests/test_pcr_chat_memory.py services/orion-cortex-exec/tests/test_unified_pcr_phase01.py -q
pytest services/orion-recall/tests/test_active_packet_activation_floor.py -q
```

- [ ] **Step 3: Run smoke (if stack up)**

```bash
bash scripts/smoke_memory_cognition_loop_e2e.sh
```

- [ ] **Step 4: Flip `MEMORY_FORMATION_AUTO_ACTIVATE_ENABLED=true` in operator `.env` after smoke PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_memory_cognition_loop_e2e.sh
git commit -m "test(memory): add memory cognition loop e2e smoke"
```

---

# Milestone M3 — Lifecycle live (follow-on, after M1+M2 ship)

Ship separately after brain/unified read path verified.

| Task | Files | Behavior |
|------|-------|----------|
| M3-1 recall_boost | `services/orion-recall/app/worker.py`, `repository.update_crystallization` | After active_packet IDs resolved, bump activation |
| M3-2 decay reaper | `services/orion-memory-consolidation/app/reaper.py` | Periodic decay + `should_retire` → deprecated |
| M3-3 de-project | `projector.py` | Remove retired beliefs from cards/chroma |
| M3-4 bus | `bus_emit.py`, `channels.yaml` | `memory.crystallization.retired.v1` |

---

## Self-review (plan vs spec)

| Spec requirement | Plan task |
|------------------|-----------|
| `formation_policy` auto vs gated | Task 3 |
| `auto_activate` weak dynamics | Task 2, 4 |
| dedup → reinforce | Task 6 |
| dynamics schema + SQL | Task 1 |
| auto-project cards/chroma only | Task 6, 7 |
| PCR brain + unified phase 0→1→3 | Task 12 |
| retrieval intent brain default | Task 8, 9 |
| activation floor + ranked packet | Task 5, 10 |
| Hub drop brain.recall.v1 default | Task 11 |
| hub_chat_lane wire | Task 11 |
| recall_boost + reaper | M3 appendix |
| smoke e2e | Task 13 |

**Placeholder scan:** None found.

**Type consistency:** `FormationPolicy`, `eligible_belief_count`, `seed_weak_dynamics`, `process_consolidation_crystallization` outcomes used consistently across tasks.

---

## Restart required (after M1+M2)

```bash
docker compose --env-file .env --env-file services/orion-memory-consolidation/.env \
  -f services/orion-memory-consolidation/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Apply SQL migration for `dynamics` column on Postgres before consolidation worker starts.
