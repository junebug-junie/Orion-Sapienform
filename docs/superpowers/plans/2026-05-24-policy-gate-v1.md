# Policy Gate v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 8 — evaluate `ProposalFrameV1` candidates against `SubstratePolicyV1` and persist deterministic `PolicyDecisionFrameV1` snapshots (governed decisions only; no execution).

**Architecture:** Schemas in `orion/schemas/policy_decision_frame.py`; pure policy logic in `orion/policy/`; polling service `orion-policy-runtime` idempotent per `source_proposal_frame_id`; optional Hub `GET /api/substrate/policy/latest`. Config from `config/policy/substrate_policy.v1.yaml`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, FastAPI/uvicorn, pytest, Postgres.

**Depends on:** Layer 7 on `main` (`ProposalFrameV1`, `substrate_proposal_frames`, port 8119). Layer 6 `SelfStateV1` in `substrate_self_state`.

**Non-goals:** Layer 9 execution, cortex-exec, bus publish, LLM, operator notify, settings mutation, autonomy mutation.

---

## Worktree and branch hygiene

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-policy-gate-v1 -b feat/policy-gate-v1 origin/main
cd .worktrees/feat-policy-gate-v1
```

**Rules:**

- All implementation commits happen only inside `.worktrees/feat-policy-gate-v1`.
- Do **not** copy changed files back to the main workspace checkout except syncing `services/orion-policy-runtime/.env` from `.env_example` on the operator machine (`.env` is gitignored).
- **Port:** `8120` (`POLICY_RUNTIME_PORT`).
- **Bus:** Register schemas in `orion/schemas/registry.py` only. **No** `orion/bus/channels.yaml` entry — policy runtime does not publish (matches Layer 7 proposal-runtime pattern).

---

## File structure

| Path | Role |
|------|------|
| `orion/schemas/policy_decision_frame.py` | `PolicyDecisionV1`, `PolicyDecisionFrameV1` |
| `orion/schemas/registry.py` | Register new schema types |
| `config/policy/substrate_policy.v1.yaml` | Substrate policy thresholds + kind rules |
| `orion/policy/__init__.py` | Package export |
| `orion/policy/policy.py` | YAML loader + `SubstratePolicyV1` |
| `orion/policy/rules.py` | Hard-block + read-only helpers |
| `orion/policy/evaluator.py` | Per-candidate `evaluate_proposal_candidate` |
| `orion/policy/builder.py` | `build_policy_decision_frame` |
| `services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql` | DDL |
| `services/orion-policy-runtime/` | Polling runtime (mirror `orion-proposal-runtime`) |
| `services/orion-hub/scripts/substrate_policy_routes.py` | Optional debug API |
| `services/orion-hub/scripts/api_routes.py` | Include policy router |
| `services/orion-hub/tests/test_substrate_policy_debug_api.py` | Hub route tests |
| `tests/test_policy_*.py` | Unit tests |
| `scripts/smoke_policy_decision_frame_v1.sh` | Live SQL smoke |
| `docs/superpowers/pr-reports/2026-05-24-policy-gate-v1-pr.md` | PR report (post-implementation) |

---

### Task 1: Policy decision schemas + registry

**Files:**

- Create: `orion/schemas/policy_decision_frame.py`
- Modify: `orion/schemas/registry.py` (import + `SCHEMA_REGISTRY` entries)

- [ ] **Step 1: Write the failing test**

Create `tests/test_policy_decision_frame_schemas.py`:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_policy_decision_validates() -> None:
    d = PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:state:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
        reasons=["read_only_low_risk"],
    )
    assert d.decision == "approved_read_only"


def test_policy_decision_frame_validates() -> None:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:state:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
    )
    frame = PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:state:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="proposal.frame:state:proposal_policy.v1",
        source_self_state_id="self.state:state",
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
    )
    assert frame.schema_version == "policy.decision.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        PolicyDecisionV1(
            decision_id="d1",
            proposal_id="p1",
            decision="rejected",
            policy_gate="none",
            risk_score=0.1,
            reversibility_score=1.0,
            confidence_score=0.9,
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyDecisionV1(
            decision_id="d1",
            proposal_id="p1",
            decision="rejected",
            policy_gate="none",
            risk_score=1.5,
            reversibility_score=1.0,
            confidence_score=0.9,
        )


def test_roundtrip_json() -> None:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:p1:substrate_policy.v1",
        proposal_id="p1",
        decision="requires_operator_review",
        policy_gate="operator_review",
        risk_score=0.3,
        reversibility_score=0.4,
        confidence_score=0.6,
        reasons=["low_reversibility"],
    )
    frame = PolicyDecisionFrameV1(
        frame_id="policy.frame:pf1:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id="pf1",
        source_self_state_id="ss1",
        decisions=[decision],
        review_required_decisions=[decision],
        overall_risk=0.3,
        operator_review_required=True,
    )
    payload = frame.model_dump(mode="json")
    restored = PolicyDecisionFrameV1.model_validate(payload)
    assert restored.frame_id == frame.frame_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_policy_decision_frame_schemas.py -v`

Expected: FAIL — `ModuleNotFoundError: orion.schemas.policy_decision_frame`

- [ ] **Step 3: Write minimal implementation**

Create `orion/schemas/policy_decision_frame.py` (full models per PR spec — `PolicyDecisionV1` and `PolicyDecisionFrameV1` with all literals and `extra="forbid"`).

Register in `orion/schemas/registry.py`:

```python
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
```

Add to `SCHEMA_REGISTRY`:

```python
    "PolicyDecisionV1": PolicyDecisionV1,
    "PolicyDecisionFrameV1": PolicyDecisionFrameV1,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_policy_decision_frame_schemas.py -q`

Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/policy_decision_frame.py orion/schemas/registry.py tests/test_policy_decision_frame_schemas.py
git commit -m "feat(policy): add PolicyDecisionFrameV1 schemas and registry"
```

---

### Task 2: Substrate policy config + loader

**Files:**

- Create: `config/policy/substrate_policy.v1.yaml` (exact YAML from PR spec)
- Create: `orion/policy/__init__.py`
- Create: `orion/policy/policy.py`
- Test: `tests/test_policy_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_policy_loader.py`:

```python
from pathlib import Path

from orion.policy.policy import load_substrate_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "policy" / "substrate_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert policy.schema_version == "substrate_policy.v1"


def test_policy_id() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert policy.policy_id == "substrate_policy.v1"


def test_execution_without_operator_disabled() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert policy.autonomy.allow_execution_without_operator is False


def test_prepare_action_requires_review() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    rule = policy.proposal_kind_rules["prepare_action"]
    assert rule.default_decision == "requires_operator_review"
    assert rule.max_autonomy_tier == "operator_review"


def test_inspect_read_only() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    rule = policy.proposal_kind_rules["inspect"]
    assert rule.default_decision == "approved_read_only"
    assert rule.allowed_scope == "inspect_only"


def test_hard_blocks_include_cortex_exec() -> None:
    policy = load_substrate_policy(POLICY_PATH)
    assert "cortex_exec_direct_call" in policy.hard_blocks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_policy_loader.py -v`

Expected: FAIL — module/config missing

- [ ] **Step 3: Write minimal implementation**

`config/policy/substrate_policy.v1.yaml` — copy verbatim from PR spec (`schema_version`, `policy_id`, `autonomy`, `thresholds`, `proposal_kind_rules`, `hard_blocks`, `read_only_effects`).

`orion/policy/policy.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AutonomyConfigV1(BaseModel):
    default_tier: str = "observe_only"
    max_tier_without_operator: str = "read_only"
    allow_execution_without_operator: bool = False


class PolicyThresholdsV1(BaseModel):
    approve_read_only_max_risk: float = 0.15
    defer_above_risk: float = 0.60
    reject_above_risk: float = 0.85
    require_review_above_risk: float = 0.20
    require_review_below_reversibility: float = 0.50
    require_review_below_confidence: float = 0.50


class ProposalKindRuleV1(BaseModel):
    allowed_scope: str
    default_decision: str
    max_autonomy_tier: str


class SubstratePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["substrate_policy.v1"] = "substrate_policy.v1"
    policy_id: str = "substrate_policy.v1"

    autonomy: AutonomyConfigV1 = Field(default_factory=AutonomyConfigV1)
    thresholds: PolicyThresholdsV1 = Field(default_factory=PolicyThresholdsV1)
    proposal_kind_rules: dict[str, ProposalKindRuleV1] = Field(default_factory=dict)
    hard_blocks: list[str] = Field(default_factory=list)
    read_only_effects: list[str] = Field(default_factory=list)


def load_substrate_policy(path: str | Path) -> SubstratePolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SubstratePolicyV1.model_validate(data)
```

`orion/policy/__init__.py`:

```python
from orion.policy.builder import build_policy_decision_frame
from orion.policy.evaluator import evaluate_proposal_candidate
from orion.policy.policy import SubstratePolicyV1, load_substrate_policy

__all__ = [
    "SubstratePolicyV1",
    "build_policy_decision_frame",
    "evaluate_proposal_candidate",
    "load_substrate_policy",
]
```

(Defer exporting builder/evaluator until Task 3–4 exist; or use lazy imports in `__init__.py` after those modules land.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_policy_loader.py -q`

Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add config/policy/substrate_policy.v1.yaml orion/policy/ tests/test_policy_loader.py
git commit -m "feat(policy): add substrate policy config and loader"
```

---

### Task 3: Policy rules + evaluator

**Files:**

- Create: `orion/policy/rules.py`
- Create: `orion/policy/evaluator.py`
- Test: `tests/test_policy_evaluator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_policy_evaluator.py` with fixtures building minimal `ProposalCandidateV1`, `ProposalFrameV1`, `SelfStateV1`, and loading policy from `config/policy/substrate_policy.v1.yaml`.

Include these tests (implement full assertions):

1. `test_read_only_inspect_low_risk_approved` — `proposal_kind="inspect"`, `risk_score=0.05`, `proposed_effect="increase_observability"`, `required_policy_gate="read_only"` → `approved_read_only`
2. `test_summarize_low_risk_approved_read_only`
3. `test_prepare_action_requires_operator_review`
4. `test_high_risk_rejected` — `risk_score=0.90` → `rejected`
5. `test_low_reversibility_requires_review` — `reversibility_score=0.30`
6. `test_low_confidence_requires_review` — `confidence_score=0.40`
7. `test_hard_blocked_execution_intent_rejected` — `execution_intent={"mode": "cortex_exec_direct_call"}`
8. `test_no_approved_for_execution_when_disabled` — loop all kinds; assert no `approved_for_execution`

Example candidate factory:

```python
def _candidate(**overrides) -> ProposalCandidateV1:
    base = dict(
        proposal_id="proposal:test:state",
        proposal_kind="inspect",
        title="Inspect",
        description="Desc",
        target_id="capability:orchestration",
        target_kind="capability",
        priority_score=0.5,
        urgency_score=0.4,
        confidence_score=0.9,
        risk_score=0.05,
        reversibility_score=1.0,
        proposed_effect="increase_observability",
        required_policy_gate="read_only",
        execution_intent={"mode": "descriptive_only"},
    )
    base.update(overrides)
    return ProposalCandidateV1(**base)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_policy_evaluator.py -v`

Expected: FAIL — `evaluate_proposal_candidate` not defined

- [ ] **Step 3: Write minimal implementation**

`orion/policy/rules.py`:

```python
from __future__ import annotations

from orion.policy.policy import SubstratePolicyV1
from orion.schemas.proposal_frame import ProposalCandidateV1


def stable_policy_decision_id(*, proposal_id: str, policy_id: str) -> str:
    return f"policy.decision:{proposal_id}:{policy_id}"


def execution_intent_blob(candidate: ProposalCandidateV1) -> str:
    parts: list[str] = []
    for key, value in candidate.execution_intent.items():
        parts.append(str(key))
        parts.append(str(value))
    return " ".join(parts).lower()


def hard_block_hits(candidate: ProposalCandidateV1, policy: SubstratePolicyV1) -> list[str]:
    blob = execution_intent_blob(candidate)
    return [block for block in policy.hard_blocks if block.lower() in blob]


def is_read_only_candidate(candidate: ProposalCandidateV1, policy: SubstratePolicyV1) -> bool:
    if candidate.proposal_kind in ("observe", "inspect", "summarize"):
        return True
    return candidate.proposed_effect in policy.read_only_effects


def kind_rule(policy: SubstratePolicyV1, proposal_kind: str):
    return policy.proposal_kind_rules.get(proposal_kind)
```

`orion/policy/evaluator.py` — evaluation order (conservative v1):

```python
def evaluate_proposal_candidate(
    *,
    candidate: ProposalCandidateV1,
    proposal_frame: ProposalFrameV1,
    self_state: SelfStateV1,
    policy: SubstratePolicyV1,
) -> PolicyDecisionV1:
    rule = kind_rule(policy, candidate.proposal_kind)
    allowed_scope = rule.allowed_scope if rule else "none"
    autonomy_tier = rule.max_autonomy_tier if rule else policy.autonomy.default_tier
    policy_gate = candidate.required_policy_gate
    reasons: list[str] = []
    blocked_by: list[str] = []
    evidence_refs = list(candidate.evidence_refs)
    evidence_refs.extend(
        [
            f"proposal_frame:{proposal_frame.frame_id}",
            f"self_state:{self_state.self_state_id}",
        ]
    )

    decision = rule.default_decision if rule else "deferred"
    blocked = hard_block_hits(candidate, policy)
    if blocked:
        return _finish(
            candidate=candidate,
            policy=policy,
            decision="rejected",
            policy_gate="execution_policy",
            autonomy_tier="observe_only",
            allowed_scope="none",
            risk_score=candidate.risk_score,
            reversibility_score=candidate.reversibility_score,
            confidence_score=candidate.confidence_score,
            reasons=["hard_block_execution_intent"],
            evidence_refs=evidence_refs,
            blocked_by=blocked,
            execution_constraints={"layer": "9_deferred"},
        )

    if candidate.proposal_kind == "defer":
        decision = "deferred"
        reasons.append("proposal_kind_defer")
    elif candidate.risk_score >= policy.thresholds.reject_above_risk:
        decision = "rejected"
        reasons.append("risk_above_reject_threshold")
    elif candidate.risk_score >= policy.thresholds.defer_above_risk:
        decision = "deferred"
        reasons.append("risk_above_defer_threshold")
    elif candidate.proposal_kind == "prepare_action":
        decision = "requires_operator_review"
        reasons.append("prepare_action_never_auto_execute_v1")
    elif candidate.required_policy_gate == "operator_review":
        decision = "requires_operator_review"
        reasons.append("candidate_requires_operator_review")
    elif candidate.risk_score >= policy.thresholds.require_review_above_risk:
        decision = "requires_operator_review"
        reasons.append("risk_above_review_threshold")
    elif candidate.reversibility_score < policy.thresholds.require_review_below_reversibility:
        decision = "requires_operator_review"
        reasons.append("reversibility_below_threshold")
    elif candidate.confidence_score < policy.thresholds.require_review_below_confidence:
        decision = "requires_operator_review"
        reasons.append("confidence_below_threshold")
    elif (
        is_read_only_candidate(candidate, policy)
        and candidate.risk_score <= policy.thresholds.approve_read_only_max_risk
        and candidate.required_policy_gate in ("none", "read_only")
    ):
        decision = "approved_read_only"
        reasons.append("read_only_low_risk")
    elif rule and rule.default_decision:
        decision = rule.default_decision
        reasons.append("proposal_kind_default")

    if (
        policy.autonomy.allow_execution_without_operator
        and decision == "approved_for_execution"
    ):
        pass  # unreachable in v1 config
    else:
        if decision == "approved_for_execution":
            decision = "requires_operator_review"
            reasons.append("execution_without_operator_disabled")

    return _finish(...)  # map decision → policy_gate, allowed_scope, execution_constraints
```

Implement `_finish(...)` to build `PolicyDecisionV1` with `decision_id=stable_policy_decision_id(...)`, clamp scores via existing `orion.proposals.scoring.clamp01` if desired, and set `execution_constraints` keys like `max_scope`, `requires_operator`, `layer` (always `"9_deferred"` in v1).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_policy_evaluator.py -q`

Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/policy/rules.py orion/policy/evaluator.py tests/test_policy_evaluator.py
git commit -m "feat(policy): add conservative proposal evaluator"
```

---

### Task 4: Policy decision frame builder

**Files:**

- Create: `orion/policy/builder.py`
- Test: `tests/test_policy_decision_builder.py`

- [ ] **Step 1: Write the failing test**

Synthetic `ProposalFrameV1` with four candidates matching PR spec keys:

- `inspect_execution_pressure` (inspect, low risk)
- `summarize_loaded_state` (summarize, low risk)
- `request_policy_review_for_action` (request_policy_review)
- `prepare_action` (synthetic — add candidate with `proposal_kind="prepare_action"`, `required_policy_gate="operator_review"`, `risk_score=0.25`)

Use `_loaded_self_state()` pattern from `tests/test_proposal_frame_builder.py`.

Assertions:

1. Builds `PolicyDecisionFrameV1`
2. `source_proposal_frame_id` / `source_self_state_id` wired
3. Partitions: read-only in `approved_decisions`; review in `review_required_decisions`; prepare in review or rejected
4. `operator_review_required is True`
5. `execution_allowed is False`
6. Stable frame id: `policy.frame:{proposal_frame.frame_id}:{policy.policy_id}`
7. At least one `approved_read_only` and one review-required

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_policy_decision_builder.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`orion/policy/builder.py`:

```python
def stable_policy_frame_id(*, proposal_frame_id: str, policy_id: str) -> str:
    return f"policy.frame:{proposal_frame_id}:{policy_id}"


def build_policy_decision_frame(
    *,
    proposal_frame: ProposalFrameV1,
    self_state: SelfStateV1,
    policy: SubstratePolicyV1,
    now: datetime | None = None,
) -> PolicyDecisionFrameV1:
    generated_at = now or datetime.now(timezone.utc)
    decisions = [
        evaluate_proposal_candidate(
            candidate=candidate,
            proposal_frame=proposal_frame,
            self_state=self_state,
            policy=policy,
        )
        for candidate in proposal_frame.candidates
    ]
    approved = [d for d in decisions if d.decision == "approved_for_execution"]
    approved_read_only = [d for d in decisions if d.decision == "approved_read_only"]
    review = [d for d in decisions if d.decision == "requires_operator_review"]
    deferred = [d for d in decisions if d.decision == "deferred"]
    rejected = [d for d in decisions if d.decision == "rejected"]
    overall_risk = max((d.risk_score for d in decisions), default=0.0)
    return PolicyDecisionFrameV1(
        frame_id=stable_policy_frame_id(
            proposal_frame_id=proposal_frame.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_proposal_frame_id=proposal_frame.frame_id,
        source_self_state_id=proposal_frame.source_self_state_id,
        source_attention_frame_id=proposal_frame.source_attention_frame_id,
        source_field_tick_id=proposal_frame.source_field_tick_id,
        policy_id=policy.policy_id,
        decisions=decisions,
        approved_decisions=approved + approved_read_only,
        review_required_decisions=review,
        deferred_decisions=deferred,
        rejected_decisions=rejected,
        overall_risk=overall_risk,
        operator_review_required=any(d.decision == "requires_operator_review" for d in decisions),
        execution_allowed=any(d.decision == "approved_for_execution" for d in decisions),
        warnings=list(proposal_frame.warnings),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_policy_decision_builder.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/policy/builder.py tests/test_policy_decision_builder.py orion/policy/__init__.py
git commit -m "feat(policy): build PolicyDecisionFrameV1 from proposals"
```

---

### Task 5: SQL migration

**Files:**

- Create: `services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql`

- [ ] **Step 1: Add migration SQL** (verbatim from PR spec — table `substrate_policy_decision_frames` + three indexes)

- [ ] **Step 2: Commit**

```bash
git add services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql
git commit -m "feat(policy): add substrate_policy_decision_frames migration"
```

---

### Task 6: `orion-policy-runtime` service

**Files:**

- Create: `services/orion-policy-runtime/app/{__init__.py,main.py,worker.py,store.py,settings.py}`
- Create: `services/orion-policy-runtime/{Dockerfile,docker-compose.yml,requirements.txt,README.md,.env_example}`
- Create: `services/orion-policy-runtime/.env` (copy from `.env_example` — local only, not committed)
- Test: `tests/test_policy_runtime_store.py`

Mirror `services/orion-proposal-runtime/` with these deltas:

| Setting | Value |
|---------|-------|
| Port | `8120` |
| Policy path | `/app/config/policy/substrate_policy.v1.yaml` |
| Idempotency key | `source_proposal_frame_id` (not self_state_id) |
| Worker inputs | latest proposal frame + matching self_state by `source_self_state_id` |
| No bus / no cortex | worker only calls `build_policy_decision_frame` + `save` |

**`.env_example`:**

```env
PROJECT=orion-athena
SERVICE_NAME=orion-policy-runtime
SERVICE_VERSION=0.1.0
POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
SUBSTRATE_POLICY_PATH=/app/config/policy/substrate_policy.v1.yaml
POLICY_POLL_INTERVAL_SEC=2.0
ENABLE_POLICY_RUNTIME=true
LOG_LEVEL=INFO
```

**Dockerfile** copies: `orion/`, `config/policy/`, `services/orion-policy-runtime/`.

**`docker-compose.yml`:** port `${POLICY_RUNTIME_PORT:-8120}:8120`, container `${PROJECT}-policy-runtime`, external `app-net`.

**Worker `_tick` logic:**

```python
def _tick(self) -> None:
    if not self._settings.enable_policy_runtime:
        return
    proposal = self._store.load_latest_proposal_frame()
    if proposal is None:
        return
    if self._store.load_policy_frame_for_proposal(proposal.frame_id) is not None:
        return
    self_state = self._store.load_self_state(proposal.source_self_state_id)
    if self_state is None:
        logger.warning("policy_skip_missing_self_state self_state_id=%s", proposal.source_self_state_id)
        return
    frame = build_policy_decision_frame(
        proposal_frame=proposal,
        self_state=self_state,
        policy=self._policy,
    )
    self._store.save_policy_decision_frame(frame)
```

- [ ] **Step 1: Write failing store tests** (`tests/test_policy_runtime_store.py` — copy pattern from `tests/test_proposal_runtime_store.py`, table `substrate_policy_decision_frames`, methods `save_policy_decision_frame`, `load_latest` via proposal id query, idempotent upsert on `frame_id`)

- [ ] **Step 2: Run tests — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_policy_runtime_store.py -v`

- [ ] **Step 3: Implement store + settings + worker + main**

`store.py` SQL (from PR spec):

- `load_latest_proposal_frame` — `substrate_proposal_frames` ORDER BY `generated_at` DESC LIMIT 1
- `load_self_state` — `substrate_self_state` WHERE `self_state_id`
- `load_policy_frame_for_proposal` — `substrate_policy_decision_frames` WHERE `source_proposal_frame_id`
- `save_policy_decision_frame` — INSERT … ON CONFLICT (`frame_id`) DO UPDATE

- [ ] **Step 4: Run store tests — expect PASS**

- [ ] **Step 5: Sync local `.env`**

```bash
cd services/orion-policy-runtime
cp -n .env_example .env
# merge any operator-specific POSTGRES_URI overrides
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-policy-runtime/ tests/test_policy_runtime_store.py
git commit -m "feat(policy): add orion-policy-runtime polling service"
```

---

### Task 7: Hub read-only debug route (low risk)

**Files:**

- Create: `services/orion-hub/scripts/substrate_policy_routes.py` (mirror `substrate_proposal_routes.py`, table `substrate_policy_decision_frames`, schema `PolicyDecisionFrameV1`, prefix `/api/substrate/policy`)
- Modify: `services/orion-hub/scripts/api_routes.py` — import + `include_router`
- Create: `services/orion-hub/tests/test_substrate_policy_debug_api.py` (mirror proposal hub tests)

- [ ] **Step 1: Write failing hub test**

- [ ] **Step 2: Run — FAIL**

Run: `PYTHONPATH=. pytest services/orion-hub/tests/test_substrate_policy_debug_api.py -v`

- [ ] **Step 3: Implement route + register**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/substrate_policy_routes.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_substrate_policy_debug_api.py
git commit -m "feat(policy): expose substrate policy latest debug API"
```

---

### Task 8: Smoke script

**Files:**

- Create: `scripts/smoke_policy_decision_frame_v1.sh` (chmod +x)

Copy structure from `scripts/smoke_proposal_frame_v1.sh` with SQL 1 (latest proposal) and SQL 2 (latest policy frame) from PR spec.

- [ ] **Commit**

```bash
git add scripts/smoke_policy_decision_frame_v1.sh
git commit -m "chore(policy): add policy decision frame smoke script"
```

---

### Task 9: Verification gate (required before PR)

- [ ] **Unit tests**

```bash
cd .worktrees/feat-policy-gate-v1
PYTHONPATH=. pytest \
  tests/test_policy_decision_frame_schemas.py \
  tests/test_policy_loader.py \
  tests/test_policy_evaluator.py \
  tests/test_policy_decision_builder.py \
  tests/test_policy_runtime_store.py \
  -q
```

Expected: all passed

- [ ] **Regression**

```bash
PYTHONPATH=. pytest tests/test_proposal_*.py -q
PYTHONPATH=. pytest tests/test_self_state_*.py -q
```

Expected: all passed

- [ ] **Compile**

```bash
PYTHONPATH=. python -m compileall \
  orion/policy \
  orion/schemas/policy_decision_frame.py \
  services/orion-policy-runtime \
  -q
```

- [ ] **No execution bleed check** (grep proof for PR report)

```bash
rg -n "cortex.exec|cortex_exec|bus\.publish|operator.notify|settings\.mutation" \
  services/orion-policy-runtime orion/policy || true
```

Expected: no matches in runtime/worker paths (evaluator may mention `cortex_exec_direct_call` only as hard-block string)

- [ ] **Commit any fixups**

---

### Task 10: Live operator steps (document in PR report; run if DB available)

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_policy_decision_frame_v1.sql

cd services/orion-policy-runtime
docker compose up -d --build

./scripts/smoke_policy_decision_frame_v1.sh
curl -s http://localhost:8080/api/substrate/policy/latest | jq
curl -s http://localhost:8120/latest | jq
```

---

### Task 11: Code review subagent + fixes

- [ ] **Dispatch code-reviewer subagent** on full diff vs `origin/main`

Required SUB-SKILL: `superpowers:requesting-code-review` then fix all substantive issues before PR.

- [ ] **Re-run Task 9 verification** after fixes

- [ ] **Commit fixes** as separate commits (do not amend unless hook requires)

---

### Task 12: PR report, push, and open PR

**Files:**

- Create: `docs/superpowers/pr-reports/2026-05-24-policy-gate-v1-pr.md`

PR report must include:

1. Layer 8 mapping in 11-layer roadmap
2. Example `PolicyDecisionFrameV1` JSON (from `/latest` or unit test fixture)
3. Proof no execution (grep + worker has no outbound clients)
4. Tests run with command output
5. Explicit Layer 9 deferred statement

```bash
cd .worktrees/feat-policy-gate-v1
git push -u origin feat/policy-gate-v1
gh pr create --title "feat: Policy Gate v1 — ProposalFrame → Governed Decisions" --body "$(cat <<'EOF'
## Summary
- Layer 8: deterministic PolicyDecisionFrameV1 from ProposalFrameV1 + SelfStateV1
- orion-policy-runtime persists to substrate_policy_decision_frames (port 8120)
- No execution, bus publish, or operator side effects

## Test plan
- [ ] pytest policy + proposal + self-state regression
- [ ] migration applied
- [ ] smoke_policy_decision_frame_v1.sh
- [ ] GET /api/substrate/policy/latest

Layer 9 execution explicitly deferred.

EOF
)"
```

Attach full PR report path in PR body or comment.

---

## Self-review (plan author checklist)

| Spec requirement | Task |
|------------------|------|
| PolicyDecisionV1 / PolicyDecisionFrameV1 | Task 1 |
| substrate_policy.v1.yaml | Task 2 |
| load_substrate_policy | Task 2 |
| evaluate_proposal_candidate rules 1–10 | Task 3 |
| build_policy_decision_frame partitions | Task 4 |
| substrate_policy_decision_frames DDL | Task 5 |
| orion-policy-runtime poll + idempotent | Task 6 |
| Hub GET /latest | Task 7 |
| All five test files | Tasks 1–6 |
| smoke script | Task 8 |
| registry | Task 1 |
| No bus publish | Worktree rules (no channels.yaml) |
| Layer 9 non-goals | Header + Task 11 grep |
| .env sync | Task 6 Step 5 |

**Placeholder scan:** No TBD steps; all code paths named with file paths.

**Type consistency:** Uses existing `ProposalCandidateV1`, `ProposalFrameV1`, `SelfStateV1` from `orion/schemas/proposal_frame.py` and `orion/schemas/self_state.py`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-policy-gate-v1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task with two-stage review between tasks.

**2. Inline Execution** — run tasks in this session using `executing-plans`, batched with checkpoints.

**Which approach?**

After implementation, the user-requested workflow also requires: code-review subagent (Task 11), PR markdown report (Task 12), and `gh pr create` push from worktree only.
