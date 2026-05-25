# Execution Dispatch v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 9 — convert `PolicyDecisionFrameV1` + `ProposalFrameV1` + `SelfStateV1` into inspectable `ExecutionDispatchFrameV1` envelopes (default `dry_run`; no cortex-exec side effects unless explicitly enabled and tested).

**Architecture:** Schemas in `orion/schemas/execution_dispatch_frame.py`; pure dispatch logic in `orion/execution_dispatch/`; polling service `orion-execution-dispatch-runtime` idempotent per `source_policy_frame_id`; optional Hub `GET /api/substrate/execution-dispatch/latest`. Config from `config/execution_dispatch/execution_dispatch_policy.v1.yaml`. **No** `router.py` live dispatch in v1 — envelopes only.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, FastAPI/uvicorn, pytest, Postgres.

**Depends on:** Layer 8 on `feat/policy-gate-v1` (`PolicyDecisionFrameV1`, `substrate_policy_decision_frames`, port 8120). Layer 7 `ProposalFrameV1` on `main`. Layer 6 `SelfStateV1` on `main`.

**Non-goals:** Layer 10 feedback, consolidation, mutating cortex-exec, bus publish (default), operator notify, file writes, network calls, LLM interpretation, mind service changes.

---

## Worktree and branch hygiene

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin feat/policy-gate-v1
git worktree add .worktrees/feat-execution-dispatch-v1 -b feat/execution-dispatch-v1 origin/feat/policy-gate-v1
cd .worktrees/feat-execution-dispatch-v1
```

**Rules:**

- All implementation commits happen only inside `.worktrees/feat-execution-dispatch-v1`.
- Do **not** copy changed files back to the main workspace checkout except syncing `services/orion-execution-dispatch-runtime/.env` from `.env_example` on the operator machine (`.env` is gitignored).
- **Port:** `8121` (`EXECUTION_DISPATCH_RUNTIME_PORT`).
- **Bus:** Register schemas in `orion/schemas/registry.py` only. **Do not** add `orion/bus/channels.yaml` entries — runtime does not publish by default (matches Layer 8 policy-runtime pattern). `CORTEX_EXEC_CHANNEL` exists in settings for future `dispatch_read_only` only; worker must not call bus in v1.

---

## File structure

| Path | Role |
|------|------|
| `orion/schemas/execution_dispatch_frame.py` | `ExecutionDispatchCandidateV1`, `ExecutionDispatchFrameV1` |
| `orion/schemas/registry.py` | Register new schema types |
| `config/execution_dispatch/execution_dispatch_policy.v1.yaml` | Dispatch policy (modes, routes, hard blocks) |
| `orion/execution_dispatch/__init__.py` | Package export |
| `orion/execution_dispatch/policy.py` | YAML loader + `ExecutionDispatchPolicyV1` |
| `orion/execution_dispatch/envelopes.py` | `build_cortex_request_envelope` |
| `orion/execution_dispatch/builder.py` | `build_execution_dispatch_frame` |
| `services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql` | DDL |
| `services/orion-execution-dispatch-runtime/` | Polling runtime (mirror `orion-proposal-runtime`) |
| `services/orion-hub/scripts/substrate_execution_dispatch_routes.py` | Optional debug API |
| `services/orion-hub/scripts/api_routes.py` | Include execution-dispatch router |
| `services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py` | Hub route tests |
| `tests/test_execution_dispatch_*.py` | Unit tests |
| `scripts/smoke_execution_dispatch_v1.sh` | Live SQL smoke |
| `docs/superpowers/pr-reports/2026-05-25-execution-dispatch-v1-pr.md` | PR report (post-implementation) |

**Explicitly omitted in v1:** `orion/execution_dispatch/router.py` (defer live cortex-exec dispatch to a follow-up; default config prevents it anyway).

---

### Task 1: Execution dispatch schemas + registry

**Files:**

- Create: `orion/schemas/execution_dispatch_frame.py`
- Modify: `orion/schemas/registry.py` (import + `SCHEMA_REGISTRY` entries)

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_dispatch_frame_schemas.py`:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_execution_dispatch_candidate_validates() -> None:
    c = ExecutionDispatchCandidateV1(
        dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
        source_decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        source_proposal_id="proposal:inspect:state",
        dispatch_status="dry_run",
        dispatch_mode="dry_run",
        dispatch_kind="inspect",
        target_id="capability:orchestration",
        target_kind="capability",
        cortex_verb="substrate.inspect",
        cortex_mode="brain",
        risk_score=0.05,
        confidence_score=0.9,
    )
    assert c.dispatch_status == "dry_run"


def test_execution_dispatch_frame_validates() -> None:
    candidate = ExecutionDispatchCandidateV1(
        dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
        source_decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        source_proposal_id="proposal:inspect:state",
        dispatch_status="dry_run",
        dispatch_mode="dry_run",
        dispatch_kind="inspect",
        target_id="capability:orchestration",
        target_kind="capability",
        risk_score=0.05,
        confidence_score=0.9,
    )
    frame = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:policy.frame:pf1:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        candidates=[candidate],
        dispatch_mode="dry_run",
        dispatch_attempted=False,
        dispatch_count=0,
        blocked_count=0,
    )
    assert frame.schema_version == "execution.dispatch.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ExecutionDispatchCandidateV1(
            dispatch_id="d1",
            source_decision_id="pd1",
            source_proposal_id="p1",
            dispatch_status="dry_run",
            dispatch_mode="dry_run",
            dispatch_kind="inspect",
            target_id="t1",
            target_kind="capability",
            risk_score=0.1,
            confidence_score=0.9,
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        ExecutionDispatchCandidateV1(
            dispatch_id="d1",
            source_decision_id="pd1",
            source_proposal_id="p1",
            dispatch_status="dry_run",
            dispatch_mode="dry_run",
            dispatch_kind="inspect",
            target_id="t1",
            target_kind="capability",
            risk_score=1.5,
            confidence_score=0.9,
        )


def test_roundtrip_json() -> None:
    frame = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:policy.frame:pf1:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        dispatch_mode="dry_run",
    )
    restored = ExecutionDispatchFrameV1.model_validate(frame.model_dump(mode="json"))
    assert restored.frame_id == frame.frame_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_frame_schemas.py -v`

Expected: FAIL — `ModuleNotFoundError: orion.schemas.execution_dispatch_frame`

- [ ] **Step 3: Write minimal implementation**

Create `orion/schemas/execution_dispatch_frame.py` (verbatim models from PR spec):

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ExecutionDispatchCandidateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_id: str

    source_decision_id: str
    source_proposal_id: str

    dispatch_status: Literal[
        "prepared",
        "dry_run",
        "blocked",
        "dispatched",
        "skipped",
    ]

    dispatch_mode: Literal[
        "dry_run",
        "prepare_only",
        "dispatch_read_only",
    ]

    dispatch_kind: Literal[
        "inspect",
        "summarize",
        "observe",
        "noop",
    ]

    target_id: str
    target_kind: str

    cortex_verb: str | None = None
    cortex_mode: str | None = None

    request_envelope: dict[str, object] = Field(default_factory=dict)

    constraints: dict[str, str] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)

    risk_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class ExecutionDispatchFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["execution.dispatch.frame.v1"] = "execution.dispatch.frame.v1"

    frame_id: str
    generated_at: datetime

    source_policy_frame_id: str
    source_proposal_frame_id: str
    source_self_state_id: str

    execution_dispatch_policy_id: str = "execution_dispatch_policy.v1"

    dispatch_mode: Literal[
        "dry_run",
        "prepare_only",
        "dispatch_read_only",
    ] = "dry_run"

    candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)
    blocked_candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)
    dispatched_candidates: list[ExecutionDispatchCandidateV1] = Field(default_factory=list)

    dispatch_attempted: bool = False
    dispatch_count: int = 0
    blocked_count: int = 0

    warnings: list[str] = Field(default_factory=list)
```

Register in `orion/schemas/registry.py`:

```python
from orion.schemas.execution_dispatch_frame import (
    ExecutionDispatchCandidateV1,
    ExecutionDispatchFrameV1,
)
```

Add to `SCHEMA_REGISTRY` (near `ProposalFrameV1`):

```python
    "ExecutionDispatchCandidateV1": ExecutionDispatchCandidateV1,
    "ExecutionDispatchFrameV1": ExecutionDispatchFrameV1,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_frame_schemas.py -q`

Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/execution_dispatch_frame.py orion/schemas/registry.py tests/test_execution_dispatch_frame_schemas.py
git commit -m "feat(execution-dispatch): add ExecutionDispatchFrameV1 schemas and registry"
```

---

### Task 2: Execution dispatch policy config + loader

**Files:**

- Create: `config/execution_dispatch/execution_dispatch_policy.v1.yaml`
- Create: `orion/execution_dispatch/__init__.py`
- Create: `orion/execution_dispatch/policy.py`
- Test: `tests/test_execution_dispatch_policy_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_dispatch_policy_loader.py`:

```python
from pathlib import Path

from orion.execution_dispatch.policy import load_execution_dispatch_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.schema_version == "execution_dispatch_policy.v1"


def test_default_mode_dry_run() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.default_dispatch_mode == "dry_run"


def test_allow_dispatch_read_only_false() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.allow_dispatch_read_only is False


def test_allow_mutating_dispatch_false() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert policy.mode.allow_mutating_dispatch is False


def test_routes_for_inspect_summarize_observe() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    assert "inspect" in policy.proposal_kind_to_cortex
    assert "summarize" in policy.proposal_kind_to_cortex
    assert "observe" in policy.proposal_kind_to_cortex
    assert policy.proposal_kind_to_cortex["inspect"].cortex_verb == "substrate.inspect"


def test_hard_blocks_include_destructive_classes() -> None:
    policy = load_execution_dispatch_policy(POLICY_PATH)
    for token in (
        "destructive_action",
        "file_write",
        "network_call",
        "service_restart",
        "settings_mutation",
        "approved_for_execution",
        "prepare_action",
    ):
        assert token in policy.hard_blocks
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_policy_loader.py -v`

Expected: FAIL — module/config missing

- [ ] **Step 3: Write minimal implementation**

`config/execution_dispatch/execution_dispatch_policy.v1.yaml` — copy verbatim from PR spec.

`orion/execution_dispatch/policy.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DispatchModeConfigV1(BaseModel):
    default_dispatch_mode: str = "dry_run"
    allow_dispatch_read_only: bool = False
    allow_mutating_dispatch: bool = False


class CortexRouteTemplateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dispatch_kind: str
    cortex_verb: str
    cortex_mode: str = "brain"
    allowed_scope: str


class DispatchLimitsV1(BaseModel):
    max_dispatch_candidates: int = 5
    max_dispatches_per_tick: int = 1


class ExecutionDispatchPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["execution_dispatch_policy.v1"] = "execution_dispatch_policy.v1"
    policy_id: str = "execution_dispatch_policy.v1"

    mode: DispatchModeConfigV1 = Field(default_factory=DispatchModeConfigV1)
    allowed_policy_decisions: list[str] = Field(default_factory=list)
    blocked_policy_decisions: list[str] = Field(default_factory=list)
    proposal_kind_to_cortex: dict[str, CortexRouteTemplateV1] = Field(default_factory=dict)
    hard_blocks: list[str] = Field(default_factory=list)
    limits: DispatchLimitsV1 = Field(default_factory=DispatchLimitsV1)


def load_execution_dispatch_policy(path: str | Path) -> ExecutionDispatchPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return ExecutionDispatchPolicyV1.model_validate(data)
```

`orion/execution_dispatch/__init__.py`:

```python
from orion.execution_dispatch.builder import build_execution_dispatch_frame, stable_execution_dispatch_frame_id
from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import ExecutionDispatchPolicyV1, load_execution_dispatch_policy

__all__ = [
    "ExecutionDispatchPolicyV1",
    "build_cortex_request_envelope",
    "build_execution_dispatch_frame",
    "load_execution_dispatch_policy",
    "stable_execution_dispatch_frame_id",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_policy_loader.py -q`

Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add config/execution_dispatch/execution_dispatch_policy.v1.yaml orion/execution_dispatch/ tests/test_execution_dispatch_policy_loader.py
git commit -m "feat(execution-dispatch): add execution dispatch policy loader"
```

---

### Task 3: Cortex request envelope builder

**Files:**

- Create: `orion/execution_dispatch/envelopes.py`
- Test: `tests/test_execution_dispatch_envelopes.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_dispatch_envelopes.py`:

```python
from datetime import datetime, timezone

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import CortexRouteTemplateV1
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1
from orion.schemas.self_state import SelfStateV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)

ROUTE = CortexRouteTemplateV1(
    dispatch_kind="inspect",
    cortex_verb="substrate.inspect",
    cortex_mode="brain",
    allowed_scope="inspect_only",
)


def _candidate() -> ProposalCandidateV1:
    return ProposalCandidateV1(
        proposal_id="proposal:inspect:state",
        proposal_kind="inspect",
        title="inspect",
        description="test",
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


def _decision() -> PolicyDecisionV1:
    return PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
    )


def _self_state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:pf1",
        generated_at=NOW,
        source_field_tick_id="tick_live",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_live",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.6,
        overall_confidence=0.9,
    )


def test_envelope_includes_verb_mode_source_dry_run() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    assert env["verb"] == "substrate.inspect"
    assert env["mode"] == "brain"
    assert env["source"] == "orion-execution-dispatch-runtime"
    assert env["dry_run"] is True


def test_envelope_includes_refs() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    ctx = env["context"]
    assert ctx["proposal_id"] == "proposal:inspect:state"
    assert ctx["decision_id"] == "policy.decision:proposal:inspect:substrate_policy.v1"
    assert ctx["self_state_id"] == "self.state:pf1"
    assert ctx["allowed_scope"] == "inspect_only"


def test_envelope_read_only_constraints() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    c = env["constraints"]
    assert c["read_only"] is True
    assert c["no_file_writes"] is True
    assert c["no_service_restarts"] is True


def test_envelope_no_field_state_or_prompts() -> None:
    env = build_cortex_request_envelope(
        candidate=_candidate(),
        decision=_decision(),
        route=ROUTE,
        self_state=_self_state(),
        dry_run=True,
    )
    blob = str(env).lower()
    assert "prompt" not in blob
    assert "llm" not in blob
    assert "field_state" not in blob
    assert "dimensions" not in blob
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_envelopes.py -v`

Expected: FAIL — `build_cortex_request_envelope` not defined

- [ ] **Step 3: Write minimal implementation**

`orion/execution_dispatch/envelopes.py`:

```python
from __future__ import annotations

from orion.execution_dispatch.policy import CortexRouteTemplateV1
from orion.schemas.policy_decision_frame import PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1
from orion.schemas.self_state import SelfStateV1


def build_cortex_request_envelope(
    *,
    candidate: ProposalCandidateV1,
    decision: PolicyDecisionV1,
    route: CortexRouteTemplateV1,
    self_state: SelfStateV1,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "verb": route.cortex_verb,
        "mode": route.cortex_mode,
        "source": "orion-execution-dispatch-runtime",
        "dry_run": dry_run,
        "context": {
            "proposal_id": candidate.proposal_id,
            "decision_id": decision.decision_id,
            "self_state_id": self_state.self_state_id,
            "target_id": candidate.target_id,
            "target_kind": candidate.target_kind,
            "allowed_scope": route.allowed_scope,
        },
        "constraints": {
            "read_only": True,
            "no_external_side_effects": True,
            "no_file_writes": True,
            "no_service_restarts": True,
            "no_operator_notifications": True,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_envelopes.py -q`

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/execution_dispatch/envelopes.py tests/test_execution_dispatch_envelopes.py
git commit -m "feat(execution-dispatch): add bounded cortex request envelope builder"
```

---

### Task 4: Execution dispatch frame builder

**Files:**

- Create: `orion/execution_dispatch/builder.py`
- Test: `tests/test_execution_dispatch_builder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_execution_dispatch_builder.py` (synthetic policy + proposal frames):

```python
from datetime import datetime, timezone
from pathlib import Path

from orion.execution_dispatch.builder import (
    build_execution_dispatch_frame,
    stable_execution_dispatch_frame_id,
)
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_execution_dispatch_policy(
    REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
)
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _loaded_self_state() -> SelfStateV1:
    def dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
        return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)

    return SelfStateV1(
        self_state_id="self.state:tick_live:frame_live:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_live",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_live:field_attention_policy.v1",
        source_attention_generated_at=NOW,
        overall_condition="loaded",
        overall_intensity=0.655,
        overall_confidence=0.9,
        dimensions={"execution_pressure": dim("execution_pressure", 1.0)},
        summary_labels=["execution_loaded"],
    )


def _candidate(proposal_id: str, proposal_kind: str, **kwargs) -> ProposalCandidateV1:
    base = dict(
        proposal_id=proposal_id,
        proposal_kind=proposal_kind,
        title=proposal_id,
        description="test",
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
    base.update(kwargs)
    return ProposalCandidateV1(**base)


def _proposal_frame() -> ProposalFrameV1:
    state = _loaded_self_state()
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id=state.self_state_id,
        source_self_state_generated_at=state.generated_at,
        source_attention_frame_id=state.source_attention_frame_id,
        source_field_tick_id=state.source_field_tick_id,
        overall_action_pressure=0.6,
        overall_risk=0.3,
        candidates=[
            _candidate("proposal:inspect:state", "inspect"),
            _candidate("proposal:summarize:state", "summarize"),
            _candidate(
                "proposal:review:state",
                "request_policy_review",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
            _candidate(
                "proposal:blocked:state",
                "prepare_action",
                required_policy_gate="operator_review",
                proposed_effect="prepare_for_policy_gate",
                risk_score=0.25,
            ),
        ],
    )


def _policy_frame(proposal: ProposalFrameV1) -> PolicyDecisionFrameV1:
    def decision(proposal_id: str, proposal_kind: str, decision_value: str) -> PolicyDecisionV1:
        return PolicyDecisionV1(
            decision_id=f"policy.decision:{proposal_id}:substrate_policy.v1",
            proposal_id=proposal_id,
            decision=decision_value,
            policy_gate="read_only" if decision_value == "approved_read_only" else "operator_review",
            risk_score=0.05 if decision_value == "approved_read_only" else 0.25,
            reversibility_score=1.0,
            confidence_score=0.9,
            allowed_scope="inspect_only" if decision_value == "approved_read_only" else "operator_review_required",
        )

    decisions = [
        decision("proposal:inspect:state", "inspect", "approved_read_only"),
        decision("proposal:summarize:state", "summarize", "approved_read_only"),
        decision("proposal:review:state", "request_policy_review", "requires_operator_review"),
        decision("proposal:blocked:state", "prepare_action", "rejected"),
    ]
    approved = [d for d in decisions if d.decision == "approved_read_only"]
    review = [d for d in decisions if d.decision == "requires_operator_review"]
    rejected = [d for d in decisions if d.decision == "rejected"]
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id=proposal.frame_id,
        source_self_state_id=proposal.source_self_state_id,
        decisions=decisions,
        approved_decisions=approved,
        review_required_decisions=review,
        rejected_decisions=rejected,
        overall_risk=0.3,
        operator_review_required=True,
        execution_allowed=False,
    )


def test_builds_execution_dispatch_frame() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    state = _loaded_self_state()
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=state,
        policy=POLICY,
        now=NOW,
    )
    assert frame.schema_version == "execution.dispatch.frame.v1"
    assert frame.source_policy_frame_id == policy_frame.frame_id
    assert frame.source_proposal_frame_id == proposal.frame_id
    assert frame.source_self_state_id == state.self_state_id


def test_approved_read_only_become_dry_run_candidates() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    kinds = {c.dispatch_kind for c in frame.candidates}
    assert "inspect" in kinds
    assert "summarize" in kinds
    assert all(c.dispatch_status == "dry_run" for c in frame.candidates)
    assert all(c.dispatch_mode == "dry_run" for c in frame.candidates)


def test_review_and_rejected_blocked() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    blocked_ids = {c.source_proposal_id for c in frame.blocked_candidates}
    assert "proposal:review:state" in blocked_ids
    assert "proposal:blocked:state" in blocked_ids
    assert frame.blocked_count >= 2


def test_default_dispatch_mode_and_no_attempt() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    assert frame.dispatch_mode == "dry_run"
    assert frame.dispatch_attempted is False
    assert frame.dispatch_count == 0


def test_no_mutating_scope_in_envelopes() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    for c in frame.candidates:
        constraints = c.request_envelope.get("constraints", {})
        assert constraints.get("read_only") is True


def test_stable_frame_id() -> None:
    proposal = _proposal_frame()
    policy_frame = _policy_frame(proposal)
    expected = stable_execution_dispatch_frame_id(
        policy_frame_id=policy_frame.frame_id,
        policy_id=POLICY.policy_id,
    )
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=_loaded_self_state(),
        policy=POLICY,
        now=NOW,
    )
    assert frame.frame_id == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_builder.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`orion/execution_dispatch/builder.py` — core logic:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.execution_dispatch.envelopes import build_cortex_request_envelope
from orion.execution_dispatch.policy import CortexRouteTemplateV1, ExecutionDispatchPolicyV1
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


def stable_execution_dispatch_frame_id(*, policy_frame_id: str, policy_id: str) -> str:
    return f"execution.dispatch.frame:{policy_frame_id}:{policy_id}"


def stable_dispatch_id(*, proposal_id: str, policy_id: str) -> str:
    return f"dispatch:{proposal_id}:{policy_id}"


def _proposal_by_id(proposal_frame: ProposalFrameV1) -> dict[str, ProposalCandidateV1]:
    return {c.proposal_id: c for c in proposal_frame.candidates}


def _is_hard_blocked(candidate: ProposalCandidateV1, policy: ExecutionDispatchPolicyV1) -> list[str]:
    hits: list[str] = []
    if candidate.proposal_kind in policy.hard_blocks:
        hits.append(f"proposal_kind:{candidate.proposal_kind}")
    blob = " ".join(
        [
            candidate.proposal_kind,
            candidate.proposed_effect,
            candidate.required_policy_gate,
            *candidate.execution_intent.values(),
        ]
    ).lower()
    for block in policy.hard_blocks:
        if block.lower() in blob:
            hits.append(block)
    if candidate.required_policy_gate in ("execution_policy", "autonomy_policy"):
        hits.append(f"policy_gate:{candidate.required_policy_gate}")
    return hits


def _resolve_dispatch_mode(
    *,
    policy: ExecutionDispatchPolicyV1,
    override_dispatch_mode: str | None,
) -> str:
    if override_dispatch_mode:
        return override_dispatch_mode
    return policy.mode.default_dispatch_mode


def _candidate_status_for_mode(dispatch_mode: str) -> tuple[str, str]:
    if dispatch_mode == "prepare_only":
        return "prepared", "prepare_only"
    if dispatch_mode == "dispatch_read_only":
        return "prepared", "dispatch_read_only"
    return "dry_run", "dry_run"


def build_execution_dispatch_frame(
    *,
    policy_frame: PolicyDecisionFrameV1,
    proposal_frame: ProposalFrameV1,
    self_state: SelfStateV1,
    policy: ExecutionDispatchPolicyV1,
    now: datetime | None = None,
    override_dispatch_mode: str | None = None,
) -> ExecutionDispatchFrameV1:
    generated_at = now or datetime.now(timezone.utc)
    dispatch_mode = _resolve_dispatch_mode(policy=policy, override_dispatch_mode=override_dispatch_mode)
    proposals = _proposal_by_id(proposal_frame)

    candidates: list[ExecutionDispatchCandidateV1] = []
    blocked: list[ExecutionDispatchCandidateV1] = []
    dispatched: list[ExecutionDispatchCandidateV1] = []
    warnings: list[str] = list(policy_frame.warnings)

    if policy.mode.allow_mutating_dispatch:
        warnings.append("mutating_dispatch_disabled_in_v1")

    dispatch_status_default, dispatch_mode_candidate = _candidate_status_for_mode(dispatch_mode)
    max_candidates = policy.limits.max_dispatch_candidates

    def make_blocked(
        decision: PolicyDecisionV1,
        candidate: ProposalCandidateV1 | None,
        *,
        reasons: list[str],
        blocked_by: list[str],
    ) -> ExecutionDispatchCandidateV1:
        return ExecutionDispatchCandidateV1(
            dispatch_id=stable_dispatch_id(proposal_id=decision.proposal_id, policy_id=policy.policy_id),
            source_decision_id=decision.decision_id,
            source_proposal_id=decision.proposal_id,
            dispatch_status="blocked",
            dispatch_mode=dispatch_mode_candidate,
            dispatch_kind="noop",
            target_id=candidate.target_id if candidate else "unknown",
            target_kind=candidate.target_kind if candidate else "system",
            reasons=reasons,
            blocked_by=blocked_by,
            risk_score=decision.risk_score,
            confidence_score=decision.confidence_score,
        )

    for decision in policy_frame.decisions:
        candidate = proposals.get(decision.proposal_id)
        if candidate is None:
            blocked.append(
                make_blocked(
                    decision,
                    None,
                    reasons=["missing_proposal_candidate"],
                    blocked_by=["proposal_not_found"],
                )
            )
            continue

        hard_hits = _is_hard_blocked(candidate, policy)
        if hard_hits:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["hard_block"],
                    blocked_by=hard_hits,
                )
            )
            continue

        if decision.decision in policy.blocked_policy_decisions:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=[f"policy_decision:{decision.decision}"],
                    blocked_by=[decision.decision],
                )
            )
            continue

        if decision.decision not in policy.allowed_policy_decisions:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["decision_not_allowed_for_dispatch_v1"],
                    blocked_by=[decision.decision],
                )
            )
            continue

        route: CortexRouteTemplateV1 | None = policy.proposal_kind_to_cortex.get(candidate.proposal_kind)
        if route is None:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["no_cortex_route_for_proposal_kind"],
                    blocked_by=[candidate.proposal_kind],
                )
            )
            continue

        if route.allowed_scope not in ("inspect_only", "summarize_only"):
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["route_scope_not_read_only"],
                    blocked_by=[route.allowed_scope],
                )
            )
            continue

        if len(candidates) >= max_candidates:
            blocked.append(
                make_blocked(
                    decision,
                    candidate,
                    reasons=["max_dispatch_candidates_exceeded"],
                    blocked_by=["limit"],
                )
            )
            continue

        dry_run = dispatch_mode != "dispatch_read_only"
        envelope = build_cortex_request_envelope(
            candidate=candidate,
            decision=decision,
            route=route,
            self_state=self_state,
            dry_run=dry_run,
        )
        dispatch_status = dispatch_status_default
        if dispatch_mode == "dispatch_read_only" and policy.mode.allow_dispatch_read_only:
            dispatch_status = "dispatched"
        elif dispatch_mode == "dispatch_read_only":
            dispatch_status = "dry_run"
            warnings.append("dispatch_read_only_disabled_by_policy")

        item = ExecutionDispatchCandidateV1(
            dispatch_id=stable_dispatch_id(proposal_id=decision.proposal_id, policy_id=policy.policy_id),
            source_decision_id=decision.decision_id,
            source_proposal_id=decision.proposal_id,
            dispatch_status=dispatch_status,
            dispatch_mode=dispatch_mode_candidate,
            dispatch_kind=route.dispatch_kind,
            target_id=candidate.target_id,
            target_kind=candidate.target_kind,
            cortex_verb=route.cortex_verb,
            cortex_mode=route.cortex_mode,
            request_envelope=envelope,
            constraints=dict(envelope.get("constraints", {})),
            reasons=["approved_read_only_dispatch_v1"],
            evidence_refs=list(decision.evidence_refs),
            risk_score=decision.risk_score,
            confidence_score=decision.confidence_score,
        )
        if dispatch_status == "dispatched":
            dispatched.append(item)
        else:
            candidates.append(item)

    dispatch_attempted = dispatch_mode == "dispatch_read_only" and policy.mode.allow_dispatch_read_only
    return ExecutionDispatchFrameV1(
        frame_id=stable_execution_dispatch_frame_id(
            policy_frame_id=policy_frame.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_policy_frame_id=policy_frame.frame_id,
        source_proposal_frame_id=proposal_frame.frame_id,
        source_self_state_id=self_state.self_state_id,
        execution_dispatch_policy_id=policy.policy_id,
        dispatch_mode=dispatch_mode,
        candidates=candidates,
        blocked_candidates=blocked,
        dispatched_candidates=dispatched,
        dispatch_attempted=dispatch_attempted,
        dispatch_count=len(dispatched),
        blocked_count=len(blocked),
        warnings=warnings,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_builder.py -q`

Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/execution_dispatch/builder.py tests/test_execution_dispatch_builder.py
git commit -m "feat(execution-dispatch): build ExecutionDispatchFrameV1 from policy decisions"
```

---

### Task 5: SQL migration

**Files:**

- Create: `services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql`

- [ ] **Step 1: Add migration SQL** (verbatim from PR spec)

```sql
create table if not exists substrate_execution_dispatch_frames (
    frame_id text primary key,
    source_policy_frame_id text not null,
    source_proposal_frame_id text not null,
    source_self_state_id text not null,
    generated_at timestamptz not null,
    policy_id text not null,
    dispatch_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_execution_dispatch_frames_generated_at
    on substrate_execution_dispatch_frames (generated_at desc);

create index if not exists idx_substrate_execution_dispatch_frames_source_policy
    on substrate_execution_dispatch_frames (source_policy_frame_id);

create index if not exists idx_substrate_execution_dispatch_frames_source_self_state
    on substrate_execution_dispatch_frames (source_self_state_id);
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql
git commit -m "feat(execution-dispatch): add substrate_execution_dispatch_frames migration"
```

---

### Task 6: `orion-execution-dispatch-runtime` service

**Files:**

- Create: `services/orion-execution-dispatch-runtime/app/{__init__.py,main.py,worker.py,store.py,settings.py}`
- Create: `services/orion-execution-dispatch-runtime/{Dockerfile,docker-compose.yml,requirements.txt,README.md,.env_example}`
- Create: `services/orion-execution-dispatch-runtime/.env` (copy from `.env_example` — local only)
- Test: `tests/test_execution_dispatch_runtime_store.py`

Mirror `services/orion-proposal-runtime/` with these deltas:

| Setting | Value |
|---------|-------|
| Port | `8121` |
| Policy path | `/app/config/execution_dispatch/execution_dispatch_policy.v1.yaml` |
| Idempotency key | `source_policy_frame_id` |
| Worker inputs | latest policy frame + matching proposal + self_state |
| No bus / no cortex | worker only builds + saves |

**`.env_example`:**

```env
# orion-execution-dispatch-runtime — Layer 9 execution dispatch (docker compose)
# Requires: substrate_policy_decision_frames + substrate_proposal_frames + substrate_self_state
# Apply migration: services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql
PROJECT=orion-athena
SERVICE_NAME=orion-execution-dispatch-runtime
SERVICE_VERSION=0.1.0
POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
EXECUTION_DISPATCH_POLICY_PATH=/app/config/execution_dispatch/execution_dispatch_policy.v1.yaml
EXECUTION_DISPATCH_MODE=dry_run
EXECUTION_DISPATCH_POLL_INTERVAL_SEC=2.0
ENABLE_EXECUTION_DISPATCH_RUNTIME=true
CORTEX_EXEC_CHANNEL=orion:cortex:request
LOG_LEVEL=INFO
EXECUTION_DISPATCH_RUNTIME_PORT=8121
```

**`app/settings.py`:**

```python
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-execution-dispatch-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    execution_dispatch_policy_path: str = Field(
        "config/execution_dispatch/execution_dispatch_policy.v1.yaml",
        alias="EXECUTION_DISPATCH_POLICY_PATH",
    )
    execution_dispatch_mode: str = Field("dry_run", alias="EXECUTION_DISPATCH_MODE")
    execution_dispatch_poll_interval_sec: float = Field(
        2.0,
        alias="EXECUTION_DISPATCH_POLL_INTERVAL_SEC",
    )
    enable_execution_dispatch_runtime: bool = Field(
        True,
        alias="ENABLE_EXECUTION_DISPATCH_RUNTIME",
    )
    cortex_exec_channel: str = Field("orion:cortex:request", alias="CORTEX_EXEC_CHANNEL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**`app/store.py`** — methods from PR spec (`load_latest_policy_frame`, `load_proposal_frame`, `load_self_state`, `load_dispatch_frame_for_policy_frame`, `save_dispatch_frame`). Use column `policy_decision_frame_json` for policy table (matches policy worktree).

**`app/worker.py` `_tick`:**

```python
def _tick(self) -> None:
    if not self._settings.enable_execution_dispatch_runtime:
        return
    policy_frame = self._store.load_latest_policy_frame()
    if policy_frame is None:
        return
    if self._store.load_dispatch_frame_for_policy_frame(policy_frame.frame_id) is not None:
        return
    proposal = self._store.load_proposal_frame(policy_frame.source_proposal_frame_id)
    if proposal is None:
        logger.warning(
            "execution_dispatch_skip_missing_proposal proposal_frame_id=%s",
            policy_frame.source_proposal_frame_id,
        )
        return
    self_state = self._store.load_self_state(policy_frame.source_self_state_id)
    if self_state is None:
        logger.warning(
            "execution_dispatch_skip_missing_self_state self_state_id=%s",
            policy_frame.source_self_state_id,
        )
        return
    frame = build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=self_state,
        policy=self._policy,
        override_dispatch_mode=self._settings.execution_dispatch_mode,
    )
    self._store.save_dispatch_frame(frame)
    logger.info(
        "execution_dispatch_frame_saved frame_id=%s policy_frame_id=%s candidates=%d blocked=%d",
        frame.frame_id,
        policy_frame.frame_id,
        len(frame.candidates),
        frame.blocked_count,
    )
```

**Dockerfile** copies: `orion/`, `config/execution_dispatch/`, `services/orion-execution-dispatch-runtime/`.

**`docker-compose.yml`:** port `${EXECUTION_DISPATCH_RUNTIME_PORT:-8121}:8121`, container `${PROJECT}-execution-dispatch-runtime`, external `app-net`.

**`README.md`** — document Layer 9, default `dry_run`, migration + smoke commands, explicit “no cortex-exec by default”.

- [ ] **Step 1: Write failing store tests** (pattern from `tests/test_proposal_runtime_store.py`, table `substrate_execution_dispatch_frames`)

- [ ] **Step 2: Run tests — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_execution_dispatch_runtime_store.py -v`

- [ ] **Step 3: Implement store + settings + worker + main** (`GET /health`, `GET /latest`)

- [ ] **Step 4: Run store tests — expect PASS**

- [ ] **Step 5: Sync local `.env`**

```bash
cd services/orion-execution-dispatch-runtime
cp -n .env_example .env
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-execution-dispatch-runtime/ tests/test_execution_dispatch_runtime_store.py
git commit -m "feat(execution-dispatch): add orion-execution-dispatch-runtime polling service"
```

---

### Task 7: Hub read-only debug route

**Files:**

- Create: `services/orion-hub/scripts/substrate_execution_dispatch_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Create: `services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py`

Mirror `substrate_proposal_routes.py`:

- Prefix: `/api/substrate/execution-dispatch`
- Route: `GET /latest`
- Table: `substrate_execution_dispatch_frames`, column `dispatch_frame_json`
- Schema: `ExecutionDispatchFrameV1`

- [ ] **Step 1: Write failing hub test**

- [ ] **Step 2: Run — FAIL**

Run: `PYTHONPATH=. pytest services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py -v`

- [ ] **Step 3: Implement route + register in `api_routes.py`**

```python
from .substrate_execution_dispatch_routes import router as substrate_execution_dispatch_router
router.include_router(substrate_execution_dispatch_router)
```

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/substrate_execution_dispatch_routes.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_substrate_execution_dispatch_debug_api.py
git commit -m "feat(execution-dispatch): expose substrate execution-dispatch latest debug API"
```

---

### Task 8: Smoke script

**Files:**

- Create: `scripts/smoke_execution_dispatch_v1.sh` (chmod +x)

```bash
#!/usr/bin/env bash
set -euo pipefail

DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest policy decision frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_proposal_frame_id,
    policy_decision_frame_json #>> '{execution_allowed}' as execution_allowed,
    policy_decision_frame_json #> '{approved_decisions}' as approved_decisions,
    policy_decision_frame_json #> '{review_required_decisions}' as review_required_decisions
from substrate_policy_decision_frames
order by generated_at desc
limit 1;
"

echo ""
echo "=== Latest execution dispatch frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select
    generated_at,
    frame_id,
    source_policy_frame_id,
    dispatch_frame_json #>> '{dispatch_mode}' as dispatch_mode,
    dispatch_frame_json #>> '{dispatch_attempted}' as dispatch_attempted,
    dispatch_frame_json #>> '{dispatch_count}' as dispatch_count,
    dispatch_frame_json #> '{candidates}' as candidates,
    dispatch_frame_json #> '{blocked_candidates}' as blocked_candidates
from substrate_execution_dispatch_frames
order by generated_at desc
limit 1;
"
```

- [ ] **Commit**

```bash
chmod +x scripts/smoke_execution_dispatch_v1.sh
git add scripts/smoke_execution_dispatch_v1.sh
git commit -m "chore(execution-dispatch): add execution dispatch frame smoke script"
```

---

### Task 9: Verification gate (required before PR)

- [ ] **Unit tests**

```bash
cd .worktrees/feat-execution-dispatch-v1
PYTHONPATH=. pytest \
  tests/test_execution_dispatch_frame_schemas.py \
  tests/test_execution_dispatch_policy_loader.py \
  tests/test_execution_dispatch_builder.py \
  tests/test_execution_dispatch_envelopes.py \
  tests/test_execution_dispatch_runtime_store.py \
  -q
```

Expected: all passed

- [ ] **Policy regression** (Layer 8 dependency)

```bash
PYTHONPATH=. pytest tests/test_policy_*.py -q
```

- [ ] **Proposal/self-state regression**

```bash
PYTHONPATH=. pytest tests/test_proposal_*.py tests/test_self_state_*.py -q
```

- [ ] **Compile**

```bash
PYTHONPATH=. python -m compileall \
  orion/execution_dispatch \
  orion/schemas/execution_dispatch_frame.py \
  services/orion-execution-dispatch-runtime \
  -q
```

- [ ] **No execution bleed check** (grep proof for PR report)

```bash
rg -n "bus\.publish|redis\.publish|httpx\.|requests\.|aiohttp|operator\.notify|settings\.mutation|service_restart" \
  services/orion-execution-dispatch-runtime orion/execution_dispatch || true
```

Expected: no matches in runtime worker/store paths

- [ ] **Commit any fixups**

---

### Task 10: Live operator steps (document in PR report; run if DB available)

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  < services/orion-sql-db/manual_migration_execution_dispatch_frame_v1.sql

cd services/orion-execution-dispatch-runtime
docker compose up -d --build

./scripts/smoke_execution_dispatch_v1.sh
curl -s http://localhost:8080/api/substrate/execution-dispatch/latest | jq
curl -s http://localhost:8121/latest | jq
```

Ensure policy runtime is running (`orion-policy-runtime` on 8120) and has produced at least one policy frame before expecting dispatch frames.

---

### Task 11: Code review subagent + fixes

- [ ] **Dispatch code-reviewer subagent** on full diff vs `origin/feat/policy-gate-v1`

Required SUB-SKILL: `superpowers:requesting-code-review` then fix all substantive issues before PR.

- [ ] **Re-run Task 9 verification** after fixes

- [ ] **Commit fixes** as separate commits (do not amend unless hook requires)

---

### Task 12: PR report, push, and open PR

**Files:**

- Create: `docs/superpowers/pr-reports/2026-05-25-execution-dispatch-v1-pr.md`

PR report must include:

1. Layer 9 mapping in 11-layer substrate roadmap
2. Example `ExecutionDispatchFrameV1` JSON (from `/latest` or unit test fixture)
3. Proof default mode is `dry_run` (`dispatch_attempted=false`, `dispatch_count=0`)
4. Proof no cortex-exec execution (grep + worker has no bus client imports)
5. Tests run with command output
6. Explicit **Layer 10 feedback deferred** statement

```bash
cd .worktrees/feat-execution-dispatch-v1
git push -u origin feat/execution-dispatch-v1
gh pr create --title "feat: Execution Dispatch v1 — PolicyDecision → Cortex-Exec Envelopes" --body "$(cat <<'EOF'
## Summary
- Layer 9: PolicyDecisionFrameV1 + ProposalFrameV1 + SelfStateV1 → ExecutionDispatchFrameV1
- orion-execution-dispatch-runtime persists to substrate_execution_dispatch_frames (port 8121)
- Default dry_run / prepare-only envelopes; no cortex-exec or bus publish by default

## Test plan
- [ ] pytest execution-dispatch + policy + proposal + self-state regression
- [ ] migration applied
- [ ] smoke_execution_dispatch_v1.sh
- [ ] GET /api/substrate/execution-dispatch/latest

Layer 10 feedback explicitly deferred.

EOF
)"
```

Base branch for PR: `feat/policy-gate-v1` (or `main` once Layer 8 merges).

---

## Self-review (plan author checklist)

| Spec requirement | Task |
|------------------|------|
| ExecutionDispatchCandidateV1 / FrameV1 | Task 1 |
| execution_dispatch_policy.v1.yaml | Task 2 |
| build_cortex_request_envelope | Task 3 |
| build_execution_dispatch_frame + v1 safety rules | Task 4 |
| substrate_execution_dispatch_frames DDL | Task 5 |
| orion-execution-dispatch-runtime | Task 6 |
| Hub GET /latest | Task 7 |
| smoke script | Task 8 |
| Default dry_run, no cortex-exec | Tasks 4, 6, 9 grep |
| No mutating dispatch | Tasks 2, 4 (`allow_mutating_dispatch=false`) |
| Layer 10 deferred | Task 12 |
| registry update | Task 1 |
| bus channels — no publish entry | Worktree rules (registry only) |
| router.py live dispatch | Omitted (v1 envelopes only) |

**Dependency note:** If `feat/policy-gate-v1` is not merged, PR should target that branch; rebase onto `main` after Layer 8 lands.
