# Feedback Frame v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 10 — observe Layer 9 `ExecutionDispatchFrameV1` outcomes and persist deterministic `FeedbackFrameV1` consequence snapshots (no learning, no policy mutation, no retries).

**Architecture:** Schemas in `orion/schemas/feedback_frame.py`; pure feedback logic in `orion/feedback/`; polling service `orion-feedback-runtime` idempotent per `source_execution_dispatch_frame_id`; optional Hub `GET /api/substrate/feedback/latest`. Config from `config/feedback/feedback_policy.v1.yaml`. **No** bus publish. **No** consolidation.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, FastAPI/uvicorn, pytest, Postgres.

**Depends on:** Layer 9 on `feat/execution-dispatch-v1` (`ExecutionDispatchFrameV1`, `substrate_execution_dispatch_frames`, port 8121). Layers 6–8 on `main` / policy branch (`SelfStateV1`, `ProposalFrameV1`, `PolicyDecisionFrameV1`).

**Non-goals:** Layer 11 consolidation, concept induction, habit learning, policy mutation, automatic retry, cortex-exec steering, proposal generation, action approval, operator notifications, LLM interpretation, bus publish (default).

---

## Worktree and branch hygiene

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin feat/execution-dispatch-v1
git worktree add .worktrees/feat-feedback-frame-v1 -b feat/feedback-frame-v1 origin/feat/execution-dispatch-v1
cd .worktrees/feat-feedback-frame-v1
```

**Rules:**

- All implementation commits happen only inside `.worktrees/feat-feedback-frame-v1`.
- Do **not** copy changed files back to the main workspace checkout except syncing `services/orion-feedback-runtime/.env` from `.env_example` on the operator machine (`.env` is gitignored).
- **Port:** `8122` (`FEEDBACK_RUNTIME_PORT`).
- **Bus:** Register schemas in `orion/schemas/registry.py` only. **Do not** add `orion/bus/channels.yaml` entries — feedback runtime does not publish (matches Layer 8–9 pattern).

---

## File structure

| Path | Role |
|------|------|
| `orion/schemas/feedback_frame.py` | `OutcomeObservationV1`, `FeedbackFrameV1` |
| `orion/schemas/registry.py` | Register new schema types |
| `config/feedback/feedback_policy.v1.yaml` | Feedback windows, scoring, pressure channels |
| `orion/feedback/__init__.py` | Package export |
| `orion/feedback/policy.py` | YAML loader + `FeedbackPolicyV1` |
| `orion/feedback/extractors.py` | Pressure snapshots + deltas |
| `orion/feedback/scoring.py` | Outcome/confidence aggregation |
| `orion/feedback/builder.py` | `build_feedback_frame` |
| `services/orion-sql-db/manual_migration_feedback_frame_v1.sql` | DDL |
| `services/orion-feedback-runtime/` | Polling runtime (mirror `orion-execution-dispatch-runtime`) |
| `services/orion-hub/scripts/substrate_feedback_routes.py` | Optional debug API |
| `services/orion-hub/scripts/api_routes.py` | Include feedback router |
| `services/orion-hub/tests/test_substrate_feedback_debug_api.py` | Hub route tests |
| `tests/test_feedback_*.py` | Unit tests |
| `scripts/smoke_feedback_frame_v1.sh` | Live SQL smoke |
| `docs/superpowers/pr-reports/2026-05-25-feedback-frame-v1-pr.md` | PR report (post-implementation) |

---

### Task 1: Feedback schemas + registry

**Files:**

- Create: `orion/schemas/feedback_frame.py`
- Modify: `orion/schemas/registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_frame_schemas.py`:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def test_outcome_observation_validates() -> None:
    obs = OutcomeObservationV1(
        observation_id="obs:dispatch:d1:dry_run",
        source_kind="dispatch_candidate",
        source_id="dispatch:proposal:inspect:feedback_policy.v1",
        outcome_kind="dry_run",
        score=0.5,
        confidence=0.9,
        observed_at=NOW,
    )
    assert obs.outcome_kind == "dry_run"


def test_feedback_frame_validates() -> None:
    obs = OutcomeObservationV1(
        observation_id="obs:dispatch:d1:dry_run",
        source_kind="dispatch_candidate",
        source_id="dispatch:proposal:inspect:feedback_policy.v1",
        outcome_kind="dry_run",
        score=0.5,
        confidence=0.9,
        observed_at=NOW,
    )
    frame = FeedbackFrameV1(
        frame_id="feedback.frame:execution.dispatch.frame:pf1:feedback_policy.v1",
        generated_at=NOW,
        source_execution_dispatch_frame_id="execution.dispatch.frame:pf1:execution_dispatch_policy.v1",
        source_policy_frame_id="policy.frame:pf1:substrate_policy.v1",
        source_proposal_frame_id="proposal.frame:pf1:proposal_policy.v1",
        source_self_state_id="self.state:pf1",
        outcome_status="dry_run_only",
        outcome_score=0.5,
        confidence_score=0.9,
        observations=[obs],
    )
    assert frame.schema_version == "feedback.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        OutcomeObservationV1(
            observation_id="o1",
            source_kind="dispatch_candidate",
            source_id="s1",
            outcome_kind="dry_run",
            score=0.5,
            confidence=0.9,
            observed_at=NOW,
            extra_field=True,
        )


def test_score_bounds_rejected() -> None:
    with pytest.raises(ValidationError):
        FeedbackFrameV1(
            frame_id="f1",
            generated_at=NOW,
            source_execution_dispatch_frame_id="d1",
            outcome_score=1.5,
            confidence_score=0.9,
        )


def test_roundtrip_json() -> None:
    frame = FeedbackFrameV1(
        frame_id="feedback.frame:execution.dispatch.frame:pf1:feedback_policy.v1",
        generated_at=NOW,
        source_execution_dispatch_frame_id="execution.dispatch.frame:pf1:execution_dispatch_policy.v1",
        outcome_status="unknown",
        outcome_score=0.25,
        confidence_score=0.5,
    )
    restored = FeedbackFrameV1.model_validate(frame.model_dump(mode="json"))
    assert restored.frame_id == frame.frame_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_feedback_frame_schemas.py -v`

Expected: FAIL — `ModuleNotFoundError: orion.schemas.feedback_frame`

- [ ] **Step 3: Write minimal implementation**

Create `orion/schemas/feedback_frame.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OutcomeObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str

    source_kind: Literal[
        "dispatch_candidate",
        "policy_decision",
        "proposal_candidate",
        "cortex_result",
        "field_delta",
        "attention_delta",
        "self_state_delta",
        "absence",
        "operator_feedback",
    ]

    source_id: str

    outcome_kind: Literal[
        "not_attempted",
        "dry_run",
        "prepared",
        "dispatched",
        "completed",
        "failed",
        "blocked",
        "deferred",
        "absent",
        "stale",
        "improved",
        "worsened",
        "unchanged",
        "unknown",
    ]

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    evidence_refs: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)

    observed_at: datetime


class FeedbackFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["feedback.frame.v1"] = "feedback.frame.v1"

    frame_id: str
    generated_at: datetime

    source_execution_dispatch_frame_id: str
    source_policy_frame_id: str | None = None
    source_proposal_frame_id: str | None = None
    source_self_state_id: str | None = None

    feedback_policy_id: str = "feedback_policy.v1"

    outcome_status: Literal[
        "dry_run_only",
        "prepared_only",
        "completed",
        "failed",
        "blocked",
        "deferred",
        "absent",
        "mixed",
        "unknown",
    ] = "unknown"

    outcome_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    observations: list[OutcomeObservationV1] = Field(default_factory=list)

    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    absence_evidence: list[str] = Field(default_factory=list)

    pressure_before: dict[str, float] = Field(default_factory=dict)
    pressure_after: dict[str, float] = Field(default_factory=dict)
    pressure_delta: dict[str, float] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)
```

Register in `orion/schemas/registry.py`:

```python
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1
```

Add to `SCHEMA_REGISTRY` (near `ExecutionDispatchFrameV1`):

```python
    "FeedbackFrameV1": FeedbackFrameV1,
    "OutcomeObservationV1": OutcomeObservationV1,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_feedback_frame_schemas.py -q`

Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/feedback_frame.py orion/schemas/registry.py tests/test_feedback_frame_schemas.py
git commit -m "feat(feedback): add FeedbackFrameV1 schemas and registry"
```

---

### Task 2: Feedback policy config + loader

**Files:**

- Create: `config/feedback/feedback_policy.v1.yaml`
- Create: `orion/feedback/__init__.py`
- Create: `orion/feedback/policy.py`
- Test: `tests/test_feedback_policy_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_policy_loader.py`:

```python
from pathlib import Path

from orion.feedback.policy import FeedbackPolicyV1, load_feedback_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "feedback" / "feedback_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.schema_version == "feedback_policy.v1"
    assert policy.policy_id == "feedback_policy.v1"


def test_windows_defaults() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.windows.field_after_window_sec == 30
    assert policy.windows.result_wait_window_sec == 30


def test_scoring_keys() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.scoring.dry_run_score == 0.50
    assert policy.scoring.completed_score == 0.85
    assert policy.scoring.absent_score == 0.15


def test_pressure_channels() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert "execution_pressure" in policy.pressure_channels
    assert policy.positive_delta_channels["agency_readiness"] == "increase"


def test_absence_rules() -> None:
    policy = load_feedback_policy(POLICY_PATH)
    assert policy.absence_rules["dry_run_needs_no_cortex_result"] is True
    assert policy.absence_rules["dispatch_read_only_requires_result"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_feedback_policy_loader.py -v`

Expected: FAIL — module/config missing

- [ ] **Step 3: Write minimal implementation**

`config/feedback/feedback_policy.v1.yaml`:

```yaml
schema_version: feedback_policy.v1
policy_id: feedback_policy.v1

windows:
  field_after_window_sec: 30
  result_wait_window_sec: 30
  stale_after_sec: 120

scoring:
  dry_run_score: 0.50
  prepared_score: 0.55
  completed_score: 0.85
  blocked_score: 0.40
  deferred_score: 0.45
  failed_score: 0.10
  absent_score: 0.15
  unknown_score: 0.25

pressure_channels:
  - execution_pressure
  - resource_pressure
  - reasoning_pressure
  - reliability_pressure
  - field_intensity
  - uncertainty
  - agency_readiness

positive_delta_channels:
  agency_readiness: increase
  coherence: increase
  uncertainty: decrease
  reliability_pressure: decrease
  execution_pressure: decrease
  resource_pressure: decrease

absence_rules:
  dry_run_needs_no_cortex_result: true
  dispatch_read_only_requires_result: true
  missing_expected_result_kind: absent
```

`orion/feedback/policy.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class FeedbackWindowsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_after_window_sec: int = 30
    result_wait_window_sec: int = 30
    stale_after_sec: int = 120


class FeedbackScoringV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dry_run_score: float = 0.50
    prepared_score: float = 0.55
    completed_score: float = 0.85
    blocked_score: float = 0.40
    deferred_score: float = 0.45
    failed_score: float = 0.10
    absent_score: float = 0.15
    unknown_score: float = 0.25


class FeedbackPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["feedback_policy.v1"] = "feedback_policy.v1"
    policy_id: str = "feedback_policy.v1"

    windows: FeedbackWindowsV1 = Field(default_factory=FeedbackWindowsV1)
    scoring: FeedbackScoringV1 = Field(default_factory=FeedbackScoringV1)
    pressure_channels: list[str] = Field(default_factory=list)
    positive_delta_channels: dict[str, str] = Field(default_factory=dict)
    absence_rules: dict[str, bool | str] = Field(default_factory=dict)


def load_feedback_policy(path: str | Path) -> FeedbackPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return FeedbackPolicyV1.model_validate(data)
```

`orion/feedback/__init__.py`:

```python
from orion.feedback.builder import build_feedback_frame, stable_feedback_frame_id
from orion.feedback.policy import FeedbackPolicyV1, load_feedback_policy

__all__ = [
    "FeedbackPolicyV1",
    "build_feedback_frame",
    "load_feedback_policy",
    "stable_feedback_frame_id",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_feedback_policy_loader.py -q`

Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add config/feedback/feedback_policy.v1.yaml orion/feedback/ tests/test_feedback_policy_loader.py
git commit -m "feat(feedback): add feedback policy config and loader"
```

---

### Task 3: Pressure extractors

**Files:**

- Create: `orion/feedback/extractors.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_feedback_builder.py` later, or create `tests/test_feedback_extractors.py`:

```python
from datetime import datetime, timezone

from orion.feedback.extractors import (
    classify_pressure_deltas,
    extract_self_state_pressure_snapshot,
    pressure_delta,
)
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
CHANNELS = ["execution_pressure", "agency_readiness", "uncertainty", "coherence"]


def _state(scores: dict[str, float]) -> SelfStateV1:
    dims = {
        k: SelfStateDimensionV1(dimension_id=k, score=v, confidence=0.9)
        for k, v in scores.items()
    }
    return SelfStateV1(
        self_state_id="self.state:test",
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="att",
        source_attention_generated_at=NOW,
        dimensions=dims,
    )


def test_extract_snapshot() -> None:
    snap = extract_self_state_pressure_snapshot(
        _state({"execution_pressure": 1.0, "agency_readiness": 0.2}),
        CHANNELS,
    )
    assert snap["execution_pressure"] == 1.0
    assert snap.get("coherence", 0.0) == 0.0


def test_pressure_delta() -> None:
    before = {"execution_pressure": 1.0, "agency_readiness": 0.2}
    after = {"execution_pressure": 0.5, "agency_readiness": 0.6}
    delta = pressure_delta(before, after)
    assert delta["execution_pressure"] == -0.5
    assert delta["agency_readiness"] == 0.4


def test_classify_positive_negative() -> None:
    delta = {"agency_readiness": 0.2, "execution_pressure": -0.3, "uncertainty": 0.1}
    pos, neg = classify_pressure_deltas(
        delta,
        {
            "agency_readiness": "increase",
            "execution_pressure": "decrease",
            "uncertainty": "decrease",
        },
    )
    assert "agency_readiness" in pos
    assert "execution_pressure" in pos
    assert "uncertainty" in neg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_feedback_extractors.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`orion/feedback/extractors.py`:

```python
from __future__ import annotations

from orion.schemas.self_state import SelfStateV1

PRESSURE_DIMENSION_IDS = frozenset({
    "execution_pressure",
    "resource_pressure",
    "reasoning_pressure",
    "reliability_pressure",
    "field_intensity",
    "uncertainty",
    "agency_readiness",
    "coherence",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def extract_self_state_pressure_snapshot(
    state: SelfStateV1 | None,
    channels: list[str],
) -> dict[str, float]:
    if state is None:
        return {ch: 0.0 for ch in channels}
    out: dict[str, float] = {}
    for ch in channels:
        dim = state.dimensions.get(ch)
        out[ch] = clamp01(dim.score) if dim is not None else 0.0
    if "coherence" in channels and "coherence" not in out:
        dim = state.dimensions.get("coherence")
        out["coherence"] = clamp01(dim.score) if dim is not None else 0.0
    return out


def pressure_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    keys = set(before) | set(after)
    return {k: clamp01(after.get(k, 0.0) - before.get(k, 0.0)) for k in keys}


def classify_pressure_deltas(
    delta: dict[str, float],
    positive_delta_channels: dict[str, str],
) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    for channel, direction in positive_delta_channels.items():
        d = delta.get(channel)
        if d is None or abs(d) < 1e-6:
            continue
        if direction == "increase" and d > 0:
            positive.append(f"pressure_delta:{channel}:+{d:.3f}")
        elif direction == "decrease" and d < 0:
            positive.append(f"pressure_delta:{channel}:{d:.3f}")
        elif direction == "increase" and d < 0:
            negative.append(f"pressure_delta:{channel}:{d:.3f}")
        elif direction == "decrease" and d > 0:
            negative.append(f"pressure_delta:{channel}:+{d:.3f}")
    return positive, negative


def normalize_cortex_result_evidence(result: dict[str, object]) -> dict[str, object]:
    """Strip raw blobs; keep status + correlation ids for FeedbackFrameV1."""
    status = str(result.get("status") or result.get("ok") or "unknown").lower()
    if status in ("true", "1"):
        status = "success"
    if status in ("false", "0"):
        status = "failed"
    return {
        "result_id": str(result.get("result_id") or result.get("correlation_id") or "unknown"),
        "dispatch_id": str(result.get("dispatch_id") or ""),
        "status": status,
        "evidence_refs": list(result.get("evidence_refs") or []),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_feedback_extractors.py -q`

Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/feedback/extractors.py tests/test_feedback_extractors.py
git commit -m "feat(feedback): add pressure extractors and cortex evidence normalizer"
```

---

### Task 4: Feedback scoring

**Files:**

- Create: `orion/feedback/scoring.py`
- Test: `tests/test_feedback_scoring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_scoring.py`:

```python
from orion.feedback.policy import FeedbackScoringV1
from orion.feedback.scoring import aggregate_confidence, score_for_outcome_status


def test_score_for_dry_run() -> None:
    scoring = FeedbackScoringV1()
    assert score_for_outcome_status("dry_run_only", scoring) == 0.50


def test_score_for_completed() -> None:
    scoring = FeedbackScoringV1()
    assert score_for_outcome_status("completed", scoring) == 0.85


def test_aggregate_confidence_empty() -> None:
    assert aggregate_confidence([]) == 0.0


def test_aggregate_confidence_mean() -> None:
    assert abs(aggregate_confidence([0.8, 0.6]) - 0.7) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_feedback_scoring.py -v`

Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

`orion/feedback/scoring.py`:

```python
from __future__ import annotations

from orion.feedback.policy import FeedbackScoringV1
from orion.schemas.feedback_frame import OutcomeObservationV1


def score_for_outcome_status(status: str, scoring: FeedbackScoringV1) -> float:
    mapping = {
        "dry_run_only": scoring.dry_run_score,
        "prepared_only": scoring.prepared_score,
        "completed": scoring.completed_score,
        "blocked": scoring.blocked_score,
        "deferred": scoring.deferred_score,
        "failed": scoring.failed_score,
        "absent": scoring.absent_score,
        "mixed": scoring.unknown_score,
        "unknown": scoring.unknown_score,
    }
    return float(mapping.get(status, scoring.unknown_score))


def aggregate_confidence(observations: list[OutcomeObservationV1]) -> float:
    if not observations:
        return 0.0
    return sum(o.confidence for o in observations) / len(observations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_feedback_scoring.py -q`

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add orion/feedback/scoring.py tests/test_feedback_scoring.py
git commit -m "feat(feedback): add outcome scoring helpers"
```

---

### Task 5: Feedback frame builder

**Files:**

- Create: `orion/feedback/builder.py`
- Test: `tests/test_feedback_builder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_builder.py` (imports shared fixtures from execution-dispatch tests pattern):

```python
from datetime import datetime, timezone
from pathlib import Path

from orion.execution_dispatch.builder import build_execution_dispatch_frame
from orion.execution_dispatch.policy import load_execution_dispatch_policy
from orion.feedback.builder import build_feedback_frame, stable_feedback_frame_id
from orion.feedback.policy import load_feedback_policy
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalCandidateV1, ProposalFrameV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

REPO = Path(__file__).resolve().parents[1]
DISPATCH_POLICY = load_execution_dispatch_policy(
    REPO / "config" / "execution_dispatch" / "execution_dispatch_policy.v1.yaml"
)
FEEDBACK_POLICY = load_feedback_policy(REPO / "config" / "feedback" / "feedback_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)


def _dim(dimension_id: str, score: float) -> SelfStateDimensionV1:
    return SelfStateDimensionV1(dimension_id=dimension_id, score=score, confidence=0.9)


def _self_state(self_state_id: str, scores: dict[str, float]) -> SelfStateV1:
    return SelfStateV1(
        self_state_id=self_state_id,
        generated_at=NOW,
        source_field_tick_id="tick",
        source_field_generated_at=NOW,
        source_attention_frame_id="att",
        source_attention_generated_at=NOW,
        dimensions={k: _dim(k, v) for k, v in scores.items()},
    )


def _proposal() -> ProposalFrameV1:
    def cand(pid: str, kind: str) -> ProposalCandidateV1:
        return ProposalCandidateV1(
            proposal_id=pid,
            proposal_kind=kind,
            title=pid,
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

    state = _self_state("self.state:before", {"execution_pressure": 1.0})
    return ProposalFrameV1(
        frame_id="proposal.frame:test:proposal_policy.v1",
        generated_at=NOW,
        source_self_state_id=state.self_state_id,
        source_self_state_generated_at=state.generated_at,
        source_attention_frame_id=state.source_attention_frame_id,
        source_field_tick_id=state.source_field_tick_id,
        overall_action_pressure=0.6,
        overall_risk=0.3,
        candidates=[cand("proposal:inspect:state", "inspect")],
    )


def _policy_frame(proposal: ProposalFrameV1) -> PolicyDecisionFrameV1:
    decision = PolicyDecisionV1(
        decision_id="policy.decision:proposal:inspect:substrate_policy.v1",
        proposal_id="proposal:inspect:state",
        decision="approved_read_only",
        policy_gate="read_only",
        risk_score=0.05,
        reversibility_score=1.0,
        confidence_score=0.9,
        allowed_scope="inspect_only",
    )
    return PolicyDecisionFrameV1(
        frame_id="policy.frame:proposal.frame:test:substrate_policy.v1",
        generated_at=NOW,
        source_proposal_frame_id=proposal.frame_id,
        source_self_state_id=proposal.source_self_state_id,
        decisions=[decision],
        approved_decisions=[decision],
        overall_risk=0.05,
    )


def _dispatch_dry_run() -> ExecutionDispatchFrameV1:
    proposal = _proposal()
    policy_frame = _policy_frame(proposal)
    before = _self_state("self.state:before", {"execution_pressure": 1.0})
    return build_execution_dispatch_frame(
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state=before,
        policy=DISPATCH_POLICY,
        now=NOW,
    )


def test_dry_run_produces_dry_run_only() -> None:
    dispatch = _dispatch_dry_run()
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=_policy_frame(_proposal()),
        proposal_frame=_proposal(),
        self_state_before=_self_state("self.state:before", {"execution_pressure": 1.0}),
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "dry_run_only"
    assert any(o.outcome_kind == "dry_run" for o in frame.observations)


def test_prepared_only_dispatch() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:prep:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:prep",
        source_proposal_frame_id="proposal.frame:prep",
        source_self_state_id="self.state:prep",
        dispatch_mode="prepare_only",
        candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="prepared",
                dispatch_mode="prepare_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
            )
        ],
        dispatch_attempted=False,
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "prepared_only"


def test_blocked_candidate_observation() -> None:
    dispatch = _dispatch_dry_run()
    dispatch = dispatch.model_copy(
        update={
            "blocked_candidates": [
                ExecutionDispatchCandidateV1(
                    dispatch_id="dispatch:proposal:blocked:execution_dispatch_policy.v1",
                    source_decision_id="pd2",
                    source_proposal_id="proposal:blocked:state",
                    dispatch_status="blocked",
                    dispatch_mode="dry_run",
                    dispatch_kind="inspect",
                    target_id="t1",
                    target_kind="capability",
                    risk_score=0.3,
                    confidence_score=0.9,
                    blocked_by=["rejected"],
                )
            ],
            "blocked_count": 1,
        }
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert any(o.outcome_kind == "blocked" for o in frame.observations)


def test_missing_cortex_result_absence() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:ro:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:ro",
        source_proposal_frame_id="proposal.frame:ro",
        source_self_state_id="self.state:ro",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=1,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status in ("absent", "mixed")
    assert len(frame.absence_evidence) >= 1


def test_successful_cortex_result_completed() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:ok:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:ok",
        source_proposal_frame_id="proposal.frame:ok",
        source_self_state_id="self.state:ok",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=1,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[{"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "success"}],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "completed"
    assert any(o.outcome_kind == "completed" for o in frame.observations)


def test_failed_cortex_result() -> None:
    dispatch = ExecutionDispatchFrameV1(
        frame_id="execution.dispatch.frame:fail:execution_dispatch_policy.v1",
        generated_at=NOW,
        source_policy_frame_id="policy.frame:fail",
        source_proposal_frame_id="proposal.frame:fail",
        source_self_state_id="self.state:fail",
        dispatch_mode="dispatch_read_only",
        dispatch_attempted=True,
        dispatch_count=1,
        dispatched_candidates=[
            ExecutionDispatchCandidateV1(
                dispatch_id="dispatch:proposal:inspect:execution_dispatch_policy.v1",
                source_decision_id="pd1",
                source_proposal_id="proposal:inspect:state",
                dispatch_status="dispatched",
                dispatch_mode="dispatch_read_only",
                dispatch_kind="inspect",
                target_id="t1",
                target_kind="capability",
                risk_score=0.05,
                confidence_score=0.9,
            )
        ],
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=[{"dispatch_id": "dispatch:proposal:inspect:execution_dispatch_policy.v1", "status": "failed"}],
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.outcome_status == "failed"


def test_self_state_improvement_positive_evidence() -> None:
    dispatch = _dispatch_dry_run()
    before = _self_state("self.state:before", {"execution_pressure": 1.0, "agency_readiness": 0.2})
    after = _self_state("self.state:after", {"execution_pressure": 0.5, "agency_readiness": 0.6})
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=before,
        self_state_after=after,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert len(frame.positive_evidence) >= 1
    assert any(o.outcome_kind == "improved" for o in frame.observations)


def test_self_state_worsening_negative_evidence() -> None:
    dispatch = _dispatch_dry_run()
    before = _self_state("self.state:before", {"agency_readiness": 0.8})
    after = _self_state("self.state:after", {"agency_readiness": 0.2})
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=before,
        self_state_after=after,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert len(frame.negative_evidence) >= 1
    assert any(o.outcome_kind == "worsened" for o in frame.observations)


def test_stable_frame_id() -> None:
    dispatch = _dispatch_dry_run()
    expected = stable_feedback_frame_id(
        dispatch_frame_id=dispatch.frame_id,
        policy_id=FEEDBACK_POLICY.policy_id,
    )
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=None,
        proposal_frame=None,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert frame.frame_id == expected


def test_no_mutation_side_effects() -> None:
    dispatch = _dispatch_dry_run()
    policy_frame = _policy_frame(_proposal())
    proposal = _proposal()
    dispatch_dump = dispatch.model_dump()
    policy_dump = policy_frame.model_dump()
    proposal_dump = proposal.model_dump()
    build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=policy_frame,
        proposal_frame=proposal,
        self_state_before=None,
        self_state_after=None,
        cortex_results=None,
        policy=FEEDBACK_POLICY,
        now=NOW,
    )
    assert dispatch.model_dump() == dispatch_dump
    assert policy_frame.model_dump() == policy_dump
    assert proposal.model_dump() == proposal_dump
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_feedback_builder.py -v`

Expected: FAIL — `build_feedback_frame` not defined

- [ ] **Step 3: Write minimal implementation**

`orion/feedback/builder.py` (core logic):

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.feedback.extractors import (
    classify_pressure_deltas,
    extract_self_state_pressure_snapshot,
    normalize_cortex_result_evidence,
    pressure_delta,
)
from orion.feedback.policy import FeedbackPolicyV1
from orion.feedback.scoring import aggregate_confidence, score_for_outcome_status
from orion.schemas.execution_dispatch_frame import ExecutionDispatchCandidateV1, ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1, OutcomeObservationV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1, PolicyDecisionV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


def stable_feedback_frame_id(*, dispatch_frame_id: str, policy_id: str) -> str:
    return f"feedback.frame:{dispatch_frame_id}:{policy_id}"


def _observation(
    *,
    observation_id: str,
    source_kind: OutcomeObservationV1.model_fields["source_kind"].annotation,
    source_id: str,
    outcome_kind: OutcomeObservationV1.model_fields["outcome_kind"].annotation,
    score: float,
    confidence: float,
    observed_at: datetime,
    reasons: list[str] | None = None,
    evidence_refs: list[str] | None = None,
) -> OutcomeObservationV1:
    return OutcomeObservationV1(
        observation_id=observation_id,
        source_kind=source_kind,
        source_id=source_id,
        outcome_kind=outcome_kind,
        score=score,
        confidence=confidence,
        reasons=reasons or [],
        evidence_refs=evidence_refs or [],
        observed_at=observed_at,
    )


def _candidate_outcome_kind(candidate: ExecutionDispatchCandidateV1) -> str:
    if candidate.dispatch_status == "blocked":
        return "blocked"
    if candidate.dispatch_status == "prepared":
        return "prepared"
    if candidate.dispatch_status == "dry_run":
        return "dry_run"
    if candidate.dispatch_status == "dispatched":
        return "dispatched"
    return "unknown"


def _policy_decision_outcome(decision: PolicyDecisionV1) -> str:
    if decision.decision == "deferred":
        return "deferred"
    if decision.decision == "rejected":
        return "blocked"
    return "not_attempted"


def _cortex_status_to_outcome(status: str) -> str:
    s = status.lower()
    if s in ("success", "ok", "completed"):
        return "completed"
    if s in ("failed", "error"):
        return "failed"
    return "unknown"


def _aggregate_outcome_status(observations: list[OutcomeObservationV1], dispatch: ExecutionDispatchFrameV1) -> str:
    kinds = {o.outcome_kind for o in observations}
    if dispatch.dispatch_mode == "dry_run" and not dispatch.dispatch_attempted:
        return "dry_run_only"
    if dispatch.dispatch_mode == "prepare_only" and not dispatch.dispatch_attempted:
        return "prepared_only"
    if "completed" in kinds and "failed" not in kinds:
        return "completed"
    if "failed" in kinds and "completed" not in kinds:
        return "failed"
    if "absent" in kinds:
        if "completed" in kinds or "failed" in kinds:
            return "mixed"
        return "absent"
    if kinds <= {"blocked", "deferred"} or (kinds and "blocked" in kinds and not dispatch.dispatch_attempted):
        if "deferred" in kinds:
            return "deferred"
        return "blocked"
    if len(kinds) > 1:
        return "mixed"
    return "unknown"


def build_feedback_frame(
    *,
    dispatch_frame: ExecutionDispatchFrameV1,
    policy_frame: PolicyDecisionFrameV1 | None,
    proposal_frame: ProposalFrameV1 | None,
    self_state_before: SelfStateV1 | None,
    self_state_after: SelfStateV1 | None,
    cortex_results: list[dict[str, object]] | None,
    policy: FeedbackPolicyV1,
    now: datetime | None = None,
) -> FeedbackFrameV1:
    generated_at = now or datetime.now(timezone.utc)
    scoring = policy.scoring
    observations: list[OutcomeObservationV1] = []
    positive_evidence: list[str] = []
    negative_evidence: list[str] = []
    absence_evidence: list[str] = []
    warnings: list[str] = list(dispatch_frame.warnings)

    channels = list(policy.pressure_channels)
    if "coherence" not in channels:
        channels.append("coherence")

    pressure_before = extract_self_state_pressure_snapshot(self_state_before, channels)
    pressure_after = extract_self_state_pressure_snapshot(self_state_after, channels)
    delta = pressure_delta(pressure_before, pressure_after)
    pos_delta, neg_delta = classify_pressure_deltas(delta, policy.positive_delta_channels)
    positive_evidence.extend(pos_delta)
    negative_evidence.extend(neg_delta)

    for candidate in (
        list(dispatch_frame.candidates)
        + list(dispatch_frame.blocked_candidates)
        + list(dispatch_frame.dispatched_candidates)
    ):
        kind = _candidate_outcome_kind(candidate)
        score = getattr(scoring, f"{kind}_score", scoring.unknown_score) if hasattr(scoring, f"{kind}_score") else scoring.unknown_score
        if kind == "dry_run":
            score = scoring.dry_run_score
        elif kind == "prepared":
            score = scoring.prepared_score
        elif kind == "blocked":
            score = scoring.blocked_score
        observations.append(
            _observation(
                observation_id=f"obs:dispatch:{candidate.dispatch_id}:{kind}",
                source_kind="dispatch_candidate",
                source_id=candidate.dispatch_id,
                outcome_kind=kind,
                score=score,
                confidence=candidate.confidence_score,
                observed_at=generated_at,
                reasons=list(candidate.reasons),
                evidence_refs=list(candidate.evidence_refs),
            )
        )

    if policy_frame is not None:
        for decision in policy_frame.decisions:
            outcome = _policy_decision_outcome(decision)
            score = scoring.deferred_score if outcome == "deferred" else scoring.blocked_score if outcome == "blocked" else scoring.unknown_score
            observations.append(
                _observation(
                    observation_id=f"obs:policy:{decision.decision_id}:{outcome}",
                    source_kind="policy_decision",
                    source_id=decision.decision_id,
                    outcome_kind=outcome,
                    score=score,
                    confidence=decision.confidence_score,
                    observed_at=generated_at,
                    reasons=list(decision.reasons),
                    evidence_refs=list(decision.evidence_refs),
                )
            )

    normalized_results = [normalize_cortex_result_evidence(r) for r in (cortex_results or [])]
    dispatched_ids = {c.dispatch_id for c in dispatch_frame.dispatched_candidates}
    matched: set[str] = set()

    for raw in normalized_results:
        status = str(raw.get("status", "unknown"))
        outcome = _cortex_status_to_outcome(status)
        dispatch_id = str(raw.get("dispatch_id") or "")
        score = scoring.completed_score if outcome == "completed" else scoring.failed_score if outcome == "failed" else scoring.unknown_score
        if dispatch_id:
            matched.add(dispatch_id)
        observations.append(
            _observation(
                observation_id=f"obs:cortex:{raw.get('result_id')}:{outcome}",
                source_kind="cortex_result",
                source_id=str(raw.get("result_id")),
                outcome_kind=outcome,
                score=score,
                confidence=0.85,
                observed_at=generated_at,
                evidence_refs=list(raw.get("evidence_refs") or []),
                reasons=[f"cortex_status:{status}"],
            )
        )

    needs_result = (
        dispatch_frame.dispatch_attempted
        and dispatch_frame.dispatch_mode == "dispatch_read_only"
        and bool(policy.absence_rules.get("dispatch_read_only_requires_result"))
    )
    if needs_result:
        for dispatch_id in dispatched_ids - matched:
            absence_evidence.append(f"missing_cortex_result:{dispatch_id}")
            observations.append(
                _observation(
                    observation_id=f"obs:absence:cortex:{dispatch_id}",
                    source_kind="absence",
                    source_id=dispatch_id,
                    outcome_kind="absent",
                    score=scoring.absent_score,
                    confidence=0.8,
                    observed_at=generated_at,
                    reasons=["expected_cortex_result_not_observed"],
                )
            )

    if self_state_before is not None and self_state_after is not None:
        overall_delta = pressure_after.get("agency_readiness", 0.0) - pressure_before.get("agency_readiness", 0.0)
        if overall_delta > 0.05:
            observations.append(
                _observation(
                    observation_id=f"obs:self_state:{self_state_after.self_state_id}:improved",
                    source_kind="self_state_delta",
                    source_id=self_state_after.self_state_id,
                    outcome_kind="improved",
                    score=scoring.completed_score,
                    confidence=self_state_after.overall_confidence,
                    observed_at=generated_at,
                    reasons=["agency_readiness_increased"],
                )
            )
        elif overall_delta < -0.05:
            observations.append(
                _observation(
                    observation_id=f"obs:self_state:{self_state_after.self_state_id}:worsened",
                    source_kind="self_state_delta",
                    source_id=self_state_after.self_state_id,
                    outcome_kind="worsened",
                    score=scoring.failed_score,
                    confidence=self_state_after.overall_confidence,
                    observed_at=generated_at,
                    reasons=["agency_readiness_decreased"],
                )
            )
        else:
            observations.append(
                _observation(
                    observation_id=f"obs:self_state:{self_state_after.self_state_id}:unchanged",
                    source_kind="self_state_delta",
                    source_id=self_state_after.self_state_id,
                    outcome_kind="unchanged",
                    score=scoring.unknown_score,
                    confidence=self_state_after.overall_confidence,
                    observed_at=generated_at,
                )
            )

    outcome_status = _aggregate_outcome_status(observations, dispatch_frame)
    outcome_score = score_for_outcome_status(outcome_status, scoring)
    confidence_score = aggregate_confidence(observations) or dispatch_frame.candidates[0].confidence_score if dispatch_frame.candidates else 0.5

    return FeedbackFrameV1(
        frame_id=stable_feedback_frame_id(
            dispatch_frame_id=dispatch_frame.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_execution_dispatch_frame_id=dispatch_frame.frame_id,
        source_policy_frame_id=dispatch_frame.source_policy_frame_id,
        source_proposal_frame_id=dispatch_frame.source_proposal_frame_id,
        source_self_state_id=dispatch_frame.source_self_state_id,
        feedback_policy_id=policy.policy_id,
        outcome_status=outcome_status,
        outcome_score=outcome_score,
        confidence_score=confidence_score,
        observations=observations,
        positive_evidence=positive_evidence,
        negative_evidence=negative_evidence,
        absence_evidence=absence_evidence,
        pressure_before=pressure_before,
        pressure_after=pressure_after,
        pressure_delta=delta,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_feedback_builder.py tests/test_feedback_extractors.py tests/test_feedback_scoring.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/feedback/builder.py tests/test_feedback_builder.py
git commit -m "feat(feedback): build FeedbackFrameV1 from dispatch and outcomes"
```

---

### Task 6: SQL migration

**Files:**

- Create: `services/orion-sql-db/manual_migration_feedback_frame_v1.sql`

- [ ] **Step 1: Add migration SQL**

```sql
create table if not exists substrate_feedback_frames (
    frame_id text primary key,
    source_execution_dispatch_frame_id text not null,
    source_policy_frame_id text,
    source_proposal_frame_id text,
    source_self_state_id text,
    generated_at timestamptz not null,
    policy_id text not null,
    feedback_frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_feedback_frames_generated_at
    on substrate_feedback_frames (generated_at desc);

create index if not exists idx_substrate_feedback_frames_source_dispatch
    on substrate_feedback_frames (source_execution_dispatch_frame_id);
```

- [ ] **Step 2: Commit**

```bash
git add services/orion-sql-db/manual_migration_feedback_frame_v1.sql
git commit -m "feat(feedback): add substrate_feedback_frames migration"
```

---

### Task 7: `orion-feedback-runtime` service

**Files:**

- Create: `services/orion-feedback-runtime/app/{__init__.py,main.py,worker.py,store.py,settings.py}`
- Create: `services/orion-feedback-runtime/{Dockerfile,docker-compose.yml,requirements.txt,README.md,.env_example}`
- Create: `services/orion-feedback-runtime/.env` (copy from `.env_example` — local only)
- Test: `tests/test_feedback_runtime_store.py`

| Setting | Value |
|---------|-------|
| Port | `8122` |
| Policy path | `/app/config/feedback/feedback_policy.v1.yaml` |
| Idempotency key | `source_execution_dispatch_frame_id` |
| Worker inputs | latest dispatch without feedback + policy + proposal + self_state before/after |
| No bus / no mutation | worker only builds + saves |

**`.env_example`:**

```env
# orion-feedback-runtime — Layer 10 feedback (docker compose)
# Requires: substrate_execution_dispatch_frames + related substrate tables
# Apply migration: services/orion-sql-db/manual_migration_feedback_frame_v1.sql
PROJECT=orion-athena
SERVICE_NAME=orion-feedback-runtime
SERVICE_VERSION=0.1.0
POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
FEEDBACK_POLICY_PATH=/app/config/feedback/feedback_policy.v1.yaml
FEEDBACK_POLL_INTERVAL_SEC=2.0
ENABLE_FEEDBACK_RUNTIME=true
LOG_LEVEL=INFO
FEEDBACK_RUNTIME_PORT=8122
```

**`app/settings.py`:**

```python
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-feedback-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    feedback_policy_path: str = Field(
        "config/feedback/feedback_policy.v1.yaml",
        alias="FEEDBACK_POLICY_PATH",
    )
    feedback_poll_interval_sec: float = Field(2.0, alias="FEEDBACK_POLL_INTERVAL_SEC")
    enable_feedback_runtime: bool = Field(True, alias="ENABLE_FEEDBACK_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**`app/store.py`** — required methods:

```python
def load_latest_dispatch_frame_without_feedback(self) -> ExecutionDispatchFrameV1 | None:
    # SELECT dispatch FROM substrate_execution_dispatch_frames d
    # LEFT JOIN substrate_feedback_frames f ON f.source_execution_dispatch_frame_id = d.frame_id
    # WHERE f.frame_id IS NULL ORDER BY d.generated_at ASC LIMIT 1

def load_policy_frame(self, frame_id: str) -> PolicyDecisionFrameV1 | None:
    # substrate_policy_decision_frames WHERE frame_id = :id

def load_proposal_frame(self, frame_id: str) -> ProposalFrameV1 | None:
    # substrate_proposal_frames WHERE frame_id = :id

def load_self_state(self, self_state_id: str) -> SelfStateV1 | None:
    # substrate_self_state WHERE self_state_id = :id

def load_latest_self_state_after(self, generated_at: datetime) -> SelfStateV1 | None:
    # substrate_self_state WHERE generated_at > :generated_at ORDER BY generated_at ASC LIMIT 1

def load_feedback_frame_for_dispatch(self, dispatch_frame_id: str) -> FeedbackFrameV1 | None:
    # substrate_feedback_frames WHERE source_execution_dispatch_frame_id = :id

def load_cortex_result_evidence(self, dispatch_frame: ExecutionDispatchFrameV1) -> list[dict[str, object]]:
    # v1: return [] — no substrate cortex-result table yet; tests inject via builder directly

def save_feedback_frame(self, frame: FeedbackFrameV1) -> None:
    # INSERT ... ON CONFLICT (frame_id) DO UPDATE
```

**`app/worker.py` `_tick`:**

```python
def _tick(self) -> None:
    if not self._settings.enable_feedback_runtime:
        return
    dispatch = self._store.load_latest_dispatch_frame_without_feedback()
    if dispatch is None:
        return
    if self._store.load_feedback_frame_for_dispatch(dispatch.frame_id) is not None:
        return
    policy_frame = self._store.load_policy_frame(dispatch.source_policy_frame_id)
    proposal_frame = self._store.load_proposal_frame(dispatch.source_proposal_frame_id)
    self_state_before = self._store.load_self_state(dispatch.source_self_state_id)
    self_state_after = self._store.load_latest_self_state_after(dispatch.generated_at)
    cortex_results = self._store.load_cortex_result_evidence(dispatch)
    frame = build_feedback_frame(
        dispatch_frame=dispatch,
        policy_frame=policy_frame,
        proposal_frame=proposal_frame,
        self_state_before=self_state_before,
        self_state_after=self_state_after,
        cortex_results=cortex_results or None,
        policy=self._policy,
    )
    self._store.save_feedback_frame(frame)
```

**Dockerfile** copies: `orion/`, `config/feedback/`, `config/execution_dispatch/` (for dispatch schema imports in tests only if needed — runtime imports `orion.feedback` + dispatch schemas transitively), `services/orion-feedback-runtime/`.

**`docker-compose.yml`:** port `${FEEDBACK_RUNTIME_PORT:-8122}:8122`, container `${PROJECT}-feedback-runtime`, external `app-net`.

**`requirements.txt`:** mirror `orion-policy-runtime` (`fastapi`, `uvicorn`, `pydantic-settings`, `sqlalchemy`, `psycopg2-binary`, `pyyaml`).

**`README.md`:** Layer 10, migration + smoke, explicit “no consolidation / no policy mutation”.

- [ ] **Step 1: Write failing store tests** (pattern from `tests/test_execution_dispatch_runtime_store.py`)

- [ ] **Step 2: Run tests — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_feedback_runtime_store.py -v`

- [ ] **Step 3: Implement store + settings + worker + main**

- [ ] **Step 4: Run store tests — expect PASS**

- [ ] **Step 5: Sync local `.env`**

```bash
cd services/orion-feedback-runtime
cp -n .env_example .env
```

- [ ] **Step 6: Commit**

```bash
git add services/orion-feedback-runtime/ tests/test_feedback_runtime_store.py
git commit -m "feat(feedback): add orion-feedback-runtime polling service"
```

---

### Task 8: Hub read-only debug route

**Files:**

- Create: `services/orion-hub/scripts/substrate_feedback_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Create: `services/orion-hub/tests/test_substrate_feedback_debug_api.py`

Mirror `substrate_policy_routes.py`:

- Prefix: `/api/substrate/feedback`
- Route: `GET /latest`
- Table: `substrate_feedback_frames`, column `feedback_frame_json`
- Schema: `FeedbackFrameV1`

Register in `api_routes.py`:

```python
from .substrate_feedback_routes import router as substrate_feedback_router
router.include_router(substrate_feedback_router)
```

- [ ] **Step 1–5:** TDD hub test + implement + commit

```bash
git add services/orion-hub/scripts/substrate_feedback_routes.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_substrate_feedback_debug_api.py
git commit -m "feat(feedback): expose substrate feedback latest debug API"
```

---

### Task 9: Smoke script

**Files:**

- Create: `scripts/smoke_feedback_frame_v1.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT:-orion-athena}"
DB="${PROJECT}-sql-db"
PSQL=(docker exec -i "$DB" psql -U postgres -d conjourney -v ON_ERROR_STOP=1)

echo "=== Latest execution dispatch frame ==="
"${PSQL[@]}" -c "
select
    generated_at,
    frame_id,
    dispatch_frame_json #>> '{dispatch_mode}' as dispatch_mode,
    dispatch_frame_json #>> '{dispatch_attempted}' as dispatch_attempted,
    dispatch_frame_json #> '{candidates}' as candidates
from substrate_execution_dispatch_frames
order by generated_at desc
limit 1;
"

echo "=== Latest feedback frame ==="
"${PSQL[@]}" -c "
select
    generated_at,
    frame_id,
    source_execution_dispatch_frame_id,
    feedback_frame_json #>> '{outcome_status}' as outcome_status,
    feedback_frame_json #>> '{outcome_score}' as outcome_score,
    feedback_frame_json #> '{observations}' as observations,
    feedback_frame_json #> '{positive_evidence}' as positive_evidence,
    feedback_frame_json #> '{negative_evidence}' as negative_evidence,
    feedback_frame_json #> '{absence_evidence}' as absence_evidence
from substrate_feedback_frames
order by generated_at desc
limit 1;
"
```

```bash
chmod +x scripts/smoke_feedback_frame_v1.sh
git add scripts/smoke_feedback_frame_v1.sh
git commit -m "chore(feedback): add smoke_feedback_frame_v1 script"
```

---

### Task 10: Final verification + PR report

- [ ] **Run required commands**

```bash
PYTHONPATH=. pytest \
  tests/test_feedback_frame_schemas.py \
  tests/test_feedback_policy_loader.py \
  tests/test_feedback_builder.py \
  tests/test_feedback_scoring.py \
  tests/test_feedback_extractors.py \
  tests/test_feedback_runtime_store.py \
  -q

PYTHONPATH=. pytest tests/test_execution_dispatch_*.py tests/test_policy_*.py -q

PYTHONPATH=. python -m compileall \
  orion/feedback \
  orion/schemas/feedback_frame.py \
  services/orion-feedback-runtime \
  -q
```

Expected: all PASS

- [ ] **Code review subagent**

Use `superpowers:requesting-code-review` — dispatch subagent, fix all reported issues, re-run tests.

- [ ] **PR report**

Create `docs/superpowers/pr-reports/2026-05-25-feedback-frame-v1-pr.md` with:

- Layer 10 roadmap mapping
- Example dry-run `FeedbackFrameV1` JSON
- Absence evidence example (`dispatch_read_only` without cortex result)
- Tests run output
- Explicit Layer 11 consolidation deferred

- [ ] **Push PR**

```bash
git push -u origin feat/feedback-frame-v1
gh pr create --base feat/execution-dispatch-v1 --title "feat(feedback): Feedback Frame v1 — Layer 10 consequence capture" --body "$(cat docs/superpowers/pr-reports/2026-05-25-feedback-frame-v1-pr.md)"
```

---

## Self-review (plan author checklist)

**Spec coverage:**

| Requirement | Task |
|-------------|------|
| `OutcomeObservationV1` + `FeedbackFrameV1` | Task 1 |
| `feedback_policy.v1.yaml` + loader | Task 2 |
| Pressure extractors | Task 3 |
| Scoring | Task 4 |
| `build_feedback_frame` rules 1–9 | Task 5 |
| SQL `substrate_feedback_frames` | Task 6 |
| `orion-feedback-runtime` worker | Task 7 |
| Hub `GET /api/substrate/feedback/latest` | Task 8 |
| Smoke script | Task 9 |
| No consolidation / mutation / retry | Non-goals + worker/store design |
| Tests 1–12 from spec | Tasks 1, 5, 7 |
| Required pytest commands | Task 10 |

**Placeholder scan:** None — all steps include concrete paths, code, and commands.

**Type consistency:** `stable_feedback_frame_id(dispatch_frame_id, policy_id)` used in builder, tests, and persistence `policy_id` column maps to `feedback_policy_id` on frame.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-25-feedback-frame-v1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
