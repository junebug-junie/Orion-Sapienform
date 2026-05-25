# Consolidation Frame v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 11 — deterministic consolidation over Layers 5–10 substrate history: motifs (11a), expectations (11b), sparse tensor slices (11c), schema candidates (11d), read-only Hub surfaces (11e). No learning, no policy mutation, no bus publish, no LLM.

**Architecture:** Schemas in `orion/schemas/consolidation_frame.py`; pure logic in `orion/consolidation/`; polling service `orion-consolidation-runtime` idempotent per consolidation window; read-only Hub routes in Phase 11e. Config from `config/consolidation/consolidation_policy.v1.yaml`. **Distinct from** `orion/substrate/consolidation.py` (graph-region consolidation — do not modify).

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, psycopg2, FastAPI/uvicorn, pytest, Postgres.

**Depends on:** Layer 10 on `main` (`FeedbackFrameV1`, `substrate_feedback_frames`, port 8122). Layers 5–9 on `main` (`FieldAttentionFrameV1`, `SelfStateV1`, `ProposalFrameV1`, `PolicyDecisionFrameV1`, `ExecutionDispatchFrameV1` and their `substrate_*` tables).

**Non-goals:** Policy/proposal/attention weight mutation, habit execution, RDF writes, cortex steering, bus publish, LLM prose, neural tensor training, automatic schema promotion.

---

## Worktree and branch hygiene

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin main
git worktree add .worktrees/feat-consolidation-frame-v1 -b feat/consolidation-frame-v1 origin/main
cd .worktrees/feat-consolidation-frame-v1
```

**Rules:**

- All implementation commits happen only inside `.worktrees/feat-consolidation-frame-v1`.
- Do **not** copy changed files back to the main workspace checkout except syncing `services/orion-consolidation-runtime/.env` from `.env_example` on the operator machine (`.env` is gitignored).
- **Port:** `8123` (`CONSOLIDATION_RUNTIME_PORT`).
- **Bus:** Register schemas in `orion/schemas/registry.py` only. **Do not** add `orion/bus/channels.yaml` entries — consolidation runtime does not publish (matches Layers 8–10).
- After all tasks: run **requesting-code-review** skill via subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-25-consolidation-frame-v1-pr.md`, push branch, `gh pr create`.

---

## File structure

| Path | Role |
|------|------|
| `orion/schemas/consolidation_frame.py` | `MotifObservationV1`, `ExpectationV1`, `SparseTensorSliceV1`, `SchemaCandidateV1`, `ConsolidationFrameV1` |
| `orion/schemas/registry.py` | Register consolidation schema types |
| `config/consolidation/consolidation_policy.v1.yaml` | Window, thresholds, motif rules, tensor axes |
| `orion/consolidation/__init__.py` | Package exports |
| `orion/consolidation/policy.py` | YAML loader + `ConsolidationPolicyV1` |
| `orion/consolidation/windows.py` | Window bounds + `ConsolidationWindowData` |
| `orion/consolidation/motif.py` | Deterministic motif detectors |
| `orion/consolidation/expectation.py` | Motif → expectation mapping (11b) |
| `orion/consolidation/tensorize.py` | Sparse tensor slices (11c) |
| `orion/consolidation/schema_candidates.py` | Schema candidates (11d) |
| `orion/consolidation/builder.py` | `build_consolidation_frame` |
| `orion/consolidation/repository.py` | Read-only DB helpers (11e) |
| `services/orion-sql-db/manual_migration_consolidation_v1.sql` | Frames table (11a) |
| `services/orion-sql-db/manual_migration_consolidation_expectations_v1.sql` | Expectations table (11b) |
| `services/orion-sql-db/manual_migration_consolidation_tensor_slices_v1.sql` | Tensor slices table (11c) |
| `services/orion-sql-db/manual_migration_consolidation_schema_candidates_v1.sql` | Schema candidates table (11d) |
| `services/orion-consolidation-runtime/` | Polling runtime (mirror `orion-feedback-runtime`) |
| `services/orion-hub/scripts/substrate_consolidation_routes.py` | Read-only debug API (11e) |
| `services/orion-hub/scripts/api_routes.py` | Include consolidation router |
| `scripts/smoke_consolidation_v1.sh` | Live SQL smoke |
| `tests/test_consolidation_*.py` | Unit tests per phase |
| `docs/superpowers/pr-reports/2026-05-25-consolidation-frame-v1-pr.md` | PR report (post-implementation) |

---

## Phase 11a — Motif observations

### Task 1: Consolidation schemas + registry

**Files:**

- Create: `orion/schemas/consolidation_frame.py`
- Modify: `orion/schemas/registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consolidation_frame_schemas.py`:

```python
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.consolidation_frame import ConsolidationFrameV1, MotifObservationV1

NOW = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)


def test_motif_observation_validates() -> None:
    motif = MotifObservationV1(
        motif_id="motif:loaded_but_reliable:consolidation_policy.v1",
        motif_kind="self_state_pattern",
        label="loaded_but_reliable",
        recurrence_count=3,
        support_score=0.6,
        confidence_score=0.7,
        evidence_frame_ids=["self.state:s1"],
    )
    assert motif.motif_kind == "self_state_pattern"


def test_consolidation_frame_validates() -> None:
    frame = ConsolidationFrameV1(
        frame_id="consolidation.frame:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00:consolidation_policy.v1",
        generated_at=NOW,
        window_start=datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc),
        window_end=NOW,
        motif_observations=[
            MotifObservationV1(
                motif_id="motif:loaded_but_reliable:consolidation_policy.v1",
                motif_kind="self_state_pattern",
                label="loaded_but_reliable",
                recurrence_count=3,
                support_score=0.6,
                confidence_score=0.7,
            )
        ],
    )
    assert frame.schema_version == "consolidation.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        MotifObservationV1(
            motif_id="m1",
            motif_kind="self_state_pattern",
            label="x",
            recurrence_count=1,
            support_score=0.5,
            confidence_score=0.5,
            extra_field=True,
        )


def test_recurrence_count_min_one() -> None:
    with pytest.raises(ValidationError):
        MotifObservationV1(
            motif_id="m1",
            motif_kind="self_state_pattern",
            label="x",
            recurrence_count=0,
            support_score=0.5,
            confidence_score=0.5,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_consolidation_frame_schemas.py -v`

Expected: FAIL — `ModuleNotFoundError: orion.schemas.consolidation_frame`

- [ ] **Step 3: Write minimal implementation**

Create `orion/schemas/consolidation_frame.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MotifObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    motif_id: str

    motif_kind: Literal[
        "field_pattern",
        "attention_pattern",
        "self_state_pattern",
        "proposal_policy_pattern",
        "dispatch_feedback_pattern",
        "absence_pattern",
        "stability_pattern",
    ]

    label: str

    recurrence_count: int = Field(ge=1)
    support_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    evidence_frame_ids: list[str] = Field(default_factory=list)
    dominant_dimensions: dict[str, float] = Field(default_factory=dict)
    dominant_channels: dict[str, float] = Field(default_factory=dict)

    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None

    reasons: list[str] = Field(default_factory=list)


class ConsolidationFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["consolidation.frame.v1"] = "consolidation.frame.v1"

    frame_id: str
    generated_at: datetime

    window_start: datetime
    window_end: datetime

    consolidation_policy_id: str = "consolidation_policy.v1"

    motif_observations: list[MotifObservationV1] = Field(default_factory=list)
    dominant_motifs: list[str] = Field(default_factory=list)

    source_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
```

Register in `orion/schemas/registry.py`:

```python
from orion.schemas.consolidation_frame import ConsolidationFrameV1, MotifObservationV1
```

Add to `_REGISTRY`:

```python
    "ConsolidationFrameV1": ConsolidationFrameV1,
    "MotifObservationV1": MotifObservationV1,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_consolidation_frame_schemas.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/schemas/consolidation_frame.py orion/schemas/registry.py tests/test_consolidation_frame_schemas.py
git commit -m "feat(consolidation): add ConsolidationFrameV1 schemas and registry"
```

---

### Task 2: Consolidation policy config + loader

**Files:**

- Create: `config/consolidation/consolidation_policy.v1.yaml`
- Create: `orion/consolidation/__init__.py`
- Create: `orion/consolidation/policy.py`
- Test: `tests/test_consolidation_policy_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consolidation_policy_loader.py`:

```python
from pathlib import Path

from orion.consolidation.policy import ConsolidationPolicyV1, load_consolidation_policy

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml"


def test_loads_yaml() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert policy.schema_version == "consolidation_policy.v1"
    assert policy.policy_id == "consolidation_policy.v1"


def test_window_config() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert policy.window.lookback_minutes == 60
    assert policy.window.min_support_count == 3


def test_motif_rules_present() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    labels = {r.label for r in policy.motif_rules}
    assert "loaded_but_reliable" in labels
    assert "dry_run_feedback_loop" in labels
    assert "stable_after_dry_run" in labels


def test_tracked_dimensions() -> None:
    policy = load_consolidation_policy(POLICY_PATH)
    assert "execution_pressure" in policy.tracked_self_dimensions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_consolidation_policy_loader.py -v`

Expected: FAIL — module/config missing

- [ ] **Step 3: Write minimal implementation**

`config/consolidation/consolidation_policy.v1.yaml`:

```yaml
schema_version: consolidation_policy.v1
policy_id: consolidation_policy.v1

window:
  lookback_minutes: 60
  min_support_count: 3
  max_frames_per_source: 500

motif_thresholds:
  min_support_score: 0.20
  min_confidence_score: 0.30
  dominant_motif_min_support: 0.50

tracked_self_dimensions:
  - field_intensity
  - execution_pressure
  - reasoning_pressure
  - resource_pressure
  - reliability_pressure
  - agency_readiness
  - uncertainty
  - coherence

tracked_attention_targets:
  - node:athena
  - capability:orchestration
  - capability:graph
  - field:recent_perturbations

tracked_feedback_outcomes:
  - dry_run_only
  - prepared_only
  - completed
  - failed
  - blocked
  - absent
  - deferred
  - mixed

motif_rules:
  loaded_but_reliable:
    kind: self_state_pattern
    label: loaded_but_reliable
    conditions:
      overall_condition: loaded
      reliability_pressure_max: 0.30
      execution_pressure_min: 0.70

  attention_saturated_execution:
    kind: attention_pattern
    label: attention_saturated_execution
    conditions:
      attention_target_any:
        - node:athena
        - capability:orchestration
      min_overall_salience: 0.70

  read_only_policy_loop:
    kind: proposal_policy_pattern
    label: read_only_policy_loop
    conditions:
      approved_read_only_min: 1

  dry_run_feedback_loop:
    kind: dispatch_feedback_pattern
    label: dry_run_feedback_loop
    conditions:
      outcome_status: dry_run_only

  blocked_review_loop:
    kind: dispatch_feedback_pattern
    label: blocked_review_loop
    conditions:
      blocked_or_review_min: 1

  stable_after_dry_run:
    kind: stability_pattern
    label: stable_after_dry_run
    conditions:
      outcome_status: dry_run_only
      self_state_delta:
        allowed:
          - unchanged
```

`orion/consolidation/policy.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ConsolidationWindowConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lookback_minutes: int = 60
    min_support_count: int = 3
    max_frames_per_source: int = 500


class MotifThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_support_score: float = 0.20
    min_confidence_score: float = 0.30
    dominant_motif_min_support: float = 0.50


class MotifRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "field_pattern",
        "attention_pattern",
        "self_state_pattern",
        "proposal_policy_pattern",
        "dispatch_feedback_pattern",
        "absence_pattern",
        "stability_pattern",
    ]
    label: str
    conditions: dict[str, Any] = Field(default_factory=dict)


class ConsolidationPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["consolidation_policy.v1"] = "consolidation_policy.v1"
    policy_id: str = "consolidation_policy.v1"

    window: ConsolidationWindowConfigV1 = Field(default_factory=ConsolidationWindowConfigV1)
    motif_thresholds: MotifThresholdsV1 = Field(default_factory=MotifThresholdsV1)

    tracked_self_dimensions: list[str] = Field(default_factory=list)
    tracked_attention_targets: list[str] = Field(default_factory=list)
    tracked_feedback_outcomes: list[str] = Field(default_factory=list)

    motif_rules: dict[str, MotifRuleV1] = Field(default_factory=dict)


def load_consolidation_policy(path: str | Path) -> ConsolidationPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw_rules = data.pop("motif_rules", {}) or {}
    rules = {key: MotifRuleV1.model_validate(val) for key, val in raw_rules.items()}
    base = ConsolidationPolicyV1.model_validate(data)
    return base.model_copy(update={"motif_rules": rules})
```

`orion/consolidation/__init__.py`:

```python
from orion.consolidation.policy import ConsolidationPolicyV1, load_consolidation_policy

__all__ = ["ConsolidationPolicyV1", "load_consolidation_policy"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_consolidation_policy_loader.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/consolidation/consolidation_policy.v1.yaml orion/consolidation/ tests/test_consolidation_policy_loader.py
git commit -m "feat(consolidation): add consolidation policy config and loader"
```

---

### Task 3: Window computation + `ConsolidationWindowData`

**Files:**

- Create: `orion/consolidation/windows.py`
- Test: extend `tests/test_consolidation_builder.py` (stub file created in Task 5) or add `tests/test_consolidation_windows.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consolidation_windows.py`:

```python
from datetime import datetime, timedelta, timezone

from orion.consolidation.windows import compute_consolidation_window, stable_consolidation_frame_id

NOW = datetime(2026, 5, 25, 15, 37, tzinfo=timezone.utc)


def test_window_lookback_60_minutes() -> None:
    start, end = compute_consolidation_window(now=NOW, lookback_minutes=60)
    assert end == NOW
    assert start == NOW - timedelta(minutes=60)


def test_stable_frame_id() -> None:
    start = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)
    end = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
    fid = stable_consolidation_frame_id(
        window_start=start,
        window_end=end,
        policy_id="consolidation_policy.v1",
    )
    assert fid == "consolidation.frame:2026-05-25T14:30:00+00:00:2026-05-25T15:30:00+00:00:consolidation_policy.v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_consolidation_windows.py -v`

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`orion/consolidation/windows.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from orion.schemas.execution_dispatch_frame import ExecutionDispatchFrameV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.proposal_frame import ProposalFrameV1
from orion.schemas.self_state import SelfStateV1


@dataclass(frozen=True)
class ConsolidationWindowData:
    window_start: datetime
    window_end: datetime
    self_states: list[SelfStateV1]
    attention_frames: list[FieldAttentionFrameV1]
    proposal_frames: list[ProposalFrameV1]
    policy_frames: list[PolicyDecisionFrameV1]
    dispatch_frames: list[ExecutionDispatchFrameV1]
    feedback_frames: list[FeedbackFrameV1]


def compute_consolidation_window(
    *,
    now: datetime | None = None,
    lookback_minutes: int,
) -> tuple[datetime, datetime]:
    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = end - timedelta(minutes=lookback_minutes)
    return start, end


def stable_consolidation_frame_id(
    *,
    window_start: datetime,
    window_end: datetime,
    policy_id: str,
) -> str:
    ws = window_start.astimezone(timezone.utc).isoformat()
    we = window_end.astimezone(timezone.utc).isoformat()
    return f"consolidation.frame:{ws}:{we}:{policy_id}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_consolidation_windows.py -q`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/consolidation/windows.py tests/test_consolidation_windows.py
git commit -m "feat(consolidation): add window bounds and stable frame id"
```

---

### Task 4: Motif detection

**Files:**

- Create: `orion/consolidation/motif.py`
- Test: `tests/test_consolidation_motif_detection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_consolidation_motif_detection.py` with fixtures building minimal `SelfStateV1`, `FieldAttentionFrameV1`, `PolicyDecisionFrameV1`, `FeedbackFrameV1` lists (mirror patterns from `tests/test_feedback_builder.py`). Required tests:

```python
# test_loaded_but_reliable_detected — 3+ self states: loaded, execution_pressure>=0.7, reliability_pressure<=0.3
# test_attention_saturated_execution_detected — attention overall_salience>=0.7 + node:athena target
# test_read_only_policy_loop_detected — policy frame approved_read_only + execution_allowed false
# test_dry_run_feedback_loop_detected — feedback outcome_status dry_run_only x3
# test_blocked_review_loop_detected — policy operator_review_required true
# test_stable_after_dry_run_detected — dry_run_only + unchanged self_state_delta obs + empty absence/negative
# test_motif_scores_and_evidence — recurrence_count, support_score, confidence_score, evidence_frame_ids populated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_consolidation_motif_detection.py -v`

Expected: FAIL — `motif` module missing

- [ ] **Step 3: Write minimal implementation**

`orion/consolidation/motif.py` — core helpers and detectors:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.consolidation.policy import ConsolidationPolicyV1, MotifRuleV1
from orion.consolidation.windows import ConsolidationWindowData
from orion.schemas.consolidation_frame import MotifObservationV1
from orion.schemas.feedback_frame import FeedbackFrameV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.policy_decision_frame import PolicyDecisionFrameV1
from orion.schemas.self_state import SelfStateV1


def _dim_score(state: SelfStateV1, dimension_id: str) -> float:
    dim = state.dimensions.get(dimension_id)
    return float(dim.score) if dim is not None else 0.0


def _motif_id(label: str, policy_id: str) -> str:
    return f"motif:{label}:{policy_id}"


def _score_motif(*, match_count: int, total: int, policy: ConsolidationPolicyV1) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    support = match_count / float(total)
    confidence = min(1.0, match_count / float(max(1, policy.window.min_support_count)))
    return support, confidence


def detect_motifs(
    *,
    window: ConsolidationWindowData,
    policy: ConsolidationPolicyV1,
) -> list[MotifObservationV1]:
    motifs: list[MotifObservationV1] = []
    for _key, rule in policy.motif_rules.items():
        detector = _DETECTORS.get(rule.label)
        if detector is None:
            continue
        motif = detector(window=window, rule=rule, policy=policy)
        if motif is None:
            continue
        if motif.support_score < policy.motif_thresholds.min_support_score:
            continue
        if motif.confidence_score < policy.motif_thresholds.min_confidence_score:
            continue
        motifs.append(motif)
    return motifs


def _detect_loaded_but_reliable(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    cond = rule.conditions
    matches = [
        s
        for s in window.self_states
        if s.overall_condition == cond.get("overall_condition", "loaded")
        and _dim_score(s, "execution_pressure") >= float(cond.get("execution_pressure_min", 0.7))
        and _dim_score(s, "reliability_pressure") <= float(cond.get("reliability_pressure_max", 0.3))
    ]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.self_states) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.self_state_id for m in matches],
        dominant_dimensions={
            "execution_pressure": sum(_dim_score(m, "execution_pressure") for m in matches) / len(matches),
            "reliability_pressure": sum(_dim_score(m, "reliability_pressure") for m in matches) / len(matches),
        },
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
        reasons=["loaded_with_high_execution_low_reliability_pressure"],
    )


def _target_ids(frame: FieldAttentionFrameV1) -> set[str]:
    ids: set[str] = set()
    for bucket in (
        frame.dominant_targets,
        frame.node_targets,
        frame.capability_targets,
        frame.system_targets,
    ):
        for t in bucket:
            ids.add(f"{t.target_kind}:{t.target_id}" if ":" not in t.target_id else t.target_id)
            if t.target_kind == "node":
                ids.add(f"node:{t.target_id}")
            if t.target_kind == "capability":
                ids.add(f"capability:{t.target_id}")
    return ids


def _detect_attention_saturated_execution(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    cond = rule.conditions
    targets_any = set(cond.get("attention_target_any", []))
    min_salience = float(cond.get("min_overall_salience", 0.7))
    matches = [
        f
        for f in window.attention_frames
        if f.overall_salience >= min_salience and _target_ids(f).intersection(targets_any)
    ]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.attention_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=["attention_salience_saturated_on_execution_targets"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


def _detect_read_only_policy_loop(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    min_count = int(rule.conditions.get("approved_read_only_min", 1))
    matches = [
        p
        for p in window.policy_frames
        if not p.execution_allowed
        and sum(1 for d in p.decisions if d.decision == "approved_read_only") >= min_count
    ]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.policy_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=["read_only_policy_decisions_without_execution"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


def _detect_dry_run_feedback_loop(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    expected = rule.conditions.get("outcome_status", "dry_run_only")
    matches = [f for f in window.feedback_frames if f.outcome_status == expected]
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.feedback_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=[f"feedback_outcome_status:{expected}"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


def _detect_blocked_review_loop(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    matches: list[str] = []
    for p in window.policy_frames:
        if p.operator_review_required or p.review_required_decisions:
            matches.append(p.frame_id)
    for d in window.dispatch_frames:
        if d.blocked_candidates:
            matches.append(d.frame_id)
    for f in window.feedback_frames:
        if f.outcome_status == "blocked":
            matches.append(f.frame_id)
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(match_count=len(matches), total=max(len(matches), 1), policy=policy)
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=matches,
        reasons=["blocked_or_operator_review_signals"],
    )


def _detect_stable_after_dry_run(
    *,
    window: ConsolidationWindowData,
    rule: MotifRuleV1,
    policy: ConsolidationPolicyV1,
) -> MotifObservationV1 | None:
    allowed = set(rule.conditions.get("self_state_delta", {}).get("allowed", ["unchanged"]))
    matches = []
    for f in window.feedback_frames:
        if f.outcome_status != rule.conditions.get("outcome_status", "dry_run_only"):
            continue
        if f.absence_evidence or f.negative_evidence:
            continue
        has_unchanged = any(
            o.source_kind == "self_state_delta" and o.outcome_kind in allowed for o in f.observations
        )
        if has_unchanged:
            matches.append(f)
    if len(matches) < policy.window.min_support_count:
        return None
    support, confidence = _score_motif(
        match_count=len(matches), total=len(window.feedback_frames) or len(matches), policy=policy
    )
    return MotifObservationV1(
        motif_id=_motif_id(rule.label, policy.policy_id),
        motif_kind=rule.kind,
        label=rule.label,
        recurrence_count=len(matches),
        support_score=support,
        confidence_score=confidence,
        evidence_frame_ids=[m.frame_id for m in matches],
        reasons=["dry_run_with_unchanged_self_state"],
        first_seen_at=min(m.generated_at for m in matches),
        last_seen_at=max(m.generated_at for m in matches),
    )


_DETECTORS = {
    "loaded_but_reliable": _detect_loaded_but_reliable,
    "attention_saturated_execution": _detect_attention_saturated_execution,
    "read_only_policy_loop": _detect_read_only_policy_loop,
    "dry_run_feedback_loop": _detect_dry_run_feedback_loop,
    "blocked_review_loop": _detect_blocked_review_loop,
    "stable_after_dry_run": _detect_stable_after_dry_run,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_consolidation_motif_detection.py -q`

Expected: PASS (all motif assertions)

- [ ] **Step 5: Commit**

```bash
git add orion/consolidation/motif.py tests/test_consolidation_motif_detection.py
git commit -m "feat(consolidation): deterministic motif detection over substrate history"
```

---

### Task 5: Consolidation frame builder (11a only)

**Files:**

- Create: `orion/consolidation/builder.py`
- Test: `tests/test_consolidation_builder.py`

- [ ] **Step 1: Write the failing test**

```python
from datetime import datetime, timezone

from orion.consolidation.builder import build_consolidation_frame
from orion.consolidation.policy import load_consolidation_policy
from orion.consolidation.windows import ConsolidationWindowData, stable_consolidation_frame_id
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY = load_consolidation_policy(REPO / "config" / "consolidation" / "consolidation_policy.v1.yaml")
NOW = datetime(2026, 5, 25, 15, 30, tzinfo=timezone.utc)
START = datetime(2026, 5, 25, 14, 30, tzinfo=timezone.utc)


def test_builder_emits_source_counts_and_stable_frame_id():
    window = ConsolidationWindowData(
        window_start=START,
        window_end=NOW,
        self_states=[],
        attention_frames=[],
        proposal_frames=[],
        policy_frames=[],
        dispatch_frames=[],
        feedback_frames=[],
    )
    frame = build_consolidation_frame(window=window, policy=POLICY, generated_at=NOW)
    assert frame.frame_id == stable_consolidation_frame_id(
        window_start=START, window_end=NOW, policy_id=POLICY.policy_id
    )
    assert frame.source_counts["self_state"] == 0
    assert frame.source_counts["feedback"] == 0
    assert frame.consolidation_policy_id == "consolidation_policy.v1"
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `builder.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.consolidation.motif import detect_motifs
from orion.consolidation.policy import ConsolidationPolicyV1
from orion.consolidation.windows import ConsolidationWindowData, stable_consolidation_frame_id
from orion.schemas.consolidation_frame import ConsolidationFrameV1


def build_consolidation_frame(
    *,
    window: ConsolidationWindowData,
    policy: ConsolidationPolicyV1,
    generated_at: datetime | None = None,
) -> ConsolidationFrameV1:
    now = generated_at or datetime.now(timezone.utc)
    motifs = detect_motifs(window=window, policy=policy)
    dominant = [
        m.label
        for m in sorted(motifs, key=lambda x: x.support_score, reverse=True)
        if m.support_score >= policy.motif_thresholds.dominant_motif_min_support
    ]
    return ConsolidationFrameV1(
        frame_id=stable_consolidation_frame_id(
            window_start=window.window_start,
            window_end=window.window_end,
            policy_id=policy.policy_id,
        ),
        generated_at=now,
        window_start=window.window_start,
        window_end=window.window_end,
        consolidation_policy_id=policy.policy_id,
        motif_observations=motifs,
        dominant_motifs=dominant,
        source_counts={
            "self_state": len(window.self_states),
            "attention": len(window.attention_frames),
            "proposal": len(window.proposal_frames),
            "policy": len(window.policy_frames),
            "dispatch": len(window.dispatch_frames),
            "feedback": len(window.feedback_frames),
        },
    )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `PYTHONPATH=. pytest tests/test_consolidation_builder.py tests/test_consolidation_motif_detection.py -q`

- [ ] **Step 5: Commit**

```bash
git add orion/consolidation/builder.py tests/test_consolidation_builder.py
git commit -m "feat(consolidation): build ConsolidationFrameV1 with motifs and source counts"
```

---

### Task 6: SQL migration (11a frames)

**Files:**

- Create: `services/orion-sql-db/manual_migration_consolidation_v1.sql`

- [ ] **Step 1: Add migration SQL** (per spec — `substrate_consolidation_frames` table + indexes)

- [ ] **Step 2: Commit**

```bash
git add services/orion-sql-db/manual_migration_consolidation_v1.sql
git commit -m "feat(consolidation): add substrate_consolidation_frames migration"
```

---

### Task 7: `orion-consolidation-runtime` service

**Files:**

- Create: `services/orion-consolidation-runtime/app/{__init__.py,main.py,worker.py,store.py,settings.py}`
- Create: `services/orion-consolidation-runtime/{Dockerfile,docker-compose.yml,requirements.txt,README.md,.env_example}`
- Create: `services/orion-consolidation-runtime/.env` (copy from `.env_example` — local only, not committed)
- Test: `tests/test_consolidation_runtime_store.py`

| Setting | Value |
|---------|-------|
| Port | `8123` |
| Policy path | `/app/config/consolidation/consolidation_policy.v1.yaml` |
| Poll interval | `60` sec default |
| Idempotency key | `(window_start, window_end)` via `frame_id` |
| Worker | load frames in window → `build_consolidation_frame` → INSERT if missing |
| No bus / no mutation | worker only builds + saves |

**`.env_example`:**

```env
# orion-consolidation-runtime — Layer 11 consolidation (docker compose)
# Requires: substrate_feedback_frames, substrate_self_state, substrate_attention_frames, etc.
# Apply migrations: services/orion-sql-db/manual_migration_consolidation_v1.sql (+ later phase migrations)
PROJECT=orion-athena
SERVICE_NAME=orion-consolidation-runtime
SERVICE_VERSION=0.1.0
POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
CONSOLIDATION_POLICY_PATH=/app/config/consolidation/consolidation_policy.v1.yaml
CONSOLIDATION_POLL_INTERVAL_SEC=60.0
ENABLE_CONSOLIDATION_RUNTIME=true
LOG_LEVEL=INFO
CONSOLIDATION_RUNTIME_PORT=8123
```

**`app/settings.py`:**

```python
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-consolidation-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    consolidation_policy_path: str = Field(
        "config/consolidation/consolidation_policy.v1.yaml",
        alias="CONSOLIDATION_POLICY_PATH",
    )
    consolidation_poll_interval_sec: float = Field(
        60.0,
        alias="CONSOLIDATION_POLL_INTERVAL_SEC",
    )
    enable_consolidation_runtime: bool = Field(
        True,
        alias="ENABLE_CONSOLIDATION_RUNTIME",
    )
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
def load_consolidation_frame_for_window(self, frame_id: str) -> ConsolidationFrameV1 | None:
    # SELECT FROM substrate_consolidation_frames WHERE frame_id = :id

def load_window_data(self, window_start: datetime, window_end: datetime, max_per_source: int) -> ConsolidationWindowData:
    # SELECT substrate_self_state WHERE generated_at >= :start AND generated_at < :end ORDER BY generated_at DESC LIMIT :n
    # Same pattern for substrate_attention_frames, substrate_proposal_frames,
    # substrate_policy_decision_frames, substrate_execution_dispatch_frames, substrate_feedback_frames

def save_consolidation_frame(self, frame: ConsolidationFrameV1) -> None:
    # INSERT INTO substrate_consolidation_frames (...) ON CONFLICT (frame_id) DO NOTHING
```

**`app/worker.py` `_tick`:**

```python
def _tick(self) -> None:
    if not self._settings.enable_consolidation_runtime:
        return
    window_start, window_end = compute_consolidation_window(
        lookback_minutes=self._policy.window.lookback_minutes,
    )
    frame_id = stable_consolidation_frame_id(
        window_start=window_start,
        window_end=window_end,
        policy_id=self._policy.policy_id,
    )
    if self._store.load_consolidation_frame_for_window(frame_id) is not None:
        return
    data = self._store.load_window_data(
        window_start,
        window_end,
        self._policy.window.max_frames_per_source,
    )
    frame = build_consolidation_frame(window=data, policy=self._policy)
    self._store.save_consolidation_frame(frame)
```

**`docker-compose.yml`** — mirror feedback; port `8123`, env vars from `.env_example`.

**`Dockerfile`** — copy from `services/orion-feedback-runtime/Dockerfile`; change service path and port.

**`requirements.txt`:**

```
fastapi>=0.110
uvicorn[standard]>=0.27
pydantic>=2.6
pydantic-settings>=2.2
sqlalchemy>=2.0
psycopg2-binary>=2.9
pyyaml>=6.0
```

**`README.md`** — document port 8123, migrations, no bus publish, idempotent windows.

- [ ] **Step 4: Run store tests**

Run: `PYTHONPATH=. pytest tests/test_consolidation_runtime_store.py -q`

- [ ] **Step 5: Commit**

```bash
git add services/orion-consolidation-runtime/ tests/test_consolidation_runtime_store.py
git commit -m "feat(consolidation): add orion-consolidation-runtime polling service"
```

---

### Task 8: Smoke script (11a)

**Files:**

- Create: `scripts/smoke_consolidation_v1.sh`

- [ ] **Step 1:** Copy SQL queries from spec (feedback frames, self-state, consolidation frame) into executable script using `psql "$POSTGRES_URI"`.

- [ ] **Step 2: Commit**

```bash
git add scripts/smoke_consolidation_v1.sh
git commit -m "chore(consolidation): add smoke_consolidation_v1.sh"
```

---

## Phase 11b — Expectations

### Task 9: `ExpectationV1` schema + frame field

**Files:**

- Modify: `orion/schemas/consolidation_frame.py`
- Modify: `orion/schemas/registry.py`
- Modify: `tests/test_consolidation_frame_schemas.py`

- [ ] Add `ExpectationV1` per spec; add `expectations: list[ExpectationV1]` to `ConsolidationFrameV1`.
- [ ] Register `ExpectationV1` in registry.
- [ ] Extend schema tests; commit: `feat(consolidation): add ExpectationV1 schema`

---

### Task 10: Expectation builder

**Files:**

- Create: `orion/consolidation/expectation.py`
- Modify: `orion/consolidation/builder.py`
- Test: `tests/test_consolidation_expectations.py`

- [ ] **Implement mapping** (deterministic):

```python
_MOTIF_TO_EXPECTATION: dict[str, str] = {
    "loaded_but_reliable": "reliability_clear",
    "attention_saturated_execution": "execution_pressure_high",
    "read_only_policy_loop": "read_only_approved",
    "dry_run_feedback_loop": "dry_run_feedback",
    "blocked_review_loop": "policy_review_required",
    "stable_after_dry_run": "dry_run_feedback",
}


def build_expectations_from_motifs(
    *,
    motifs: list[MotifObservationV1],
    feedback_frames: list[FeedbackFrameV1],
    policy: ConsolidationPolicyV1,
) -> list[ExpectationV1]:
    out: list[ExpectationV1] = []
    for motif in motifs:
        kind = _MOTIF_TO_EXPECTATION.get(motif.label)
        if kind is None:
            continue
        out.append(
            ExpectationV1(
                expectation_id=f"expectation:{motif.motif_id}:{kind}",
                trigger_motif_id=motif.motif_id,
                expected_outcome_kind=kind,
                confidence_score=motif.confidence_score,
                support_count=motif.recurrence_count,
                evidence_refs=list(motif.evidence_frame_ids),
                reasons=[f"derived_from_motif:{motif.label}"],
            )
        )
    return out
```

- [ ] Wire into `build_consolidation_frame` (pass `window.feedback_frames`).
- [ ] Tests per spec assertions 1–6; commit.

---

### Task 11: Expectations migration + runtime upsert

**Files:**

- Create: `services/orion-sql-db/manual_migration_consolidation_expectations_v1.sql`
- Modify: `services/orion-consolidation-runtime/app/store.py`
- Modify: `services/orion-consolidation-runtime/app/worker.py`

- [ ] `upsert_expectations(expectations)` — `INSERT ... ON CONFLICT (expectation_id) DO UPDATE` on `substrate_expectations`.
- [ ] Worker calls upsert after saving consolidation frame.
- [ ] Test idempotent upsert in `tests/test_consolidation_runtime_store.py`; commit.

---

## Phase 11c — Sparse tensor slices

### Task 12: `SparseTensorSliceV1` + policy tensor config

**Files:**

- Modify: `orion/schemas/consolidation_frame.py`, `config/consolidation/consolidation_policy.v1.yaml`, `orion/consolidation/policy.py`

- [ ] Add `SparseTensorSliceV1` and `tensor_slices` field on frame.
- [ ] Extend policy models:

```yaml
tensor:
  enabled: true
  max_coordinates: 200

tensor_axes:
  field_attention_self:
    - time_bucket
    - self_condition
    - attention_target
    - dimension
  policy_dispatch_feedback:
    - proposal_kind
    - policy_decision
    - dispatch_status
    - feedback_outcome
  motif_condition_outcome:
    - motif
    - self_condition
    - outcome_status
```

- [ ] Commit: `feat(consolidation): add SparseTensorSliceV1 and tensor policy config`

---

### Task 13: `tensorize.py`

**Files:**

- Create: `orion/consolidation/tensorize.py`
- Modify: `orion/consolidation/builder.py`
- Test: `tests/test_consolidation_tensorize.py`

- [ ] Implement `build_sparse_tensor_slices` producing three slice kinds; cap coordinates at `policy.tensor.max_coordinates`; set `tensor_id` stable per window+kind.
- [ ] `field_attention_self`: join self_state + attention on time buckets (ISO minute), value = dimension score.
- [ ] `policy_dispatch_feedback`: join dispatch candidates + feedback outcome paths, value `1.0`.
- [ ] `motif_condition_outcome`: motif label × self_condition × feedback outcome, value = `motif.support_score`.
- [ ] Assert tests 1–8 from spec; commit.

---

### Task 14: Tensor migration + runtime persist

**Files:**

- Create: `services/orion-sql-db/manual_migration_consolidation_tensor_slices_v1.sql`
- Modify: store + worker

- [ ] `save_tensor_slices(slices, window_start, window_end)` with `ON CONFLICT DO NOTHING` on `tensor_id`.
- [ ] Commit.

---

## Phase 11d — Schema candidates

### Task 15: `SchemaCandidateV1` + `schema_candidates.py`

**Files:**

- Modify: `orion/schemas/consolidation_frame.py`
- Create: `orion/consolidation/schema_candidates.py`
- Test: `tests/test_consolidation_schema_candidates.py`

- [ ] Implement `build_schema_candidates` with three initial candidates from spec; **`promotion_status` always `candidate_only`**.
- [ ] Wire into builder; commit.

---

### Task 16: Schema candidates migration + runtime upsert

**Files:**

- Create: `services/orion-sql-db/manual_migration_consolidation_schema_candidates_v1.sql`
- Modify: store + worker

- [ ] Upsert `substrate_schema_candidates`; commit.

---

## Phase 11e — Read-only surfaces

### Task 17: `repository.py`

**Files:**

- Create: `orion/consolidation/repository.py`
- Test: `tests/test_consolidation_repository.py`

- [ ] Read-only SQL helpers: `load_recent_motifs`, `load_expectations_for_motif`, `load_schema_candidates`, `load_latest_tensor_slices` — accept `postgres_uri` or engine param; no writes.
- [ ] Commit.

---

### Task 18: Hub debug routes

**Files:**

- Create: `services/orion-hub/scripts/substrate_consolidation_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Test: `services/orion-hub/tests/test_substrate_consolidation_debug_api.py`

Routes (GET only):

```text
/api/substrate/consolidation/latest
/api/substrate/consolidation/motifs
/api/substrate/consolidation/expectations
/api/substrate/consolidation/schema-candidates
/api/substrate/consolidation/tensor-slices/latest
```

Mirror `substrate_feedback_routes.py` pattern — SQL → Pydantic validate → JSON. Register router in `api_routes.py`.

- [ ] Assert no POST routes in tests; commit.

---

## Post-implementation: code review, PR report, push

### Task 19: Verification gate

- [ ] Apply all SQL migrations on dev Postgres.
- [ ] Run full consolidation test suite:

```bash
PYTHONPATH=. pytest \
  tests/test_consolidation_frame_schemas.py \
  tests/test_consolidation_policy_loader.py \
  tests/test_consolidation_windows.py \
  tests/test_consolidation_motif_detection.py \
  tests/test_consolidation_builder.py \
  tests/test_consolidation_expectations.py \
  tests/test_consolidation_tensorize.py \
  tests/test_consolidation_schema_candidates.py \
  tests/test_consolidation_repository.py \
  tests/test_consolidation_runtime_store.py \
  services/orion-hub/tests/test_substrate_consolidation_debug_api.py \
  -q
```

Expected: all PASS

- [ ] **REQUIRED SUB-SKILL:** `requesting-code-review` — dispatch subagent with plan + diff summary; fix all blocking issues before PR.

---

### Task 20: PR report + GitHub PR

**Files:**

- Create: `docs/superpowers/pr-reports/2026-05-25-consolidation-frame-v1-pr.md`

- [ ] Write PR report (Summary, Architecture diagram, Files changed, Test plan, Non-goals verified).
- [ ] Push and open PR:

```bash
git push -u origin feat/consolidation-frame-v1
gh pr create --title "feat(consolidation): Layer 11 consolidation frame v1 (11a–11e)" --body "$(cat <<'EOF'
## Summary
- Layer 11 deterministic consolidation: motifs, expectations, sparse tensor slices, schema candidates, read-only Hub APIs.
- New `orion-consolidation-runtime` on port 8123; idempotent per window; no bus publish; no behavior mutation.

## Test plan
- [ ] `pytest tests/test_consolidation_* services/orion-hub/tests/test_substrate_consolidation_debug_api.py`
- [ ] Apply SQL migrations and run `scripts/smoke_consolidation_v1.sh`
- [ ] Confirm no `orion/bus/channels.yaml` changes

EOF
)"
```

---

## Self-review (plan author checklist)

| Spec requirement | Task |
|------------------|------|
| MotifObservationV1 + ConsolidationFrameV1 | Task 1 |
| consolidation_policy.v1.yaml + 6 motif rules | Tasks 2, 4 |
| loaded_but_reliable … stable_after_dry_run | Task 4 |
| Runtime idempotent per window | Tasks 3, 7 |
| ExpectationV1 + mappings | Tasks 9–11 |
| SparseTensorSliceV1 + 3 tensor kinds | Tasks 12–14 |
| SchemaCandidateV1 candidate_only | Tasks 15–16 |
| Read-only Hub + repository | Tasks 17–18 |
| No LLM / no policy mutation / no bus | Worktree rules + all tasks |
| SQL migrations | Tasks 6, 11, 14, 16 |
| .env sync from .env_example | Task 7 |
| Code review + PR | Tasks 19–20 |

**Placeholder scan:** No TBD steps. All motif rules have detector implementations specified.

**Type consistency:** `ConsolidationPolicyV1`, `ConsolidationWindowData`, `build_consolidation_frame`, `stable_consolidation_frame_id` used consistently across tasks.

---

## Final acceptance (Layer 11 v1)

```text
1. Recent substrate history produces consolidation frames.
2. Repeated conditions produce motif observations.
3. Motifs produce expectations.
4. Sparse tensor-shaped history slices are persisted.
5. Schema candidates emitted without automatic promotion.
6. Read-only Hub surfaces expose artifacts.
7. Nothing mutates behavior automatically.
8. No LLM required.
9. No policy/proposal/execution weight changes.
10. No RDF concept creation without explicit later review.
```
