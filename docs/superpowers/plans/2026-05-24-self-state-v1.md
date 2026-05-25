# Self-State v1 (Attention + Field → Operating Condition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 6 of the Orion cognition substrate — a deterministic, read-only self-state layer that consumes `FieldStateV1` + `FieldAttentionFrameV1` and emits `SelfStateV1` snapshots persisted to Postgres, without proposals, policy gates, LLM interpretation, or bus publish.

**Architecture:** Shared Pydantic schemas in `orion/schemas/self_state.py`; pure synthesis in `orion/self_state/` (policy loader, scoring, builder); minimal polling service `orion-self-state-runtime` idempotent per `source_attention_frame_id`; optional Hub debug route mirroring attention routes. Policy from `config/self_state/self_state_policy.v1.yaml`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, FastAPI/uvicorn, pytest, Postgres (`substrate_field_state` + `substrate_attention_frames` → `substrate_self_state`).

**Design source:** User spec “PR: Self-State v1 — Attention + Field → Orion Operating Condition” (2026-05-24).

**Depends on:** Layer 5 (`feat/attention-frame-v1`) merged to `main` — `FieldAttentionFrameV1`, `orion-attention-runtime`, `substrate_attention_frames` (verified 2026-05-24).

**Non-goals:** `ProposalFrameV1`, policy gates, cortex-exec steering, mind service, LLM interpretation, bus publish, memory consolidation, mutating field/attention state, operator notifications.

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-self-state-v1 \
  -b feat/self-state-v1 \
  main
cd .worktrees/feat-self-state-v1
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

**Rules:**
- All commits only in `.worktrees/feat-self-state-v1`.
- Never bleed files to the main checkout **except** copying `.env_example` → local `.env` for `orion-self-state-runtime` (and Hub if Hub route added).
- PR title: `PR: Self-State v1 — Attention + Field → Orion Operating Condition`.
- When done: run `requesting-code-review` subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-24-self-state-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| Layer 5 | Live on `main`: `FieldAttentionFrameV1`, `orion-attention-runtime` (port 8117), `substrate_attention_frames` |
| `FieldStateV1` | `orion/schemas/field_state.py` — `tick_id`, `node_vectors`, `capability_vectors`, `recent_perturbations` |
| Field persistence | `substrate_field_state` has **`tick_id` text PK** — use `WHERE tick_id = :tick_id` (not JSON path) |
| Attention persistence | `substrate_attention_frames` — `frame_id` PK, `source_field_tick_id` indexed |
| **Name collision** | `orion/schemas/self_study.py` has `SelfSnapshotV1`, `SelfKnowledgeItemV1` (self-study pipeline). **Do not conflate.** Layer 6 uses `SelfStateV1` / `SelfStateDimensionV1` in `orion/schemas/self_state.py`. |
| `orion/substrate/attention/` | Chat/curiosity detector — **out of scope** |
| Bus channels v1 | **No publish** — do not add channels; registry gets new schema IDs only |
| Port | Use **8118** (8115 substrate-runtime, 8116 field-digester, 8117 attention-runtime) |
| Base branch | `main` (attention-frame-v1 merged) |

### Layer 6 placement (11-layer roadmap)

```text
1. Organs → 2. Grammar → 3. Reducers → 4. Field digestion →
5. Attention → 6. Self-state (THIS PR) → 7. Proposals → 8. Policy →
9. Execution → 10. Feedback → 11. Consolidation
```

Mental model: **Self-state is the substrate’s estimate of operating condition** — not agency, not action, not personality prose.

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/self_state.py` | `SelfStateDimensionV1`, `SelfStateV1` |
| `orion/schemas/registry.py` | Register substrate self-state schemas |
| `config/self_state/self_state_policy.v1.yaml` | Deterministic dimension/threshold maps |
| `orion/self_state/__init__.py` | Package exports |
| `orion/self_state/policy.py` | Pydantic policy models + `load_self_state_policy()` |
| `orion/self_state/scoring.py` | Channel pressure collection, dimension mapping, condition helpers |
| `orion/self_state/builder.py` | `build_self_state()` orchestration |
| `services/orion-sql-db/manual_migration_self_state_v1.sql` | `substrate_self_state` DDL |
| `services/orion-self-state-runtime/app/settings.py` | Env settings |
| `services/orion-self-state-runtime/app/store.py` | Postgres load/save |
| `services/orion-self-state-runtime/app/worker.py` | Poll attention → build self-state |
| `services/orion-self-state-runtime/app/main.py` | FastAPI lifespan + `/health`, `/latest` |
| `services/orion-self-state-runtime/Dockerfile` | Copy `orion/`, `config/self_state/`, service |
| `services/orion-self-state-runtime/docker-compose.yml` | `app-net`, port 8118 |
| `services/orion-self-state-runtime/.env_example` | Operator template |
| `services/orion-self-state-runtime/.env` | Local sync from `.env_example` (gitignored) |
| `services/orion-self-state-runtime/README.md` | Runbook |
| `services/orion-self-state-runtime/requirements.txt` | Same stack as attention-runtime |
| `services/orion-hub/scripts/substrate_self_state_routes.py` | Optional `GET /api/substrate/self-state/latest` |
| `services/orion-hub/scripts/api_routes.py` | `include_router` for self-state routes |
| `services/orion-hub/tests/test_substrate_self_state_debug_api.py` | Hub route tests (mock engine) |
| `tests/test_self_state_schemas.py` | Schema validation |
| `tests/test_self_state_policy_loader.py` | Policy loader |
| `tests/test_self_state_scoring.py` | Scoring unit tests |
| `tests/test_self_state_builder.py` | Builder integration tests |
| `tests/test_self_state_runtime_store.py` | Store tests (mock SQLAlchemy) |
| `scripts/smoke_self_state_v1.sh` | Live SQL smoke |

---

# Phase 0 — Worktree + branch

### Task 0: Create isolated worktree

- [ ] **Step 1: Create worktree from main**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-self-state-v1 \
  -b feat/self-state-v1 \
  main
cd .worktrees/feat-self-state-v1
```

Expected: `git branch --show-current` → `feat/self-state-v1`

- [ ] **Step 2: Verify isolation**

```bash
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

- [ ] **Step 3: Commit**

```bash
# no code yet — skip commit until Task 1
```

---

# Phase 1 — Substrate self-state schemas

### Task 1: SelfStateDimensionV1 + SelfStateV1

**Files:**
- Create: `orion/schemas/self_state.py`
- Modify: `orion/schemas/registry.py`
- Test: `tests/test_self_state_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_self_state_schemas.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def test_self_state_dimension_v1_validates() -> None:
    d = SelfStateDimensionV1(
        dimension_id="execution_pressure",
        score=0.8,
        confidence=0.7,
        dominant_evidence=["node:athena.execution_load"],
        reasons=["execution_load elevated"],
    )
    assert d.dimension_id == "execution_pressure"


def test_self_state_v1_roundtrip() -> None:
    state = SelfStateV1(
        self_state_id="self.state:tick_a:frame_a:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        source_attention_frame_id="attention.frame:tick_a:field_attention_policy.v1",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.6,
    )
    payload = state.model_dump(mode="json")
    restored = SelfStateV1.model_validate(payload)
    assert restored.schema_version == "self.state.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        SelfStateDimensionV1(
            dimension_id="coherence",
            score=0.5,
            confidence=0.5,
            bogus=True,  # type: ignore[call-arg]
        )


def test_scores_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        SelfStateDimensionV1(
            dimension_id="coherence",
            score=1.5,
            confidence=0.5,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_self_state_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: orion.schemas.self_state`

- [ ] **Step 3: Implement schemas**

```python
# orion/schemas/self_state.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SelfStateDimensionV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension_id: Literal[
        "field_intensity",
        "coherence",
        "uncertainty",
        "agency_readiness",
        "resource_pressure",
        "execution_pressure",
        "reasoning_pressure",
        "reliability_pressure",
        "continuity_pressure",
        "introspection_pressure",
        "social_pressure",
        "policy_pressure",
    ]

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    dominant_evidence: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class SelfStateV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self.state.v1"] = "self.state.v1"

    self_state_id: str
    generated_at: datetime

    source_field_tick_id: str
    source_field_generated_at: datetime

    source_attention_frame_id: str
    source_attention_generated_at: datetime

    self_state_policy_id: str = "self_state_policy.v1"

    overall_condition: Literal[
        "quiet",
        "steady",
        "loaded",
        "strained",
        "unstable",
        "unknown",
    ] = "unknown"

    overall_intensity: float = Field(ge=0.0, le=1.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)

    dimensions: dict[str, SelfStateDimensionV1] = Field(default_factory=dict)

    dominant_attention_targets: list[str] = Field(default_factory=list)
    dominant_field_channels: dict[str, float] = Field(default_factory=dict)

    unresolved_pressures: list[str] = Field(default_factory=list)
    stabilizing_factors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    summary_labels: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Register in registry**

In `orion/schemas/registry.py` add import:

```python
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1
```

Add to `_REGISTRY` dict (near `FieldAttentionFrameV1`):

```python
    "SelfStateDimensionV1": SelfStateDimensionV1,
    "SelfStateV1": SelfStateV1,
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. pytest tests/test_self_state_schemas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/self_state.py orion/schemas/registry.py tests/test_self_state_schemas.py
git commit -m "feat(self-state): add SelfStateV1 substrate schemas"
```

---

# Phase 2 — Policy config + loader

### Task 2: YAML policy and loader

**Files:**
- Create: `config/self_state/self_state_policy.v1.yaml`
- Create: `orion/self_state/__init__.py`
- Create: `orion/self_state/policy.py`
- Test: `tests/test_self_state_policy_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_self_state_policy_loader.py
from pathlib import Path

from orion.self_state.policy import SelfStatePolicyV1, load_self_state_policy

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config" / "self_state" / "self_state_policy.v1.yaml"


def test_load_self_state_policy_v1() -> None:
    policy = load_self_state_policy(POLICY)
    assert isinstance(policy, SelfStatePolicyV1)
    assert policy.policy_id == "self_state_policy.v1"
    assert policy.channel_dimension_map["execution_load"] == "execution_pressure"
    assert policy.condition_thresholds.quiet_max == 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_self_state_policy_loader.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Add policy YAML** (exact content from spec)

```yaml
# config/self_state/self_state_policy.v1.yaml
schema_version: self_state_policy.v1
policy_id: self_state_policy.v1

condition_thresholds:
  quiet_max: 0.15
  steady_max: 0.40
  loaded_max: 0.70
  strained_max: 0.90

dimension_weights:
  field_intensity: 0.15
  coherence: 0.15
  uncertainty: 0.10
  agency_readiness: 0.10
  resource_pressure: 0.10
  execution_pressure: 0.15
  reasoning_pressure: 0.05
  reliability_pressure: 0.15
  continuity_pressure: 0.03
  introspection_pressure: 0.02

attention_target_weights:
  system: 0.30
  node: 0.35
  capability: 0.35

channel_dimension_map:
  execution_load: execution_pressure
  execution_friction: reliability_pressure
  failure_pressure: reliability_pressure
  reasoning_load: reasoning_pressure
  cpu_pressure: resource_pressure
  gpu_pressure: resource_pressure
  memory_pressure: resource_pressure
  disk_pressure: resource_pressure
  thermal_pressure: resource_pressure
  staleness: continuity_pressure
  availability: coherence
  expected_offline_suppression: coherence
  pressure: resource_pressure
  execution_pressure: execution_pressure
  reasoning_pressure: reasoning_pressure
  reliability_pressure: reliability_pressure
  confidence: coherence
  available_capacity: coherence

stabilizing_channels:
  availability: 0.50
  confidence: 0.50
  available_capacity: 0.50
  expected_offline_suppression: 0.30

unresolved_pressure_threshold: 0.60
dominant_channel_threshold: 0.25
```

- [ ] **Step 4: Implement policy loader**

```python
# orion/self_state/policy.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SelfStateConditionThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quiet_max: float = 0.15
    steady_max: float = 0.40
    loaded_max: float = 0.70
    strained_max: float = 0.90


class SelfStatePolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["self_state_policy.v1"] = "self_state_policy.v1"
    policy_id: str = "self_state_policy.v1"

    condition_thresholds: SelfStateConditionThresholdsV1 = Field(
        default_factory=SelfStateConditionThresholdsV1
    )

    dimension_weights: dict[str, float] = Field(default_factory=dict)
    attention_target_weights: dict[str, float] = Field(default_factory=dict)
    channel_dimension_map: dict[str, str] = Field(default_factory=dict)
    stabilizing_channels: dict[str, float] = Field(default_factory=dict)

    unresolved_pressure_threshold: float = 0.60
    dominant_channel_threshold: float = 0.25


def load_self_state_policy(path: str | Path) -> SelfStatePolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SelfStatePolicyV1.model_validate(data)
```

```python
# orion/self_state/__init__.py
from orion.self_state.builder import build_self_state
from orion.self_state.policy import SelfStatePolicyV1, load_self_state_policy

__all__ = [
    "SelfStatePolicyV1",
    "build_self_state",
    "load_self_state_policy",
]
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. pytest tests/test_self_state_policy_loader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config/self_state/self_state_policy.v1.yaml orion/self_state/ tests/test_self_state_policy_loader.py
git commit -m "feat(self-state): add self_state_policy.v1 loader"
```

---

# Phase 3 — Scoring helpers

### Task 3: Channel pressure collection and dimension mapping

**Files:**
- Create: `orion/self_state/scoring.py`
- Test: `tests/test_self_state_scoring.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_self_state_scoring.py
from datetime import datetime, timezone

from orion.self_state.policy import load_self_state_policy
from orion.self_state.scoring import (
    agency_readiness_score,
    clamp01,
    collect_attention_channel_pressures,
    collect_field_channel_pressures,
    condition_from_intensity,
    coherence_score,
    map_channels_to_dimensions,
    uncertainty_score,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _field_high_execution() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_scoring",
        node_vectors={"node:athena": {"execution_load": 1.0, "execution_friction": 0.0}},
        capability_vectors={},
    )


def test_high_execution_load_raises_execution_pressure() -> None:
    channels = collect_field_channel_pressures(_field_high_execution())
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("execution_pressure", 0.0) > 0.5


def test_high_execution_friction_raises_reliability_pressure() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_friction",
        node_vectors={"node:athena": {"execution_friction": 1.0}},
    )
    channels = collect_field_channel_pressures(field)
    dims = map_channels_to_dimensions(channel_pressures=channels, policy=POLICY)
    assert dims.get("reliability_pressure", 0.0) > 0.5


def test_high_failure_pressure_raises_reliability() -> None:
    field = FieldStateV1(
        generated_at=NOW,
        tick_id="tick_fail",
        node_vectors={"node:athena": {"failure_pressure": 1.0}},
    )
    dims = map_channels_to_dimensions(
        channel_pressures=collect_field_channel_pressures(field),
        policy=POLICY,
    )
    assert dims.get("reliability_pressure", 0.0) > 0.5


def test_available_capacity_improves_coherence() -> None:
    low = coherence_score(
        channel_pressures={"cpu_pressure": 0.9},
        policy=POLICY,
    )
    high = coherence_score(
        channel_pressures={"available_capacity": 1.0, "confidence": 1.0},
        policy=POLICY,
    )
    assert high > low


def test_high_salience_low_coherence_raises_uncertainty() -> None:
    u = uncertainty_score(overall_salience=1.0, coherence=0.1)
    assert u > 0.5


def test_agency_readiness_falls_with_reliability_pressure() -> None:
    high_rel = agency_readiness_score(
        coherence=0.9,
        execution_pressure=0.2,
        reliability_pressure=0.9,
        uncertainty=0.1,
        resource_pressure=0.1,
    )
    low_rel = agency_readiness_score(
        coherence=0.9,
        execution_pressure=0.2,
        reliability_pressure=0.1,
        uncertainty=0.1,
        resource_pressure=0.1,
    )
    assert low_rel > high_rel


def test_condition_thresholds() -> None:
    t = POLICY.condition_thresholds
    assert condition_from_intensity(0.10, t) == "quiet"
    assert condition_from_intensity(0.30, t) == "steady"
    assert condition_from_intensity(0.55, t) == "loaded"
    assert condition_from_intensity(0.85, t) == "strained"
    assert condition_from_intensity(0.95, t) == "unstable"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_self_state_scoring.py -v`

- [ ] **Step 3: Implement scoring.py**

```python
# orion/self_state/scoring.py
from __future__ import annotations

from orion.self_state.policy import SelfStateConditionThresholdsV1, SelfStatePolicyV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1

PRESSURE_CHANNELS = frozenset({
    "execution_load",
    "execution_friction",
    "execution_pressure",
    "failure_pressure",
    "reasoning_load",
    "reasoning_pressure",
    "reliability_pressure",
    "cpu_pressure",
    "gpu_pressure",
    "memory_pressure",
    "disk_pressure",
    "thermal_pressure",
    "staleness",
    "pressure",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def condition_from_intensity(
    intensity: float,
    thresholds: SelfStateConditionThresholdsV1,
) -> str:
    x = clamp01(intensity)
    if x <= thresholds.quiet_max:
        return "quiet"
    if x <= thresholds.steady_max:
        return "steady"
    if x <= thresholds.loaded_max:
        return "loaded"
    if x <= thresholds.strained_max:
        return "strained"
    return "unstable"


def _merge_max(dest: dict[str, float], src: dict[str, float]) -> None:
    for k, v in src.items():
        dest[k] = max(dest.get(k, 0.0), clamp01(float(v)))


def collect_field_channel_pressures(field: FieldStateV1) -> dict[str, float]:
    out: dict[str, float] = {}
    for vector in list(field.node_vectors.values()) + list(field.capability_vectors.values()):
        for channel, value in vector.items():
            if channel in PRESSURE_CHANNELS or float(value) > 0:
                out[channel] = max(out.get(channel, 0.0), clamp01(float(value)))
    # recent perturbation saturation boosts field:recent_perturbations proxy
    n = len(field.recent_perturbations)
    if n > 0:
        out["recent_perturbation_count"] = clamp01(min(1.0, n / 20.0))
    return out


def collect_attention_channel_pressures(
    attention: FieldAttentionFrameV1,
    policy: SelfStatePolicyV1,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in attention.dominant_targets:
        kind_weight = float(policy.attention_target_weights.get(target.target_kind, 0.0))
        salience = clamp01(target.salience_score)
        for channel, contrib in target.dominant_channels.items():
            key = channel
            weighted = clamp01(float(contrib) * salience * kind_weight)
            out[key] = max(out.get(key, 0.0), weighted)
    out["overall_salience"] = clamp01(attention.overall_salience)
    return out


def map_channels_to_dimensions(
    *,
    channel_pressures: dict[str, float],
    policy: SelfStatePolicyV1,
) -> dict[str, float]:
    dims: dict[str, float] = {}
    for channel, pressure in channel_pressures.items():
        dim_id = policy.channel_dimension_map.get(channel)
        if not dim_id:
            continue
        dims[dim_id] = max(dims.get(dim_id, 0.0), clamp01(pressure))
    return dims


def coherence_score(
    *,
    channel_pressures: dict[str, float],
    policy: SelfStatePolicyV1,
) -> float:
    stabilizing = 0.0
    for channel, weight in policy.stabilizing_channels.items():
        stabilizing += clamp01(channel_pressures.get(channel, 0.0)) * float(weight)
    penalty = 0.0
    for ch in ("failure_pressure", "execution_friction", "staleness", "pressure"):
        penalty += clamp01(channel_pressures.get(ch, 0.0)) * 0.25
    return clamp01(stabilizing - penalty)


def uncertainty_score(*, overall_salience: float, coherence: float) -> float:
    return clamp01(clamp01(overall_salience) * (1.0 - clamp01(coherence)))


def agency_readiness_score(
    *,
    coherence: float,
    execution_pressure: float,
    reliability_pressure: float,
    uncertainty: float,
    resource_pressure: float,
) -> float:
    base = clamp01(coherence)
    base -= execution_pressure * 0.25
    base -= reliability_pressure * 0.35
    base -= uncertainty * 0.25
    base -= resource_pressure * 0.15
    return clamp01(base)


def field_intensity_score(
    *,
    overall_salience: float,
    recent_perturbation_saturation: float,
) -> float:
    return clamp01(0.6 * overall_salience + 0.4 * recent_perturbation_saturation)


def weighted_overall_intensity(
    dimension_scores: dict[str, float],
    policy: SelfStatePolicyV1,
) -> float:
    total_w = 0.0
    acc = 0.0
    for dim_id, weight in policy.dimension_weights.items():
        w = float(weight)
        if w <= 0:
            continue
        total_w += w
        acc += w * clamp01(dimension_scores.get(dim_id, 0.0))
    if total_w <= 0:
        return 0.0
    return clamp01(acc / total_w)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `PYTHONPATH=. pytest tests/test_self_state_scoring.py -v`

- [ ] **Step 5: Commit**

```bash
git add orion/self_state/scoring.py tests/test_self_state_scoring.py
git commit -m "feat(self-state): add deterministic scoring helpers"
```

---

# Phase 4 — Builder

### Task 4: build_self_state orchestration

**Files:**
- Create: `orion/self_state/builder.py`
- Modify: `orion/self_state/__init__.py` (export already wired)
- Test: `tests/test_self_state_builder.py`

- [ ] **Step 1: Write failing builder tests**

```python
# tests/test_self_state_builder.py
from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
ATTENTION_POLICY = load_attention_policy(REPO / "config" / "attention" / "field_attention_policy.v1.yaml")
SELF_POLICY = load_self_state_policy(REPO / "config" / "self_state" / "self_state_policy.v1.yaml")
NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _synthetic_field() -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="tick_exec_attention",
        node_vectors={
            "node:athena": {
                "execution_load": 1.0,
                "reasoning_load": 0.35,
                "availability": 1.0,
            },
        },
        capability_vectors={
            "capability:orchestration": {
                "execution_pressure": 1.0,
                "reliability_pressure": 0.0,
            }
        },
        recent_perturbations=["state_delta:exec_1", "state_delta:exec_2"],
    )


def test_builder_references_sources() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert state.source_field_tick_id == field.tick_id
    assert state.source_attention_frame_id == attention.frame_id


def test_execution_pressure_nonzero() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "execution_pressure" in state.dimensions
    assert state.dimensions["execution_pressure"].score > 0.0


def test_agency_readiness_present() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "agency_readiness" in state.dimensions


def test_dominant_attention_targets() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    ids = set(state.dominant_attention_targets)
    assert "node:athena" in ids or "capability:orchestration" in ids


def test_summary_labels_execution_loaded() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    state = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert "execution_loaded" in state.summary_labels


def test_self_state_id_stable() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    a = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    b = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW)
    assert a.self_state_id == b.self_state_id


def test_no_action_outputs() -> None:
    field = _synthetic_field()
    attention = build_attention_frame(field=field, policy=ATTENTION_POLICY, now=NOW)
    payload = build_self_state(field=field, attention=attention, policy=SELF_POLICY, now=NOW).model_dump()
    forbidden = ("proposal", "action", "policy_gate", "cortex", "selected_action")
    for key in payload:
        assert not any(f in key.lower() for f in forbidden)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_self_state_builder.py -v`

- [ ] **Step 3: Implement builder.py**

```python
# orion/self_state/builder.py
from __future__ import annotations

from datetime import datetime, timezone

from orion.self_state.policy import SelfStatePolicyV1
from orion.self_state.scoring import (
    agency_readiness_score,
    clamp01,
    collect_attention_channel_pressures,
    collect_field_channel_pressures,
    condition_from_intensity,
    coherence_score,
    field_intensity_score,
    map_channels_to_dimensions,
    uncertainty_score,
    weighted_overall_intensity,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1
from orion.schemas.field_state import FieldStateV1
from orion.schemas.self_state import SelfStateDimensionV1, SelfStateV1


def stable_self_state_id(
    *,
    source_field_tick_id: str,
    source_attention_frame_id: str,
    policy_id: str,
) -> str:
    return f"self.state:{source_field_tick_id}:{source_attention_frame_id}:{policy_id}"


def _emit_summary_labels(
    *,
    dimension_scores: dict[str, float],
    overall_condition: str,
    policy: SelfStatePolicyV1,
) -> list[str]:
    labels: list[str] = []
    if dimension_scores.get("execution_pressure", 0.0) >= 0.5:
        labels.append("execution_loaded")
    if dimension_scores.get("resource_pressure", 0.0) >= 0.5:
        labels.append("resource_pressurized")
    if dimension_scores.get("reliability_pressure", 0.0) < 0.3:
        labels.append("reliability_clear")
    if dimension_scores.get("field_intensity", 0.0) >= 0.5:
        labels.append("field_active")
    if overall_condition in ("loaded", "strained", "unstable"):
        labels.append("orchestration_pressurized")
    if policy.unresolved_pressure_threshold and dimension_scores.get("execution_pressure", 0) >= policy.unresolved_pressure_threshold:
        pass  # unresolved list handled separately
    return sorted(set(labels))


def build_self_state(
    *,
    field: FieldStateV1,
    attention: FieldAttentionFrameV1,
    policy: SelfStatePolicyV1,
    previous_self_state: SelfStateV1 | None = None,
    now: datetime | None = None,
) -> SelfStateV1:
    generated_at = now or datetime.now(timezone.utc)

    if attention.source_field_tick_id != field.tick_id:
        warnings = [
            f"attention_source_tick_mismatch:{attention.source_field_tick_id}!={field.tick_id}"
        ]
    else:
        warnings = []

    field_channels = collect_field_channel_pressures(field)
    attn_channels = collect_attention_channel_pressures(attention, policy)
    merged_channels: dict[str, float] = dict(field_channels)
    for k, v in attn_channels.items():
        merged_channels[k] = max(merged_channels.get(k, 0.0), v)

    mapped = map_channels_to_dimensions(channel_pressures=merged_channels, policy=policy)

    coherence = coherence_score(channel_pressures=merged_channels, policy=policy)
    uncertainty = uncertainty_score(
        overall_salience=attention.overall_salience,
        coherence=coherence,
    )
    field_intensity = field_intensity_score(
        overall_salience=attention.overall_salience,
        recent_perturbation_saturation=merged_channels.get("recent_perturbation_count", 0.0),
    )

    execution_p = mapped.get("execution_pressure", 0.0)
    reliability_p = mapped.get("reliability_pressure", 0.0)
    resource_p = mapped.get("resource_pressure", 0.0)
    reasoning_p = mapped.get("reasoning_pressure", 0.0)
    continuity_p = mapped.get("continuity_pressure", 0.0)

    agency = agency_readiness_score(
        coherence=coherence,
        execution_pressure=execution_p,
        reliability_pressure=reliability_p,
        uncertainty=uncertainty,
        resource_pressure=resource_p,
    )

    dimension_scores: dict[str, float] = {
        "field_intensity": field_intensity,
        "coherence": coherence,
        "uncertainty": uncertainty,
        "agency_readiness": agency,
        "resource_pressure": resource_p,
        "execution_pressure": execution_p,
        "reasoning_pressure": reasoning_p,
        "reliability_pressure": reliability_p,
        "continuity_pressure": continuity_p,
        "introspection_pressure": 0.0,
        "social_pressure": 0.0,
        "policy_pressure": 0.0,
    }
    dimension_scores.update(mapped)

    overall_intensity = weighted_overall_intensity(dimension_scores, policy)
    overall_condition = condition_from_intensity(
        overall_intensity, policy.condition_thresholds
    )

    dominant_targets = [t.target_id for t in attention.dominant_targets[:5]]
    evidence_density = clamp01(len(dominant_targets) / 5.0)
    overall_confidence = clamp01(0.5 + 0.5 * evidence_density)

    dominant_field_channels = {
        ch: v
        for ch, v in sorted(field_channels.items(), key=lambda kv: kv[1], reverse=True)
        if v >= policy.dominant_channel_threshold
    }[:8]

    unresolved: list[str] = []
    stabilizing: list[str] = []
    for ch, v in merged_channels.items():
        if ch in policy.stabilizing_channels and v >= 0.3:
            stabilizing.append(f"{ch}={v:.2f}")
        dim = policy.channel_dimension_map.get(ch)
        if dim and v >= policy.unresolved_pressure_threshold:
            unresolved.append(f"{ch}→{dim}")

    dimensions: dict[str, SelfStateDimensionV1] = {}
    for dim_id, score in dimension_scores.items():
        if dim_id not in policy.dimension_weights and dim_id not in (
            "introspection_pressure",
            "social_pressure",
            "policy_pressure",
        ):
            continue
        dimensions[dim_id] = SelfStateDimensionV1(
            dimension_id=dim_id,  # type: ignore[arg-type]
            score=clamp01(score),
            confidence=overall_confidence,
            dominant_evidence=[f"channel:{c}" for c, _ in dominant_field_channels.items()][:3],
            reasons=[f"{dim_id} derived from field+attention channels"],
        )

    summary_labels = _emit_summary_labels(
        dimension_scores=dimension_scores,
        overall_condition=overall_condition,
        policy=policy,
    )

    return SelfStateV1(
        self_state_id=stable_self_state_id(
            source_field_tick_id=field.tick_id,
            source_attention_frame_id=attention.frame_id,
            policy_id=policy.policy_id,
        ),
        generated_at=generated_at,
        source_field_tick_id=field.tick_id,
        source_field_generated_at=field.generated_at,
        source_attention_frame_id=attention.frame_id,
        source_attention_generated_at=attention.generated_at,
        self_state_policy_id=policy.policy_id,
        overall_condition=overall_condition,  # type: ignore[arg-type]
        overall_intensity=overall_intensity,
        overall_confidence=overall_confidence,
        dimensions=dimensions,
        dominant_attention_targets=dominant_targets,
        dominant_field_channels=dominant_field_channels,
        unresolved_pressures=unresolved,
        stabilizing_factors=stabilizing,
        warnings=warnings,
        summary_labels=summary_labels,
    )
```

- [ ] **Step 4: Run builder + scoring + schema tests**

Run:
```bash
PYTHONPATH=. pytest \
  tests/test_self_state_schemas.py \
  tests/test_self_state_policy_loader.py \
  tests/test_self_state_scoring.py \
  tests/test_self_state_builder.py \
  -q
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/self_state/builder.py tests/test_self_state_builder.py
git commit -m "feat(self-state): add build_self_state from field+attention"
```

---

# Phase 5 — SQL migration

### Task 5: substrate_self_state table

**Files:**
- Create: `services/orion-sql-db/manual_migration_self_state_v1.sql`

- [ ] **Step 1: Add migration SQL** (from spec)

```sql
-- Self-state v1 (apply before orion-self-state-runtime)
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_self_state_v1.sql

create table if not exists substrate_self_state (
    self_state_id text primary key,
    source_field_tick_id text not null,
    source_attention_frame_id text not null,
    generated_at timestamptz not null,
    policy_id text not null,
    self_state_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_self_state_generated_at
    on substrate_self_state (generated_at desc);

create index if not exists idx_substrate_self_state_source_field_tick
    on substrate_self_state (source_field_tick_id);

create index if not exists idx_substrate_self_state_source_attention_frame
    on substrate_self_state (source_attention_frame_id);
```

- [ ] **Step 2: Apply locally (operator)**

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney \
  -f - < services/orion-sql-db/manual_migration_self_state_v1.sql
```

Expected: `CREATE TABLE` / `CREATE INDEX` (idempotent)

- [ ] **Step 3: Commit**

```bash
git add services/orion-sql-db/manual_migration_self_state_v1.sql
git commit -m "feat(self-state): add substrate_self_state migration"
```

---

# Phase 6 — orion-self-state-runtime service

### Task 6: Settings, store, worker, main

**Files:**
- Create: `services/orion-self-state-runtime/app/settings.py`
- Create: `services/orion-self-state-runtime/app/store.py`
- Create: `services/orion-self-state-runtime/app/worker.py`
- Create: `services/orion-self-state-runtime/app/main.py`
- Create: `services/orion-self-state-runtime/app/__init__.py`
- Test: `tests/test_self_state_runtime_store.py`

Mirror `services/orion-attention-runtime/` — poll **latest attention frame**, load field by `source_field_tick_id`, noop if self-state exists for that frame.

- [ ] **Step 1: Write store tests**

```python
# tests/test_self_state_runtime_store.py
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
SVC = REPO / "services" / "orion-self-state-runtime"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SVC))

from app.store import SelfStateRuntimeStore  # noqa: E402
from orion.schemas.self_state import SelfStateV1  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _state() -> SelfStateV1:
    return SelfStateV1(
        self_state_id="self.state:tick_a:frame_a:self_state_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        source_attention_frame_id="frame_a",
        source_attention_generated_at=NOW,
        overall_intensity=0.5,
        overall_confidence=0.6,
    )


def test_save_and_load_latest(monkeypatch) -> None:
    payload = _state().model_dump(mode="json")
    store = SelfStateRuntimeStore("postgresql://test:test@localhost/test")
    fake_engine = MagicMock()
    conn = MagicMock()
    fake_engine.connect.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
    fake_engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    fake_engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    saved = []

    def execute_side_effect(stmt, params=None):
        sql = str(stmt)
        result = MagicMock()
        if "INSERT INTO substrate_self_state" in sql:
            saved.append(params)
            result.rowcount = 1
        elif "source_attention_frame_id" in sql:
            result.mappings.return_value.first.return_value = None
        else:
            result.mappings.return_value.first.return_value = {"self_state_json": payload}
        return result

    conn.execute.side_effect = execute_side_effect
    monkeypatch.setattr(store, "_engine", fake_engine)

    store.save_self_state(_state())
    loaded = store.load_latest_self_state()
    assert loaded is not None
    assert loaded.self_state_id == "self.state:tick_a:frame_a:self_state_policy.v1"
```

- [ ] **Step 2: Implement settings.py**

```python
# services/orion-self-state-runtime/app/settings.py
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    service_name: str = Field("orion-self-state-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")

    self_state_policy_path: str = Field(
        "config/self_state/self_state_policy.v1.yaml",
        alias="SELF_STATE_POLICY_PATH",
    )
    self_state_poll_interval_sec: float = Field(2.0, alias="SELF_STATE_POLL_INTERVAL_SEC")
    enable_self_state_runtime: bool = Field(True, alias="ENABLE_SELF_STATE_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3: Implement store.py**

Use `WHERE tick_id = :tick_id` on `substrate_field_state` (column exists per `manual_migration_field_digester_v1.sql`).

Key SQL for `load_field_for_tick`:

```sql
SELECT field_json FROM substrate_field_state
WHERE tick_id = :tick_id
ORDER BY generated_at DESC
LIMIT 1
```

Key SQL for `load_self_state_for_attention_frame`:

```sql
SELECT self_state_json FROM substrate_self_state
WHERE source_attention_frame_id = :frame_id
ORDER BY generated_at DESC
LIMIT 1
```

`save_self_state`: `ON CONFLICT (self_state_id) DO UPDATE SET self_state_json = EXCLUDED.self_state_json` — mirror attention store insert pattern.

- [ ] **Step 4: Implement worker.py**

```python
# services/orion-self-state-runtime/app/worker.py — poll logic
def _tick(self) -> None:
    if not self._settings.enable_self_state_runtime:
        return
    attention = self._store.load_latest_attention_frame()
    if attention is None:
        return
    if self._store.load_self_state_for_attention_frame(attention.frame_id) is not None:
        return
    field = self._store.load_field_for_tick(attention.source_field_tick_id)
    if field is None:
        return
    previous = self._store.load_latest_self_state()
    state = build_self_state(
        field=field,
        attention=attention,
        policy=self._policy,
        previous_self_state=previous,
    )
    self._store.save_self_state(state)
```

- [ ] **Step 5: Implement main.py** — copy structure from `services/orion-attention-runtime/app/main.py`; swap store/worker types; port **8118** only in compose/Dockerfile.

- [ ] **Step 6: Run store tests**

Run: `PYTHONPATH=. pytest tests/test_self_state_runtime_store.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add services/orion-self-state-runtime/app/ tests/test_self_state_runtime_store.py
git commit -m "feat(self-state): add orion-self-state-runtime worker and store"
```

---

### Task 7: Docker, compose, env, README

**Files:**
- Create: `services/orion-self-state-runtime/Dockerfile`
- Create: `services/orion-self-state-runtime/docker-compose.yml`
- Create: `services/orion-self-state-runtime/.env_example`
- Create: `services/orion-self-state-runtime/requirements.txt`
- Create: `services/orion-self-state-runtime/README.md`
- Copy: `.env_example` → `.env` (local only, gitignored)

- [ ] **Step 1: requirements.txt** — identical to attention-runtime:

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
PyYAML==6.0.2
```

- [ ] **Step 2: Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY services/orion-self-state-runtime/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY orion /app/orion
COPY config/self_state /app/config/self_state
COPY services/orion-self-state-runtime /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8118"]
```

- [ ] **Step 3: docker-compose.yml**

```yaml
services:
  self-state-runtime:
    build:
      context: ../..
      dockerfile: services/orion-self-state-runtime/Dockerfile
    container_name: ${PROJECT}-self-state-runtime
    restart: unless-stopped
    networks:
      - app-net
    ports:
      - "${SELF_STATE_RUNTIME_PORT:-8118}:8118"
    environment:
      - PROJECT=${PROJECT}
      - SERVICE_NAME=${SERVICE_NAME:-orion-self-state-runtime}
      - SERVICE_VERSION=${SERVICE_VERSION:-0.1.0}
      - POSTGRES_URI=${POSTGRES_URI}
      - SELF_STATE_POLICY_PATH=${SELF_STATE_POLICY_PATH:-/app/config/self_state/self_state_policy.v1.yaml}
      - SELF_STATE_POLL_INTERVAL_SEC=${SELF_STATE_POLL_INTERVAL_SEC:-2.0}
      - ENABLE_SELF_STATE_RUNTIME=${ENABLE_SELF_STATE_RUNTIME:-true}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}

networks:
  app-net:
    external: true
```

- [ ] **Step 4: .env_example**

```bash
SERVICE_NAME=orion-self-state-runtime
SERVICE_VERSION=0.1.0
POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
SELF_STATE_POLICY_PATH=/app/config/self_state/self_state_policy.v1.yaml
SELF_STATE_POLL_INTERVAL_SEC=2.0
ENABLE_SELF_STATE_RUNTIME=true
LOG_LEVEL=INFO
SELF_STATE_RUNTIME_PORT=8118
PROJECT=orion-athena
```

- [ ] **Step 5: Sync local .env**

```bash
cp services/orion-self-state-runtime/.env_example services/orion-self-state-runtime/.env
```

- [ ] **Step 6: README** — document Layer 6, deps (field-digester + attention-runtime), migration apply, curl `/health` and `/latest`.

- [ ] **Step 7: Commit**

```bash
git add services/orion-self-state-runtime/
git commit -m "feat(self-state): add docker compose and operator docs"
```

---

# Phase 7 — Hub debug route (optional, low risk)

### Task 8: GET /api/substrate/self-state/latest

**Files:**
- Create: `services/orion-hub/scripts/substrate_self_state_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Create: `services/orion-hub/tests/test_substrate_self_state_debug_api.py`

Mirror `substrate_attention_routes.py` — prefix `/api/substrate/self-state`, read-only, 404 if empty.

- [ ] **Step 1: Implement route** (copy attention pattern, swap table/json column names)

- [ ] **Step 2: Register router** in `api_routes.py`:

```python
from .substrate_self_state_routes import router as substrate_self_state_router
router.include_router(substrate_self_state_router)
```

- [ ] **Step 3: Hub tests** — mock `_engine`, assert 200 + schema_version `self.state.v1`

Run: `PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_substrate_self_state_debug_api.py -q`

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/scripts/substrate_self_state_routes.py \
  services/orion-hub/scripts/api_routes.py \
  services/orion-hub/tests/test_substrate_self_state_debug_api.py
git commit -m "feat(self-state): add hub debug route for latest self-state"
```

---

# Phase 8 — Smoke script

### Task 9: scripts/smoke_self_state_v1.sh

**Files:**
- Create: `scripts/smoke_self_state_v1.sh`

- [ ] **Step 1: Add executable script** (exact SQL from spec; `chmod +x`)

- [ ] **Step 2: Dry-run locally**

```bash
./scripts/smoke_self_state_v1.sh
```

Expected: prints latest attention row + latest self-state row (or empty self-state before runtime runs)

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_self_state_v1.sh
git commit -m "chore(self-state): add smoke_self_state_v1.sh"
```

---

# Phase 9 — Verification gate

### Task 10: Full test + regression suite

- [ ] **Step 1: Self-state unit tests**

```bash
PYTHONPATH=. pytest \
  tests/test_self_state_schemas.py \
  tests/test_self_state_policy_loader.py \
  tests/test_self_state_scoring.py \
  tests/test_self_state_builder.py \
  tests/test_self_state_runtime_store.py \
  -q
```

Expected: all PASS

- [ ] **Step 2: Attention regression**

```bash
PYTHONPATH=. pytest tests/test_attention_*.py -q
```

Expected: PASS

- [ ] **Step 3: Field regression**

```bash
PYTHONPATH=.:services/orion-field-digester pytest \
  tests/test_field_topology_reconciliation.py \
  tests/test_field_execution_perturbations.py \
  tests/test_field_state_schemas.py \
  tests/test_field_digestion_rules.py \
  -q
```

Expected: PASS

- [ ] **Step 4: Compile**

```bash
PYTHONPATH=. python -m compileall \
  orion/self_state \
  orion/schemas/self_state.py \
  services/orion-self-state-runtime
```

Expected: no errors

- [ ] **Step 5: Optional runtime smoke**

Start stack, wait for poll, run `./scripts/smoke_self_state_v1.sh` — capture `overall_condition`, `execution_pressure`, `agency_readiness` for PR report.

---

# Phase 10 — Code review, PR report, push

### Task 11: Subagent code review + PR

- [ ] **Step 1: Run `requesting-code-review` subagent** on full diff vs `main`; fix all findings in worktree.

- [ ] **Step 2: Write PR report** — `docs/superpowers/pr-reports/2026-05-24-self-state-v1-pr.md` including:
  - Layer 6 mapping in 11-layer roadmap
  - Example `SelfStateV1` JSON from synthetic field+attention (paste from test or builder REPL)
  - Live/dry-run evidence: attention frame + self-state dimensions
  - Explicit non-goals (no proposal/policy/cortex/LLM)
  - Tests run with command output
  - Gaps deferred to Layer 7 (`ProposalFrameV1`)

- [ ] **Step 3: Push and open PR**

```bash
cd /mnt/scripts/Orion-Sapienform/.worktrees/feat-self-state-v1
git push -u origin feat/self-state-v1
gh pr create --title "PR: Self-State v1 — Attention + Field → Orion Operating Condition" --body "$(cat <<'EOF'
## Summary
- Layer 6: deterministic SelfStateV1 from FieldStateV1 + FieldAttentionFrameV1
- orion-self-state-runtime persists to substrate_self_state (read-only w.r.t. field/attention)
- No proposals, policy gates, cortex steering, or LLM interpretation

## Test plan
- [ ] pytest self-state + attention + field regressions
- [ ] manual_migration applied
- [ ] smoke_self_state_v1.sh after runtime poll

EOF
)"
```

---

## Self-review checklist

| Spec requirement | Task |
|------------------|------|
| SelfStateDimensionV1 + SelfStateV1 | Task 1 |
| Registry | Task 1 |
| Policy YAML + loader | Task 2 |
| scoring.py formulas | Task 3 |
| builder.py stable id + labels | Task 4 |
| migration substrate_self_state | Task 5 |
| runtime poll + idempotent | Task 6–7 |
| Hub GET latest | Task 8 |
| smoke script | Task 9 |
| tests per spec | Tasks 1–6, 10 |
| no bus channels | Preflight — skip channels.yaml |
| .env sync | Task 7 |

**Placeholder scan:** No TBD steps; all code blocks are complete starter implementations.

**Type consistency:** `self_state_policy_id` on `SelfStateV1`; `policy_id` column in SQL; `stable_self_state_id` uses same `policy_id` string as attention uses for frames.

---

## Example synthetic SelfStateV1 (for PR report)

After Task 4 tests pass, capture:

```bash
cd .worktrees/feat-self-state-v1
PYTHONPATH=. python -c "
from datetime import datetime, timezone
from pathlib import Path
from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.self_state.builder import build_self_state
from orion.self_state.policy import load_self_state_policy
from orion.schemas.field_state import FieldStateV1
import json
REPO = Path('.')
NOW = datetime(2026,5,24,12,0,tzinfo=timezone.utc)
field = FieldStateV1(generated_at=NOW, tick_id='tick_exec_attention',
  node_vectors={'node:athena': {'execution_load': 1.0, 'reasoning_load': 0.35, 'availability': 1.0}},
  capability_vectors={'capability:orchestration': {'execution_pressure': 1.0}},
  recent_perturbations=['d1','d2'])
attn = build_attention_frame(field=field, policy=load_attention_policy(REPO/'config/attention/field_attention_policy.v1.yaml'), now=NOW)
state = build_self_state(field=field, attention=attn, policy=load_self_state_policy(REPO/'config/self_state/self_state_policy.v1.yaml'), now=NOW)
print(json.dumps(state.model_dump(mode='json'), indent=2))
"
```

Paste output into PR report as evidence.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-self-state-v1.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. **REQUIRED SUB-SKILL:** superpowers:subagent-driven-development

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints. **REQUIRED SUB-SKILL:** superpowers:executing-plans

**Which approach?**
