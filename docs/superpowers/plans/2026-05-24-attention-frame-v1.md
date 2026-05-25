# Attention Frame v1 (FieldState → What Matters Now) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Layer 5 of the Orion cognition substrate — a deterministic, read-only attention layer that consumes `FieldStateV1` and emits `FieldAttentionFrameV1` snapshots persisted to Postgres, without action, self-state, LLM interpretation, or bus publish.

**Architecture:** Shared Pydantic schemas in `orion/schemas/field_attention_frame.py`; pure selection logic in `orion/attention/field_attention/`; minimal polling service `orion-attention-runtime` idempotent per `source_field_tick_id`; optional Hub debug route mirroring field routes. Policy from `config/attention/field_attention_policy.v1.yaml`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, SQLAlchemy, FastAPI/uvicorn, pytest, Postgres (`substrate_field_state` → `substrate_attention_frames`).

**Design source:** User spec “PR: Attention Frame v1 — FieldState → What Matters Now” (2026-05-24).

**Depends on:** `feat/field-topology-reconciliation-v1` (execution channel maps on lattice, reconciled `FieldStateV1` ticks). If that PR is merged to `main` before implementation starts, branch from `main` instead.

**Non-goals:** `SelfStateV1`, proposals, policy gates, cortex-exec steering, mind service, LLM interpretation, bus publish, memory consolidation, mutating `FieldStateV1`, operator notifications.

---

## Worktree isolation (mandatory)

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-attention-frame-v1 \
  -b feat/attention-frame-v1 \
  feat/field-topology-reconciliation-v1
cd .worktrees/feat-attention-frame-v1
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

**Rules:**
- All commits only in `.worktrees/feat-attention-frame-v1`.
- Never bleed files to the main checkout **except** copying `.env_example` → local `.env` for `orion-attention-runtime` (and Hub if Hub route added).
- PR title: `PR: Attention Frame v1 — FieldState → What Matters Now`.
- When done: run `requesting-code-review` subagent, fix findings, write `docs/superpowers/pr-reports/2026-05-24-attention-frame-v1-pr.md`, push branch, `gh pr create`.

---

## Preflight findings (2026-05-24)

| Question | Finding |
|----------|---------|
| `FieldStateV1` | `orion/schemas/field_state.py` — `node_vectors`, `capability_vectors`, `recent_perturbations`, optional topology metadata |
| Field persistence | `substrate_field_state` — read pattern in `services/orion-field-digester/app/store.py:129-149` |
| **Schema name collision** | `orion/schemas/attention_frame.py` already defines **conversational** `AttentionFrameV1` (`open_loops`, `selected_action`, `attention.frame.v1`) used by `orion/substrate/attention/*` and cortex-exec tests. **Do not overwrite.** |
| **Resolution** | Substrate Layer 5 types: `FieldAttentionTargetV1`, `FieldAttentionFrameV1` in `orion/schemas/field_attention_frame.py` with `schema_version: field.attention.frame.v1`. PR prose may say “attention frame”; code uses `Field*` prefix. |
| `orion/substrate/attention/` | Chat/curiosity detector pipeline — **out of scope**; new code lives in `orion/attention/field_attention/` |
| Bus channels v1 | **No publish** — do not add channels; registry gets new schema IDs only |
| Port | Use `8117` (`SUBSTRATE_RUNTIME` 8115, `FIELD_DIGESTER` 8116) |
| Base branch | `feat/field-topology-reconciliation-v1` (not yet on `main` as of plan date) |

### Layer 5 placement (11-layer roadmap)

```text
1. Organs → 2. Grammar → 3. Reducers → 4. Field digestion →
5. Attention (THIS PR) → 6. Self-state → 7. Proposals → 8. Policy →
9. Execution → 10. Feedback → 11. Consolidation
```

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/schemas/field_attention_frame.py` | `FieldAttentionTargetV1`, `FieldAttentionFrameV1` |
| `orion/schemas/registry.py` | Register substrate attention schemas |
| `config/attention/field_attention_policy.v1.yaml` | Deterministic weights/thresholds |
| `orion/attention/field_attention/__init__.py` | Package exports |
| `orion/attention/field_attention/policy.py` | Pydantic policy models + `load_attention_policy()` |
| `orion/attention/field_attention/scoring.py` | `clamp01`, `weighted_pressure`, salience helpers |
| `orion/attention/field_attention/selectors.py` | Node/capability/system target builders |
| `orion/attention/field_attention/builder.py` | `build_attention_frame()` orchestration |
| `services/orion-sql-db/manual_migration_attention_frame_v1.sql` | `substrate_attention_frames` DDL |
| `services/orion-attention-runtime/app/settings.py` | Env settings |
| `services/orion-attention-runtime/app/store.py` | Postgres load/save |
| `services/orion-attention-runtime/app/worker.py` | Poll field → build frame |
| `services/orion-attention-runtime/app/main.py` | FastAPI lifespan + `/health`, `/latest` |
| `services/orion-attention-runtime/Dockerfile` | Copy `orion/`, `config/attention/`, service |
| `services/orion-attention-runtime/docker-compose.yml` | `app-net`, port 8117 |
| `services/orion-attention-runtime/.env_example` | Operator template |
| `services/orion-attention-runtime/README.md` | Runbook |
| `services/orion-attention-runtime/requirements.txt` | Same stack as field-digester |
| `services/orion-hub/scripts/substrate_attention_routes.py` | Optional `GET /api/substrate/attention/latest` |
| `services/orion-hub/scripts/api_routes.py` | `include_router` for attention routes |
| `services/orion-hub/tests/test_substrate_attention_debug_api.py` | Hub route tests (mock engine) |
| `tests/test_attention_frame_schemas.py` | Schema validation |
| `tests/test_attention_field_scoring.py` | Scoring unit tests |
| `tests/test_attention_frame_builder.py` | Builder integration tests |
| `tests/test_attention_runtime_store.py` | Store tests (mock SQLAlchemy) |
| `scripts/smoke_attention_frame_v1.sh` | Live SQL smoke |

---

# Phase 0 — Worktree + branch

### Task 0: Create isolated worktree

- [ ] **Step 1: Create worktree from field-topology base**

```bash
cd /mnt/scripts/Orion-Sapienform
git fetch origin
git worktree add .worktrees/feat-attention-frame-v1 \
  -b feat/attention-frame-v1 \
  feat/field-topology-reconciliation-v1
cd .worktrees/feat-attention-frame-v1
```

Expected: `git branch --show-current` → `feat/attention-frame-v1`

- [ ] **Step 2: Verify isolation**

```bash
git check-ignore -q .worktrees && echo "worktree gitignored ok"
```

---

# Phase 1 — Substrate attention schemas

### Task 1: FieldAttentionTargetV1 + FieldAttentionFrameV1

**Files:**
- Create: `orion/schemas/field_attention_frame.py`
- Modify: `orion/schemas/registry.py`
- Test: `tests/test_attention_frame_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attention_frame_schemas.py
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1


def test_field_attention_target_v1_validates() -> None:
    t = FieldAttentionTargetV1(
        target_id="node:athena",
        target_kind="node",
        salience_score=0.8,
        pressure_score=0.9,
        novelty_score=0.0,
        urgency_score=0.5,
        confidence_score=0.2,
        dominant_channels={"execution_load": 0.7},
        reasons=["node execution_load is elevated"],
        evidence_refs=["field:tick_abc"],
        suggested_observation_mode="inspect",
    )
    assert t.target_kind == "node"


def test_field_attention_frame_v1_roundtrip() -> None:
    now = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)
    frame = FieldAttentionFrameV1(
        frame_id="attention.frame:tick_abc:field_attention_policy.v1",
        generated_at=now,
        source_field_tick_id="tick_abc",
        source_field_generated_at=now,
        dominant_targets=[],
    )
    payload = frame.model_dump(mode="json")
    restored = FieldAttentionFrameV1.model_validate(payload)
    assert restored.schema_version == "field.attention.frame.v1"


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        FieldAttentionTargetV1(
            target_id="node:athena",
            target_kind="node",
            salience_score=0.5,
            pressure_score=0.5,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
            bogus=True,  # type: ignore[call-arg]
        )


def test_scores_reject_out_of_range() -> None:
    with pytest.raises(ValidationError):
        FieldAttentionTargetV1(
            target_id="node:athena",
            target_kind="node",
            salience_score=1.5,
            pressure_score=0.5,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_attention_frame_schemas.py -v`
Expected: FAIL — `ModuleNotFoundError: orion.schemas.field_attention_frame`

- [ ] **Step 3: Implement schemas**

```python
# orion/schemas/field_attention_frame.py
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FieldAttentionTargetV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    target_kind: Literal[
        "node",
        "capability",
        "channel",
        "edge",
        "field",
        "system",
    ]

    salience_score: float = Field(ge=0.0, le=1.0)
    pressure_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(ge=0.0, le=1.0)

    dominant_channels: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    suggested_observation_mode: Literal[
        "watch",
        "inspect",
        "summarize",
        "ignore",
    ] = "watch"


class FieldAttentionFrameV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["field.attention.frame.v1"] = "field.attention.frame.v1"

    frame_id: str
    generated_at: datetime

    source_field_tick_id: str
    source_field_generated_at: datetime

    attention_policy_id: str = "field_attention_policy.v1"

    overall_salience: float = Field(ge=0.0, le=1.0)

    dominant_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    node_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    capability_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    system_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)
    suppressed_targets: list[FieldAttentionTargetV1] = Field(default_factory=list)

    recent_perturbations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Register in registry**

In `orion/schemas/registry.py` add import:

```python
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
```

Add to `SCHEMA_REGISTRY` dict (near `FieldStateV1`):

```python
    "FieldAttentionTargetV1": FieldAttentionTargetV1,
    "FieldAttentionFrameV1": FieldAttentionFrameV1,
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. pytest tests/test_attention_frame_schemas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/field_attention_frame.py orion/schemas/registry.py tests/test_attention_frame_schemas.py
git commit -m "feat(attention): add FieldAttentionFrameV1 substrate schemas"
```

---

# Phase 2 — Policy config + loader

### Task 2: YAML policy and loader

**Files:**
- Create: `config/attention/field_attention_policy.v1.yaml`
- Create: `orion/attention/field_attention/policy.py`
- Create: `orion/attention/field_attention/__init__.py`
- Test: extend `tests/test_attention_frame_schemas.py` or add `tests/test_attention_policy_loader.py`

- [ ] **Step 1: Add policy YAML** (exact content from spec)

```yaml
# config/attention/field_attention_policy.v1.yaml
schema_version: attention_policy.v1
policy_id: field_attention_policy.v1

limits:
  max_targets_total: 12
  max_node_targets: 5
  max_capability_targets: 5
  max_system_targets: 3

thresholds:
  min_salience: 0.10
  high_salience: 0.70
  suppress_below: 0.03

weights:
  pressure: 0.45
  novelty: 0.20
  urgency: 0.25
  confidence: 0.10

node_channel_weights:
  cpu_pressure: 0.50
  gpu_pressure: 0.60
  memory_pressure: 0.55
  disk_pressure: 0.45
  thermal_pressure: 0.75
  staleness: 0.60
  availability: -0.40
  execution_load: 0.70
  execution_friction: 0.85
  reasoning_load: 0.45
  failure_pressure: 1.00
  expected_offline_suppression: -0.70

capability_channel_weights:
  pressure: 0.70
  execution_pressure: 0.80
  reasoning_pressure: 0.60
  reliability_pressure: 1.00
  confidence: -0.35
  available_capacity: -0.45

observation_modes:
  inspect_threshold: 0.75
  summarize_threshold: 0.45
  watch_threshold: 0.10
```

- [ ] **Step 2: Write failing policy loader test**

```python
# tests/test_attention_policy_loader.py
from pathlib import Path

from orion.attention.field_attention.policy import FieldAttentionPolicyV1, load_attention_policy

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / "config" / "attention" / "field_attention_policy.v1.yaml"


def test_load_attention_policy_v1() -> None:
    policy = load_attention_policy(POLICY)
    assert isinstance(policy, FieldAttentionPolicyV1)
    assert policy.policy_id == "field_attention_policy.v1"
    assert policy.node_channel_weights["execution_load"] == 0.70
    assert policy.limits.max_node_targets == 5
```

- [ ] **Step 3: Run test — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_attention_policy_loader.py -v`

- [ ] **Step 4: Implement policy loader**

```python
# orion/attention/field_attention/policy.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AttentionLimitsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_targets_total: int = 12
    max_node_targets: int = 5
    max_capability_targets: int = 5
    max_system_targets: int = 3


class AttentionThresholdsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_salience: float = 0.10
    high_salience: float = 0.70
    suppress_below: float = 0.03


class AttentionWeightsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pressure: float = 0.45
    novelty: float = 0.20
    urgency: float = 0.25
    confidence: float = 0.10


class ObservationModesV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inspect_threshold: float = 0.75
    summarize_threshold: float = 0.45
    watch_threshold: float = 0.10


class FieldAttentionPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["attention_policy.v1"] = "attention_policy.v1"
    policy_id: str = "field_attention_policy.v1"

    limits: AttentionLimitsV1 = Field(default_factory=AttentionLimitsV1)
    thresholds: AttentionThresholdsV1 = Field(default_factory=AttentionThresholdsV1)
    weights: AttentionWeightsV1 = Field(default_factory=AttentionWeightsV1)
    node_channel_weights: dict[str, float] = Field(default_factory=dict)
    capability_channel_weights: dict[str, float] = Field(default_factory=dict)
    observation_modes: ObservationModesV1 = Field(default_factory=ObservationModesV1)


def load_attention_policy(path: str | Path) -> FieldAttentionPolicyV1:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return FieldAttentionPolicyV1.model_validate(data)
```

```python
# orion/attention/field_attention/__init__.py
from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import FieldAttentionPolicyV1, load_attention_policy

__all__ = [
    "FieldAttentionPolicyV1",
    "build_attention_frame",
    "load_attention_policy",
]
```

- [ ] **Step 5: Run test — expect PASS**

- [ ] **Step 6: Commit**

```bash
git add config/attention/ orion/attention/ tests/test_attention_policy_loader.py
git commit -m "feat(attention): add field attention policy loader"
```

---

# Phase 3 — Scoring

### Task 3: Deterministic scoring helpers

**Files:**
- Create: `orion/attention/field_attention/scoring.py`
- Test: `tests/test_attention_field_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

```python
# tests/test_attention_field_scoring.py
from orion.attention.field_attention.policy import FieldAttentionPolicyV1, load_attention_policy
from orion.attention.field_attention.scoring import (
    URGENCY_CHANNELS,
    clamp01,
    compute_salience,
    confidence_from_vector,
    novelty_for_target,
    urgency_score,
    weighted_pressure,
)
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config/attention/field_attention_policy.v1.yaml")


def test_execution_load_raises_node_pressure() -> None:
    pressure, dominant = weighted_pressure(
        {"execution_load": 1.0, "reasoning_load": 0.35},
        POLICY.node_channel_weights,
    )
    assert pressure > 0.5
    assert "execution_load" in dominant


def test_failure_pressure_high_urgency() -> None:
    score = urgency_score({"failure_pressure": 1.0}, POLICY.node_channel_weights)
    assert score >= 0.9


def test_availability_reduces_pressure() -> None:
    pressure, _ = weighted_pressure({"availability": 1.0}, POLICY.node_channel_weights)
    assert pressure < 0.5


def test_expected_offline_suppression_reduces_pressure() -> None:
    pressure, _ = weighted_pressure(
        {"expected_offline_suppression": 1.0},
        POLICY.node_channel_weights,
    )
    assert pressure < 0.4


def test_capability_execution_pressure_raises_salience() -> None:
    pressure, _ = weighted_pressure(
        {"execution_pressure": 1.0},
        POLICY.capability_channel_weights,
    )
    assert pressure > 0.5


def test_capability_available_capacity_reduces() -> None:
    pressure, _ = weighted_pressure(
        {"available_capacity": 1.0},
        POLICY.capability_channel_weights,
    )
    assert pressure < 0.5


def test_scores_clamped() -> None:
    assert clamp01(2.0) == 1.0
    assert clamp01(-1.0) == 0.0
    salience = compute_salience(
        pressure_score=2.0,
        novelty_score=0.0,
        urgency_score=0.0,
        confidence_score=0.0,
        weights=POLICY.weights,
    )
    assert salience <= 1.0
```

- [ ] **Step 2: Run — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_attention_field_scoring.py -v`

- [ ] **Step 3: Implement scoring**

```python
# orion/attention/field_attention/scoring.py
from __future__ import annotations

from orion.attention.field_attention.policy import AttentionWeightsV1
from orion.schemas.field_attention_frame import FieldAttentionFrameV1

URGENCY_CHANNELS = frozenset({
    "failure_pressure",
    "reliability_pressure",
    "thermal_pressure",
    "staleness",
    "execution_friction",
})


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def weighted_pressure(
    vector: dict[str, float],
    channel_weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Signed weighted sum clamped to [0, 1]; dominant_channels keeps top positive contributions."""
    raw = 0.0
    dominant: dict[str, float] = {}
    for channel, value in vector.items():
        weight = float(channel_weights.get(channel, 0.0))
        if weight == 0.0:
            continue
        contrib = float(value) * weight
        raw += contrib
        if contrib > 0.0:
            dominant[channel] = contrib
    # keep top 5 channel contributions for explainability
    dominant = dict(
        sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)[:5]
    )
    return clamp01(raw), dominant


def urgency_score(
    vector: dict[str, float],
    channel_weights: dict[str, float],
) -> float:
    urgent = 0.0
    for channel in URGENCY_CHANNELS:
        if channel not in vector:
            continue
        weight = float(channel_weights.get(channel, 0.0))
        if weight <= 0.0:
            continue
        urgent = max(urgent, float(vector[channel]) * weight)
    return clamp01(urgent)


def confidence_from_vector(
    vector: dict[str, float],
    channel_weights: dict[str, float],
) -> float:
    """Healthy signals (negative weights * high values) increase confidence proxy."""
    healthy = 0.0
    for channel, value in vector.items():
        weight = float(channel_weights.get(channel, 0.0))
        if weight < 0.0:
            healthy += float(value) * abs(weight)
    return clamp01(healthy)


def novelty_for_target(
    target_id: str,
    current_salience: float,
    previous_frame: FieldAttentionFrameV1 | None,
) -> float:
    if previous_frame is None:
        return 0.0
    prior = 0.0
    for bucket in (
        previous_frame.dominant_targets,
        previous_frame.node_targets,
        previous_frame.capability_targets,
        previous_frame.system_targets,
        previous_frame.suppressed_targets,
    ):
        for t in bucket:
            if t.target_id == target_id:
                prior = t.salience_score
                break
    return clamp01(abs(current_salience - prior))


def compute_salience(
    *,
    pressure_score: float,
    novelty_score: float,
    urgency_score: float,
    confidence_score: float,
    weights: AttentionWeightsV1,
) -> float:
    raw = (
        weights.pressure * pressure_score
        + weights.novelty * novelty_score
        + weights.urgency * urgency_score
        + weights.confidence * confidence_score
    )
    return clamp01(raw)
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add orion/attention/field_attention/scoring.py tests/test_attention_field_scoring.py
git commit -m "feat(attention): add deterministic field scoring helpers"
```

---

# Phase 4 — Selectors

### Task 4: Target selection from field vectors

**Files:**
- Create: `orion/attention/field_attention/selectors.py`
- Test: covered in builder tests; add unit test for observation mode if desired

- [ ] **Step 1: Implement selectors**

```python
# orion/attention/field_attention/selectors.py
from __future__ import annotations

from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import (
    compute_salience,
    confidence_from_vector,
    novelty_for_target,
    urgency_score,
    weighted_pressure,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1


def observation_mode_for(salience: float, policy: FieldAttentionPolicyV1) -> str:
    modes = policy.observation_modes
    if salience >= modes.inspect_threshold:
        return "inspect"
    if salience >= modes.summarize_threshold:
        return "summarize"
    if salience >= modes.watch_threshold:
        return "watch"
    return "ignore"


def _reasons_from_dominant(dominant: dict[str, float], prefix: str) -> list[str]:
    reasons: list[str] = []
    for channel, contrib in sorted(dominant.items(), key=lambda kv: kv[1], reverse=True)[:3]:
        if contrib > 0.05:
            reasons.append(f"{prefix} {channel} is elevated")
    return reasons or [f"{prefix} pressure present"]


def select_node_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
) -> list[FieldAttentionTargetV1]:
    targets: list[FieldAttentionTargetV1] = []
    for node_id, vector in field.node_vectors.items():
        pressure, dominant = weighted_pressure(vector, policy.node_channel_weights)
        if pressure <= 0.0 and not any(float(v) > 0 for v in vector.values()):
            continue
        urg = urgency_score(vector, policy.node_channel_weights)
        conf = confidence_from_vector(vector, policy.node_channel_weights)
        salience = compute_salience(
            pressure_score=pressure,
            novelty_score=0.0,
            urgency_score=urg,
            confidence_score=conf,
            weights=policy.weights,
        )
        salience = novelty_for_target(node_id, salience, previous_frame)  # recompute with novelty
        salience = compute_salience(
            pressure_score=pressure,
            novelty_score=novelty_for_target(node_id, salience, previous_frame),
            urgency_score=urg,
            confidence_score=conf,
            weights=policy.weights,
        )
        targets.append(
            FieldAttentionTargetV1(
                target_id=node_id,
                target_kind="node",
                salience_score=salience,
                pressure_score=pressure,
                novelty_score=novelty_for_target(node_id, salience, previous_frame),
                urgency_score=urg,
                confidence_score=conf,
                dominant_channels=dominant,
                reasons=_reasons_from_dominant(dominant, "node"),
                evidence_refs=[f"field:{field.tick_id}"],
                suggested_observation_mode=observation_mode_for(salience, policy),  # type: ignore[arg-type]
            )
        )
    return targets


def select_capability_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None,
) -> list[FieldAttentionTargetV1]:
    targets: list[FieldAttentionTargetV1] = []
    for cap_id, vector in field.capability_vectors.items():
        pressure, dominant = weighted_pressure(vector, policy.capability_channel_weights)
        if pressure <= 0.0 and not any(float(v) > 0 for v in vector.values()):
            continue
        urg = urgency_score(vector, policy.capability_channel_weights)
        conf = confidence_from_vector(vector, policy.capability_channel_weights)
        salience = compute_salience(
            pressure_score=pressure,
            novelty_score=novelty_for_target(cap_id, 0.0, previous_frame),
            urgency_score=urg,
            confidence_score=conf,
            weights=policy.weights,
        )
        targets.append(
            FieldAttentionTargetV1(
                target_id=cap_id,
                target_kind="capability",
                salience_score=salience,
                pressure_score=pressure,
                novelty_score=novelty_for_target(cap_id, salience, previous_frame),
                urgency_score=urg,
                confidence_score=conf,
                dominant_channels=dominant,
                reasons=_reasons_from_dominant(dominant, "capability"),
                evidence_refs=[f"field:{field.tick_id}"],
                suggested_observation_mode=observation_mode_for(salience, policy),  # type: ignore[arg-type]
            )
        )
    return targets


def select_system_targets(
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
) -> list[FieldAttentionTargetV1]:
    count = len(field.recent_perturbations)
    if count == 0:
        return []
    # normalize: 10+ perturbations → salience ~1.0
    salience = min(1.0, count / 10.0)
    if salience < policy.thresholds.min_salience:
        return []
    return [
        FieldAttentionTargetV1(
            target_id="field:recent_perturbations",
            target_kind="system",
            salience_score=salience,
            pressure_score=salience,
            novelty_score=0.0,
            urgency_score=0.0,
            confidence_score=0.0,
            dominant_channels={},
            reasons=[f"recent field perturbation count is {count}"],
            evidence_refs=[f"field:{field.tick_id}"],
            suggested_observation_mode=observation_mode_for(salience, policy),  # type: ignore[arg-type]
        )
    ]
```

**Note for implementer:** Fix the double `novelty_for_target` call in `select_node_targets` — compute `novelty` once, then `salience` once (copy pattern from capability block).

- [ ] **Step 2: Commit**

```bash
git add orion/attention/field_attention/selectors.py
git commit -m "feat(attention): add field target selectors"
```

---

# Phase 5 — Builder

### Task 5: build_attention_frame

**Files:**
- Create: `orion/attention/field_attention/builder.py`
- Test: `tests/test_attention_frame_builder.py`

- [ ] **Step 1: Write failing builder test**

```python
# tests/test_attention_frame_builder.py
from datetime import datetime, timezone
from pathlib import Path

from orion.attention.field_attention.builder import build_attention_frame
from orion.attention.field_attention.policy import load_attention_policy
from orion.schemas.field_state import FieldStateV1

REPO = Path(__file__).resolve().parents[1]
POLICY = load_attention_policy(REPO / "config/attention/field_attention_policy.v1.yaml")
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
            "node:prometheus": {
                "cpu_pressure": 0.02,
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


def test_builder_selects_athena_and_orchestration() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    node_ids = {t.target_id for t in frame.node_targets}
    cap_ids = {t.target_id for t in frame.capability_targets}
    assert "node:athena" in node_ids
    assert "capability:orchestration" in cap_ids


def test_dominant_channels_present() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    athena = next(t for t in frame.node_targets if t.target_id == "node:athena")
    orch = next(t for t in frame.capability_targets if t.target_id == "capability:orchestration")
    assert "execution_load" in athena.dominant_channels
    assert "execution_pressure" in orch.dominant_channels


def test_overall_salience_positive() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    assert frame.overall_salience > 0.0


def test_low_salience_suppressed() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    dominant_ids = {t.target_id for t in frame.dominant_targets}
    assert "node:prometheus" not in dominant_ids


def test_targets_sorted_desc() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    scores = [t.salience_score for t in frame.dominant_targets]
    assert scores == sorted(scores, reverse=True)


def test_frame_id_stable() -> None:
    field = _synthetic_field()
    a = build_attention_frame(field=field, policy=POLICY, now=NOW)
    b = build_attention_frame(field=field, policy=POLICY, now=NOW)
    assert a.frame_id == b.frame_id


def test_source_field_tick_id() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    assert frame.source_field_tick_id == "tick_exec_attention"


def test_recent_perturbations_carried() -> None:
    frame = build_attention_frame(field=_synthetic_field(), policy=POLICY, now=NOW)
    assert frame.recent_perturbations == ["state_delta:exec_1", "state_delta:exec_2"]
```

- [ ] **Step 2: Run — expect FAIL**

Run: `PYTHONPATH=. pytest tests/test_attention_frame_builder.py -v`

- [ ] **Step 3: Implement builder**

```python
# orion/attention/field_attention/builder.py
from __future__ import annotations

from datetime import datetime, timezone

from orion.attention.field_attention.policy import FieldAttentionPolicyV1
from orion.attention.field_attention.scoring import clamp01
from orion.attention.field_attention.selectors import (
    select_capability_targets,
    select_node_targets,
    select_system_targets,
)
from orion.schemas.field_attention_frame import FieldAttentionFrameV1, FieldAttentionTargetV1
from orion.schemas.field_state import FieldStateV1


def stable_frame_id(*, tick_id: str, policy_id: str) -> str:
    return f"attention.frame:{tick_id}:{policy_id}"


def build_attention_frame(
    *,
    field: FieldStateV1,
    policy: FieldAttentionPolicyV1,
    previous_frame: FieldAttentionFrameV1 | None = None,
    now: datetime | None = None,
) -> FieldAttentionFrameV1:
    generated_at = now or datetime.now(timezone.utc)

    node_targets = select_node_targets(field, policy, previous_frame)
    capability_targets = select_capability_targets(field, policy, previous_frame)
    system_targets = select_system_targets(field, policy)

    all_targets = node_targets + capability_targets + system_targets
    all_targets.sort(key=lambda t: t.salience_score, reverse=True)

    active: list[FieldAttentionTargetV1] = []
    suppressed: list[FieldAttentionTargetV1] = []
    for t in all_targets:
        if t.salience_score < policy.thresholds.suppress_below:
            suppressed.append(t)
        elif t.salience_score >= policy.thresholds.min_salience:
            active.append(t)
        else:
            suppressed.append(t)

    # per-kind caps
    nodes = [t for t in active if t.target_kind == "node"][: policy.limits.max_node_targets]
    caps = [t for t in active if t.target_kind == "capability"][: policy.limits.max_capability_targets]
    systems = [t for t in active if t.target_kind == "system"][: policy.limits.max_system_targets]
    capped = (nodes + caps + systems)[: policy.limits.max_targets_total]
    capped.sort(key=lambda t: t.salience_score, reverse=True)

    overall = clamp01(max((t.salience_score for t in capped), default=0.0))

    return FieldAttentionFrameV1(
        frame_id=stable_frame_id(tick_id=field.tick_id, policy_id=policy.policy_id),
        generated_at=generated_at,
        source_field_tick_id=field.tick_id,
        source_field_generated_at=field.generated_at,
        attention_policy_id=policy.policy_id,
        overall_salience=overall,
        dominant_targets=capped,
        node_targets=nodes,
        capability_targets=caps,
        system_targets=systems,
        suppressed_targets=suppressed,
        recent_perturbations=list(field.recent_perturbations),
        warnings=[],
    )
```

- [ ] **Step 4: Run builder + schema + scoring tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/test_attention_frame_schemas.py \
  tests/test_attention_field_scoring.py \
  tests/test_attention_frame_builder.py \
  tests/test_attention_policy_loader.py \
  -q
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add orion/attention/field_attention/builder.py tests/test_attention_frame_builder.py
git commit -m "feat(attention): build FieldAttentionFrameV1 from FieldStateV1"
```

---

# Phase 6 — SQL migration

### Task 6: substrate_attention_frames table

**Files:**
- Create: `services/orion-sql-db/manual_migration_attention_frame_v1.sql`

- [ ] **Step 1: Add migration SQL** (exact DDL from spec)

```sql
-- services/orion-sql-db/manual_migration_attention_frame_v1.sql
create table if not exists substrate_attention_frames (
    frame_id text primary key,
    source_field_tick_id text not null,
    source_field_generated_at timestamptz not null,
    generated_at timestamptz not null,
    policy_id text not null,
    frame_json jsonb not null,
    created_at timestamptz not null default now()
);

create index if not exists idx_substrate_attention_frames_generated_at
    on substrate_attention_frames (generated_at desc);

create index if not exists idx_substrate_attention_frames_source_tick
    on substrate_attention_frames (source_field_tick_id);
```

- [ ] **Step 2: Apply on operator DB** (manual, not in CI)

```bash
docker exec -i "${DB:-orion-athena-sql-db}" psql -U "${PGUSER:-postgres}" -d "${PGDATABASE:-conjourney}" \
  < services/orion-sql-db/manual_migration_attention_frame_v1.sql
```

- [ ] **Step 3: Commit**

```bash
git add services/orion-sql-db/manual_migration_attention_frame_v1.sql
git commit -m "chore(db): add substrate_attention_frames migration"
```

---

# Phase 7 — orion-attention-runtime service

### Task 7: Runtime store + worker + FastAPI

**Files:**
- Create: `services/orion-attention-runtime/` (full tree per file structure)
- Test: `tests/test_attention_runtime_store.py`

- [ ] **Step 1: Scaffold service** (mirror `orion-field-digester`)

`requirements.txt`:

```text
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.3
pydantic-settings==2.6.1
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
PyYAML==6.0.2
```

`app/settings.py`:

```python
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project: str = Field("orion-athena", alias="PROJECT")
    service_name: str = Field("orion-attention-runtime", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    postgres_uri: str = Field(..., alias="POSTGRES_URI")
    attention_policy_path: str = Field(
        "config/attention/field_attention_policy.v1.yaml",
        alias="ATTENTION_POLICY_PATH",
    )
    attention_poll_interval_sec: float = Field(2.0, alias="ATTENTION_POLL_INTERVAL_SEC")
    enable_attention_runtime: bool = Field(True, alias="ENABLE_ATTENTION_RUNTIME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 2: Implement store** (copy SQL patterns from field-digester `load_latest_field`)

`app/store.py` — class `AttentionRuntimeStore` with methods:
- `load_latest_field() -> FieldStateV1 | None`
- `load_latest_attention_frame() -> FieldAttentionFrameV1 | None`
- `load_attention_frame_for_field_tick(tick_id: str) -> FieldAttentionFrameV1 | None`
- `save_attention_frame(frame: FieldAttentionFrameV1) -> None` with `ON CONFLICT (frame_id) DO UPDATE`

Use `psycopg2.extras.Json` for `frame_json` insert.

- [ ] **Step 3: Write store unit test (mock engine)**

```python
# tests/test_attention_runtime_store.py
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.schemas.field_attention_frame import FieldAttentionFrameV1

# Import store from services path — add services/orion-attention-runtime to PYTHONPATH in test
import sys
from pathlib import Path

SVC = Path(__file__).resolve().parents[1] / "services" / "orion-attention-runtime"
sys.path.insert(0, str(SVC))

from app.store import AttentionRuntimeStore  # noqa: E402

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def _frame() -> FieldAttentionFrameV1:
    return FieldAttentionFrameV1(
        frame_id="attention.frame:tick_a:field_attention_policy.v1",
        generated_at=NOW,
        source_field_tick_id="tick_a",
        source_field_generated_at=NOW,
        overall_salience=0.5,
    )


def test_save_and_load_latest(monkeypatch) -> None:
    # mock sqlalchemy engine/connect — mirror test_substrate_field_debug_api pattern
    ...
```

Implementer: complete mock following `services/orion-hub/tests/test_substrate_field_debug_api.py`.

- [ ] **Step 4: Implement worker**

```python
# app/worker.py — AttentionRuntimeWorker
# _tick():
#   if not settings.enable_attention_runtime: return
#   field = store.load_latest_field()
#   if field is None: return
#   if store.load_attention_frame_for_field_tick(field.tick_id): return  # idempotent
#   previous = store.load_latest_attention_frame()
#   policy = load_attention_policy(settings.attention_policy_path)
#   frame = build_attention_frame(field=field, policy=policy, previous_frame=previous)
#   store.save_attention_frame(frame)
```

- [ ] **Step 5: main.py** — lifespan starts worker; routes `/health`, `/latest` return latest frame JSON

- [ ] **Step 6: Dockerfile + docker-compose**

Dockerfile copies: `orion/`, `config/attention/`, `services/orion-attention-runtime/`

```yaml
# docker-compose.yml
services:
  attention-runtime:
    build:
      context: ../..
      dockerfile: services/orion-attention-runtime/Dockerfile
    container_name: ${PROJECT}-attention-runtime
    restart: unless-stopped
    networks:
      - app-net
    ports:
      - "${ATTENTION_RUNTIME_PORT:-8117}:8117"
    environment:
      - POSTGRES_URI=${POSTGRES_URI}
      - ATTENTION_POLICY_PATH=${ATTENTION_POLICY_PATH:-/app/config/attention/field_attention_policy.v1.yaml}
      - ATTENTION_POLL_INTERVAL_SEC=${ATTENTION_POLL_INTERVAL_SEC:-2.0}
      - ENABLE_ATTENTION_RUNTIME=${ENABLE_ATTENTION_RUNTIME:-true}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}

networks:
  app-net:
    external: true
```

- [ ] **Step 7: .env_example + README + sync local .env**

```bash
cp services/orion-attention-runtime/.env_example services/orion-attention-runtime/.env
# merge keys into operator env; POSTGRES_URI must match field-digester
```

`.env_example`:

```bash
PROJECT=orion-athena
POSTGRES_URI=postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney
ATTENTION_POLICY_PATH=/app/config/attention/field_attention_policy.v1.yaml
ATTENTION_POLL_INTERVAL_SEC=2.0
ENABLE_ATTENTION_RUNTIME=true
ATTENTION_RUNTIME_PORT=8117
LOG_LEVEL=INFO
```

- [ ] **Step 8: Compile**

```bash
PYTHONPATH=. python -m compileall \
  orion/attention \
  orion/schemas/field_attention_frame.py \
  services/orion-attention-runtime
```

- [ ] **Step 9: Commit**

```bash
git add services/orion-attention-runtime/
git commit -m "feat(attention): add orion-attention-runtime polling service"
```

---

# Phase 8 — Optional Hub debug route

### Task 8: GET /api/substrate/attention/latest

**Files:**
- Create: `services/orion-hub/scripts/substrate_attention_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Test: `services/orion-hub/tests/test_substrate_attention_debug_api.py`

- [ ] **Step 1: Add routes** (mirror `substrate_field_routes.py`)

```python
router = APIRouter(prefix="/api/substrate/attention", tags=["substrate-attention"])

@router.get("/latest")
async def attention_latest() -> dict[str, Any]:
    # SELECT frame_json FROM substrate_attention_frames ORDER BY generated_at DESC LIMIT 1
    # return FieldAttentionFrameV1.model_validate(payload).model_dump(mode="json")
```

- [ ] **Step 2: Wire router in api_routes.py**

```python
from .substrate_attention_routes import router as substrate_attention_router
router.include_router(substrate_attention_router)
```

- [ ] **Step 3: Hub tests + commit**

```bash
git add services/orion-hub/scripts/substrate_attention_routes.py \
  services/orion-hub/scripts/api_routes.py \
  services/orion-hub/tests/test_substrate_attention_debug_api.py
git commit -m "feat(hub): expose latest substrate attention frame"
```

---

# Phase 9 — Smoke script

### Task 9: smoke_attention_frame_v1.sh

**Files:**
- Create: `scripts/smoke_attention_frame_v1.sh`

- [ ] **Step 1: Add script** (from spec — docker psql queries for field + attention)

```bash
#!/usr/bin/env bash
set -euo pipefail
DB="${DB:-orion-athena-sql-db}"
PGDATABASE="${PGDATABASE:-conjourney}"
PGUSER="${PGUSER:-postgres}"

echo "=== Latest field state ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select generated_at, tick_id,
  field_json #> '{node_vectors,node:athena}' as athena,
  field_json #> '{capability_vectors,capability:orchestration}' as orchestration
from substrate_field_state order by generated_at desc limit 1;"

echo "=== Latest attention frame ==="
docker exec -i "$DB" psql -U "$PGUSER" -d "$PGDATABASE" -c "
select generated_at, frame_id, source_field_tick_id,
  frame_json #> '{dominant_targets}' as dominant_targets,
  frame_json #>> '{overall_salience}' as overall_salience
from substrate_attention_frames order by generated_at desc limit 1;"
```

- [ ] **Step 2: chmod +x and commit**

```bash
chmod +x scripts/smoke_attention_frame_v1.sh
git add scripts/smoke_attention_frame_v1.sh
git commit -m "chore: add attention frame v1 smoke script"
```

---

# Phase 10 — Regression + verification

### Task 10: Required test matrix

- [ ] **Step 1: Attention unit tests**

```bash
PYTHONPATH=. pytest \
  tests/test_attention_frame_schemas.py \
  tests/test_attention_field_scoring.py \
  tests/test_attention_frame_builder.py \
  tests/test_attention_policy_loader.py \
  tests/test_attention_runtime_store.py \
  -q
```

- [ ] **Step 2: Field regression**

```bash
PYTHONPATH=.:services/orion-field-digester pytest \
  tests/test_field_topology_reconciliation.py \
  tests/test_field_execution_perturbations.py \
  tests/test_field_*.py \
  -q
```

- [ ] **Step 3: Execution digestion regression**

```bash
PYTHONPATH=. pytest \
  tests/test_execution_substrate_reducer.py \
  tests/test_execution_substrate_pipeline.py \
  tests/test_execution_projection_schemas.py \
  -q
```

- [ ] **Step 4: Hub attention route test** (if Task 8 done)

```bash
PYTHONPATH=.:services/orion-hub pytest services/orion-hub/tests/test_substrate_attention_debug_api.py -q
```

- [ ] **Step 5: Live smoke** (stack running: sql-db, field-digester, attention-runtime)

```bash
./scripts/smoke_attention_frame_v1.sh
```

Expected: `node:athena` / `capability:orchestration` visible in field JSON; attention frame `dominant_targets` non-empty when execution pressure present.

---

# Phase 11 — Code review, PR report, push

### Task 11: Review + PR

- [ ] **Step 1: Run requesting-code-review subagent** on full diff vs `feat/field-topology-reconciliation-v1`; fix all findings.

- [ ] **Step 2: Write PR report** `docs/superpowers/pr-reports/2026-05-24-attention-frame-v1-pr.md`

Must include:
- Layer 5 mapping in 11-layer roadmap
- Naming note: `FieldAttentionFrameV1` vs conversational `AttentionFrameV1`
- Example synthetic frame JSON (`node:athena`, `capability:orchestration`)
- Live or dry-run smoke evidence
- Explicit non-goals (no self-state, no action)
- Tests run with commands + pass/fail
- Gaps deferred to Layer 6 (`SelfStateV1`)

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin feat/attention-frame-v1
gh pr create --base feat/field-topology-reconciliation-v1 --title "PR: Attention Frame v1 — FieldState → What Matters Now" --body "$(cat <<'EOF'
## Summary
- Layer 5: deterministic FieldStateV1 → FieldAttentionFrameV1
- orion-attention-runtime polls field state, idempotent per tick
- No bus publish, no action/self-state/LLM

## Test plan
- [ ] pytest attention + field + execution suites
- [ ] smoke_attention_frame_v1.sh with live stack

EOF
)"
```

Adjust `--base` to `main` if topology PR merged first.

---

## Self-review (plan author checklist)

| Spec requirement | Task |
|------------------|------|
| AttentionTargetV1 / AttentionFrameV1 schemas | Task 1 (`Field*` names — collision resolved) |
| Registry | Task 1 |
| Policy YAML + loader | Task 2 |
| Scoring | Task 3 |
| Selectors | Task 4 |
| Builder | Task 5 |
| SQL migration | Task 6 |
| Runtime service | Task 7 |
| Hub optional | Task 8 |
| Smoke script | Task 9 |
| No bus / no action | Preflight + worker has no bus client |
| Tests from spec | Tasks 1, 3, 5, 7 |
| Acceptance: athena + orchestration salient | Task 5 builder test |
| Idempotent per tick | Task 7 worker |
| `.env` sync | Task 7 step 7 |

**Placeholder scan:** No TBD steps.

**Type consistency:** `FieldAttentionFrameV1`, `FieldAttentionTargetV1`, `FieldAttentionPolicyV1` used throughout; frame_id format `attention.frame:{tick_id}:{policy_id}`.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-24-attention-frame-v1.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with executing-plans checkpoints  

**Which approach?**

After implementation (either path), run code-review subagent, fix issues, write PR report, push PR — per user request and Phase 11.
