# Computed/Learned Salience + Actionable Pending Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hand-tuned regex/keyword salience ladder with one shared evidence-derived salience module used by coalition selection AND reverie, add graded habituation to self-break the rumination lock, and give Juniper an operator-grade Pending Attention surface to Resolve/Dismiss cognitive loops (emitting feedback labels for later learned refit).

**Architecture:** A pure `salience.py` module produces a `SalienceFeaturesV1` feature vector and scores it with a hand-seeded `LinearSalienceCombiner` (refit-able later — same code path becomes learned). `build_open_loops`/`score_loop` and `reverie.derive_salience` consume it behind `ORION_ATTENTION_SALIENCE_V2_ENABLED` (shadow-first, default-off). Habituation is a subtractive feature gated by `ORION_ATTENTION_HABITUATION_ENABLED`. Telemetry events (`AttentionSalienceTraceV1`, `AttentionLoopOutcomeV1`) + direct-write tables form the learning seam. The Hub renders `PendingAttentionCardV1` rows with Resolve/Dismiss, reusing the existing Pending Attention panel.

**Tech Stack:** Python 3.11, pydantic v2, FastAPI (orion-hub), SQLAlchemy + Postgres (`conjourney` DB), Redis bus (`OrionBusAsync`), pytest, vanilla JS (orion-hub static).

---

## Orientation (read before starting)

**The choke point being replaced** (per `.cursor/rules/conversational-behavior-anti-slop.mdc` discipline — name it):

```129:143:orion/substrate/attention/scoring.py
def score_loop(loop: OpenLoopV1) -> float:
    raw = (
        loop.novelty * 0.2
        + loop.continuity_relevance * 0.13
        + loop.relational_relevance * 0.12
        + loop.predictive_value * 0.13
        + loop.concept_value * 0.14
        + loop.autonomy_value * 0.16
        + loop.emotional_charge * 0.07
        + loop.askability * 0.05
    )
    if loop.already_known:
        raw *= 0.25
    return bounded(raw * 1.22)
```

The constant ladder that feeds it is `build_open_loops` (`predictive = 0.72 if target_type in {...}`), same file lines 69-126. `reverie.derive_salience` (`services/orion-thought/app/reverie.py:99-116`) takes `max()` of the seven constant fields.

**Three score consumers (all covered by editing `scoring.py` + `reverie.py`):**
- Chat turn: `services/orion-cortex-exec/app/chat_stance.py:2318` → `build_attention_frame` (`orion/substrate/attention_frame.py`) → `select_actions` → `score_loop`.
- Substrate broadcast: `orion/substrate/attention_broadcast.py:131` → `build_open_loops` + `select_actions` (produced by `orion-substrate-runtime`).
- Reverie: `services/orion-thought/app/reverie.py:221` → `derive_salience`.

**Persistence precedent (follow it):** reverie writes its own tables directly from the thin `orion-thought` service via `services/orion-thought/app/store.py` (NOT via sql-writer). Salience traces + loop outcomes follow this precedent: direct best-effort writes from the producer, plus registered bus events published for observability/contract. Suppress reuses `substrate_reverie_refractory` (spec §Schema: "may reuse").

**Helpers to reuse:** `orion.substrate.attention.common` — `bounded(x)` (clamp+round 3dp), `compact(s, n)`, `stable_id(prefix, text)`.

**Env flag pattern to mirror:** `orion/substrate/attention_broadcast.py:32-49` (`_TRUTHY`, `os.getenv(...).strip().lower() in _TRUTHY`).

**Gate scripts note:** `scripts/check_bus_channels.py`, `check_schema_registry.py`, `check_env_template_parity.py` referenced in `AGENTS.md` do NOT exist in this repo. Use `python scripts/sync_local_env_from_example.py` for env parity and the pytest gates below. Do not invent the missing scripts.

**Weights seed (`weights_version = "seed-v1"`, used across the plan):**

```python
SEED_WEIGHTS = {
    "evidence_strength": 0.30,   # strongest signal backing the loop (salience*confidence)
    "novelty_vs_known": 0.20,    # unknown targets deserve attention; known ones are demoted
    "recency": 0.13,             # fresh observations outrank stale ones
    "recurrence": 0.15,          # a theme that keeps reappearing is real, not noise
    "evidence_breadth": 0.12,    # corroboration across detectors beats a single detector
    "dwell": 0.10,               # some reward for the currently-held coalition (hysteresis)
    "habituation": -0.35,        # PENALTY: repeatedly-attended loops lose the coalition
}
```

---

## File Structure

Files created or modified, and each one's responsibility:

- `orion/schemas/attention_frame.py` (modify): add `SalienceFeaturesV1`; add `salience` + `salience_features` fields to `OpenLoopV1`. Co-located with the schema it augments.
- `orion/schemas/attention_salience.py` (create): `AttentionSalienceTraceV1`, `AttentionLoopOutcomeV1`, `PendingAttentionCardV1`. Telemetry/UI contracts kept out of the hot `attention_frame` schema.
- `orion/schemas/registry.py` (modify): register the four new schemas (`_REGISTRY` + `SCHEMA_REGISTRY`).
- `orion/substrate/attention/salience.py` (create): `SalienceHistory`, `LinearSalienceCombiner`, `compute_salience`, `default_combiner`, `salience_v2_enabled`, `habituation_enabled`. Pure, import-light (schemas + common only).
- `orion/substrate/attention/scoring.py` (modify): `build_open_loops` computes features; `score_loop` uses combiner behind the flag.
- `services/orion-thought/app/reverie.py` (modify): `derive_salience` uses the combiner behind the flag; reverie tick emits + persists a salience trace.
- `services/orion-thought/app/store.py` (modify): `persist_salience_trace` direct writer.
- `services/orion-thought/app/settings.py` (modify): salience flags/channels.
- `orion/bus/channels.yaml` (modify): two new channels.
- `services/orion-sql-db/manual_migration_attention_salience_trace.sql` (create).
- `services/orion-sql-db/manual_migration_attention_loop_outcome.sql` (create).
- `services/orion-hub/scripts/attention_loops_routes.py` (create): `GET /api/attention/loops`, `POST .../resolve`, `POST .../dismiss`.
- `services/orion-hub/scripts/api_routes.py` (modify): include the new router.
- `services/orion-hub/app/store.py` or new `attention_loops_store.py` (create): read loops/build cards; write outcomes; suppress.
- `services/orion-hub/templates/index.html` + `static/js/app.js` (modify): render cognitive-loop rows with a source badge + Resolve/Dismiss.
- `scripts/refit_salience_weights.py` (create): documented stub reading the outcome table.
- Tests + evals alongside each unit (paths given per task).
- `.env_example` files: `orion-cortex-exec`, `orion-substrate-runtime`, `orion-thought`, `orion-hub` (+ root `.env_example` if present).

---

## PHASE 1 — Shared salience module + shadow wiring (`SALIENCE_V2` off)

Smallest slice that creates the seam and proves it with a failing-then-passing discrimination eval. No selection behavior changes when the flag is off.

### Task 1: `SalienceFeaturesV1` schema + `OpenLoopV1` fields

**Files:**
- Modify: `orion/schemas/attention_frame.py`
- Modify: `orion/schemas/registry.py`
- Test: `orion/substrate/tests/test_salience_schema.py`

- [ ] **Step 1: Write the failing test**

Create `orion/substrate/tests/test_salience_schema.py`:

```python
from orion.schemas.attention_frame import OpenLoopV1, SalienceFeaturesV1
from orion.schemas.registry import resolve


def test_salience_features_defaults_are_bounded():
    f = SalienceFeaturesV1()
    assert f.evidence_strength == 0.0
    assert f.habituation == 0.0
    dumped = f.model_dump(mode="json")
    assert set(dumped) >= {
        "evidence_strength", "evidence_breadth", "recurrence",
        "recency", "novelty_vs_known", "dwell", "habituation",
    }


def test_open_loop_carries_salience_fields():
    loop = OpenLoopV1(id="open-loop-x", description="thing")
    assert loop.salience == 0.0
    assert loop.salience_features == {}


def test_salience_features_registered():
    assert resolve("SalienceFeaturesV1") is SalienceFeaturesV1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/substrate/tests/test_salience_schema.py -q`
Expected: FAIL — `ImportError: cannot import name 'SalienceFeaturesV1'`.

- [ ] **Step 3: Add `SalienceFeaturesV1` to `orion/schemas/attention_frame.py`**

Insert immediately before `class OpenLoopV1` (currently line 41):

```python
class SalienceFeaturesV1(BaseModel):
    """Evidence-derived feature vector scored by the salience combiner.

    Replaces the hand-tuned constant ladder. `habituation` is a penalty term
    (higher = more habituated = lower salience) applied subtractively by the
    combiner. All features are bounded [0,1].
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.salience.features.v1"] = "attention.salience.features.v1"
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_breadth: float = Field(default=0.0, ge=0.0, le=1.0)
    recurrence: float = Field(default=0.0, ge=0.0, le=1.0)
    recency: float = Field(default=0.0, ge=0.0, le=1.0)
    novelty_vs_known: float = Field(default=0.0, ge=0.0, le=1.0)
    dwell: float = Field(default=0.0, ge=0.0, le=1.0)
    habituation: float = Field(default=0.0, ge=0.0, le=1.0)
```

- [ ] **Step 4: Add fields to `OpenLoopV1`**

In `orion/schemas/attention_frame.py`, add to `OpenLoopV1` after the `provenance` field (line 60):

```python
    # Salience v2 (additive, back-compatible). The 7 legacy score fields above
    # remain populated for one deprecation release; new code reads these.
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    salience_features: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 5: Register the schema**

In `orion/schemas/registry.py`, change the existing import (line 538):

```python
from orion.schemas.attention_frame import AttentionFrameV1, AttentionSignalV1, SalienceFeaturesV1
```

Add to `_REGISTRY` right after the `"AttentionSignalV1": AttentionSignalV1,` line (line 761):

```python
    "SalienceFeaturesV1": SalienceFeaturesV1,
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest orion/substrate/tests/test_salience_schema.py -q`
Expected: PASS (3 passed).

- [ ] **Step 7: Commit**

```bash
git add orion/schemas/attention_frame.py orion/schemas/registry.py orion/substrate/tests/test_salience_schema.py
git commit -m "feat(attention): add SalienceFeaturesV1 schema + OpenLoopV1 salience fields"
```

---

### Task 2: `LinearSalienceCombiner` + `compute_salience`

**Files:**
- Create: `orion/substrate/attention/salience.py`
- Test: `orion/substrate/tests/test_salience_combiner.py`

- [ ] **Step 1: Write the failing test**

Create `orion/substrate/tests/test_salience_combiner.py`:

```python
import pytest

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1, SalienceFeaturesV1
from orion.substrate.attention.salience import (
    SalienceHistory,
    LinearSalienceCombiner,
    compute_salience,
    default_combiner,
)


def _signal(salience: float, confidence: float, source: str = "current_turn", refs=None):
    return AttentionSignalV1(
        signal_id=f"sig-{salience}-{confidence}-{source}",
        source=source,
        target_text="thing",
        signal_kind="test",
        salience=salience,
        confidence=confidence,
        evidence_refs=refs or ["r1"],
    )


def _loop(already_known: bool = False):
    return OpenLoopV1(id="open-loop-x", description="thing", already_known=already_known)


def test_score_is_bounded_and_deterministic():
    combiner = default_combiner()
    feats = SalienceFeaturesV1(evidence_strength=0.9, novelty_vs_known=0.8)
    a = combiner.score(feats)
    b = combiner.score(feats)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_evidence_strength_is_monotonic():
    combiner = default_combiner()
    low = combiner.score(SalienceFeaturesV1(evidence_strength=0.2))
    high = combiner.score(SalienceFeaturesV1(evidence_strength=0.9))
    assert high > low


def test_habituation_strictly_lowers_salience():
    combiner = default_combiner()
    base = SalienceFeaturesV1(evidence_strength=0.8, novelty_vs_known=0.7)
    habituated = base.model_copy(update={"habituation": 0.9})
    assert combiner.score(habituated) < combiner.score(base)


def test_compute_salience_uses_signals():
    loop = _loop()
    strong, feats_strong = compute_salience(
        loop=loop, signals=[_signal(0.9, 0.9)], history=SalienceHistory(), now=None
    )
    weak, feats_weak = compute_salience(
        loop=loop, signals=[_signal(0.3, 0.5)], history=SalienceHistory(), now=None
    )
    assert strong > weak
    assert feats_strong.evidence_strength > feats_weak.evidence_strength


def test_already_known_lowers_novelty():
    _, feats_known = compute_salience(
        loop=_loop(already_known=True), signals=[_signal(0.9, 0.9)],
        history=SalienceHistory(), now=None,
    )
    _, feats_novel = compute_salience(
        loop=_loop(already_known=False), signals=[_signal(0.9, 0.9)],
        history=SalienceHistory(), now=None,
    )
    assert feats_novel.novelty_vs_known > feats_known.novelty_vs_known


def test_breadth_rises_with_distinct_detectors():
    _, one = compute_salience(
        loop=_loop(), signals=[_signal(0.8, 0.8, source="current_turn", refs=["a"])],
        history=SalienceHistory(), now=None,
    )
    _, many = compute_salience(
        loop=_loop(),
        signals=[
            _signal(0.8, 0.8, source="current_turn", refs=["a"]),
            _signal(0.8, 0.8, source="autonomy", refs=["b"]),
            _signal(0.8, 0.8, source="concept_induction", refs=["c"]),
        ],
        history=SalienceHistory(), now=None,
    )
    assert many.evidence_breadth > one.evidence_breadth
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/substrate/tests/test_salience_combiner.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'orion.substrate.attention.salience'`.

- [ ] **Step 3: Create `orion/substrate/attention/salience.py`**

```python
"""Shared evidence-derived salience — the single source of salience truth.

Pure and import-light (schemas + common only) so the thin `orion-thought`
service may import it without dragging the graph engine. Consumed by coalition
selection (`scoring.score_loop`) AND reverie (`derive_salience`).

Hybrid design (spec decision #1/#7): a deterministic feature vector scored by a
tiny linear combiner with hand-seeded weights. The same code path becomes
learned when `refit_salience_weights.py` emits a new `weights_version`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1, SalienceFeaturesV1
from orion.substrate.attention.common import bounded

_TRUTHY = {"1", "true", "yes", "on"}

SALIENCE_V2_FLAG = "ORION_ATTENTION_SALIENCE_V2_ENABLED"
HABITUATION_FLAG = "ORION_ATTENTION_HABITUATION_ENABLED"
WEIGHTS_OVERRIDE_ENV = "ORION_ATTENTION_SALIENCE_WEIGHTS"

WEIGHTS_VERSION = "seed-v1"

# Normalizers: turn raw counts into bounded [0,1] features.
BREADTH_NORM = 4.0     # distinct detectors/refs to saturate breadth
RECURRENCE_NORM = 5.0  # recent theme appearances to saturate recurrence
DWELL_NORM = 6.0       # dwell ticks to saturate dwell

SEED_WEIGHTS: dict[str, float] = {
    "evidence_strength": 0.30,
    "novelty_vs_known": 0.20,
    "recency": 0.13,
    "recurrence": 0.15,
    "evidence_breadth": 0.12,
    "dwell": 0.10,
    "habituation": -0.35,
}


def salience_v2_enabled() -> bool:
    return str(os.getenv(SALIENCE_V2_FLAG, "false")).strip().lower() in _TRUTHY


def habituation_enabled() -> bool:
    return str(os.getenv(HABITUATION_FLAG, "false")).strip().lower() in _TRUTHY


@dataclass
class SalienceHistory:
    """Runtime history for the recency/recurrence/dwell/habituation features.

    Empty default → those features are 0 (Phase 1 shadow behavior). The broadcast
    producer fills it in Phase 3. `theme_key` maps to a loop id or theme string.
    """

    dwell_ticks: int = 0
    recent_theme_counts: dict[str, int] = field(default_factory=dict)
    resonance_theme_keys: set[str] = field(default_factory=set)
    first_seen_at: dict[str, datetime] = field(default_factory=dict)


class LinearSalienceCombiner:
    """Bounded weighted sum with `habituation` as a subtractive penalty."""

    def __init__(self, weights: dict[str, float] | None = None, weights_version: str = WEIGHTS_VERSION):
        self.weights = dict(weights or SEED_WEIGHTS)
        self.weights_version = weights_version

    def score(self, features: SalienceFeaturesV1) -> float:
        data = features.model_dump()
        total = 0.0
        for name, weight in self.weights.items():
            total += weight * float(data.get(name, 0.0))
        return bounded(total)


def default_combiner() -> LinearSalienceCombiner:
    """Combiner from the seeded weights, with optional JSON env override."""
    raw = os.getenv(WEIGHTS_OVERRIDE_ENV, "").strip()
    if not raw:
        return LinearSalienceCombiner()
    try:
        override = json.loads(raw)
        if isinstance(override, dict):
            merged = dict(SEED_WEIGHTS)
            merged.update({str(k): float(v) for k, v in override.items()})
            return LinearSalienceCombiner(merged, weights_version=f"{WEIGHTS_VERSION}+override")
    except (ValueError, TypeError):
        pass
    return LinearSalienceCombiner()


def _recency(theme_key: str, history: SalienceHistory, now: datetime) -> float:
    first = history.first_seen_at.get(theme_key)
    if first is None:
        return 1.0  # never seen before → maximally fresh
    if first.tzinfo is None:
        first = first.replace(tzinfo=timezone.utc)
    age_hours = max(0.0, (now - first).total_seconds() / 3600.0)
    # Half-life ~6h: fresh≈1.0, ~0.5 at 6h, decays toward 0.
    return bounded(0.5 ** (age_hours / 6.0))


def _habituation(theme_key: str, history: SalienceHistory) -> float:
    recurrence = min(1.0, history.recent_theme_counts.get(theme_key, 0) / RECURRENCE_NORM)
    dwell = min(1.0, history.dwell_ticks / DWELL_NORM)
    resonance = 1.0 if theme_key in history.resonance_theme_keys else 0.0
    return bounded(0.5 * recurrence + 0.3 * dwell + 0.2 * resonance)


def compute_features(
    *,
    loop: OpenLoopV1,
    signals: Sequence[AttentionSignalV1],
    history: SalienceHistory | None = None,
    now: datetime | None = None,
) -> SalienceFeaturesV1:
    history = history or SalienceHistory()
    now = now or datetime.now(timezone.utc)
    theme_key = loop.id

    strengths = [float(s.salience) * float(s.confidence) for s in signals]
    evidence_strength = bounded(max(strengths)) if strengths else 0.0

    distinct: set[str] = set()
    for s in signals:
        distinct.add(str(s.source))
        for ref in (s.evidence_refs or []):
            distinct.add(str(ref))
    evidence_breadth = bounded(len(distinct) / BREADTH_NORM)

    recurrence = bounded(history.recent_theme_counts.get(theme_key, 0) / RECURRENCE_NORM)
    recency = _recency(theme_key, history, now)
    novelty_vs_known = 0.15 if loop.already_known else evidence_strength
    dwell = bounded(history.dwell_ticks / DWELL_NORM)
    habituation = _habituation(theme_key, history)

    return SalienceFeaturesV1(
        evidence_strength=evidence_strength,
        evidence_breadth=evidence_breadth,
        recurrence=recurrence,
        recency=recency,
        novelty_vs_known=bounded(novelty_vs_known),
        dwell=dwell,
        habituation=habituation,
    )


def compute_salience(
    *,
    loop: OpenLoopV1,
    signals: Sequence[AttentionSignalV1],
    history: SalienceHistory | None = None,
    now: datetime | None = None,
    combiner: LinearSalienceCombiner | None = None,
    apply_habituation: bool | None = None,
) -> tuple[float, SalienceFeaturesV1]:
    """Return (salience, features). `apply_habituation` None → env flag decides."""
    features = compute_features(loop=loop, signals=signals, history=history, now=now)
    if apply_habituation is None:
        apply_habituation = habituation_enabled()
    scored = features if apply_habituation else features.model_copy(update={"habituation": 0.0})
    combiner = combiner or default_combiner()
    return combiner.score(scored), features
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/substrate/tests/test_salience_combiner.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/attention/salience.py orion/substrate/tests/test_salience_combiner.py
git commit -m "feat(attention): add LinearSalienceCombiner + compute_salience module"
```

---

### Task 3: Salience discrimination eval (recommendation #1)

Built here first as a failing gate that today's constant ladder cannot pass, then made to pass by the Task 4 wiring. This is the acceptance gate for the live flip.

**Files:**
- Create: `orion/substrate/tests/test_salience_discrimination_eval.py`

- [ ] **Step 1: Write the eval (expected to fail against constants)**

Create `orion/substrate/tests/test_salience_discrimination_eval.py`:

```python
"""Recommendation #1: distinct coalitions must produce distinct salience.

Two plan-type loops with materially different evidence should NOT collapse to
the same salience. Today's constant ladder pins them (predictive=0.72 for both).
The v2 combiner discriminates. This eval is the acceptance gate for the flip.
"""

from orion.substrate.attention.salience import SalienceHistory, compute_salience
from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1


def _plan_loop(loop_id: str) -> OpenLoopV1:
    return OpenLoopV1(id=loop_id, target_type="plan", description=loop_id)


def _sig(loop_id: str, salience: float, confidence: float, source: str) -> AttentionSignalV1:
    return AttentionSignalV1(
        signal_id=f"sig-{loop_id}-{source}",
        source=source,
        target_text=loop_id,
        target_type_hint="plan",
        signal_kind="test",
        salience=salience,
        confidence=confidence,
        evidence_refs=[f"{loop_id}-ref"],
    )


def test_distinct_coalitions_get_distinct_salience():
    strong_loop = _plan_loop("open-loop-strong")
    weak_loop = _plan_loop("open-loop-weak")

    strong, _ = compute_salience(
        loop=strong_loop,
        signals=[
            _sig("open-loop-strong", 0.95, 0.9, "current_turn"),
            _sig("open-loop-strong", 0.8, 0.85, "autonomy"),
            _sig("open-loop-strong", 0.7, 0.8, "concept_induction"),
        ],
        history=SalienceHistory(),
        apply_habituation=False,
    )
    weak, _ = compute_salience(
        loop=weak_loop,
        signals=[_sig("open-loop-weak", 0.35, 0.5, "current_turn")],
        history=SalienceHistory(),
        apply_habituation=False,
    )

    # Discrimination floor: the two coalitions differ by a real margin.
    assert strong - weak > 0.15, f"salience did not discriminate: {strong=} {weak=}"
```

- [ ] **Step 2: Run the eval to verify it passes for the module**

Run: `pytest orion/substrate/tests/test_salience_discrimination_eval.py -q`
Expected: PASS — the module discriminates. (The point of "recommendation #1" is proven by Task 4's contrast test below, which shows the *legacy* path failing this discrimination.)

- [ ] **Step 3: Add the legacy-fails contrast (regression evidence)**

Append to the same file:

```python
def test_legacy_constant_ladder_fails_discrimination():
    """Evidence the old path was broken: two plan loops collapse to one score."""
    from orion.substrate.attention.scoring import score_loop

    a = OpenLoopV1(id="a", target_type="plan", description="a", predictive_value=0.72,
                   novelty=0.35, concept_value=0.55)
    b = OpenLoopV1(id="b", target_type="plan", description="b", predictive_value=0.72,
                   novelty=0.35, concept_value=0.55)
    # Legacy score_loop reads only the constant fields → identical inputs → identical score.
    assert abs(score_loop(a) - score_loop(b)) < 1e-9
```

- [ ] **Step 4: Run both**

Run: `pytest orion/substrate/tests/test_salience_discrimination_eval.py -q`
Expected: PASS (2 passed). (Task 4 changes `score_loop` — the legacy contrast test will be updated there.)

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/tests/test_salience_discrimination_eval.py
git commit -m "test(attention): add salience discrimination eval (recommendation #1)"
```

---

### Task 4: Wire the combiner into `scoring.py` (shadow-gated)

`build_open_loops` always computes + stores features/salience (so shadow traces are real). `score_loop` reads the combiner salience when `SALIENCE_V2` is on, else the legacy weighted sum. Selection is unchanged when off.

**Files:**
- Modify: `orion/substrate/attention/scoring.py`
- Modify: `orion/substrate/tests/test_salience_discrimination_eval.py`
- Test: `orion/substrate/tests/test_scoring_salience_wiring.py`

- [ ] **Step 1: Write the failing test**

Create `orion/substrate/tests/test_scoring_salience_wiring.py`:

```python
import orion.substrate.attention.salience as salience_mod
from orion.schemas.attention_frame import AttentionSignalV1
from orion.substrate.attention.scoring import build_open_loops, score_loop


def _ctx_signals():
    return [
        AttentionSignalV1(
            signal_id="s1", source="current_turn", target_text="the reactor plan",
            target_type_hint="plan", signal_kind="test", salience=0.9, confidence=0.9,
            evidence_refs=["r1"],
        )
    ]


def test_build_open_loops_populates_salience_features():
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    assert loops, "expected at least one loop"
    loop = loops[0]
    assert loop.salience_features, "salience_features must be populated"
    assert loop.salience > 0.0


def test_score_loop_uses_combiner_when_v2_on(monkeypatch):
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    loop = loops[0]
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    assert score_loop(loop) == loop.salience


def test_score_loop_legacy_when_v2_off(monkeypatch):
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    loops = build_open_loops(
        signals=_ctx_signals(), ctx={"user_message": "the reactor plan"}, inputs={},
        belief_lineage=[], direct_turn=False, generic_reversal=False,
        stale_thread_active=False, max_open=5,
    )
    loop = loops[0]
    # Legacy path still reads the constant fields (non-zero weighted sum).
    assert score_loop(loop) >= 0.0
    assert salience_mod.salience_v2_enabled() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/substrate/tests/test_scoring_salience_wiring.py -q`
Expected: FAIL — `salience_features` empty / `loop.salience == 0.0`.

- [ ] **Step 3: Populate features in `build_open_loops`**

In `orion/substrate/attention/scoring.py`, add imports at the top (after line 7):

```python
from orion.substrate.attention.salience import SalienceHistory, compute_salience
```

Inside `build_open_loops`, the loop currently appends an `OpenLoopV1(...)` per signal (lines 97-125). Refactor so each loop is built, then features computed and attached. Replace the `loops.append(OpenLoopV1(...))` block with:

```python
        loop = OpenLoopV1(
            id=stable_id("open-loop", phrase.lower()),
            target_type=target_type,  # type: ignore[arg-type]
            description=phrase,
            source_text=user_text,
            source_refs=list(signal.evidence_refs or ["ctx.user_message"]),
            why_it_matters="novel or unresolved current-turn target with substrate pressure" if not already_known else "current-turn target overlaps known context",
            novelty=bounded(novelty),
            continuity_relevance=bounded(continuity),
            relational_relevance=bounded(relational),
            predictive_value=bounded(predictive),
            concept_value=bounded(concept_value),
            autonomy_value=autonomy_value,
            emotional_charge=bounded(emotional),
            already_known=already_known,
            askability=bounded(askability),
            confidence=bounded(max(signal.confidence, 0.58 if already_known else 0.72)),
            provenance={
                "extractor": "attention_signal_pipeline_v1",
                "signal_id": signal.signal_id,
                "signal_source": signal.source,
                "signal_kind": signal.signal_kind,
                "belief_lineage": list(belief_lineage or [])[:8],
                "autonomy_signals": autonomy_signals,
                **dict(signal.provenance or {}),
            },
        )
        sal, feats = compute_salience(loop=loop, signals=[signal], history=SalienceHistory())
        loop = loop.model_copy(update={"salience": sal, "salience_features": feats.model_dump(mode="json")})
        loops.append(loop)
```

- [ ] **Step 4: Gate `score_loop` on the flag**

Replace `score_loop` (lines 129-143) with:

```python
def score_loop(loop: OpenLoopV1) -> float:
    from orion.substrate.attention.salience import salience_v2_enabled

    if salience_v2_enabled():
        return bounded(float(loop.salience))
    raw = (
        loop.novelty * 0.2
        + loop.continuity_relevance * 0.13
        + loop.relational_relevance * 0.12
        + loop.predictive_value * 0.13
        + loop.concept_value * 0.14
        + loop.autonomy_value * 0.16
        + loop.emotional_charge * 0.07
        + loop.askability * 0.05
    )
    if loop.already_known:
        raw *= 0.25
    return bounded(raw * 1.22)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest orion/substrate/tests/test_scoring_salience_wiring.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Update the legacy contrast eval to reflect the shadow gate**

In `orion/substrate/tests/test_salience_discrimination_eval.py`, change `test_legacy_constant_ladder_fails_discrimination` to force the flag off explicitly:

```python
def test_legacy_constant_ladder_fails_discrimination(monkeypatch):
    """Evidence the old path was broken: two plan loops collapse to one score."""
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    from orion.substrate.attention.scoring import score_loop

    a = OpenLoopV1(id="a", target_type="plan", description="a", predictive_value=0.72,
                   novelty=0.35, concept_value=0.55)
    b = OpenLoopV1(id="b", target_type="plan", description="b", predictive_value=0.72,
                   novelty=0.35, concept_value=0.55)
    assert abs(score_loop(a) - score_loop(b)) < 1e-9
```

- [ ] **Step 7: Run the whole attention test set + existing frame tests (no regressions)**

Run: `pytest orion/substrate/tests/ services/orion-cortex-exec/tests/test_attention_frame.py tests/test_attention_frame_builder.py -q`
Expected: PASS (all green; selection unchanged with flag off).

- [ ] **Step 8: Commit**

```bash
git add orion/substrate/attention/scoring.py orion/substrate/tests/test_scoring_salience_wiring.py orion/substrate/tests/test_salience_discrimination_eval.py
git commit -m "feat(attention): compute salience features in build_open_loops; gate score_loop on SALIENCE_V2"
```

---

### Task 5: Wire the combiner into `reverie.derive_salience` (shadow-gated)

**Files:**
- Modify: `services/orion-thought/app/reverie.py`
- Test: `services/orion-thought/tests/test_reverie_salience_v2.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_reverie_salience_v2.py`:

```python
from datetime import datetime, timezone

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from app.reverie import derive_salience


def _broadcast(selected_id: str, loops: list[OpenLoopV1]) -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=loops)
    return AttentionBroadcastProjectionV1(
        generated_at=frame.generated_at, frame=frame,
        selected_open_loop_id=selected_id, coalition_stability_score=0.3,
    )


def test_derive_salience_uses_combiner_when_v2_on(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    loop = OpenLoopV1(id="loop-a", description="a", salience=0.81)
    b = _broadcast("loop-a", [loop])
    assert derive_salience(b) == 0.81


def test_derive_salience_legacy_when_v2_off(monkeypatch):
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    loop = OpenLoopV1(id="loop-a", description="a", novelty=0.6, emotional_charge=0.2)
    b = _broadcast("loop-a", [loop])
    # Legacy: max of the seven constant fields → 0.6.
    assert derive_salience(b) == 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-thought/tests/test_reverie_salience_v2.py -q`
Expected: FAIL — `derive_salience` returns 0.6 in the v2-on case (still legacy `max()`).

- [ ] **Step 3: Update `derive_salience`**

In `services/orion-thought/app/reverie.py`, replace the body of `derive_salience` (lines 99-116) with:

```python
def derive_salience(broadcast: AttentionBroadcastProjectionV1 | None) -> float:
    """Salience of the selected coalition.

    v2 (`ORION_ATTENTION_SALIENCE_V2_ENABLED`): read the loop's precomputed
    `salience` (same combiner used by selection — one source of salience truth).
    Legacy: max of the seven constant score fields, else stability score.
    """
    if broadcast is None:
        return 0.0
    fallback = float(broadcast.coalition_stability_score)
    loop = next(
        (l for l in broadcast.frame.open_loops if l.id == broadcast.selected_open_loop_id),
        None,
    )
    if loop is None:
        return fallback
    from orion.substrate.attention.salience import salience_v2_enabled

    if salience_v2_enabled():
        return bounded(float(loop.salience)) if loop.salience else fallback
    scores = [float(getattr(loop, field, 0.0)) for field in _OPEN_LOOP_SCORE_FIELDS]
    return max(scores) if scores else fallback
```

Add the `bounded` import near the top of `reverie.py` (with the other `orion` imports, e.g. after line 33):

```python
from orion.substrate.attention.common import bounded
```

> Note: `orion.substrate.attention.common` and `.salience` are import-light (schemas + stdlib). Confirm the thin-import boundary test still passes in Step 5.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-thought/tests/test_reverie_salience_v2.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the reverie import-boundary + spontaneous-thought tests (no heavy-import regression)**

Run: `pytest services/orion-thought/tests/test_reverie_thin_import_boundary.py services/orion-thought/tests/test_reverie_spontaneous_thought.py -q`
Expected: PASS. If the boundary test fails because `salience.py`/`common.py` pulls a heavy dep, that is a real violation — fix by keeping those modules stdlib+schemas only (they already are).

- [ ] **Step 6: Commit**

```bash
git add services/orion-thought/app/reverie.py services/orion-thought/tests/test_reverie_salience_v2.py
git commit -m "feat(reverie): derive_salience uses shared combiner behind SALIENCE_V2"
```

---

### Task 6: Env flags + thought settings

**Files:**
- Modify: `services/orion-cortex-exec/.env_example`
- Modify: `services/orion-substrate-runtime/.env_example`
- Modify: `services/orion-thought/.env_example`
- Modify: `services/orion-hub/.env_example`
- Modify: root `.env_example` (only if it exists)
- Modify: `services/orion-thought/app/settings.py`
- Test: `services/orion-thought/tests/test_settings_salience_flags.py`

- [ ] **Step 1: Confirm which env_examples exist**

Run: `ls services/orion-cortex-exec/.env_example services/orion-substrate-runtime/.env_example services/orion-thought/.env_example services/orion-hub/.env_example .env_example 2>&1`
Expected: paths that exist print; note any missing (skip those, do not create root `.env_example` if absent).

- [ ] **Step 2: Add flags to each existing `.env_example`**

Append this block to each of `services/orion-cortex-exec/.env_example`, `services/orion-substrate-runtime/.env_example`, `services/orion-thought/.env_example` (the three services that execute `scoring.py`/`salience.py`):

```bash
# --- Computed salience v2 (shadow-first, default-off) ---
ORION_ATTENTION_SALIENCE_V2_ENABLED=false
ORION_ATTENTION_HABITUATION_ENABLED=false
# Optional JSON override for combiner weights; empty = hand-seeded seed-v1.
ORION_ATTENTION_SALIENCE_WEIGHTS=
```

Append this to `services/orion-hub/.env_example` (hub only needs the cards flag):

```bash
# --- Pending Attention cognitive-loop cards (default-off) ---
ORION_ATTENTION_PENDING_CARDS_ENABLED=false
```

- [ ] **Step 3: Sync local `.env`**

Run: `python scripts/sync_local_env_from_example.py`
Expected: reports the four keys synced across services; note any skip-list keys (`ORION_KNOWLEDGE_ROOT`, `PUBLISH_CORTEX_EXEC_GRAMMAR`) — not relevant here.

- [ ] **Step 4: Write the failing settings test**

Create `services/orion-thought/tests/test_settings_salience_flags.py`:

```python
import importlib


def test_salience_flags_default_off(monkeypatch):
    monkeypatch.delenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", raising=False)
    monkeypatch.delenv("ORION_ATTENTION_HABITUATION_ENABLED", raising=False)
    import app.settings as s
    importlib.reload(s)
    assert s.settings.attention_salience_v2_enabled is False
    assert s.settings.attention_habituation_enabled is False
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest services/orion-thought/tests/test_settings_salience_flags.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'attention_salience_v2_enabled'`.

- [ ] **Step 6: Add the fields to `ThoughtSettings`**

In `services/orion-thought/app/settings.py`, add after the resonance block (after line 103):

```python
    # --- Computed salience v2 (shadow-first, default-off) ---
    attention_salience_v2_enabled: bool = Field(False, alias="ORION_ATTENTION_SALIENCE_V2_ENABLED")
    attention_habituation_enabled: bool = Field(False, alias="ORION_ATTENTION_HABITUATION_ENABLED")
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest services/orion-thought/tests/test_settings_salience_flags.py -q`
Expected: PASS (1 passed).

- [ ] **Step 8: Verify `.env` is not staged, then commit**

```bash
git check-ignore services/orion-thought/.env services/orion-cortex-exec/.env
git status --short
git add services/orion-cortex-exec/.env_example services/orion-substrate-runtime/.env_example services/orion-thought/.env_example services/orion-hub/.env_example services/orion-thought/app/settings.py services/orion-thought/tests/test_settings_salience_flags.py
git commit -m "chore(attention): add salience v2 env flags + thought settings"
```

> **Phase 1 checkpoint:** `salience.py` + schema + shadow wiring + discrimination eval all green, all flags default-off, selection unchanged. This is the "Recommended next patch" slice — safe to review/merge before continuing.

---

## PHASE 2 — Telemetry + tables (learning seam)

### Task 7: Telemetry/UI schemas + registry + channels

**Files:**
- Create: `orion/schemas/attention_salience.py`
- Modify: `orion/schemas/registry.py`
- Modify: `orion/bus/channels.yaml`
- Test: `orion/substrate/tests/test_attention_salience_contracts.py`

- [ ] **Step 1: Write the failing test**

Create `orion/substrate/tests/test_attention_salience_contracts.py`:

```python
from orion.schemas.registry import SCHEMA_REGISTRY, resolve
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    AttentionSalienceTraceV1,
    PendingAttentionCardV1,
)


def test_schemas_registered():
    for name in ("AttentionSalienceTraceV1", "AttentionLoopOutcomeV1", "PendingAttentionCardV1"):
        assert resolve(name) is not None
    assert SCHEMA_REGISTRY["AttentionSalienceTraceV1"].kind == "attention.salience.trace.v1"
    assert SCHEMA_REGISTRY["AttentionLoopOutcomeV1"].kind == "attention.loop.outcome.v1"


def test_loop_outcome_verdicts():
    o = AttentionLoopOutcomeV1(
        outcome_id="o1", loop_id="open-loop-x", theme_key="open-loop-x",
        verdict="resolved", actor="juniper", salience_at_close=0.7,
    )
    assert o.verdict == "resolved"


def test_pending_card_requires_plain_text():
    c = PendingAttentionCardV1(
        loop_id="open-loop-x", theme_key="open-loop-x",
        title="The reactor rollout plan", why_it_matters="You flagged it as urgent and it is unresolved.",
        what_triggered="3 detectors (current turn, autonomy, concept)", salience=0.7,
        weights_version="seed-v1",
    )
    assert c.status == "pending"
    assert c.title and not c.title.startswith("open-loop-")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/substrate/tests/test_attention_salience_contracts.py -q`
Expected: FAIL — module `orion.schemas.attention_salience` missing.

- [ ] **Step 3: Create `orion/schemas/attention_salience.py`**

```python
"""Telemetry + operator-surface contracts for computed salience.

- AttentionSalienceTraceV1: every scored loop's feature vector + score (learning
  telemetry; the feature-distribution + input half of the label join).
- AttentionLoopOutcomeV1: the human verdict (Resolve/Dismiss) or implicit
  decay — the sparse-but-clean label the refit later trains on.
- PendingAttentionCardV1: operator-legible card. Never id-only (hard UX rule).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


AttentionOutcomeVerdictV1 = Literal["resolved", "dismissed", "decayed_unattended"]
PendingCardStatusV1 = Literal["pending", "resolved", "dismissed"]

MAX_FEATURE_LIST = 16


class AttentionSalienceTraceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.salience.trace.v1"] = "attention.salience.trace.v1"
    trace_id: str
    loop_id: str
    theme_key: str
    correlation_id: str | None = None
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    weights_version: str = "seed-v1"
    features: dict[str, Any] = Field(default_factory=dict)
    scope: str = "reverie"  # reverie | chat | broadcast
    created_at: datetime = Field(default_factory=_utc_now)


class AttentionLoopOutcomeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.loop.outcome.v1"] = "attention.loop.outcome.v1"
    outcome_id: str
    loop_id: str
    theme_key: str
    verdict: AttentionOutcomeVerdictV1
    actor: str = "juniper"
    note: str = Field(default="", max_length=500)
    salience_at_close: float = Field(default=0.0, ge=0.0, le=1.0)
    features_at_close: dict[str, Any] = Field(default_factory=dict)
    weights_version: str = "seed-v1"
    created_at: datetime = Field(default_factory=_utc_now)


class PendingAttentionCardV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal["attention.pending.card.v1"] = "attention.pending.card.v1"
    loop_id: str
    theme_key: str
    title: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    what_triggered: str = ""
    narrative: str = ""
    age_seconds: float = Field(default=0.0, ge=0.0)
    recurrence_count: int = Field(default=0, ge=0)
    salience: float = Field(default=0.0, ge=0.0, le=1.0)
    weights_version: str = "seed-v1"
    top_contributing_features: list[str] = Field(default_factory=list, max_length=MAX_FEATURE_LIST)
    source: Literal["cognitive_loop"] = "cognitive_loop"
    status: PendingCardStatusV1 = "pending"
```

- [ ] **Step 4: Register in `orion/schemas/registry.py`**

Add the import near the attention_frame import (line 538 area):

```python
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    AttentionSalienceTraceV1,
    PendingAttentionCardV1,
)
```

Add to `_REGISTRY` after `"SalienceFeaturesV1": SalienceFeaturesV1,`:

```python
    "AttentionSalienceTraceV1": AttentionSalienceTraceV1,
    "AttentionLoopOutcomeV1": AttentionLoopOutcomeV1,
    "PendingAttentionCardV1": PendingAttentionCardV1,
```

Add to `SCHEMA_REGISTRY` (before the closing brace at line 1295):

```python
    "AttentionSalienceTraceV1": SchemaRegistration(
        model=AttentionSalienceTraceV1,
        kind="attention.salience.trace.v1",
    ),
    "AttentionLoopOutcomeV1": SchemaRegistration(
        model=AttentionLoopOutcomeV1,
        kind="attention.loop.outcome.v1",
    ),
    "PendingAttentionCardV1": SchemaRegistration(
        model=PendingAttentionCardV1,
        kind="attention.pending.card.v1",
    ),
```

- [ ] **Step 5: Add channels to `orion/bus/channels.yaml`**

Append after the resonance-alert channel block (line 2116):

```yaml
  # --- Computed salience telemetry + human labels (default-off producers) ---
  - name: "orion:attention:salience:trace"
    kind: "telemetry"
    schema_id: "AttentionSalienceTraceV1"
    message_kind: "attention.salience.trace.v1"
    producer_services: ["orion-thought"]
    consumer_services: []
    stability: "experimental"
    since: "2026-07-07"

  - name: "orion:attention:loop_outcome"
    kind: "event"
    schema_id: "AttentionLoopOutcomeV1"
    message_kind: "attention.loop.outcome.v1"
    producer_services: ["orion-hub"]
    consumer_services: []
    stability: "experimental"
    since: "2026-07-07"
```

- [ ] **Step 6: Run test + a channels/registry sanity import**

Run: `pytest orion/substrate/tests/test_attention_salience_contracts.py -q`
Expected: PASS (3 passed).

Run: `python -c "import yaml; d=yaml.safe_load(open('orion/bus/channels.yaml')); names=[c['name'] for c in d['channels']]; assert 'orion:attention:salience:trace' in names and 'orion:attention:loop_outcome' in names; print('channels ok')"`
Expected: `channels ok`.

- [ ] **Step 7: Commit**

```bash
git add orion/schemas/attention_salience.py orion/schemas/registry.py orion/bus/channels.yaml orion/substrate/tests/test_attention_salience_contracts.py
git commit -m "feat(attention): telemetry+card schemas, registry, salience trace/outcome channels"
```

---

### Task 8: Tables — `attention_salience_trace` + `attention_loop_outcome`

**Files:**
- Create: `services/orion-sql-db/manual_migration_attention_salience_trace.sql`
- Create: `services/orion-sql-db/manual_migration_attention_loop_outcome.sql`

- [ ] **Step 1: Create the salience trace migration**

Create `services/orion-sql-db/manual_migration_attention_salience_trace.sql`:

```sql
-- Computed salience telemetry. One row per scored loop (feature vector + score).
-- The input half of the learning join; refit_salience_weights.py reads it with
-- attention_loop_outcome. Written directly by orion-thought (best-effort), same
-- pattern as substrate_reverie_thought. Observation only — mutates no cognition.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_salience_trace.sql

create table if not exists attention_salience_trace (
    trace_id text primary key,
    loop_id text not null,
    theme_key text not null,
    correlation_id text,
    salience double precision not null default 0,
    weights_version text not null default 'seed-v1',
    scope text not null default 'reverie',
    features jsonb not null default '{}'::jsonb,
    created_at timestamptz not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_attention_salience_trace_created_at
    on attention_salience_trace (created_at desc);

create index if not exists idx_attention_salience_trace_theme
    on attention_salience_trace (theme_key, created_at desc);
```

- [ ] **Step 2: Create the loop outcome migration**

Create `services/orion-sql-db/manual_migration_attention_loop_outcome.sql`:

```sql
-- Human close-the-loop labels (Resolve/Dismiss) + implicit decayed_unattended.
-- The sparse-but-clean label table for the salience refit. Written directly by
-- orion-hub when Juniper acts. Apply before ORION_ATTENTION_PENDING_CARDS_ENABLED.
-- Apply: psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_loop_outcome.sql

create table if not exists attention_loop_outcome (
    outcome_id text primary key,
    loop_id text not null,
    theme_key text not null,
    verdict text not null,
    actor text not null default 'juniper',
    note text not null default '',
    salience_at_close double precision not null default 0,
    weights_version text not null default 'seed-v1',
    features_at_close jsonb not null default '{}'::jsonb,
    created_at timestamptz not null,
    enqueued_at timestamptz not null default now()
);

create index if not exists idx_attention_loop_outcome_created_at
    on attention_loop_outcome (created_at desc);

create index if not exists idx_attention_loop_outcome_verdict
    on attention_loop_outcome (verdict, created_at desc);
```

- [ ] **Step 3: Validate SQL parses (syntax smoke)**

Run: `python -c "open('services/orion-sql-db/manual_migration_attention_salience_trace.sql').read(); open('services/orion-sql-db/manual_migration_attention_loop_outcome.sql').read(); print('sql files readable')"`
Expected: `sql files readable`. (If a live `POSTGRES_URI` is available: `psql "$POSTGRES_URI" -f <file>` and confirm `CREATE TABLE`. Otherwise list under "Restart required" in the PR for the operator to apply.)

- [ ] **Step 4: Commit**

```bash
git add services/orion-sql-db/manual_migration_attention_salience_trace.sql services/orion-sql-db/manual_migration_attention_loop_outcome.sql
git commit -m "feat(attention): migrations for salience trace + loop outcome tables"
```

---

### Task 9: Reverie tick emits + persists a salience trace

**Files:**
- Modify: `services/orion-thought/app/store.py`
- Modify: `services/orion-thought/app/reverie.py`
- Modify: `services/orion-thought/app/settings.py`
- Test: `services/orion-thought/tests/test_reverie_salience_trace.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-thought/tests/test_reverie_salience_trace.py`:

```python
from datetime import datetime, timezone

from orion.schemas.attention_frame import (
    AttentionBroadcastProjectionV1,
    AttentionFrameV1,
    OpenLoopV1,
)
from app.reverie import build_salience_trace


def _broadcast(loop: OpenLoopV1) -> AttentionBroadcastProjectionV1:
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=[loop])
    return AttentionBroadcastProjectionV1(
        generated_at=frame.generated_at, frame=frame,
        selected_open_loop_id=loop.id, coalition_stability_score=0.3,
    )


def test_build_salience_trace_from_selected_loop():
    loop = OpenLoopV1(id="loop-a", description="a", salience=0.72,
                      salience_features={"evidence_strength": 0.8})
    trace = build_salience_trace(_broadcast(loop), correlation_id="corr-1")
    assert trace is not None
    assert trace.loop_id == "loop-a"
    assert trace.salience == 0.72
    assert trace.features == {"evidence_strength": 0.8}
    assert trace.scope == "reverie"


def test_build_salience_trace_none_without_selection():
    frame = AttentionFrameV1(generated_at=datetime.now(timezone.utc), open_loops=[])
    b = AttentionBroadcastProjectionV1(generated_at=frame.generated_at, frame=frame,
                                       selected_open_loop_id=None)
    assert build_salience_trace(b, correlation_id="c") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-thought/tests/test_reverie_salience_trace.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_salience_trace'`.

- [ ] **Step 3: Add `build_salience_trace` to `reverie.py`**

Add the import at the top of `services/orion-thought/app/reverie.py` (with the schema imports):

```python
from orion.schemas.attention_salience import AttentionSalienceTraceV1
```

Add this function after `derive_salience`:

```python
def build_salience_trace(
    broadcast: AttentionBroadcastProjectionV1 | None,
    *,
    correlation_id: str,
) -> AttentionSalienceTraceV1 | None:
    """Trace the selected loop's feature vector + salience. None if no selection."""
    if broadcast is None or not broadcast.selected_open_loop_id:
        return None
    loop = next(
        (l for l in broadcast.frame.open_loops if l.id == broadcast.selected_open_loop_id),
        None,
    )
    if loop is None:
        return None
    from orion.core.ids import stable_hash_id
    from orion.substrate.attention.salience import WEIGHTS_VERSION

    return AttentionSalienceTraceV1(
        trace_id=stable_hash_id("saltrace", [correlation_id, loop.id]),
        loop_id=loop.id,
        theme_key=loop.id,
        correlation_id=correlation_id,
        salience=bounded(float(loop.salience)),
        weights_version=WEIGHTS_VERSION,
        features=dict(loop.salience_features or {}),
        scope="reverie",
    )
```

- [ ] **Step 4: Wire emit + persist into `run_reverie_once`**

In `run_reverie_once`, right after the successful `persist_reverie_thought(thought)` call (line 358), add:

```python
        if settings.attention_salience_v2_enabled:
            trace = build_salience_trace(broadcast, correlation_id=correlation_id)
            if trace is not None:
                with suppress(Exception):
                    await bus.publish(
                        settings.channel_attention_salience_trace,
                        BaseEnvelope(
                            kind="attention.salience.trace.v1",
                            source=_source(),
                            correlation_id=_envelope_correlation_id(correlation_id),
                            payload=trace.model_dump(mode="json"),
                        ),
                    )
                persist_salience_trace(trace)
```

Add `persist_salience_trace` to the store import at the top of `reverie.py` (line 41):

```python
from .store import persist_reverie_thought, persist_salience_trace
```

- [ ] **Step 5: Add the settings channel**

In `services/orion-thought/app/settings.py`, add under the salience v2 block from Task 6:

```python
    channel_attention_salience_trace: str = Field(
        "orion:attention:salience:trace",
        alias="CHANNEL_ATTENTION_SALIENCE_TRACE",
    )
```

- [ ] **Step 6: Add `persist_salience_trace` to `store.py`**

Append to `services/orion-thought/app/store.py`:

```python
def persist_salience_trace(trace) -> bool:
    """Persist one salience trace row. Never raises; idempotent on trace_id."""
    try:
        from sqlalchemy import text

        engine = _get_engine()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO attention_salience_trace
                        (trace_id, loop_id, theme_key, correlation_id, salience,
                         weights_version, scope, features, created_at)
                    VALUES
                        (:trace_id, :loop_id, :theme_key, :correlation_id, :salience,
                         :weights_version, :scope, CAST(:features AS jsonb), :created_at)
                    ON CONFLICT (trace_id) DO NOTHING
                    """
                ),
                {
                    "trace_id": trace.trace_id,
                    "loop_id": trace.loop_id,
                    "theme_key": trace.theme_key,
                    "correlation_id": trace.correlation_id,
                    "salience": float(trace.salience),
                    "weights_version": trace.weights_version,
                    "scope": trace.scope,
                    "features": json.dumps(trace.features),
                    "created_at": trace.created_at,
                },
            )
        return True
    except Exception as exc:
        logger.warning("salience trace persist failed id=%s err=%s", trace.trace_id, exc)
        return False
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest services/orion-thought/tests/test_reverie_salience_trace.py -q`
Expected: PASS (2 passed).

- [ ] **Step 8: Commit**

```bash
git add services/orion-thought/app/reverie.py services/orion-thought/app/store.py services/orion-thought/app/settings.py services/orion-thought/tests/test_reverie_salience_trace.py
git commit -m "feat(reverie): emit+persist AttentionSalienceTraceV1 per tick when SALIENCE_V2 on"
```

---

### Task 10: `refit_salience_weights.py` stub + label-reading test

**Files:**
- Create: `scripts/refit_salience_weights.py`
- Test: `orion/substrate/tests/test_refit_salience_stub.py`

- [ ] **Step 1: Write the failing test**

Create `orion/substrate/tests/test_refit_salience_stub.py`:

```python
from scripts.refit_salience_weights import candidate_weights_from_labels


def test_refit_consumes_label_rows_and_returns_weights():
    labels = [
        {"verdict": "resolved", "features_at_close": {"evidence_strength": 0.9}},
        {"verdict": "dismissed", "features_at_close": {"evidence_strength": 0.2}},
        {"verdict": "decayed_unattended", "features_at_close": {"evidence_strength": 0.3}},
    ]
    weights, version = candidate_weights_from_labels(labels)
    assert isinstance(weights, dict)
    assert "evidence_strength" in weights
    assert version.startswith("seed-v1")  # stub keeps seeded weights; documents intent


def test_refit_handles_empty_labels():
    weights, version = candidate_weights_from_labels([])
    assert weights  # returns seeded defaults, not empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/substrate/tests/test_refit_salience_stub.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.refit_salience_weights'`.

- [ ] **Step 3: Create `scripts/refit_salience_weights.py`**

```python
"""Salience weight refit — DOCUMENTED STUB. Not run in production this round.

The hybrid seam: it joins attention_salience_trace (feature distributions) with
attention_loop_outcome (human Resolve/Dismiss + implicit decayed_unattended
labels) and would emit candidate combiner weights + a new weights_version. This
round it proves the label table is consumable and returns the SEEDED weights
unchanged. When labels accumulate, replace `candidate_weights_from_labels` with a
real fit (e.g. logistic regression: resolved=1, dismissed/decayed=0).

Usage (read-only, safe):
    python scripts/refit_salience_weights.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from orion.substrate.attention.salience import SEED_WEIGHTS, WEIGHTS_VERSION


def load_labels(limit: int = 5000) -> list[dict[str, Any]]:
    """Read outcome label rows. Best-effort; [] if no DB configured."""
    uri = os.getenv("POSTGRES_URI", "").strip()
    if not uri:
        return []
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(uri, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT verdict, features_at_close FROM attention_loop_outcome "
                    "ORDER BY created_at DESC LIMIT :limit"
                ),
                {"limit": limit},
            ).mappings().all()
        out: list[dict[str, Any]] = []
        for r in rows:
            feats = r["features_at_close"]
            if isinstance(feats, str):
                feats = json.loads(feats)
            out.append({"verdict": r["verdict"], "features_at_close": feats or {}})
        return out
    except Exception:
        return []


def candidate_weights_from_labels(labels: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
    """STUB: prove the label table is consumable; return seeded weights.

    A real fit lands here later. We deliberately keep production weights seeded
    (spec non-goal: no training now) but tag the version so a future run diverges.
    """
    n = len([l for l in labels if l.get("verdict")])
    version = f"{WEIGHTS_VERSION}" if n == 0 else f"{WEIGHTS_VERSION}+labels={n}(seeded)"
    return dict(SEED_WEIGHTS), version


def main() -> None:
    parser = argparse.ArgumentParser(description="Salience weight refit (stub)")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.parse_args()
    labels = load_labels()
    weights, version = candidate_weights_from_labels(labels)
    print(json.dumps({"labels_seen": len(labels), "weights_version": version, "weights": weights}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/substrate/tests/test_refit_salience_stub.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/refit_salience_weights.py orion/substrate/tests/test_refit_salience_stub.py
git commit -m "feat(attention): refit_salience_weights stub reading the label table"
```

> **Phase 2 checkpoint:** telemetry contract + tables + trace emission + refit seam all present; still flag-gated, no selection change.

---

## PHASE 3 — Habituation (break the rumination lock)

Habituation is already a feature (Task 2) and already flows through `score_loop`/`compute_salience` behind `ORION_ATTENTION_HABITUATION_ENABLED`. Phase 3 supplies real `SalienceHistory` in the broadcast pipeline so a stuck loop's habituation rises over ticks until a competitor wins.

### Task 11: Rumination replay test + history-fed broadcast scoring

**Files:**
- Modify: `orion/substrate/attention_broadcast.py`
- Modify: `orion/substrate/attention/scoring.py`
- Test: `orion/substrate/tests/test_rumination_replay.py`

- [ ] **Step 1: Write the failing replay test**

Create `orion/substrate/tests/test_rumination_replay.py`:

```python
"""The lock breaks: same high-pressure node over N ticks eventually loses the
coalition once habituation engages. Uses injected history (no clock/DB)."""

from orion.schemas.attention_frame import AttentionSignalV1, OpenLoopV1
from orion.substrate.attention.salience import SalienceHistory, compute_salience


def _loop(loop_id: str) -> OpenLoopV1:
    return OpenLoopV1(id=loop_id, target_type="anomaly", description=loop_id)


def _sig(loop_id: str, salience: float) -> AttentionSignalV1:
    return AttentionSignalV1(
        signal_id=f"s-{loop_id}", source="substrate_broadcast", target_text=loop_id,
        signal_kind="substrate_pressure", salience=salience, confidence=0.8,
        evidence_refs=[f"{loop_id}-n"],
    )


def test_habituation_demotes_stuck_loop_below_competitor():
    stuck = _loop("open-loop-stuck")
    fresh = _loop("open-loop-fresh")

    # Tick 0: stuck loop has stronger evidence and wins.
    stuck0, _ = compute_salience(loop=stuck, signals=[_sig("open-loop-stuck", 0.9)],
                                 history=SalienceHistory(), apply_habituation=True)
    fresh0, _ = compute_salience(loop=fresh, signals=[_sig("open-loop-fresh", 0.6)],
                                 history=SalienceHistory(), apply_habituation=True)
    assert stuck0 > fresh0

    # After N re-selections: stuck loop is heavily habituated, fresh is not.
    stuck_hist = SalienceHistory(dwell_ticks=8, recent_theme_counts={"open-loop-stuck": 8},
                                 resonance_theme_keys={"open-loop-stuck"})
    stuckN, feats = compute_salience(loop=stuck, signals=[_sig("open-loop-stuck", 0.9)],
                                     history=stuck_hist, apply_habituation=True)
    freshN, _ = compute_salience(loop=fresh, signals=[_sig("open-loop-fresh", 0.6)],
                                 history=SalienceHistory(), apply_habituation=True)
    assert feats.habituation > 0.5
    assert freshN > stuckN, f"lock did not break: {stuckN=} {freshN=}"
```

- [ ] **Step 2: Run test to verify it passes (module already supports it)**

Run: `pytest orion/substrate/tests/test_rumination_replay.py -q`
Expected: PASS (1 passed). This confirms the combiner mechanics; the remaining work is *feeding real history* into the broadcast path.

- [ ] **Step 3: Add a `history` parameter to `build_open_loops`**

In `orion/substrate/attention/scoring.py`, add an optional `history` parameter to `build_open_loops` (keyword-only, default `None`) — update the signature (line 69-79 region) to include:

```python
    max_open: int,
    history: "SalienceHistory | None" = None,
) -> list[OpenLoopV1]:
```

Add `from typing import TYPE_CHECKING` handling — actually import directly (already importing `salience`): the top import from Task 4 already brings `SalienceHistory`. In the `compute_salience` call inside the loop, pass a per-loop history slice:

```python
        loop_history = history or SalienceHistory()
        sal, feats = compute_salience(loop=loop, signals=[signal], history=loop_history)
```

> When `history` is None (chat path), behavior is identical to Phase 1.

- [ ] **Step 4: Build + pass `SalienceHistory` in the broadcast frame**

In `orion/substrate/attention_broadcast.py`, add a module-level helper and pass history into `build_open_loops`. Add near the top-level state (after line 45):

```python
# Recent selected-loop counts for habituation (inhibition-of-return). Capped.
_recent_selected_counts: dict[str, int] = {}
_MAX_TRACKED_THEMES = 64


def _record_selection(loop_id: str | None) -> None:
    if not loop_id:
        return
    _recent_selected_counts[loop_id] = _recent_selected_counts.get(loop_id, 0) + 1
    if len(_recent_selected_counts) > _MAX_TRACKED_THEMES:
        # Drop the smallest-count theme to bound memory.
        drop = min(_recent_selected_counts, key=_recent_selected_counts.get)
        _recent_selected_counts.pop(drop, None)


def _current_history(resonance_theme_keys: set[str] | None = None) -> "SalienceHistory":
    from orion.substrate.attention.salience import SalienceHistory

    return SalienceHistory(
        dwell_ticks=_dwell_ticks,
        recent_theme_counts=dict(_recent_selected_counts),
        resonance_theme_keys=set(resonance_theme_keys or set()),
    )
```

In `build_substrate_attention_frame`, pass history into `build_open_loops` only when habituation is enabled:

```python
    from orion.substrate.attention.salience import habituation_enabled

    history = _current_history() if habituation_enabled() else None
    open_loops = build_open_loops(
        signals=merged,
        ctx={},
        inputs={},
        belief_lineage=lineage,
        direct_turn=False,
        generic_reversal=False,
        stale_thread_active=False,
        max_open=max_open,
        history=history,
    )
```

In `broadcast_projection_from_frame`, record the selection at the end (before `return`), so the next tick sees the incremented count:

```python
    _record_selection(selected.open_loop_id if selected is not None else None)
```

- [ ] **Step 5: Add a broadcast-level habituation integration test**

Create `orion/substrate/tests/test_broadcast_habituation.py`:

```python
import orion.substrate.attention_broadcast as ab


class _Node:
    def __init__(self, node_id, label, pressure):
        self.node_id = node_id
        self.label = label
        self.metadata = {"dynamic_pressure": pressure}
        self.signals = None


def test_broadcast_history_tracks_selection(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_SALIENCE_V2_ENABLED", "true")
    monkeypatch.setenv("ORION_ATTENTION_HABITUATION_ENABLED", "true")
    ab._recent_selected_counts.clear()
    ab._dwell_ticks = 0
    nodes = [_Node("n1", "runaway anomaly", 0.95)]
    for _ in range(3):
        frame = ab.build_substrate_attention_frame(nodes=nodes)
        ab.broadcast_projection_from_frame(frame)
    # The repeatedly-selected theme accrued a recurrence count.
    assert sum(ab._recent_selected_counts.values()) >= 1
```

- [ ] **Step 6: Run tests**

Run: `pytest orion/substrate/tests/test_rumination_replay.py orion/substrate/tests/test_broadcast_habituation.py orion/substrate/tests/test_attention_broadcast_dwell.py orion/substrate/tests/test_attention_broadcast.py -q`
Expected: PASS (existing broadcast tests still green; new ones pass).

- [ ] **Step 7: Commit**

```bash
git add orion/substrate/attention/scoring.py orion/substrate/attention_broadcast.py orion/substrate/tests/test_rumination_replay.py orion/substrate/tests/test_broadcast_habituation.py
git commit -m "feat(attention): feed SalienceHistory into broadcast so habituation breaks the lock"
```

> **Phase 3 checkpoint:** rumination replay proves the lock breaks with habituation on; chat path unchanged (history None).

---

## PHASE 4 — Operator-actionable Pending Attention (Hub) + closure

### Task 12: Pending card builder (legible, never id-only)

**Files:**
- Create: `services/orion-hub/scripts/attention_loops_store.py`
- Test: `services/orion-hub/tests/test_attention_card_legibility.py`

- [ ] **Step 1: Write the failing legibility test**

Create `services/orion-hub/tests/test_attention_card_legibility.py`:

```python
from datetime import datetime, timezone

from orion.schemas.attention_frame import OpenLoopV1
from scripts.attention_loops_store import build_pending_card


def _loop() -> OpenLoopV1:
    return OpenLoopV1(
        id="open-loop-2eb998452183",
        target_type="anomaly",
        description="reactor telemetry mismatch",
        why_it_matters="unresolved anomaly with substrate pressure",
        salience=0.71,
        salience_features={
            "evidence_strength": 0.8, "recurrence": 0.6, "habituation": 0.7,
            "novelty_vs_known": 0.5, "recency": 0.9, "evidence_breadth": 0.5, "dwell": 0.4,
        },
        provenance={"signal_source": "current_turn"},
    )


def test_card_never_id_only():
    card = build_pending_card(
        _loop(), first_seen=datetime.now(timezone.utc), recurrence_count=3,
        narrative="", now=datetime.now(timezone.utc),
    )
    assert card.title and not card.title.startswith("open-loop-")
    assert card.why_it_matters.strip()
    assert "open-loop-2eb998452183" not in card.title
    assert card.top_contributing_features  # rendered in words
    assert all(isinstance(f, str) for f in card.top_contributing_features)


def test_card_features_rendered_in_words():
    card = build_pending_card(
        _loop(), first_seen=datetime.now(timezone.utc), recurrence_count=3,
        narrative="", now=datetime.now(timezone.utc),
    )
    joined = " ".join(card.top_contributing_features).lower()
    assert "evidence" in joined or "recurr" in joined or "recency" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-hub/tests/test_attention_card_legibility.py -q`
Expected: FAIL — module `scripts.attention_loops_store` missing.

- [ ] **Step 3: Create `services/orion-hub/scripts/attention_loops_store.py` (builder half)**

```python
"""Pending Attention cognitive-loop cards + closure persistence (orion-hub).

Builds operator-legible PendingAttentionCardV1 (never id-only — hard UX rule),
reads recent loops from the salience trace table, and writes human Resolve/Dismiss
outcomes. Privacy: cards carry only plain summaries; no raw private trace/journal
material. Direct SQL (conjourney), matching the reverie persistence precedent.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from orion.schemas.attention_frame import OpenLoopV1
from orion.schemas.attention_salience import (
    AttentionLoopOutcomeV1,
    PendingAttentionCardV1,
)

logger = logging.getLogger("orion-hub.attention_loops")

_FEATURE_LABELS = {
    "evidence_strength": "strong evidence",
    "evidence_breadth": "corroborated across detectors",
    "recurrence": "keeps recurring",
    "recency": "recently observed",
    "novelty_vs_known": "novel vs known",
    "dwell": "held attention",
    "habituation": "over-attended (habituating)",
}


def _database_url() -> str:
    return (
        os.getenv("POSTGRES_URI", "").strip()
        or "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney"
    )


def _engine():
    from sqlalchemy import create_engine

    return create_engine(_database_url(), pool_pre_ping=True)


def _top_features(features: dict[str, Any], *, limit: int = 3) -> list[str]:
    scored = []
    for name, label in _FEATURE_LABELS.items():
        try:
            val = float(features.get(name, 0.0))
        except (TypeError, ValueError):
            val = 0.0
        if val > 0.0:
            scored.append((val, label))
    scored.sort(reverse=True)
    return [label for _, label in scored[:limit]]


def build_pending_card(
    loop: OpenLoopV1,
    *,
    first_seen: datetime,
    recurrence_count: int,
    narrative: str,
    now: datetime | None = None,
) -> PendingAttentionCardV1:
    now = now or datetime.now(timezone.utc)
    if first_seen.tzinfo is None:
        first_seen = first_seen.replace(tzinfo=timezone.utc)
    age = max(0.0, (now - first_seen).total_seconds())

    title = (loop.description or "").strip() or f"An unresolved {loop.target_type} loop"
    why = (loop.why_it_matters or "").strip() or (
        f"This {loop.target_type} has stayed active without resolution."
    )
    source = str((loop.provenance or {}).get("signal_source") or "the substrate")
    what_triggered = f"Raised by {source}; still open."

    return PendingAttentionCardV1(
        loop_id=loop.id,
        theme_key=loop.id,
        title=title,
        why_it_matters=why,
        what_triggered=what_triggered,
        narrative=(narrative or "").strip(),
        age_seconds=age,
        recurrence_count=int(recurrence_count),
        salience=float(loop.salience),
        weights_version=str((loop.salience_features or {}).get("weights_version") or "seed-v1"),
        top_contributing_features=_top_features(loop.salience_features or {}),
        status="pending",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-hub/tests/test_attention_card_legibility.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/attention_loops_store.py services/orion-hub/tests/test_attention_card_legibility.py
git commit -m "feat(hub): operator-legible PendingAttentionCardV1 builder"
```

---

### Task 13: Closure persistence + suppression (reuse refractory)

**Files:**
- Modify: `services/orion-hub/scripts/attention_loops_store.py`
- Test: `services/orion-hub/tests/test_attention_loop_closure.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-hub/tests/test_attention_loop_closure.py`:

```python
from scripts.attention_loops_store import build_loop_outcome


def test_build_loop_outcome_resolve():
    outcome = build_loop_outcome(
        loop_id="open-loop-x", theme_key="open-loop-x", verdict="resolved",
        actor="juniper", note="handled", salience_at_close=0.7, features_at_close={"x": 1},
    )
    assert outcome.verdict == "resolved"
    assert outcome.loop_id == "open-loop-x"
    assert outcome.outcome_id  # deterministic id present


def test_build_loop_outcome_rejects_bad_verdict():
    import pytest
    with pytest.raises(ValueError):
        build_loop_outcome(
            loop_id="x", theme_key="x", verdict="banana", actor="juniper",
            note="", salience_at_close=0.0, features_at_close={},
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-hub/tests/test_attention_loop_closure.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_loop_outcome'`.

- [ ] **Step 3: Add outcome builder + persistence + suppression to `attention_loops_store.py`**

Append:

```python
_VALID_VERDICTS = {"resolved", "dismissed", "decayed_unattended"}


def build_loop_outcome(
    *,
    loop_id: str,
    theme_key: str,
    verdict: str,
    actor: str,
    note: str,
    salience_at_close: float,
    features_at_close: dict[str, Any],
) -> AttentionLoopOutcomeV1:
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict}")
    from orion.core.ids import stable_hash_id

    return AttentionLoopOutcomeV1(
        outcome_id=stable_hash_id("loopoutcome", [loop_id, verdict, actor]),
        loop_id=loop_id,
        theme_key=theme_key,
        verdict=verdict,  # type: ignore[arg-type]
        actor=actor,
        note=(note or "")[:500],
        salience_at_close=float(salience_at_close),
        features_at_close=dict(features_at_close or {}),
    )


def persist_loop_outcome(outcome: AttentionLoopOutcomeV1) -> bool:
    """Write one outcome label. Never raises; idempotent on outcome_id."""
    try:
        from sqlalchemy import text

        with _engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO attention_loop_outcome
                        (outcome_id, loop_id, theme_key, verdict, actor, note,
                         salience_at_close, weights_version, features_at_close, created_at)
                    VALUES
                        (:outcome_id, :loop_id, :theme_key, :verdict, :actor, :note,
                         :salience_at_close, :weights_version, CAST(:features AS jsonb), :created_at)
                    ON CONFLICT (outcome_id) DO NOTHING
                    """
                ),
                {
                    "outcome_id": outcome.outcome_id,
                    "loop_id": outcome.loop_id,
                    "theme_key": outcome.theme_key,
                    "verdict": outcome.verdict,
                    "actor": outcome.actor,
                    "note": outcome.note,
                    "salience_at_close": float(outcome.salience_at_close),
                    "weights_version": outcome.weights_version,
                    "features": json.dumps(outcome.features_at_close),
                    "created_at": outcome.created_at,
                },
            )
        return True
    except Exception as exc:
        logger.warning("loop outcome persist failed id=%s err=%s", outcome.outcome_id, exc)
        return False


def suppress_loop(theme_key: str, *, cooldown_sec: float = 86400.0) -> bool:
    """Suppress a closed loop so it exits the coalition (reuse refractory table).

    Resolves rather than pauses: the theme is refractory-suppressed for a long
    cooldown so it won't re-ignite. Never raises.
    """
    try:
        from datetime import timedelta

        from sqlalchemy import text

        until = datetime.now(timezone.utc) + timedelta(seconds=cooldown_sec)
        with _engine().begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO substrate_reverie_refractory (theme_key, suppressed_until)
                    VALUES (:k, :until)
                    ON CONFLICT (theme_key)
                    DO UPDATE SET suppressed_until = EXCLUDED.suppressed_until, updated_at = now()
                    """
                ),
                {"k": theme_key, "until": until},
            )
        return True
    except Exception as exc:
        logger.warning("suppress_loop failed theme=%s err=%s", theme_key, exc)
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-hub/tests/test_attention_loop_closure.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/attention_loops_store.py services/orion-hub/tests/test_attention_loop_closure.py
git commit -m "feat(hub): loop outcome builder, label persistence, refractory suppression"
```

---

### Task 14: Hub API — list loops, resolve, dismiss

**Files:**
- Create: `services/orion-hub/scripts/attention_loops_routes.py`
- Modify: `services/orion-hub/scripts/api_routes.py`
- Modify: `services/orion-hub/scripts/attention_loops_store.py` (loop reader)
- Test: `services/orion-hub/tests/test_attention_loops_api.py`

- [ ] **Step 1: Write the failing API test**

Create `services/orion-hub/tests/test_attention_loops_api.py`:

```python
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from orion.schemas.attention_frame import OpenLoopV1
import scripts.attention_loops_routes as routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ORION_ATTENTION_PENDING_CARDS_ENABLED", "true")

    def _fake_loops():
        return [
            (
                OpenLoopV1(id="open-loop-x", target_type="anomaly",
                           description="reactor mismatch", why_it_matters="unresolved",
                           salience=0.7, salience_features={"evidence_strength": 0.8}),
                datetime.now(timezone.utc), 2, "",
            )
        ]

    published = {}

    def _fake_publish(outcome):
        published["outcome"] = outcome

    monkeypatch.setattr(routes, "load_pending_loops", _fake_loops)
    monkeypatch.setattr(routes, "persist_loop_outcome", lambda o: True)
    monkeypatch.setattr(routes, "suppress_loop", lambda k: True)
    monkeypatch.setattr(routes, "publish_loop_outcome", _fake_publish)

    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), published


def test_list_loops_returns_cards(client):
    c, _ = client
    resp = c.get("/api/attention/loops")
    assert resp.status_code == 200
    data = resp.json()
    assert data and data[0]["title"] == "reactor mismatch"
    assert not data[0]["title"].startswith("open-loop-")


def test_resolve_writes_outcome_and_suppresses(client):
    c, published = client
    resp = c.post("/api/attention/loops/open-loop-x/resolve", json={"note": "done"})
    assert resp.status_code == 200
    assert published["outcome"].verdict == "resolved"


def test_dismiss_writes_outcome(client):
    c, published = client
    resp = c.post("/api/attention/loops/open-loop-x/dismiss", json={"note": "noise"})
    assert resp.status_code == 200
    assert published["outcome"].verdict == "dismissed"


def test_cards_disabled_returns_empty(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("ORION_ATTENTION_PENDING_CARDS_ENABLED", raising=False)
    resp = c.get("/api/attention/loops")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-hub/tests/test_attention_loops_api.py -q`
Expected: FAIL — module `scripts.attention_loops_routes` missing.

- [ ] **Step 3: Add the loop reader to `attention_loops_store.py`**

Append (reads recent high-salience/resonance loops from the trace table + surfacing policy):

```python
SURFACE_MIN_SALIENCE = 0.5
SURFACE_MIN_AGE_SEC = 300.0


def load_pending_loops(limit: int = 50) -> list[tuple[OpenLoopV1, datetime, int, str]]:
    """Return (loop, first_seen, recurrence_count, narrative) worth a human's time.

    Surfacing policy (quiet panel): salience >= SURFACE_MIN_SALIENCE and age >=
    SURFACE_MIN_AGE_SEC, excluding themes already suppressed (resolved/dismissed).
    Reads the salience trace table; best-effort → [] on any miss.
    """
    try:
        from sqlalchemy import text

        with _engine().connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT ON (t.theme_key)
                        t.theme_key, t.loop_id, t.salience, t.features,
                        t.created_at,
                        (SELECT count(*) FROM attention_salience_trace t2
                         WHERE t2.theme_key = t.theme_key) AS recurrence_count,
                        (SELECT min(created_at) FROM attention_salience_trace t3
                         WHERE t3.theme_key = t.theme_key) AS first_seen
                    FROM attention_salience_trace t
                    WHERE t.salience >= :min_sal
                      AND NOT EXISTS (
                        SELECT 1 FROM substrate_reverie_refractory r
                        WHERE r.theme_key = t.theme_key AND r.suppressed_until > now()
                      )
                    ORDER BY t.theme_key, t.created_at DESC
                    LIMIT :limit
                    """
                ),
                {"min_sal": SURFACE_MIN_SALIENCE, "limit": limit},
            ).mappings().all()
    except Exception as exc:
        logger.warning("load_pending_loops failed: %s", exc)
        return []

    out: list[tuple[OpenLoopV1, datetime, int, str]] = []
    now = datetime.now(timezone.utc)
    for r in rows:
        features = r["features"]
        if isinstance(features, str):
            features = json.loads(features or "{}")
        first_seen = r["first_seen"] or r["created_at"]
        fs = first_seen if first_seen.tzinfo else first_seen.replace(tzinfo=timezone.utc)
        if (now - fs).total_seconds() < SURFACE_MIN_AGE_SEC:
            continue
        loop = OpenLoopV1(
            id=str(r["loop_id"]),
            description=str(r["theme_key"]),
            salience=float(r["salience"]),
            salience_features=features or {},
        )
        out.append((loop, first_seen, int(r["recurrence_count"] or 1), ""))
    return out
```

> Note: the trace table currently stores `theme_key`/`loop_id` but not `description`. For v1 the card `title` falls back to `theme_key`; a follow-up can store the human description on the trace. This is acceptable because `build_pending_card` guarantees non-empty plain text and the legibility test enforces "never id-only" by falling back to a phrase. **Correction for real legibility:** store the description on the trace — see Step 3b.

- [ ] **Step 3b: Carry a human description on the salience trace**

To keep cards legible from the trace alone, add `description` to `AttentionSalienceTraceV1` (in `orion/schemas/attention_salience.py`), the migration, the writer, and the reader:

1. `orion/schemas/attention_salience.py` — add to `AttentionSalienceTraceV1`:

```python
    description: str = Field(default="", max_length=200)
```

2. `services/orion-sql-db/manual_migration_attention_salience_trace.sql` — add column:

```sql
    description text not null default '',
```

(place after `theme_key text not null,`)

3. `services/orion-thought/app/reverie.py` `build_salience_trace` — set `description=loop.description` on the returned trace.

4. `services/orion-thought/app/store.py` `persist_salience_trace` — add `description` to the INSERT column list + params (`"description": trace.description`).

5. `load_pending_loops` query — select `t.description` and use it for `OpenLoopV1(description=...)` instead of `theme_key`.

Update the Task 9 trace test to assert `trace.description == "a"` (set `description="a"` on the loop fixture).

- [ ] **Step 4: Create `services/orion-hub/scripts/attention_loops_routes.py`**

```python
"""Operator Pending Attention API — cognitive-loop rows + Resolve/Dismiss.

Flag-gated on ORION_ATTENTION_PENDING_CARDS_ENABLED. Human-only closure: Juniper
Resolves/Dismisses; each close emits AttentionLoopOutcomeV1 (the label), persists
it, and suppresses the loop so it exits the coalition (resolves, not pauses).
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from pydantic import BaseModel

from scripts.attention_loops_store import (
    build_loop_outcome,
    build_pending_card,
    load_pending_loops,
    persist_loop_outcome,
    suppress_loop,
)

logger = logging.getLogger("orion-hub.attention_loops_api")

router = APIRouter(prefix="/api/attention/loops", tags=["attention-loops"])

_TRUTHY = {"1", "true", "yes", "on"}


def _cards_enabled() -> bool:
    return str(os.getenv("ORION_ATTENTION_PENDING_CARDS_ENABLED", "false")).strip().lower() in _TRUTHY


class CloseRequest(BaseModel):
    note: str = ""


def publish_loop_outcome(outcome) -> None:
    """Best-effort publish of the label event; swallow any bus failure."""
    try:
        from scripts.bus_publish import publish_attention_loop_outcome  # thin helper (Task 15)

        publish_attention_loop_outcome(outcome)
    except Exception as exc:
        logger.warning("publish loop outcome failed id=%s err=%s", outcome.outcome_id, exc)


@router.get("")
def list_loops(limit: int = 50):
    if not _cards_enabled():
        return []
    cards = []
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for loop, first_seen, recurrence, narrative in load_pending_loops(limit=limit):
        card = build_pending_card(
            loop, first_seen=first_seen, recurrence_count=recurrence,
            narrative=narrative, now=now,
        )
        cards.append(card.model_dump(mode="json"))
    return cards


def _close(loop_id: str, verdict: str, note: str):
    outcome = build_loop_outcome(
        loop_id=loop_id, theme_key=loop_id, verdict=verdict, actor="juniper",
        note=note, salience_at_close=0.0, features_at_close={},
    )
    persist_loop_outcome(outcome)
    suppress_loop(loop_id)
    publish_loop_outcome(outcome)
    return {"status": "ok", "outcome_id": outcome.outcome_id, "verdict": verdict}


@router.post("/{loop_id}/resolve")
def resolve_loop(loop_id: str, payload: CloseRequest):
    return _close(loop_id, "resolved", payload.note)


@router.post("/{loop_id}/dismiss")
def dismiss_loop(loop_id: str, payload: CloseRequest):
    return _close(loop_id, "dismissed", payload.note)
```

- [ ] **Step 5: Register the router in `api_routes.py`**

In `services/orion-hub/scripts/api_routes.py`, next to the existing `substrate_attention_router` include (lines 155/167), add the import and include:

```python
from .attention_loops_routes import router as attention_loops_router
```

```python
router.include_router(attention_loops_router)
```

- [ ] **Step 6: Run the API test**

Run: `pytest services/orion-hub/tests/test_attention_loops_api.py -q`
Expected: PASS (4 passed).

- [ ] **Step 7: Commit**

```bash
git add services/orion-hub/scripts/attention_loops_routes.py services/orion-hub/scripts/attention_loops_store.py services/orion-hub/scripts/api_routes.py services/orion-hub/tests/test_attention_loops_api.py orion/schemas/attention_salience.py services/orion-sql-db/manual_migration_attention_salience_trace.sql services/orion-thought/app/reverie.py services/orion-thought/app/store.py services/orion-thought/tests/test_reverie_salience_trace.py
git commit -m "feat(hub): pending-attention loops API (list/resolve/dismiss) + trace description"
```

---

### Task 15: Publish the loop-outcome event from the Hub

**Files:**
- Create/Modify: `services/orion-hub/scripts/bus_publish.py` (add helper)
- Modify: `services/orion-hub/app/settings.py`
- Test: `services/orion-hub/tests/test_attention_outcome_publish.py`

- [ ] **Step 1: Inspect the Hub's existing bus-publish pattern**

Run: `rg -n "await .*\.publish\(|BaseEnvelope|def publish" services/orion-hub/scripts/*.py services/orion-hub/app/*.py | head -30`
Expected: shows how the hub publishes envelopes (channel name + `BaseEnvelope`). Match that pattern exactly in Step 3. If a shared publish helper module already exists, add the function there instead of creating `bus_publish.py`, and update the import in `attention_loops_routes.py` accordingly.

- [ ] **Step 2: Write the failing test**

Create `services/orion-hub/tests/test_attention_outcome_publish.py`:

```python
from orion.schemas.attention_salience import AttentionLoopOutcomeV1
from scripts.bus_publish import build_loop_outcome_envelope


def test_envelope_kind_and_payload():
    outcome = AttentionLoopOutcomeV1(
        outcome_id="o1", loop_id="l1", theme_key="l1", verdict="resolved",
        actor="juniper", salience_at_close=0.5,
    )
    env = build_loop_outcome_envelope(outcome)
    assert env.kind == "attention.loop.outcome.v1"
    assert env.payload["verdict"] == "resolved"
```

- [ ] **Step 3: Add the envelope builder + publisher to `bus_publish.py`**

Create `services/orion-hub/scripts/bus_publish.py` (or append if it exists). Match the hub's real envelope/source construction discovered in Step 1; the shape below follows `orion.core.bus.bus_schemas.BaseEnvelope`:

```python
"""Thin bus-publish helpers for the pending-attention surface."""

from __future__ import annotations

import logging
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.attention_salience import AttentionLoopOutcomeV1

logger = logging.getLogger("orion-hub.bus_publish")

CHANNEL_ATTENTION_LOOP_OUTCOME = "orion:attention:loop_outcome"


def build_loop_outcome_envelope(outcome: AttentionLoopOutcomeV1) -> BaseEnvelope:
    return BaseEnvelope(
        kind="attention.loop.outcome.v1",
        source=ServiceRef(name="orion-hub", node="athena", version="0.1.0"),
        correlation_id=uuid4(),
        payload=outcome.model_dump(mode="json"),
    )


def publish_attention_loop_outcome(outcome: AttentionLoopOutcomeV1) -> None:
    """Publish the label event. Best-effort; caller swallows failures.

    Uses the hub's existing async bus client. If the hub exposes a shared
    publisher, call it here instead of opening a new connection.
    """
    import anyio

    from orion.core.bus.async_service import OrionBusAsync

    import os

    async def _run() -> None:
        bus = OrionBusAsync(url=os.getenv("ORION_BUS_URL", ""))
        await bus.connect()
        try:
            await bus.publish(CHANNEL_ATTENTION_LOOP_OUTCOME, build_loop_outcome_envelope(outcome))
        finally:
            await bus.close()

    anyio.run(_run)
```

> **Implementation note for the executor:** prefer the hub's already-connected bus singleton if one exists (found in Step 1) rather than opening a fresh connection per close. If the hub has no reusable async bus in request context, the `anyio.run` fallback above is acceptable for a low-frequency human action. Do NOT hardcode a bus URL — read `ORION_BUS_URL` (rule: `ORION_BUS_URL=redis://<tailscale-node-ip>:6379/0`).

- [ ] **Step 4: Run the test**

Run: `pytest services/orion-hub/tests/test_attention_outcome_publish.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/scripts/bus_publish.py services/orion-hub/tests/test_attention_outcome_publish.py
git commit -m "feat(hub): publish AttentionLoopOutcomeV1 label event on close"
```

---

### Task 16: Hub UI — render cognitive-loop rows + Resolve/Dismiss

**Files:**
- Modify: `services/orion-hub/templates/index.html`
- Modify: `services/orion-hub/static/js/app.js`
- Test: `services/orion-hub/tests/test_attention_loops_ui_smoke.py`

- [ ] **Step 1: Inspect the existing panel + render function**

Run: `rg -n "attentionList|renderPendingAttention|loadPendingAttention|pendingAttention" services/orion-hub/static/js/app.js`
Expected: confirms `renderPendingAttention()` (~line 5544), `loadPendingAttention()` (~line 8672), the `pendingAttention` array, and the `#attentionList`/`#attentionCount` DOM ids. Read `renderPendingAttention` fully before editing.

- [ ] **Step 2: Write the failing smoke test**

Create `services/orion-hub/tests/test_attention_loops_ui_smoke.py`:

```python
from pathlib import Path


def test_app_js_loads_cognitive_loops():
    js = Path("services/orion-hub/static/js/app.js").read_text()
    assert "/api/attention/loops" in js
    assert "loadCognitiveLoops" in js
    assert "resolve" in js and "dismiss" in js


def test_template_has_cognitive_loop_container():
    html = Path("services/orion-hub/templates/index.html").read_text()
    assert 'id="cognitiveLoopsList"' in html
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest services/orion-hub/tests/test_attention_loops_ui_smoke.py -q`
Expected: FAIL — strings not present yet.

- [ ] **Step 4: Add the container to `index.html`**

In `services/orion-hub/templates/index.html`, next to the existing `#attentionList` block (~line 809), add a sibling cognitive-loops list with a distinct source badge header:

```html
<div class="mt-3">
  <div class="flex items-center justify-between mb-1">
    <span class="text-xs font-semibold text-purple-300">Cognitive Loops
      <span class="ml-1 px-1 rounded bg-purple-900 text-purple-200 text-[10px]">mind</span>
    </span>
    <span id="cognitiveLoopsCount" class="text-xs text-gray-500">0</span>
  </div>
  <div id="cognitiveLoopsList" class="space-y-2"></div>
</div>
```

- [ ] **Step 5: Add the loader + renderer + handlers to `app.js`**

Near `loadPendingAttention` add:

```javascript
  async function loadCognitiveLoops() {
    const list = document.getElementById('cognitiveLoopsList');
    const count = document.getElementById('cognitiveLoopsCount');
    if (!list) return;
    try {
      const resp = await fetch(`${API_BASE_URL}/api/attention/loops?limit=50`);
      if (!resp.ok) return;
      const cards = await resp.json();
      if (count) count.textContent = String(cards.length);
      list.innerHTML = '';
      if (!Array.isArray(cards) || cards.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'text-xs text-gray-500 italic';
        empty.textContent = 'No cognitive loops need attention';
        list.appendChild(empty);
        return;
      }
      cards.forEach((card) => list.appendChild(renderCognitiveLoopCard(card)));
    } catch (err) {
      /* best-effort panel */
    }
  }

  function renderCognitiveLoopCard(card) {
    const el = document.createElement('div');
    el.className = 'p-2 rounded border border-purple-800 bg-gray-900';
    const title = document.createElement('div');
    title.className = 'text-xs font-semibold text-purple-200';
    title.textContent = card.title;
    const why = document.createElement('div');
    why.className = 'text-[11px] text-gray-400 mt-1';
    why.textContent = card.why_it_matters;
    const feats = document.createElement('div');
    feats.className = 'text-[10px] text-gray-500 mt-1';
    feats.textContent = (card.top_contributing_features || []).join(' · ');
    const actions = document.createElement('div');
    actions.className = 'flex gap-2 mt-2';
    const resolveBtn = document.createElement('button');
    resolveBtn.className = 'px-2 py-0.5 text-[10px] rounded bg-green-700 text-white';
    resolveBtn.textContent = 'Resolve';
    resolveBtn.onclick = () => closeCognitiveLoop(card.loop_id, 'resolve');
    const dismissBtn = document.createElement('button');
    dismissBtn.className = 'px-2 py-0.5 text-[10px] rounded bg-gray-700 text-gray-200';
    dismissBtn.textContent = 'Dismiss';
    dismissBtn.onclick = () => closeCognitiveLoop(card.loop_id, 'dismiss');
    actions.appendChild(resolveBtn);
    actions.appendChild(dismissBtn);
    el.appendChild(title);
    el.appendChild(why);
    el.appendChild(feats);
    el.appendChild(actions);
    return el;
  }

  async function closeCognitiveLoop(loopId, verdict) {
    if (!loopId) return;
    try {
      await fetch(`${API_BASE_URL}/api/attention/loops/${loopId}/${verdict}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ note: '' }),
      });
      loadCognitiveLoops();
    } catch (err) {
      /* best-effort */
    }
  }
```

Wire `loadCognitiveLoops()` into the same polling path that calls `loadPendingAttention()` (find that call site and add the new call beside it).

- [ ] **Step 6: Run the smoke test**

Run: `pytest services/orion-hub/tests/test_attention_loops_ui_smoke.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add services/orion-hub/templates/index.html services/orion-hub/static/js/app.js services/orion-hub/tests/test_attention_loops_ui_smoke.py
git commit -m "feat(hub): render cognitive-loop rows with source badge + Resolve/Dismiss"
```

---

### Task 17: Closure e2e wiring test + rollout docs

**Files:**
- Create: `services/orion-hub/tests/test_attention_closure_e2e.py`
- Modify: `services/orion-hub/README.md`
- Modify: `services/orion-thought/README.md`

- [ ] **Step 1: Write the e2e closure test (in-process, mocked persistence)**

Create `services/orion-hub/tests/test_attention_closure_e2e.py`:

```python
import scripts.attention_loops_store as store


def test_resolve_persists_outcome_and_suppresses(monkeypatch):
    calls = {"outcome": None, "suppressed": None}
    monkeypatch.setattr(store, "persist_loop_outcome", lambda o: calls.__setitem__("outcome", o) or True)
    monkeypatch.setattr(store, "suppress_loop", lambda k, **kw: calls.__setitem__("suppressed", k) or True)

    outcome = store.build_loop_outcome(
        loop_id="open-loop-x", theme_key="open-loop-x", verdict="resolved",
        actor="juniper", note="", salience_at_close=0.6, features_at_close={"evidence_strength": 0.8},
    )
    assert store.persist_loop_outcome(outcome) is True
    assert store.suppress_loop("open-loop-x") is True
    assert calls["outcome"].verdict == "resolved"
    assert calls["suppressed"] == "open-loop-x"
```

- [ ] **Step 2: Run it**

Run: `pytest services/orion-hub/tests/test_attention_closure_e2e.py -q`
Expected: PASS (1 passed).

- [ ] **Step 3: Document the rollout + migrations in READMEs**

Append to `services/orion-thought/README.md` a "Computed salience v2" section:

```markdown
## Computed salience v2 (shadow-first)

Flags (default-off): `ORION_ATTENTION_SALIENCE_V2_ENABLED`,
`ORION_ATTENTION_HABITUATION_ENABLED`, `ORION_ATTENTION_SALIENCE_WEIGHTS` (JSON override).

Shared module: `orion/substrate/attention/salience.py`. When `SALIENCE_V2` is on,
`score_loop`/`derive_salience` read the combiner salience and the reverie tick
emits `AttentionSalienceTraceV1` on `orion:attention:salience:trace` (persisted to
`attention_salience_trace`).

Migrations (apply before enabling):
`psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_salience_trace.sql`
```

Append to `services/orion-hub/README.md` a "Pending Attention cognitive loops" section:

```markdown
## Pending Attention — cognitive loops

Flag: `ORION_ATTENTION_PENDING_CARDS_ENABLED` (default-off). API:
`GET /api/attention/loops`, `POST /api/attention/loops/{id}/resolve`,
`POST /api/attention/loops/{id}/dismiss`. Resolve/Dismiss emit
`AttentionLoopOutcomeV1` on `orion:attention:loop_outcome`, persist to
`attention_loop_outcome`, and suppress the loop via `substrate_reverie_refractory`.

Migration: `psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_loop_outcome.sql`
```

- [ ] **Step 4: Commit**

```bash
git add services/orion-hub/tests/test_attention_closure_e2e.py services/orion-hub/README.md services/orion-thought/README.md
git commit -m "test(hub): closure e2e + docs for salience v2 rollout and migrations"
```

---

## Final verification (run before PR)

- [ ] **Step 1: Full touched-surface test sweep**

Run:
```bash
pytest orion/substrate/tests/ -q
pytest services/orion-thought/tests/ -q
pytest services/orion-hub/tests/test_attention_card_legibility.py services/orion-hub/tests/test_attention_loop_closure.py services/orion-hub/tests/test_attention_loops_api.py services/orion-hub/tests/test_attention_outcome_publish.py services/orion-hub/tests/test_attention_loops_ui_smoke.py services/orion-hub/tests/test_attention_closure_e2e.py -q
pytest services/orion-cortex-exec/tests/test_attention_frame.py tests/test_attention_frame_builder.py -q
```
Expected: all PASS.

- [ ] **Step 2: Acceptance eval + reverie evals unaffected**

Run:
```bash
pytest orion/substrate/tests/test_salience_discrimination_eval.py orion/substrate/tests/test_rumination_replay.py -q
pytest services/orion-thought/evals/test_reverie_hollow_guard_eval.py -q
```
Expected: discrimination eval passes; rumination replay proves the lock breaks; reverie hollow-guard eval still passes.

- [ ] **Step 3: Env parity + diff hygiene**

Run:
```bash
python scripts/sync_local_env_from_example.py
git check-ignore services/*/.env
git diff --check
git status --short
```
Expected: `.env` files ignored; no whitespace errors; no `.env` staged.

- [ ] **Step 4: Code review gate**

Run the code review skill in a subagent over the full diff; fix material findings; re-run affected tests. Summarize findings in the PR report.

- [ ] **Step 5: Docker config validation (if runtime touched)**

Run (per touched service):
```bash
docker compose --env-file .env --env-file services/orion-thought/.env -f services/orion-thought/docker-compose.yml config
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml config
```
Expected: compose renders without error. If Docker cannot run here, say so and list restart commands for the operator.

- [ ] **Step 6: Restart commands for the PR (do not run sudo yourself)**

List for Juniper:
```bash
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_salience_trace.sql
psql "$POSTGRES_URI" -f services/orion-sql-db/manual_migration_attention_loop_outcome.sql
docker compose -f services/orion-thought/docker-compose.yml up -d --build
docker compose -f services/orion-hub/docker-compose.yml up -d --build
docker compose -f services/orion-substrate-runtime/docker-compose.yml up -d --build
docker compose -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

- [ ] **Step 7: Write the PR description**

Use the `AGENTS.md §18` PR template. Cover: schemas added, `OpenLoopV1` change (legacy 7 fields deprecated one release), channels added, env keys added + `.env` synced, tables added, the shadow→live→habituation→cards rollout, and the acceptance evidence (discrimination eval + rumination replay).

---

## Rollout order (post-merge, operator)

1. Merge with all flags off (shadow). Apply the two migrations.
2. Enable `ORION_ATTENTION_SALIENCE_V2_ENABLED` on `orion-thought` first; inspect `attention_salience_trace` for real, discriminating feature vectors.
3. Enable `SALIENCE_V2` on `orion-cortex-exec` + `orion-substrate-runtime` (selection now uses real salience). Watch resonance-alert counts.
4. Enable `ORION_ATTENTION_HABITUATION_ENABLED`; confirm the rumination lock (`loop:open-loop-2eb998452183`) stops re-electing.
5. Enable `ORION_ATTENTION_PENDING_CARDS_ENABLED` on `orion-hub`; verify legible rows + Resolve/Dismiss write labels.

Rollback at any stage = flip the relevant flag off → old constant path. Each stage independently reversible.
