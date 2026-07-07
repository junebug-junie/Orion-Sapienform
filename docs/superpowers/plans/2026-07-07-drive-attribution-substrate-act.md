# Drive Tick Attribution + Substrate Act Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the live substrate-fed motivation path so per-tick drive dominance and goal `drive_origin` come from event attribution (not saturated flat pressures), and Tier B fetch + episode journal run on world-pulse metabolism without requiring a fresh goal publish.

**Architecture:** Approach A adds `drive_attribution.py` — sum `tension.magnitude × drive_impacts` per tick, merge `metabolism.drive_deltas`, compute `dominant_drive` with structural tension-kind tie-break. `DriveAuditV1` emits `tick_attribution`; `bus_worker.py` wires it before goal propose. Approach B adds `SubstrateEpisodeIntentV1` + `maybe_execute_substrate_act_after_metabolism` in `policy_act.py`, decoupling fetch/journal from `goal_decision.proposal is not None`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, existing `DriveEngine` / `GoalProposalEngine` / `capability_policy.evaluate()` / `metabolize_substrate_signals()`.

**Spec:** [`docs/superpowers/specs/2026-07-07-drive-attribution-substrate-act-design.md`](../specs/2026-07-07-drive-attribution-substrate-act-design.md)

**Related (v1 shipped):** [`docs/superpowers/plans/2026-07-06-substrate-fed-motivation.md`](./2026-07-06-substrate-fed-motivation.md)

**Branch:** `feat/drive-attribution-substrate-act-v1.1`

**Worktree setup (run once before Task 1):**

```bash
cd /mnt/scripts/Orion-Sapienform
git status --short
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-drive-attribution -b feat/drive-attribution-substrate-act-v1.1
cd ../Orion-Sapienform-drive-attribution
```

---

## File map

| File | Phase | Action | Responsibility |
|------|-------|--------|----------------|
| `orion/spark/concept_induction/drive_attribution.py` | 1 | **Create** | `compute_tick_attribution`, `dominant_drive_from_attribution`, `select_lead_tension` |
| `orion/spark/concept_induction/tests/test_drive_attribution.py` | 1 | **Create** | Acceptance tests 1–3 from spec |
| `orion/core/schemas/drives.py` | 1 | Modify | `DriveAuditV1.tick_attribution` optional field |
| `orion/spark/concept_induction/audit.py` | 2 | Modify | Accept attribution + dominant override; emit `tick_attribution` |
| `orion/spark/concept_induction/goals.py` | 2 | Modify | `tick_attribution` / deprecated `audit_dominant` origin source |
| `orion/spark/concept_induction/settings.py` | 2 | Modify | Default `GOAL_DRIVE_ORIGIN_SOURCE=tick_attribution` |
| `orion/spark/concept_induction/bus_worker.py` | 2–3 | Modify | Wire attribution + metabolism deltas; decouple substrate act |
| `orion/autonomy/models.py` | 3 | Modify | `SubstrateEpisodeIntentV1`, `SubstrateActResultV1` |
| `orion/autonomy/policy_act.py` | 3 | Modify | `resolve_episode_intent`, `maybe_execute_substrate_act_after_metabolism` |
| `orion/autonomy/tests/test_policy_act.py` | 3 | Modify | Tests 4–5 from spec |
| `orion/spark/concept_induction/tests/test_metabolism_hook.py` | 3 | Modify | Substrate act without goal publish |
| `services/orion-spark-concept-induction/.env_example` | 4 | Modify | `GOAL_DRIVE_ORIGIN_SOURCE=tick_attribution` |
| `scripts/sync_local_env_from_example.py` | 4 | Modify | Add env keys to `SYNC_EXACT` / `SYNC_PREFIXES` |

**Non-goals (this plan):** DriveEngine saturation rework (Appendix C), new drive taxonomy, keyword triggers, goal cooldown duration change.

---

## Schema note (read before coding)

- `metabolism.drive_deltas` is already computed in `orion/autonomy/substrate_metabolism.py` (`_PREDICTIVE_DELTA = 0.15` per gap section) but **never merged** into dominance in `bus_worker.py` today.
- Saturated pressures (~0.73 on all six drives) remain on `DriveStateV1` for history/UI; **decisions use `tick_attribution` only**.
- Policy `web.fetch.readonly` requires `drive_origin=predictive` and `goal` with `proposal_status=proposed` — synthetic goal from episode intent must satisfy that.
- `clamp01` already exists in `orion/spark/concept_induction/tensions.py` — reuse it, do not duplicate.

---

# Phase 1 — Drive tick attribution module

Ships: pure attribution functions + unit tests. No bus wiring yet.

## Task 1: `drive_attribution.py` core + GPU gap test

**Files:**
- Create: `orion/spark/concept_induction/drive_attribution.py`
- Create: `orion/spark/concept_induction/tests/test_drive_attribution.py`

- [ ] **Step 1: Write the failing test**

Create `orion/spark/concept_induction/tests/test_drive_attribution.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.drives import TensionEventV1
from orion.spark.concept_induction.drive_attribution import (
    DRIVE_KEYS,
    compute_tick_attribution,
    dominant_drive_from_attribution,
    select_lead_tension,
)


def _gap_tension() -> TensionEventV1:
    return TensionEventV1.model_validate(
        {
            "artifact_id": "tension-gap-gpu",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "substrate.world_coverage_gap",
            "magnitude": 0.65,
            "drive_impacts": {"predictive": 0.15},
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


def _contradiction_tension() -> TensionEventV1:
    return TensionEventV1.model_validate(
        {
            "artifact_id": "tension-coh",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "tension.contradiction.v1",
            "magnitude": 0.5,
            "drive_impacts": {"coherence": 0.4, "autonomy": 0.4},
            "provenance": {"intake_channel": "orion:metacognition:tick"},
        }
    )


def test_attribution_gpu_gap_predictive_dominant() -> None:
    """Spec acceptance 1: saturated pressures irrelevant; gap → predictive."""
    gap = _gap_tension()
    attribution = compute_tick_attribution(
        [gap],
        metabolism_deltas={"predictive": 0.15},
    )
    dominant = dominant_drive_from_attribution(attribution, lead_tension=gap)
    assert dominant == "predictive"
    assert attribution["predictive"] > attribution.get("autonomy", 0.0)


def test_attribution_no_alphabetical_autonomy() -> None:
    """Spec acceptance 2: equal attribution ties break on lead tension kind, not 'autonomy'."""
    tension = _contradiction_tension()
    # Equal per-drive scores — legacy max(sorted(pressures)) would pick autonomy.
    attribution = {key: 0.1 for key in DRIVE_KEYS}
    dominant = dominant_drive_from_attribution(attribution, lead_tension=tension)
    assert dominant == "coherence"
    assert dominant != "autonomy"


def test_metabolism_deltas_merge_into_attribution() -> None:
    """Spec acceptance 3: drive_deltas alone shift dominance when tensions empty."""
    attribution = compute_tick_attribution([], metabolism_deltas={"predictive": 0.2})
    dominant = dominant_drive_from_attribution(attribution, lead_tension=None)
    assert dominant == "predictive"
    assert attribution["predictive"] == 0.2


def test_select_lead_tension_prefers_gap_on_magnitude_tie() -> None:
    gap = _gap_tension()
    other = _contradiction_tension().model_copy(update={"magnitude": gap.magnitude})
    lead = select_lead_tension([other, gap])
    assert lead is not None
    assert lead.kind == "substrate.world_coverage_gap"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
pytest orion/spark/concept_induction/tests/test_drive_attribution.py -v
```

Expected: `ModuleNotFoundError: No module named 'orion.spark.concept_induction.drive_attribution'`

- [ ] **Step 3: Write minimal implementation**

Create `orion/spark/concept_induction/drive_attribution.py`:

```python
from __future__ import annotations

from typing import Sequence

from orion.core.schemas.drives import TensionEventV1
from orion.spark.concept_induction.tensions import clamp01

DRIVE_KEYS = ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")

_GAP_KIND = "substrate.world_coverage_gap"

_PRIMARY_DRIVE_BY_KIND: dict[str, str] = {
    _GAP_KIND: "predictive",
    "tension.contradiction.v1": "coherence",
    "tension.distress.v1": "relational",
    "tension.identity_drift.v1": "continuity",
    "tension.cognitive_load.v1": "capability",
}


def compute_tick_attribution(
    tensions: Sequence[TensionEventV1],
    *,
    metabolism_deltas: dict[str, float] | None = None,
) -> dict[str, float]:
    """Sum magnitude × drive_impact weight per drive for THIS tick only."""
    attribution: dict[str, float] = {key: 0.0 for key in DRIVE_KEYS}
    for tension in tensions:
        mag = clamp01(tension.magnitude)
        for drive, weight in (tension.drive_impacts or {}).items():
            if drive not in attribution:
                continue
            attribution[drive] += mag * clamp01(float(weight))
    for drive, delta in (metabolism_deltas or {}).items():
        if drive in attribution:
            attribution[drive] += float(delta)
    return attribution


def primary_drive_for_tension_kind(
    kind: str,
    *,
    drive_impacts: dict[str, float] | None = None,
) -> str | None:
    """Structural map for tie-break; not digest keyword matching."""
    if kind == "tension.drive_competition.v1" and drive_impacts:
        ranked = sorted(drive_impacts.items(), key=lambda item: (-float(item[1]), item[0]))
        return ranked[0][0] if ranked else None
    return _PRIMARY_DRIVE_BY_KIND.get(kind)


def _tied_at_max(attribution: dict[str, float]) -> list[str]:
    if not attribution:
        return []
    max_val = max(attribution.values())
    if max_val <= 0.0:
        return []
    return sorted([drive for drive, val in attribution.items() if val == max_val])


def dominant_drive_from_attribution(
    attribution: dict[str, float],
    *,
    lead_tension: TensionEventV1 | None = None,
) -> str | None:
    """Argmax attribution; tie-break via primary_drive(lead_tension.kind)."""
    tied = _tied_at_max(attribution)
    if not tied:
        return None
    if len(tied) == 1:
        return tied[0]

    if lead_tension is not None:
        primary = primary_drive_for_tension_kind(
            lead_tension.kind,
            drive_impacts=lead_tension.drive_impacts,
        )
        if primary and primary in tied:
            return primary
        impacts = lead_tension.drive_impacts or {}
        ranked = sorted(
            ((drive, float(impacts.get(drive, 0.0))) for drive in tied),
            key=lambda item: (-item[1], item[0]),
        )
        if ranked and ranked[0][1] > 0.0:
            return ranked[0][0]

    return tied[0]


def select_lead_tension(tensions: Sequence[TensionEventV1]) -> TensionEventV1 | None:
    """Highest magnitude this tick; substrate gap preferred on magnitude tie."""
    if not tensions:
        return None

    def _sort_key(t: TensionEventV1) -> tuple[float, int, str]:
        gap_bias = 1 if t.kind == _GAP_KIND else 0
        return (-float(t.magnitude), gap_bias, t.kind)

    return sorted(tensions, key=_sort_key)[0]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/spark/concept_induction/tests/test_drive_attribution.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/drive_attribution.py \
        orion/spark/concept_induction/tests/test_drive_attribution.py
git commit -m "$(cat <<'EOF'
feat: add per-tick drive attribution for dominance decisions

EOF
)"
```

---

## Task 2: `DriveAuditV1.tick_attribution` schema

**Files:**
- Modify: `orion/core/schemas/drives.py` (after `dominant_drive` on `DriveAuditV1`, ~line 106)
- Test: extend `orion/spark/concept_induction/tests/test_drive_attribution.py`

- [ ] **Step 1: Write the failing test**

Append to `orion/spark/concept_induction/tests/test_drive_attribution.py`:

```python
from orion.core.schemas.drives import DriveAuditV1


def test_drive_audit_v1_accepts_tick_attribution() -> None:
    audit = DriveAuditV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.drives.audit.v1",
            "dominant_drive": "predictive",
            "tick_attribution": {"predictive": 0.25, "autonomy": 0.1},
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )
    assert audit.tick_attribution["predictive"] == 0.25
    assert audit.dominant_drive == "predictive"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/spark/concept_induction/tests/test_drive_attribution.py::test_drive_audit_v1_accepts_tick_attribution -v
```

Expected: validation error — `tick_attribution` extra/forbidden field

- [ ] **Step 3: Write minimal implementation**

In `orion/core/schemas/drives.py`, on `DriveAuditV1` add after `dominant_drive`:

```python
    tick_attribution: Dict[str, float] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/spark/concept_induction/tests/test_drive_attribution.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add orion/core/schemas/drives.py orion/spark/concept_induction/tests/test_drive_attribution.py
git commit -m "$(cat <<'EOF'
feat: add tick_attribution field to DriveAuditV1 schema

EOF
)"
```

---

# Phase 2 — Audit / goals / bus_worker wiring

Ships: live `tick_attribution` on audit; goal `drive_origin` from attribution; metabolism deltas merged.

## Task 3: `build_drive_audit` uses attribution

**Files:**
- Modify: `orion/spark/concept_induction/audit.py`
- Create test in: `orion/spark/concept_induction/tests/test_drive_attribution.py`

- [ ] **Step 1: Write the failing test**

Append to `test_drive_attribution.py`:

```python
from datetime import datetime, timezone
from uuid import uuid4

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import DriveStateV1
from orion.spark.concept_induction.audit import build_drive_audit


def test_audit_emits_tick_attribution() -> None:
    """Spec acceptance 6."""
    gap = _gap_tension()
    env = BaseEnvelope(
        kind="world.pulse.run.result.v1",
        source=ServiceRef(name="test", version="0"),
        correlation_id=uuid4(),
        payload={},
    )
    drive_state = DriveStateV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.drives.state.v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "pressures": {k: 0.73 for k in DRIVE_KEYS},
            "activations": {k: True for k in DRIVE_KEYS},
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )
    attribution = compute_tick_attribution([gap], metabolism_deltas={"predictive": 0.15})
    dominant = dominant_drive_from_attribution(attribution, lead_tension=gap)
    audit = build_drive_audit(
        env=env,
        intake_channel="orion:world_pulse:run:result",
        drive_state=drive_state,
        tensions=[gap],
        tick_attribution=attribution,
        dominant_drive=dominant,
    )
    assert audit.tick_attribution["predictive"] > 0.0
    assert audit.dominant_drive == "predictive"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/spark/concept_induction/tests/test_drive_attribution.py::test_audit_emits_tick_attribution -v
```

Expected: `TypeError: build_drive_audit() got an unexpected keyword argument 'tick_attribution'`

- [ ] **Step 3: Write minimal implementation**

Replace `build_drive_audit` in `orion/spark/concept_induction/audit.py`:

```python
def build_drive_audit(
    *,
    env: BaseEnvelope,
    intake_channel: str,
    drive_state: DriveStateV1,
    tensions: Iterable[TensionEventV1],
    tick_attribution: dict[str, float] | None = None,
    dominant_drive: str | None = None,
) -> DriveAuditV1:
    tension_list = list(tensions)
    if dominant_drive is None:
        if tick_attribution:
            from .drive_attribution import dominant_drive_from_attribution, select_lead_tension

            dominant_drive = dominant_drive_from_attribution(
                tick_attribution,
                lead_tension=select_lead_tension(tension_list),
            )
        elif drive_state.pressures:
            dominant_drive = max(
                sorted(drive_state.pressures),
                key=lambda key: drive_state.pressures.get(key, 0.0),
            )
    active_drives = [key for key, active in sorted(drive_state.activations.items()) if active]
    evidence_items = build_evidence_items(env, intake_channel, drive_state.provenance.evidence_text)
    source_event_ref = build_source_event_ref(env, intake_channel)
    tension_refs = [tension.artifact_id for tension in tension_list]
    tension_kinds = [tension.kind for tension in tension_list]
    summary = None
    if dominant_drive:
        summary = f"{drive_state.subject} pressure concentrates on {dominant_drive}"
    return DriveAuditV1(
        artifact_id=_artifact_id(drive_state.subject, drive_state.correlation_id, "drive-audit"),
        subject=drive_state.subject,
        model_layer=drive_state.model_layer,
        entity_id=drive_state.entity_id,
        kind="memory.drives.audit.v1",
        ts=drive_state.updated_at,
        confidence=drive_state.confidence,
        correlation_id=drive_state.correlation_id,
        trace_id=drive_state.trace_id or extract_trace_id(env),
        turn_id=drive_state.turn_id or extract_turn_id(env),
        provenance=drive_state.provenance.model_copy(update={
            "source_event_refs": [source_event_ref],
            "evidence_items": evidence_items,
            "tension_refs": tension_refs,
        }),
        related_nodes=drive_state.related_nodes + tension_refs,
        drive_pressures=drive_state.pressures,
        drive_activations=drive_state.activations,
        active_drives=active_drives,
        dominant_drive=dominant_drive,
        tick_attribution=dict(tick_attribution or {}),
        tension_kinds=tension_kinds,
        source_event_refs=[source_event_ref],
        evidence_items=evidence_items,
        summary=summary,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/spark/concept_induction/tests/test_drive_attribution.py::test_audit_emits_tick_attribution -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/audit.py orion/spark/concept_induction/tests/test_drive_attribution.py
git commit -m "$(cat <<'EOF'
feat: compute drive audit dominance from tick attribution

EOF
)"
```

---

## Task 4: Goal drive origin source `tick_attribution`

**Files:**
- Modify: `orion/spark/concept_induction/goals.py` (`_drive_origin`, ~lines 35–45)
- Modify: `orion/spark/concept_induction/settings.py` (line 132)
- Modify: `orion/spark/concept_induction/tests/test_goals.py`

- [ ] **Step 1: Write the failing test**

Append to `orion/spark/concept_induction/tests/test_goals.py`:

```python
def test_drive_origin_from_tick_attribution_dominant():
    engine = GoalProposalEngine(cooldown_minutes=0)
    drive_state = DriveStateV1.model_validate({
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "kind": "memory.drives.state.v1",
        "pressures": {"autonomy": 0.95, "predictive": 0.73},
        "activations": {},
        "updated_at": datetime.now(timezone.utc),
        "provenance": {
            "intake_channel": "x",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
    })
    origin = engine._drive_origin(
        drive_state,
        dominant_drive="predictive",
        source="tick_attribution",
    )
    assert origin == "predictive"


def test_audit_dominant_alias_maps_to_tick_attribution():
    engine = GoalProposalEngine(cooldown_minutes=0)
    drive_state = _drive_state("trace-aaa")
    origin = engine._drive_origin(
        drive_state,
        dominant_drive="relational",
        source="audit_dominant",
    )
    assert origin == "relational"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/spark/concept_induction/tests/test_goals.py::test_drive_origin_from_tick_attribution_dominant -v
```

Expected: FAIL — returns `autonomy` (legacy pressures path)

- [ ] **Step 3: Write minimal implementation**

In `orion/spark/concept_induction/goals.py`, replace `_drive_origin`:

```python
    @staticmethod
    def _normalize_drive_origin_source(source: str) -> str:
        if source == "audit_dominant":
            return "tick_attribution"
        return source

    @staticmethod
    def _drive_origin(
        drive_state: DriveStateV1,
        *,
        dominant_drive: str | None = None,
        source: str = "pressures",
    ) -> str:
        normalized = GoalProposalEngine._normalize_drive_origin_source(source)
        if normalized == "tick_attribution" and dominant_drive:
            return dominant_drive.strip().lower()
        if not drive_state.pressures:
            return "continuity"
        return max(sorted(drive_state.pressures), key=lambda key: drive_state.pressures.get(key, 0.0))
```

In `orion/spark/concept_induction/settings.py` line 132:

```python
    goal_drive_origin_source: str = Field("tick_attribution", alias="GOAL_DRIVE_ORIGIN_SOURCE")
```

- [ ] **Step 4: Run tests**

```bash
pytest orion/spark/concept_induction/tests/test_goals.py -v
```

Expected: all passed (including existing `test_drive_origin_from_audit_dominant`)

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/goals.py \
        orion/spark/concept_induction/settings.py \
        orion/spark/concept_induction/tests/test_goals.py
git commit -m "$(cat <<'EOF'
feat: default goal drive_origin to tick attribution dominant drive

EOF
)"
```

---

## Task 5: Wire attribution + metabolism deltas in `bus_worker.py`

**Files:**
- Modify: `orion/spark/concept_induction/bus_worker.py` (~lines 535–621)

- [ ] **Step 1: Add imports at top of `bus_worker.py`**

```python
from .drive_attribution import (
    compute_tick_attribution,
    dominant_drive_from_attribution,
    select_lead_tension,
)
```

- [ ] **Step 2: Track metabolism drive_deltas**

After `metabolism_tensions: List[TensionEventV1] = []` (~line 535), add:

```python
        metabolism_drive_deltas: dict[str, float] = {}
```

Inside the metabolism block after `metabolism = metabolize_substrate_signals(...)`:

```python
                    metabolism_drive_deltas = dict(metabolism.drive_deltas)
```

- [ ] **Step 3: Compute attribution before `build_drive_audit`**

Replace the bare `build_drive_audit(...)` call (~line 621) with:

```python
        tick_attribution = compute_tick_attribution(
            all_tensions,
            metabolism_deltas=metabolism_drive_deltas or None,
        )
        lead_tension = select_lead_tension(all_tensions)
        dominant_drive = dominant_drive_from_attribution(
            tick_attribution,
            lead_tension=lead_tension,
        )
        drive_audit = build_drive_audit(
            env=env,
            intake_channel=intake_channel,
            drive_state=drive_state,
            tensions=all_tensions,
            tick_attribution=tick_attribution,
            dominant_drive=dominant_drive,
        )
```

- [ ] **Step 4: Run focused tests**

```bash
pytest orion/spark/concept_induction/tests/test_drive_attribution.py \
       orion/spark/concept_induction/tests/test_metabolism_hook.py \
       orion/spark/concept_induction/tests/test_goals.py -q
```

Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/bus_worker.py
git commit -m "$(cat <<'EOF'
feat: wire metabolism drive_deltas into tick attribution on bus worker

EOF
)"
```

---

# Phase 3 — Substrate act path (decouple fetch from goal publish)

Ships: fetch + journal on world-pulse metabolism when gap signals present, even if goal cooldown suppresses publish.

## Task 6: Episode intent models

**Files:**
- Modify: `orion/autonomy/models.py` (append after `MetabolismResultV1`, ~line 169)
- Create: `orion/autonomy/tests/test_substrate_act_models.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_substrate_act_models.py`:

```python
from orion.autonomy.models import SubstrateActResultV1, SubstrateEpisodeIntentV1


def test_substrate_episode_intent_v1_fields() -> None:
    intent = SubstrateEpisodeIntentV1(
        goal_artifact_id="episode-wp-run-1",
        drive_origin="predictive",
        spawned_correlation_id="wp-run-1",
        subject="orion",
    )
    assert intent.drive_origin == "predictive"
    assert intent.spawned_correlation_id == "wp-run-1"


def test_substrate_act_result_v1_optional_outcomes() -> None:
    result = SubstrateActResultV1(fetch_attempted=True, journal_attempted=False)
    assert result.fetch_attempted is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/autonomy/tests/test_substrate_act_models.py -v
```

Expected: `ImportError: cannot import name 'SubstrateEpisodeIntentV1'`

- [ ] **Step 3: Write minimal implementation**

Append to `orion/autonomy/models.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SubstrateEpisodeIntentV1:
    goal_artifact_id: str
    drive_origin: str
    spawned_correlation_id: str
    subject: str


class SubstrateActResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    fetch_attempted: bool = False
    journal_attempted: bool = False
    fetch_outcome_id: str | None = None
    journal_entry_id: str | None = None
```

Move the `dataclass` import to the top of `models.py` with existing imports (file already uses pydantic; add `from dataclasses import dataclass`).

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/autonomy/tests/test_substrate_act_models.py -v
```

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/models.py orion/autonomy/tests/test_substrate_act_models.py
git commit -m "$(cat <<'EOF'
feat: add substrate episode intent and act result models

EOF
)"
```

---

## Task 7: `maybe_execute_substrate_act_after_metabolism`

**Files:**
- Modify: `orion/autonomy/policy_act.py`
- Modify: `orion/autonomy/tests/test_policy_act.py`

- [ ] **Step 1: Write the failing tests**

Append to `orion/autonomy/tests/test_policy_act.py`:

```python
from orion.autonomy.models import SubstrateEpisodeIntentV1
from orion.autonomy.policy_act import (
    maybe_execute_substrate_act_after_metabolism,
    resolve_episode_intent,
)


def _intent() -> SubstrateEpisodeIntentV1:
    return SubstrateEpisodeIntentV1(
        goal_artifact_id="episode-wp-run-gap-gpu",
        drive_origin="predictive",
        spawned_correlation_id="wp-run-gap-gpu",
        subject="orion",
    )


class _FakeStore:
    def __init__(self, slot: dict | None = None) -> None:
        self._slot = slot or {}

    def load_goal_slot(self, subject: str, drive_origin: str) -> dict:
        return dict(self._slot)


def test_resolve_episode_intent_uses_predictive_slot() -> None:
    store = _FakeStore({"artifact_id": "goal-predictive-slot", "signature": "sig"})
    intent = resolve_episode_intent(store=store, subject="orion", run_id="wp-run-1")
    assert intent.goal_artifact_id == "goal-predictive-slot"
    assert intent.drive_origin == "predictive"


def test_resolve_episode_intent_synthetic_when_slot_empty() -> None:
    intent = resolve_episode_intent(store=_FakeStore(), subject="orion", run_id="wp-run-1")
    assert intent.goal_artifact_id == "episode-wp-run-1"
    assert intent.spawned_correlation_id == "wp-run-1"


@pytest.mark.asyncio
async def test_substrate_act_runs_when_goal_suppressed(monkeypatch, tmp_path) -> None:
    """Spec acceptance 4: proposal=None path still executes fetch."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"success": True, "urls": ["https://example.com/a"]})
    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[_gap_signal()],
        fetch_backend=backend,
    )
    assert result.fetch_attempted is True
    backend.assert_awaited_once()


@pytest.mark.asyncio
async def test_substrate_act_denied_without_gap_signal(monkeypatch) -> None:
    """Spec acceptance 5."""
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    result = await maybe_execute_substrate_act_after_metabolism(
        episode_intent=_intent(),
        drive_state=_drive_state(),
        curiosity_signals=[],
        fetch_backend=AsyncMock(),
    )
    assert result.fetch_attempted is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest orion/autonomy/tests/test_policy_act.py::test_substrate_act_runs_when_goal_suppressed -v
```

Expected: `ImportError: cannot import name 'maybe_execute_substrate_act_after_metabolism'`

- [ ] **Step 3: Write implementation**

Append to `orion/autonomy/policy_act.py`:

```python
from orion.autonomy.models import SubstrateActResultV1, SubstrateEpisodeIntentV1


def resolve_episode_intent(
    *,
    store,
    subject: str,
    run_id: str,
    drive_origin: str = "predictive",
) -> SubstrateEpisodeIntentV1:
    slot = store.load_goal_slot(subject, drive_origin)
    artifact_id = slot.get("artifact_id") if isinstance(slot, dict) else None
    if isinstance(artifact_id, str) and artifact_id.strip():
        return SubstrateEpisodeIntentV1(
            goal_artifact_id=artifact_id.strip(),
            drive_origin=drive_origin,
            spawned_correlation_id=run_id,
            subject=subject,
        )
    return SubstrateEpisodeIntentV1(
        goal_artifact_id=f"episode-{run_id}",
        drive_origin="predictive",
        spawned_correlation_id=run_id,
        subject=subject,
    )


def goal_proposal_from_episode_intent(intent: SubstrateEpisodeIntentV1) -> GoalProposalV1:
    return GoalProposalV1.model_validate(
        {
            "artifact_id": intent.goal_artifact_id,
            "subject": intent.subject,
            "model_layer": "self-model",
            "entity_id": f"self:{intent.subject}",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "Substrate episode intent (synthetic goal for policy).",
            "proposal_signature": f"episode-{intent.spawned_correlation_id}",
            "drive_origin": intent.drive_origin,
            "proposal_status": "proposed",
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


async def maybe_execute_substrate_act_after_metabolism(
    *,
    episode_intent: SubstrateEpisodeIntentV1,
    drive_state: DriveStateV1,
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
    spawned_correlation_id: str | None = None,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
    journal_dispatch: Callable[..., Awaitable[dict[str, Any]]] | None = None,
    budget_used: dict[str, int] | None = None,
    episode_journal_enabled: bool = False,
) -> SubstrateActResultV1:
    run_id = spawned_correlation_id or episode_intent.spawned_correlation_id
    synthetic_goal = goal_proposal_from_episode_intent(episode_intent)
    result = SubstrateActResultV1()

    fetch_decision, fetch_outcome = await maybe_execute_readonly_fetch_after_goal(
        goal=synthetic_goal,
        drive_state=drive_state,
        curiosity_signals=curiosity_signals,
        spawned_correlation_id=run_id,
        fetch_backend=fetch_backend,
        budget_used=budget_used,
    )
    if fetch_decision.outcome == "allowed" and fetch_outcome is not None:
        result = result.model_copy(update={"fetch_attempted": True, "fetch_outcome_id": fetch_outcome.action_id})

    if not episode_journal_enabled or fetch_outcome is None:
        return result

    journal_decision, journal_payload = await maybe_compose_autonomy_episode_after_fetch(
        goal=synthetic_goal,
        drive_state=drive_state,
        curiosity_signals=curiosity_signals,
        spawned_correlation_id=run_id,
        fetch_outcome=fetch_outcome,
        journal_dispatch=journal_dispatch,
        budget_used=budget_used,
    )
    if journal_decision.outcome == "allowed" and journal_payload is not None:
        entry_id = None
        if isinstance(journal_payload.get("write"), dict):
            entry_id = journal_payload["write"].get("entry_id")
        result = result.model_copy(update={"journal_attempted": True, "journal_entry_id": entry_id})
    return result
```

- [ ] **Step 4: Run policy act tests**

```bash
pytest orion/autonomy/tests/test_policy_act.py -v
```

Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/policy_act.py orion/autonomy/tests/test_policy_act.py
git commit -m "$(cat <<'EOF'
feat: substrate act path decoupled from fresh goal publish

EOF
)"
```

---

## Task 8: Decouple fetch/journal in `bus_worker.py`

**Files:**
- Modify: `orion/spark/concept_induction/bus_worker.py` (~lines 650–699)
- Modify: `orion/spark/concept_induction/tests/test_metabolism_hook.py`

- [ ] **Step 1: Write the failing test**

Append to `orion/spark/concept_induction/tests/test_metabolism_hook.py`:

```python
@pytest.mark.asyncio
async def test_substrate_act_runs_when_goal_suppressed(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    substrate_mock = AsyncMock(return_value=MagicMock(fetch_attempted=True, journal_attempted=False))
    monkeypatch.setattr(
        "orion.spark.concept_induction.bus_worker.maybe_execute_substrate_act_after_metabolism",
        substrate_mock,
    )
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {
        "pressures": {k: 0.73 for k in ("coherence", "continuity", "capability", "relational", "predictive", "autonomy")},
        "activations": {"predictive": True},
    }
    worker.store.load_goal_slot.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({}, {"predictive": True}))
    worker._publish_tension_event = AsyncMock(return_value=None)
    worker._publish_drive_state = AsyncMock(return_value=None)
    worker._publish_artifact = AsyncMock(return_value=None)
    worker._publish_dossier = AsyncMock(return_value=None)
    # Goal suppressed by cooldown
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature="sig-cooldown"))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    substrate_mock.assert_awaited_once()
    assert substrate_mock.await_args.kwargs["spawned_correlation_id"] == "wp-run-hook"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/spark/concept_induction/tests/test_metabolism_hook.py::test_substrate_act_runs_when_goal_suppressed -v
```

Expected: FAIL — `substrate_mock.assert_awaited_once()` assertion error (not called today)

- [ ] **Step 3: Refactor `bus_worker.py`**

Add import:

```python
from orion.autonomy.policy_act import (
    maybe_compose_autonomy_episode_after_fetch,
    maybe_execute_readonly_fetch_after_goal,
    maybe_execute_substrate_act_after_metabolism,
    resolve_episode_intent,
)
```

(Consolidate with existing `policy_act` imports if already present.)

Replace the block inside `if goal_decision.proposal is not None:` that runs fetch/journal (~lines 655–699) with:

```python
        if goal_decision.proposal is not None:
            await self._publish_artifact(goal_decision.proposal, self.cfg.goal_proposal_channel, env.correlation_id)
            published_artifacts.append(goal_decision.proposal)
            maybe_archive_after_goal_publish(subject=subject)
        elif goal_decision.suppressed_signature:
            suppressed_signatures.append(goal_decision.suppressed_signature)

        if (
            env.kind == "world.pulse.run.result.v1"
            and metabolism_enabled()
            and metabolism_curiosity_signals
            and spawned_correlation_id
        ):
            policy_budget: dict[str, int] = {}
            intent = resolve_episode_intent(
                store=self.store,
                subject=subject,
                run_id=spawned_correlation_id,
            )
            try:

                async def _journal_dispatch(**kwargs):
                    return await self._dispatch_autonomy_episode_journal(env, **kwargs)

                await maybe_execute_substrate_act_after_metabolism(
                    episode_intent=intent,
                    drive_state=drive_state,
                    curiosity_signals=metabolism_curiosity_signals,
                    spawned_correlation_id=spawned_correlation_id,
                    fetch_backend=self._fetch_backend,
                    journal_dispatch=_journal_dispatch,
                    budget_used=policy_budget,
                    episode_journal_enabled=self.cfg.autonomy_episode_journal_enabled,
                )
            except Exception:
                logger.warning(
                    "substrate_act_failed run_id=%s intent=%s",
                    spawned_correlation_id,
                    intent.goal_artifact_id,
                    exc_info=True,
                )
```

Remove the old nested fetch/journal block that lived only under `goal_decision.proposal is not None`.

Update `test_policy_fetch_runs_after_goal_publish` in `test_metabolism_hook.py` to patch `maybe_execute_substrate_act_after_metabolism` instead of the two separate mocks (fetch still runs, now via substrate act).

- [ ] **Step 4: Run metabolism hook + policy tests**

```bash
pytest orion/spark/concept_induction/tests/test_metabolism_hook.py \
       orion/autonomy/tests/test_policy_act.py -q
```

Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/bus_worker.py \
        orion/spark/concept_induction/tests/test_metabolism_hook.py
git commit -m "$(cat <<'EOF'
feat: run substrate act after metabolism regardless of goal publish

EOF
)"
```

---

# Phase 4 — Env parity + agent gate

## Task 9: Env template + sync script

**Files:**
- Modify: `services/orion-spark-concept-induction/.env_example` (line 61)
- Modify: `scripts/sync_local_env_from_example.py`

- [ ] **Step 1: Update `.env_example`**

Change:

```text
GOAL_DRIVE_ORIGIN_SOURCE=audit_dominant
```

to:

```text
# tick_attribution | pressures (legacy) | audit_dominant (deprecated alias)
GOAL_DRIVE_ORIGIN_SOURCE=tick_attribution
```

- [ ] **Step 2: Add sync keys**

In `scripts/sync_local_env_from_example.py`, add to `SYNC_EXACT`:

```python
        "GOAL_DRIVE_ORIGIN_SOURCE",
```

Add to `SYNC_PREFIXES` tuple (if not already covered):

```python
    "ORION_SUBSTRATE_",
    "ORION_AUTONOMY_",
    "ORION_METABOLISM_",
    "ORION_CAPABILITY_POLICY_",
    "ORION_EPISODE_",
```

- [ ] **Step 3: Sync local env**

```bash
cd /mnt/scripts/Orion-Sapienform
python scripts/sync_local_env_from_example.py
```

Report any skipped keys (`ORION_BUS_URL` is expected skip).

- [ ] **Step 4: Run agent-check gate**

```bash
make agent-check SERVICE=orion-spark-concept-induction
```

If `make agent-check` missing, run manually:

```bash
git diff --check
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
pytest orion/spark/concept_induction/tests/test_drive_attribution.py \
       orion/spark/concept_induction/tests/test_metabolism_hook.py \
       orion/spark/concept_induction/tests/test_goals.py \
       orion/autonomy/tests/test_policy_act.py \
       orion/autonomy/tests/test_substrate_act_models.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/orion-spark-concept-induction/.env_example scripts/sync_local_env_from_example.py
git commit -m "$(cat <<'EOF'
chore: sync GOAL_DRIVE_ORIGIN_SOURCE and substrate env keys

EOF
)"
```

---

## Task 10: Code review + smoke

- [ ] **Run code review subagent** (superpowers:requesting-code-review) on full diff vs `main`
- [ ] **Fix material findings** and re-run Task 9 Step 4 tests
- [ ] **Optional integration smoke** (requires live stack):

```bash
python scripts/smoke_substrate_motivation_golden_path.py
```

Verify after world-pulse with GPU gap:
- `DriveAudit.dominant_drive == predictive`
- `tick_attribution.predictive > 0`
- action outcome store has `web.fetch.readonly` entry even when goal cooldown suppresses publish

- [ ] **Push branch + open PR** with AGENTS.md PR template

**Restart required after deploy:**

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml \
  up -d --build
```

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| `compute_tick_attribution` + structural tie-break | Task 1 |
| `DriveAuditV1.tick_attribution` | Task 2 |
| Dominance from attribution, not `max(sorted(pressures))` | Tasks 3, 5 |
| Merge `metabolism.drive_deltas` | Task 5 |
| `GOAL_DRIVE_ORIGIN_SOURCE=tick_attribution` default | Task 4 |
| `SubstrateEpisodeIntentV1` + resolve from slot / synthetic | Tasks 6–7 |
| Fetch/journal outside goal-publish gate | Task 8 |
| Tests 1–6 from spec acceptance | Tasks 1–3, 7–8 |
| Env sync keys | Task 9 |
| Appendix C (saturation rework) deferred | Not in plan ✓ |

**Placeholder scan:** No TBD/TODO/similar-to tasks. All steps include code or exact commands.

**Type consistency:** `SubstrateEpisodeIntentV1` dataclass in `models.py`; `goal_proposal_from_episode_intent` → `GoalProposalV1` for policy; `bus_worker` passes `spawned_correlation_id` from `wp_result.run.run_id`.
