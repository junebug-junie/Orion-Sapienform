# Substrate-fed Motivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop from substrate/world-pulse signals → six canonical drives → curiosity → capability-gated readonly fetch → episode journal → sleep crystallization, without keyword triggers or a fixed workflow pipe.

**Architecture:** A flag-gated `substrate_metabolism` adapter converts `WorldPulseRunResultV1` section gaps into `TensionEventV1` (predictive pressure) and `world_coverage_gap` frontier signals. Existing `DriveEngine` / `GoalProposalEngine` in concept-induction consume metabolism output unchanged. Layer C `capability_policy.evaluate()` gates readonly capabilities using Layer A thresholds + Layer B goal state. Episode actions write `ActionOutcomeRefV1`; journal and crystallizer gain `autonomy_episode` lineage via `spawned_correlation_id`.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, YAML policy config, Redis bus (`OrionBusAsync`), existing planner verb `autonomy.goal.execute.v1`, optional fcc-claude + Firecrawl MCP (flag-gated).

**Spec:** [`docs/superpowers/specs/2026-07-06-substrate-fed-motivation-design.md`](../specs/2026-07-06-substrate-fed-motivation-design.md)

**Branches (one per phase):**
- `feat/substrate-motivation-phase-0` — metabolism adapter only
- `feat/substrate-motivation-phase-1` — world_coverage_gap + endogenous bridge + goal lineage
- `feat/substrate-motivation-phase-2` — capability policy evaluator
- `feat/substrate-motivation-phase-3` — Tier B readonly fetch + action outcomes
- `feat/substrate-motivation-phase-4` — episode journal + crystallization intake

**Worktree setup (run once before Phase 0):**

```bash
cd /mnt/scripts/Orion-Sapienform
git switch main && git pull --ff-only
git worktree add ../Orion-Sapienform-substrate-motivation -b feat/substrate-motivation-phase-0
cd ../Orion-Sapienform-substrate-motivation
```

---

## Schema note (read before coding)

The approved spec prose mentions rollup `status in ("sparse", "empty")`. **Runtime truth** in `orion/schemas/world_pulse.py` uses:

- Digest-level `coverage_status`: `complete | partial | sparse | empty`
- Section rollup `status`: `covered | missing | source_unavailable | no_articles`

**v1 gap rule (deterministic):**

```text
digest_item_count == 0 AND status != "covered"
```

Do not keyword-match digest text. Do not branch on `"GPU"` in summaries.

---

## File Map

| File | Phase | Action | Responsibility |
|------|-------|--------|----------------|
| `orion/autonomy/models.py` | 0,2 | Modify | `MetabolismResultV1`, `CapabilityDecisionV1`, policy rule models |
| `orion/autonomy/substrate_metabolism.py` | 0 | Create | `metabolize_substrate_signals()` |
| `orion/autonomy/tests/test_substrate_metabolism.py` | 0 | Create | Gate tests for gap → predictive tension |
| `orion/core/schemas/frontier_curiosity.py` | 1 | Modify | Add `world_coverage_gap` to signal type union |
| `orion/substrate/endogenous_curiosity.py` | 1 | Modify | Accept `coverage_gap_signals` input |
| `orion/substrate/tests/test_endogenous_curiosity.py` | 1 | Modify | Gap signal → curiosity_candidate test |
| `orion/core/schemas/drives.py` | 1 | Modify | `spawned_correlation_id`, `episode_id` on `ArtifactProvenance` |
| `orion/spark/concept_induction/goals.py` | 1 | Modify | Pass `spawned_correlation_id` into goal provenance |
| `orion/spark/concept_induction/bus_worker.py` | 0,1,2 | Modify | Metabolism hook before drive/goal engines |
| `orion/spark/concept_induction/settings.py` | 0,1,2 | Modify | Metabolism + policy env keys |
| `services/orion-spark-concept-induction/.env_example` | 0,1,2 | Modify | Same env keys + world_pulse intake channel |
| `config/autonomy/capability_policy.v1.yaml` | 2 | Create | Layer C policy table |
| `orion/autonomy/capability_policy.py` | 2 | Create | `evaluate()` → `CapabilityDecisionV1` |
| `orion/autonomy/tests/test_capability_policy.py` | 2 | Create | Allow/deny/promote gate tests |
| `orion/autonomy/action_outcomes.py` | 3 | Create | Append + load recent `ActionOutcomeRefV1` per subject |
| `orion/autonomy/episode_fetch.py` | 3 | Create | Readonly web fetch executor (mockable) |
| `orion/autonomy/tests/test_action_outcomes.py` | 3 | Create | Store + reducer clears unknown flag |
| `orion/autonomy/tests/test_episode_fetch.py` | 3 | Create | Mock fetch writes outcome |
| `services/orion-cortex-exec/app/chat_stance.py` | 3 | Modify | Load action outcomes into reducer input |
| `orion/journaler/schemas.py` | 4 | Modify | `autonomy_episode` trigger + source kinds |
| `orion/journaler/worker.py` | 4 | Modify | Episode narrative contract + lineage fields |
| `orion/journaler/tests/test_autonomy_episode_journal.py` | 4 | Create | `spawned_correlation_id` on write payload |
| `orion/memory/crystallization/schemas.py` | 4 | Modify | `autonomy_episode` evidence source kind |
| `orion/memory/crystallization/intake_autonomy_episode.py` | 4 | Create | Build crystallization propose from journal |
| `tests/test_autonomy_episode_crystallization.py` | 4 | Create | Intake gate test |
| `.env_example` | 0–4 | Modify | Root env keys if shared across services |

---

# Phase 0 — Metabolism Adapter (PR 1)

Ships: `metabolize_substrate_signals()` + tests. Flag `ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED=false` (no runtime behavior change).

## Task 1: Metabolism models

**Files:**
- Modify: `orion/autonomy/models.py` (append after `ActionOutcomeRefV1`)
- Create: `orion/autonomy/tests/test_metabolism_models.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_metabolism_models.py`:

```python
from orion.autonomy.models import MetabolismResultV1
from orion.core.schemas.drives import TensionEventV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def test_metabolism_result_v1_accepts_tensions_and_curiosity() -> None:
    tension = TensionEventV1(
        artifact_id="tension-gap-1",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="substrate.world_coverage_gap",
        magnitude=0.65,
        drive_impacts={"predictive": 0.15},
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    signal = FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=["section:hardware_compute_gpu"],
        signal_strength=0.65,
        evidence_summary="hardware_compute_gpu had zero digest items",
        confidence=0.7,
        notes=["run_id:wp-run-gap-1"],
    )
    result = MetabolismResultV1(
        drive_deltas={"predictive": 0.15},
        tensions=[tension],
        curiosity_signals=[signal],
    )
    assert result.drive_deltas["predictive"] == 0.15
    assert result.tensions[0].kind == "substrate.world_coverage_gap"
    assert result.curiosity_signals[0].signal_type == "world_coverage_gap"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /mnt/scripts/Orion-Sapienform
pytest orion/autonomy/tests/test_metabolism_models.py -v
```

Expected: `ImportError: cannot import name 'MetabolismResultV1'`

- [ ] **Step 3: Write minimal implementation**

Append to `orion/autonomy/models.py`:

```python
class MetabolismResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    drive_deltas: dict[str, float] = Field(default_factory=dict)
    tensions: list[TensionEventV1] = Field(default_factory=list)
    curiosity_signals: list[FrontierInvocationSignalV1] = Field(default_factory=list)
```

Add imports at top of `orion/autonomy/models.py`:

```python
from orion.core.schemas.drives import TensionEventV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/autonomy/tests/test_metabolism_models.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/models.py orion/autonomy/tests/test_metabolism_models.py
git commit -m "feat(autonomy): add MetabolismResultV1 schema"
```

---

## Task 2: `metabolize_substrate_signals()` core

**Files:**
- Create: `orion/autonomy/substrate_metabolism.py`
- Create: `orion/autonomy/tests/test_substrate_metabolism.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_substrate_metabolism.py`:

```python
from datetime import datetime, timezone

from orion.autonomy.substrate_metabolism import metabolize_substrate_signals
from orion.schemas.world_pulse import (
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionRollupV1,
    WorldPulseRunResultV1,
    WorldPulseRunV1,
)


def _gpu_gap_result() -> WorldPulseRunResultV1:
    run_id = "wp-run-gap-gpu"
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    digest = DailyWorldPulseV1(
        run_id=run_id,
        date="2026-07-06",
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="Sparse GPU coverage.",
        sections=DailyWorldPulseSectionsV1(),
        items=[],
        orion_analysis_layer="deterministic",
        coverage_status="sparse",
        section_rollups=[
            SectionRollupV1(
                section="hardware_compute_gpu",
                status="missing",
                article_count=0,
                digest_item_count=0,
                confidence=0.35,
            ),
            SectionRollupV1(
                section="ai_technology",
                status="covered",
                article_count=2,
                digest_item_count=1,
                confidence=1.0,
            ),
        ],
        created_at=now,
    )
    return WorldPulseRunResultV1(
        run=WorldPulseRunV1(
            run_id=run_id,
            date="2026-07-06",
            started_at=now,
            completed_at=now,
            status="completed",
            dry_run=False,
        ),
        digest=digest,
    )


def test_metabolism_sparse_gpu_section_raises_predictive() -> None:
    result = metabolize_substrate_signals(world_pulse_result=_gpu_gap_result())
    assert result.drive_deltas.get("predictive", 0.0) > 0.0
    assert any(t.kind == "substrate.world_coverage_gap" for t in result.tensions)
    assert any(
        s.signal_type == "world_coverage_gap" and "hardware_compute_gpu" in s.focal_node_refs[0]
        for s in result.curiosity_signals
    )


def test_metabolism_skips_covered_sections() -> None:
    covered_only = _gpu_gap_result()
    covered_only.digest.section_rollups = [
        SectionRollupV1(section="hardware_compute_gpu", status="covered", digest_item_count=2)
    ]
    result = metabolize_substrate_signals(world_pulse_result=covered_only)
    assert result.drive_deltas == {}
    assert result.tensions == []
    assert result.curiosity_signals == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/autonomy/tests/test_substrate_metabolism.py -v
```

Expected: `ModuleNotFoundError: orion.autonomy.substrate_metabolism`

- [ ] **Step 3: Write minimal implementation**

Create `orion/autonomy/substrate_metabolism.py`:

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import yaml

from orion.autonomy.models import MetabolismResultV1
from orion.core.schemas.drives import ArtifactProvenance, TensionEventV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.world_pulse import WorldPulseRunResultV1, SectionRollupV1
from orion.signals.models import OrionSignalV1

_TRUTHY = {"1", "true", "yes", "on"}
_GAP_STATUSES = frozenset({"missing", "no_articles", "source_unavailable"})
_PREDICTIVE_DELTA = 0.15
_RECOMMENDED_STRENGTH = 0.65
_DEFAULT_STRENGTH = 0.45


def metabolism_enabled() -> bool:
    return str(os.getenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "false")).strip().lower() in _TRUTHY


def _load_recommended_sections() -> set[str]:
    path = Path(__file__).resolve().parents[2] / "config" / "world_pulse" / "sources.yaml"
    if not path.is_file():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return set(data.get("recommended_sections") or [])


def _gap_strength(section: str, recommended: set[str]) -> float:
    return _RECOMMENDED_STRENGTH if section in recommended else _DEFAULT_STRENGTH


def _tension_from_gap(*, section: str, run_id: str, strength: float) -> TensionEventV1:
    return TensionEventV1(
        artifact_id=f"tension-gap-{section}-{run_id}"[:80],
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="substrate.world_coverage_gap",
        magnitude=strength,
        drive_impacts={"predictive": _PREDICTIVE_DELTA},
        provenance=ArtifactProvenance(
            intake_channel="orion:world_pulse:run:result",
            correlation_id=run_id,
            evidence_summary=f"section {section} digest_item_count=0",
        ),
        related_nodes=[f"world_pulse:section:{section}"],
    )


def _signal_from_gap(*, section: str, run_id: str, strength: float) -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=[f"section:{section}"],
        signal_strength=strength,
        evidence_summary=f"world coverage gap: {section} had zero digest items",
        confidence=strength,
        notes=[f"run_id:{run_id}", "source:substrate_metabolism"],
    )


def _gaps_from_rollups(
    rollups: Sequence[SectionRollupV1],
    *,
    run_id: str,
    recommended: set[str],
) -> tuple[list[TensionEventV1], list[FrontierInvocationSignalV1], dict[str, float]]:
    tensions: list[TensionEventV1] = []
    signals: list[FrontierInvocationSignalV1] = []
    deltas: dict[str, float] = {}
    for rollup in rollups:
        if int(rollup.digest_item_count or 0) != 0:
            continue
        status = str(rollup.status or "missing")
        if status == "covered":
            continue
        if status not in _GAP_STATUSES and status != "missing":
            continue
        section = str(rollup.section or "").strip()
        if not section:
            continue
        strength = _gap_strength(section, recommended)
        tensions.append(_tension_from_gap(section=section, run_id=run_id, strength=strength))
        signals.append(_signal_from_gap(section=section, run_id=run_id, strength=strength))
        deltas["predictive"] = min(1.0, deltas.get("predictive", 0.0) + _PREDICTIVE_DELTA)
    return tensions, signals, deltas


def metabolize_substrate_signals(
    *,
    signals: Sequence[OrionSignalV1] = (),
    molecules: Sequence[object] | None = None,
    world_pulse_result: WorldPulseRunResultV1 | None = None,
) -> MetabolismResultV1:
    if world_pulse_result is None:
        return MetabolismResultV1()

    digest = world_pulse_result.digest
    if digest is None:
        return MetabolismResultV1()

    run_id = str(world_pulse_result.run.run_id)
    recommended = _load_recommended_sections()
    tensions, curiosity_signals, drive_deltas = _gaps_from_rollups(
        digest.section_rollups,
        run_id=run_id,
        recommended=recommended,
    )
    return MetabolismResultV1(
        drive_deltas=drive_deltas,
        tensions=tensions,
        curiosity_signals=curiosity_signals,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/autonomy/tests/test_substrate_metabolism.py -v
```

Expected: `2 passed` (test 1 may fail until `world_coverage_gap` is added to schema in Phase 1 — if so, temporarily use `signal_type="curiosity_candidate"` in implementation and update in Task 5 of Phase 1; **prefer adding schema in Phase 0 follow-up commit**)

**Phase 0 schema unblock:** Add `world_coverage_gap` to `FrontierInvocationSignalTypeV1` in this task if tests fail validation:

```python
# orion/core/schemas/frontier_curiosity.py — add to union
"world_coverage_gap",
```

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/substrate_metabolism.py orion/autonomy/tests/test_substrate_metabolism.py
git add orion/core/schemas/frontier_curiosity.py  # if modified
git commit -m "feat(autonomy): substrate metabolism adapter for world-pulse gaps"
```

---

## Task 3: Concept-worker metabolism hook (flag off)

**Files:**
- Modify: `orion/spark/concept_induction/settings.py`
- Modify: `services/orion-spark-concept-induction/.env_example`
- Modify: `orion/spark/concept_induction/bus_worker.py:466-510`
- Create: `orion/spark/concept_induction/tests/test_metabolism_hook.py`

- [ ] **Step 1: Write the failing test**

Create `orion/spark/concept_induction/tests/test_metabolism_hook.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.spark.concept_induction.bus_worker import ConceptWorker
from orion.spark.concept_induction.settings import ConceptSettings


def _world_pulse_envelope() -> BaseEnvelope:
    now = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    return BaseEnvelope(
        id="env-wp-1",
        kind="world.pulse.run.result.v1",
        channel="orion:world_pulse:run:result",
        correlation_id="corr-wp-1",
        ts=now,
        source=ServiceRef(name="orion-world-pulse", version="0.1.0", node="athena"),
        payload={
            "run": {
                "run_id": "wp-run-hook",
                "date": "2026-07-06",
                "started_at": now.isoformat(),
                "completed_at": now.isoformat(),
                "status": "completed",
                "dry_run": False,
            },
            "digest": {
                "run_id": "wp-run-hook",
                "date": "2026-07-06",
                "generated_at": now.isoformat(),
                "title": "t",
                "executive_summary": "e",
                "sections": {},
                "items": [],
                "orion_analysis_layer": "deterministic",
                "coverage_status": "sparse",
                "section_rollups": [
                    {
                        "section": "hardware_compute_gpu",
                        "status": "missing",
                        "article_count": 0,
                        "digest_item_count": 0,
                        "confidence": 0.35,
                    }
                ],
                "created_at": now.isoformat(),
            },
        },
    )


@pytest.mark.asyncio
async def test_metabolism_disabled_does_not_add_tensions(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "false")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({}, {}))
    worker._publish_tension_event = MagicMock(return_value=None)
    worker._publish_drive_state = MagicMock(return_value=None)
    worker._publish_artifact = MagicMock(return_value=None)
    worker._publish_dossier = MagicMock(return_value=None)
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    call_kwargs = worker.drive_engine.update.call_args.kwargs
    assert call_kwargs["tensions"] == []


@pytest.mark.asyncio
async def test_metabolism_enabled_merges_gap_tensions(monkeypatch) -> None:
    monkeypatch.setenv("ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED", "true")
    cfg = ConceptSettings()
    worker = ConceptWorker(cfg)
    worker.store = MagicMock()
    worker.store.load_drive_state.return_value = {}
    worker.drive_engine.update = MagicMock(return_value=({"predictive": 0.2}, {"predictive": False}))
    worker._publish_tension_event = MagicMock(return_value=None)
    worker._publish_drive_state = MagicMock(return_value=None)
    worker._publish_artifact = MagicMock(return_value=None)
    worker._publish_dossier = MagicMock(return_value=None)
    worker.goal_engine.propose = MagicMock(return_value=MagicMock(proposal=None, suppressed_signature=None))

    await worker.handle_envelope(_world_pulse_envelope(), "orion:world_pulse:run:result")

    tensions = worker.drive_engine.update.call_args.kwargs["tensions"]
    assert any(getattr(t, "kind", "") == "substrate.world_coverage_gap" for t in tensions)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/spark/concept_induction/tests/test_metabolism_hook.py -v
```

Expected: second test fails — metabolism tensions not merged

- [ ] **Step 3: Add settings + hook**

In `orion/spark/concept_induction/settings.py`, add after goal settings block:

```python
    substrate_autonomy_metabolism_enabled: bool = Field(
        False, alias="ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED"
    )
```

Append to `services/orion-spark-concept-induction/.env_example`:

```bash
# Substrate-fed motivation (v1 — default off)
ORION_SUBSTRATE_AUTONOMY_METABOLISM_ENABLED=false
ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE=0.55
ORION_METABOLISM_MIN_CURIOSITY_STRENGTH=0.5
```

Add `orion:world_pulse:run:result` to `BUS_INTAKE_CHANNELS` in `.env_example`.

In `bus_worker.py`, add imports:

```python
from orion.autonomy.substrate_metabolism import metabolize_substrate_signals, metabolism_enabled
from orion.schemas.world_pulse import WorldPulseRunResultV1
```

In `handle_envelope`, after `spark_tensions` is built (before drive update ~line 495), insert:

```python
        metabolism_tensions = []
        if metabolism_enabled() and env.kind == "world.pulse.run.result.v1":
            try:
                wp_result = WorldPulseRunResultV1.model_validate(self._payload_dict(env))
                metabolism = metabolize_substrate_signals(world_pulse_result=wp_result)
                metabolism_tensions = list(metabolism.tensions)
                for tension in metabolism_tensions:
                    await self._publish_tension_event(tension, env.correlation_id)
                    published_artifacts.append(tension)
            except Exception:
                logger.warning("substrate_metabolism_failed kind=%s", env.kind, exc_info=True)
```

Change drive update to merge tensions:

```python
        all_spark_tensions = spark_tensions + metabolism_tensions
        pressures, activations = self.drive_engine.update(
            previous_pressures=prior_drive_state.get("pressures"),
            previous_activations=prior_drive_state.get("activations"),
            tensions=all_spark_tensions,
            now=now,
            previous_ts=previous_ts,
        )
```

Update downstream `all_tensions = spark_tensions + pressure_tensions` to:

```python
        all_tensions = all_spark_tensions + pressure_tensions
```

Run env sync from repo root:

```bash
python scripts/sync_local_env_from_example.py
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest orion/spark/concept_induction/tests/test_metabolism_hook.py -v
pytest orion/autonomy/tests/test_substrate_metabolism.py -v
```

Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/settings.py orion/spark/concept_induction/bus_worker.py
git add orion/spark/concept_induction/tests/test_metabolism_hook.py
git add services/orion-spark-concept-induction/.env_example
git commit -m "feat(concept-induction): flag-gated substrate metabolism hook on world-pulse"
```

**Phase 0 PR gate:**

```bash
make agent-check SERVICE=orion-spark-concept-induction  # if target exists
pytest orion/autonomy/tests/test_substrate_metabolism.py orion/spark/concept_induction/tests/test_metabolism_hook.py -q
python scripts/check_env_template_parity.py
```

---

# Phase 1 — Curiosity Bridge + Goal Lineage (PR 2)

Branch: `git switch -c feat/substrate-motivation-phase-1`

## Task 4: `world_coverage_gap` in endogenous curiosity

**Files:**
- Modify: `orion/substrate/endogenous_curiosity.py`
- Modify: `orion/substrate/tests/test_endogenous_curiosity.py`

- [ ] **Step 1: Write the failing test**

Append to `orion/substrate/tests/test_endogenous_curiosity.py`:

```python
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1


def test_world_coverage_gap_passes_through_as_curiosity_seed() -> None:
    gap = FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope="orion",
        subject_ref="entity:orion",
        target_zone="concept_graph",
        task_type_candidate="concept_expand",
        focal_node_refs=["section:hardware_compute_gpu"],
        signal_strength=0.65,
        evidence_summary="gpu section empty",
        confidence=0.65,
        notes=["run_id:wp-1"],
    )
    candidates = endogenous_curiosity_candidates(
        coverage_gap_signals=[gap],
        config=_enabled(),
    )
    assert len(candidates) == 1
    assert candidates[0].signal_type == "curiosity_candidate"
    assert "hardware_compute_gpu" in candidates[0].focal_node_refs[0]
    assert "world_coverage_gap" in candidates[0].notes
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest orion/substrate/tests/test_endogenous_curiosity.py::test_world_coverage_gap_passes_through_as_curiosity_seed -v
```

Expected: `TypeError: unexpected keyword argument 'coverage_gap_signals'`

- [ ] **Step 3: Implement pass-through**

Add helper in `endogenous_curiosity.py`:

```python
def _coverage_gap_candidates(
    signals: Sequence[FrontierInvocationSignalV1],
    *,
    anchor_scope: str,
    subject_ref: str | None,
) -> list[FrontierInvocationSignalV1]:
    out: list[FrontierInvocationSignalV1] = []
    for sig in signals:
        if str(sig.signal_type) != "world_coverage_gap":
            continue
        out.append(
            FrontierInvocationSignalV1(
                signal_type="curiosity_candidate",
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                target_zone="concept_graph",
                task_type_candidate=sig.task_type_candidate,
                focal_node_refs=list(sig.focal_node_refs[:8]),
                signal_strength=_clamp01(sig.signal_strength),
                evidence_summary=sig.evidence_summary,
                confidence=_clamp01(sig.confidence),
                notes=["endogenous_seed", "source:world_coverage_gap", *list(sig.notes[:6])],
            )
        )
    return out
```

Extend `endogenous_curiosity_candidates` signature:

```python
    coverage_gap_signals: Sequence[FrontierInvocationSignalV1] = (),
```

After repair candidate block, before attention loops:

```python
    candidates.extend(
        _coverage_gap_candidates(
            coverage_gap_signals,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
        )
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest orion/substrate/tests/test_endogenous_curiosity.py -q
```

Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add orion/substrate/endogenous_curiosity.py orion/substrate/tests/test_endogenous_curiosity.py
git commit -m "feat(substrate): endogenous curiosity accepts world_coverage_gap signals"
```

---

## Task 5: `spawned_correlation_id` on goals

**Files:**
- Modify: `orion/core/schemas/drives.py`
- Modify: `orion/spark/concept_induction/goals.py`
- Modify: `orion/spark/concept_induction/bus_worker.py`
- Create: `orion/spark/concept_induction/tests/test_goal_spawned_correlation.py`

- [ ] **Step 1: Write the failing test**

Create `orion/spark/concept_induction/tests/test_goal_spawned_correlation.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock

from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.schemas.drives import DriveStateV1
from orion.spark.concept_induction.goals import GoalProposalEngine


def test_goal_provenance_carries_spawned_correlation_id() -> None:
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    env = BaseEnvelope(
        id="e1",
        kind="world.pulse.run.result.v1",
        channel="orion:world_pulse:run:result",
        correlation_id="corr-parent",
        ts=now,
        source=ServiceRef(name="t", version="0", node="n"),
        payload={},
    )
    drive_state = DriveStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.drives.state.v1",
        pressures={"predictive": 0.7},
        activations={"predictive": True},
        updated_at=now,
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    engine = GoalProposalEngine(cooldown_minutes=0)
    store = MagicMock()
    store.load_goal_cooldown.return_value = None
    store.load_goal_slot.return_value = {}
    decision = engine.propose(
        env=env,
        intake_channel="orion:world_pulse:run:result",
        drive_state=drive_state,
        tensions=[],
        store=store,
        dominant_drive="predictive",
        window_summary="hardware_compute_gpu gap",
        spawned_correlation_id="wp-run-gap-gpu",
    )
    assert decision.proposal is not None
    assert decision.proposal.provenance.spawned_correlation_id == "wp-run-gap-gpu"
```

- [ ] **Step 2: Run test — expect fail on unknown field / missing param**

- [ ] **Step 3: Implement**

In `ArtifactProvenance` (`orion/core/schemas/drives.py`):

```python
    spawned_correlation_id: Optional[str] = None
    episode_id: Optional[str] = None
```

In `GoalProposalEngine.propose`, add kwarg `spawned_correlation_id: str | None = None` and set on provenance when building `GoalProposalV1`.

In `bus_worker.py`, when metabolism runs, stash `spawned_correlation_id = wp_result.run.run_id` and pass to `goal_engine.propose(...)`.

- [ ] **Step 4: Run test — expect pass**

```bash
pytest orion/spark/concept_induction/tests/test_goal_spawned_correlation.py -v
```

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(goals): spawned_correlation_id lineage from world-pulse run_id"
```

---

# Phase 2 — Capability Policy (PR 3)

Branch: `git switch -c feat/substrate-motivation-phase-2`

## Task 6: Policy YAML + models

**Files:**
- Create: `config/autonomy/capability_policy.v1.yaml`
- Modify: `orion/autonomy/models.py`

- [ ] **Step 1: Create policy file**

Create `config/autonomy/capability_policy.v1.yaml`:

```yaml
version: v1
rules:
  - capability_id: web.fetch.readonly
    side_effect_class: readonly
    auto_execute: true
    requires_goal_status: proposed
    required_drive_origins: [predictive]
    required_signal_kinds: [world_coverage_gap]
    budget_per_cycle: 2
  - capability_id: journal.compose.episode
    side_effect_class: write
    auto_execute: true
    requires_goal_status: proposed
    required_drive_origins: [predictive]
    required_signal_kinds: []
    budget_per_cycle: 1
  - capability_id: world_pulse.run
    side_effect_class: readonly
    auto_execute: true
    requires_goal_status: none
    required_drive_origins: []
    required_signal_kinds: []
    budget_per_cycle: 1
  - capability_id: web.fetch.write
    side_effect_class: external
    auto_execute: false
    requires_goal_status: planned
    required_drive_origins: []
    required_signal_kinds: []
    budget_per_cycle: 0
```

- [ ] **Step 2: Add policy models to `orion/autonomy/models.py`**

```python
CapabilityDecisionOutcome = Literal["allowed", "denied", "requires_promote"]


class CapabilityPolicyRuleV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    capability_id: str
    side_effect_class: Literal["readonly", "write", "external"]
    auto_execute: bool = False
    requires_goal_status: str = "none"
    required_drive_origins: list[str] = Field(default_factory=list)
    required_signal_kinds: list[str] = Field(default_factory=list)
    budget_per_cycle: int = 0


class CapabilityPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    version: str = "v1"
    rules: list[CapabilityPolicyRuleV1] = Field(default_factory=list)


class CapabilityDecisionV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    capability_id: str
    outcome: CapabilityDecisionOutcome
    reason_code: str
    auto_execute: bool = False
    notes: list[str] = Field(default_factory=list)
```

- [ ] **Step 3: Commit**

```bash
git add config/autonomy/capability_policy.v1.yaml orion/autonomy/models.py
git commit -m "feat(autonomy): capability policy schema and v1 rules file"
```

---

## Task 7: `capability_policy.evaluate()`

**Files:**
- Create: `orion/autonomy/capability_policy.py`
- Create: `orion/autonomy/tests/test_capability_policy.py`

- [ ] **Step 1: Write failing tests**

Create `orion/autonomy/tests/test_capability_policy.py`:

```python
from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.core.schemas.drives import GoalProposalV1


def _goal(**kwargs) -> GoalProposalV1:
    base = dict(
        artifact_id="goal-test",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.goals.proposed.v1",
        goal_statement="Reduce predictive uncertainty for hardware_compute_gpu.",
        proposal_signature="sig",
        drive_origin="predictive",
        proposal_status="proposed",
        provenance={"intake_channel": "orion:world_pulse:run:result"},
    )
    base.update(kwargs)
    return GoalProposalV1.model_validate(base)


def test_capability_policy_allows_readonly_when_goal_proposed(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    monkeypatch.setenv("ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE", "0.55")
    monkeypatch.setenv("ORION_METABOLISM_MIN_CURIOSITY_STRENGTH", "0.5")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=_goal(),
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "allowed"
    assert decision.auto_execute is True


def test_capability_policy_denies_without_goal(monkeypatch) -> None:
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    ctx = CapabilityEvaluationContext(
        predictive_pressure=0.7,
        curiosity_strength=0.65,
        signal_kinds=["world_coverage_gap"],
        goal=None,
        budget_used={},
    )
    decision = evaluate_capability("web.fetch.readonly", ctx)
    assert decision.outcome == "denied"
    assert decision.reason_code == "missing_goal"
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Implement evaluator**

Create `orion/autonomy/capability_policy.py` with:

- `CapabilityEvaluationContext` dataclass: `predictive_pressure`, `curiosity_strength`, `signal_kinds`, `goal`, `budget_used`
- `load_capability_policy()` reading `config/autonomy/capability_policy.v1.yaml`
- `evaluate_capability(capability_id, ctx)` implementing Layer A checks:
  - `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED` must be true for auto
  - `predictive_pressure >= ORION_METABOLISM_MIN_PREDICTIVE_PRESSURE`
  - `curiosity_strength >= ORION_METABOLISM_MIN_CURIOSITY_STRENGTH`
- Layer B: goal present, `drive_origin` in `required_drive_origins`, `proposal_status` meets `requires_goal_status`
- Budget: deny with `capability_budget_exhausted` when `budget_used[capability_id] >= budget_per_cycle`
- External/write without `planned` → `requires_promote`

- [ ] **Step 4: Run tests**

```bash
pytest orion/autonomy/tests/test_capability_policy.py -v
```

- [ ] **Step 5: Commit + env**

Add to `services/orion-spark-concept-induction/.env_example`:

```bash
ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=false
```

```bash
git add orion/autonomy/capability_policy.py orion/autonomy/tests/test_capability_policy.py
git commit -m "feat(autonomy): Layer C capability policy evaluator"
```

---

# Phase 3 — Tier B Act + Action Outcomes (PR 4)

Branch: `git switch -c feat/substrate-motivation-phase-3`

## Task 8: Action outcome store

**Files:**
- Create: `orion/autonomy/action_outcomes.py`
- Create: `orion/autonomy/tests/test_action_outcomes.py`

- [ ] **Step 1: Write failing test**

```python
from datetime import datetime, timezone

from orion.autonomy.action_outcomes import append_action_outcome, load_action_outcomes
from orion.autonomy.models import ActionOutcomeRefV1
from orion.autonomy.reducer import AutonomyReducerInputV1, reduce_autonomy_state


def test_action_outcome_clears_reducer_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    fixed = datetime(2026, 7, 6, 12, 0, tzinfo=timezone.utc)
    ref = ActionOutcomeRefV1(
        action_id="fetch-1",
        kind="web.fetch.readonly",
        summary="fetched 2 articles",
        success=True,
        surprise=0.1,
        observed_at=fixed,
    )
    append_action_outcome(subject="orion", outcome=ref)
    loaded = load_action_outcomes(subject="orion")
    assert len(loaded) == 1

    r = reduce_autonomy_state(
        AutonomyReducerInputV1(
            subject="orion",
            previous_state=None,
            evidence=[],
            action_outcomes=loaded,
            now=fixed,
        )
    )
    assert "no_action_outcome_history" not in r.state.unknowns
```

- [ ] **Step 2–4: Implement JSON file store (max 12 per subject), run test**

- [ ] **Step 5: Wire `chat_stance._run_autonomy_reducer`**

In `services/orion-cortex-exec/app/chat_stance.py`:

```python
from orion.autonomy.action_outcomes import load_action_outcomes
```

Replace `action_outcomes=[]` with:

```python
action_outcomes=load_action_outcomes(subject=subject),
```

Add env to `services/orion-cortex-exec/.env_example`:

```bash
ORION_ACTION_OUTCOME_STORE_PATH=/tmp/orion-action-outcomes.json
```

- [ ] **Commit**

---

## Task 9: Episode readonly fetch executor

**Files:**
- Create: `orion/autonomy/episode_fetch.py`
- Create: `orion/autonomy/tests/test_episode_fetch.py`

- [ ] **Step 1: Write failing test with injectable fetch backend**

```python
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.models import ActionOutcomeRefV1


@pytest.mark.asyncio
async def test_execute_readonly_fetch_writes_action_outcome(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ORION_ACTION_OUTCOME_STORE_PATH", str(tmp_path / "outcomes.json"))
    backend = AsyncMock(return_value={"urls": ["https://example.com/a"], "success": True})
    req = EpisodeFetchRequest(
        subject="orion",
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        query="hardware GPU supply chain news",
        max_articles=2,
    )
    outcome = await execute_readonly_fetch(req, fetch_backend=backend)
    assert isinstance(outcome, ActionOutcomeRefV1)
    assert outcome.success is True
    assert outcome.kind == "web.fetch.readonly"
    backend.assert_awaited_once()
```

- [ ] **Step 2–4: Implement**

`execute_readonly_fetch`:
1. Calls injectable `fetch_backend` (production: thin wrapper around planner/cortex fcc-claude turn — **mock in tests**)
2. Builds `ActionOutcomeRefV1` with `surprise=1.0` on failure
3. Calls `append_action_outcome`
4. Returns outcome ref

Production backend stub (flag-gated, raises if MCP disabled):

```python
async def default_fetch_backend(query: str, *, max_articles: int) -> dict:
    raise RuntimeError("episode_fetch_backend_not_configured")
```

Wire from concept-worker or a small `orion/autonomy/episode_runner.py` after policy `allowed` — **do not** hardcode `if gpu_section`.

- [ ] **Step 5: Commit**

---

# Phase 4 — Episode Journal + Crystallization (PR 5)

Branch: `git switch -c feat/substrate-motivation-phase-4`

## Task 10: `autonomy_episode` journal trigger

**Files:**
- Modify: `orion/journaler/schemas.py`
- Modify: `orion/journaler/worker.py`
- Create: `orion/journaler/tests/test_autonomy_episode_journal.py`

- [ ] **Step 1: Write failing test**

```python
from orion.journaler.worker import build_autonomy_episode_trigger, build_compose_request


def test_episode_journal_carries_spawned_correlation_id() -> None:
    trigger = build_autonomy_episode_trigger(
        goal_artifact_id="goal-gap-gpu",
        spawned_correlation_id="wp-run-gap-gpu",
        narrative_seed="gap → curiosity → fetch → learnings",
    )
    assert trigger.trigger_kind == "autonomy_episode"
    assert trigger.source_ref == "goal-gap-gpu"
    req = build_compose_request(trigger, session_id="orion", user_id="juniper", trace_id="wp-run-gap-gpu")
    assert req.context.metadata["spawned_correlation_id"] == "wp-run-gap-gpu"
    assert "gap" in (trigger.prompt_seed or "").lower() or "curiosity" in (trigger.prompt_seed or "").lower()
```

- [ ] **Step 2–4: Implement**

Extend `JournalTriggerKind` with `"autonomy_episode"`.
Extend `JournalSourceKind` with `"autonomy_episode"`.
Add `spawned_correlation_id` and `episode_id` optional fields to `JournalEntryWriteV1`.
Add `build_autonomy_episode_trigger()` with narrative contract sections:
`digest → gap → curiosity → fetch → learnings → next steps`

Map trigger to `JournalMode` `"digest"` (or new `"response"` if preferred — stay consistent in `_TRIGGER_TO_MODE`).

- [ ] **Step 5: Commit**

---

## Task 11: Crystallization intake from autonomy episode

**Files:**
- Modify: `orion/memory/crystallization/schemas.py`
- Create: `orion/memory/crystallization/intake_autonomy_episode.py`
- Create: `tests/test_autonomy_episode_crystallization.py`

- [ ] **Step 1: Write failing test**

```python
from orion.memory.crystallization.intake_autonomy_episode import build_crystallization_from_episode
from orion.journaler.schemas import JournalEntryWriteV1


def test_crystallization_proposal_marks_autonomy_episode_source() -> None:
    entry = JournalEntryWriteV1(
        author="orion",
        mode="digest",
        title="Episode: GPU coverage gap",
        body="## Gap\nhardware_compute_gpu empty\n## Learnings\nFetched two articles.",
        source_kind="autonomy_episode",
        source_ref="goal-gap-gpu",
        correlation_id="wp-run-gap-gpu",
    )
    proposal = build_crystallization_from_episode(
        journal_entry=entry,
        spawned_correlation_id="wp-run-gap-gpu",
        grammar_event_ids=["gram-1", "gram-2"],
    )
    assert proposal.governance.status == "proposed"
    assert any(ev.source_kind == "grammar_event" for ev in proposal.evidence)
    assert proposal.provenance.get("spawned_correlation_id") == "wp-run-gap-gpu"
```

- [ ] **Step 2–4: Implement**

Add `"autonomy_episode"` to `CrystallizationSourceKind` if used as evidence kind; episode body becomes primary evidence with `grammar_event` refs.

`build_crystallization_from_episode` returns `MemoryCrystallizationV1` with `kind="episode"`, `source_kind` metadata in provenance/governance.

- [ ] **Step 5: Commit**

---

## Task 12: Golden-path integration smoke (optional, flags on)

**Files:**
- Create: `scripts/smoke_substrate_motivation_golden_path.py`

- [ ] **Step 1: Script replays fixture chain**

```text
WorldPulseRunResultV1 (gpu gap fixture)
  → metabolize_substrate_signals
  → endogenous_curiosity_candidates
  → evaluate_capability (mock allow)
  → execute_readonly_fetch (mock backend)
  → build_autonomy_episode_trigger
  → build_crystallization_from_episode
```

- [ ] **Step 2: Run**

```bash
python scripts/smoke_substrate_motivation_golden_path.py
```

Expected: prints `GOLDEN_PATH_OK` and exits 0

- [ ] **Step 3: Commit**

---

# Phase completion checklist

| Phase | Acceptance | Command |
|-------|------------|---------|
| 0 | Predictive delta from empty GPU section | `pytest orion/autonomy/tests/test_substrate_metabolism.py -q` |
| 1 | Gap signal + goal lineage | `pytest orion/substrate/tests/test_endogenous_curiosity.py orion/spark/concept_induction/tests/test_goal_spawned_correlation.py -q` |
| 2 | Policy allow/deny | `pytest orion/autonomy/tests/test_capability_policy.py -q` |
| 3 | Outcome clears reducer unknown | `pytest orion/autonomy/tests/test_action_outcomes.py -q` |
| 4 | Journal + crystallization lineage | `pytest orion/journaler/tests/test_autonomy_episode_journal.py tests/test_autonomy_episode_crystallization.py -q` |

**Restart after Phase 0+ (when flags enabled):**

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml \
  up -d --build

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build
```

---

## Spec self-review (plan vs spec)

| Spec requirement | Plan task |
|------------------|-----------|
| Metabolism adapter | Phase 0 Tasks 1–3 |
| `world_coverage_gap` signal | Phase 0 Task 2 + Phase 1 Task 4 |
| Six canonical drives via DriveEngine | Phase 0 — tensions → existing `DriveEngine.update` |
| C on A+B capability policy | Phase 2 Tasks 6–7 |
| `spawned_correlation_id` lineage | Phase 1 Task 5, Phase 4 Task 10 |
| Action outcome feedback | Phase 3 Tasks 8–9 |
| Episode journal + crystallization | Phase 4 Tasks 10–11 |
| No keyword cathedral | Schema note + gap rule uses counts only |
| Flag defaults false | All new env keys default off |
| Gate tests <2s | Unit tests only, no Docker in gate path |

**Deferred (explicit):** Tier A internal auto (EXP-A), general MCP router (EXP-C), full `CognitiveEpisodeV1` orchestrator (EXP-EP).

---

## Risks

| Severity | Concern | Mitigation |
|----------|---------|------------|
| Medium | fcc-claude MCP may not be merged when Phase 3 lands | Injectable `fetch_backend`; production stub until MCP plan ships |
| Low | Section status vocabulary differs from spec prose | Plan uses runtime `SectionCoverageState` values |
| Low | Concept worker did not subscribe to world_pulse | Added to `BUS_INTAKE_CHANNELS` in Task 3 |
| Medium | `chat_stance` only reducer consumer of outcomes | File-backed store + cortex-exec wiring in Task 8 |
