# World-Pulse Curiosity Followups ("Orion went looking") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a non-dry world-pulse run detects under-covered sections, fetch gap-filling articles inline, attach them to the digest as an "Orion went looking" block, and carry those findings on the published run result so the reactive episode-journal loop reuses them instead of fetching again.

**Architecture:** Producer side (`orion-world-pulse`) gains a gated, dry-run-skipped inline fetch that reuses the existing `execute_readonly_fetch` + `web.fetch.readonly` capability policy and writes results to a new `DailyWorldPulseV1.curiosity_followups` field. The findings ride the existing `world.pulse.run.result.v1` stream event. Consumer side (`orion-spark-concept-induction`) reads matching followups and passes a `prefetched_outcome` into `maybe_execute_substrate_act_after_metabolism`, skipping a second Firecrawl call; a live fetch remains the fallback.

**Tech Stack:** Python 3.11, Pydantic v2, FastAPI (world-pulse), Redis Streams (bus), pytest. Firecrawl via `orion/autonomy/fetch_backends.py`.

**Spec:** `docs/superpowers/specs/2026-07-07-world-pulse-curiosity-followups-design.md`

---

## File Structure

**New files:**
- `orion/autonomy/curiosity_reuse.py` — pure helpers to select a reusable followup for a gap and rebuild an `ActionOutcomeRefV1` from it (consumer side).
- `services/orion-world-pulse/app/services/curiosity.py` — `build_curiosity_followups(...)`: capability gate + per-section inline fetch + mapping to `CuriosityFollowupV1` (producer side).
- Tests: `orion/autonomy/tests/test_salience_tokenize.py`, `orion/autonomy/tests/test_curiosity_reuse.py`, `orion/autonomy/tests/test_policy_act_prefetched.py`, `services/orion-world-pulse/tests/test_curiosity_followups_schema.py`, `services/orion-world-pulse/tests/test_curiosity.py`, `services/orion-world-pulse/tests/test_renderers_curiosity.py`.

**Modified files:**
- `orion/schemas/world_pulse.py` — add `CuriosityFindingV1`, `CuriosityFollowupV1`, `DailyWorldPulseV1.curiosity_followups`.
- `orion/autonomy/salience.py` — add public `tokenize_terms`.
- `orion/autonomy/policy_act.py` — add `prefetched_outcome` to `maybe_execute_substrate_act_after_metabolism`.
- `orion/spark/concept_induction/bus_worker.py` — wire followup reuse into the `world.pulse.run.result.v1` branch.
- `services/orion-world-pulse/app/settings.py` + `.env_example` — new flags.
- `services/orion-world-pulse/app/services/pipeline.py` — call `build_curiosity_followups`, set `digest.curiosity_followups`.
- `services/orion-world-pulse/app/services/digest.py` — `build_digest` optional passthrough.
- `services/orion-world-pulse/app/services/renderers.py` — render the "Orion went looking" block.
- `services/orion-world-pulse/tests/test_world_pulse_pipeline.py` — new pipeline test.

---

## Task 0: Branch / worktree setup

**Files:** none (git only)

- [ ] **Step 1: Create a clean worktree + branch**

Run:
```bash
cd /mnt/scripts/Orion-Sapienform
git status --short
git worktree add ../Orion-Sapienform-wp-curiosity -b feat/world-pulse-curiosity-followups
cd ../Orion-Sapienform-wp-curiosity
```
Expected: new worktree created on branch `feat/world-pulse-curiosity-followups`, clean tree. All subsequent paths are relative to this worktree.

---

## Task 1: Schema — `CuriosityFindingV1`, `CuriosityFollowupV1`, digest field

**Files:**
- Modify: `orion/schemas/world_pulse.py` (add models after `SectionRollupV1` ~line 466; add field to `DailyWorldPulseV1` ~line 483)
- Test: `services/orion-world-pulse/tests/test_curiosity_followups_schema.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-world-pulse/tests/test_curiosity_followups_schema.py`:
```python
from datetime import datetime, timezone

from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
)


def _minimal_digest(**overrides) -> DailyWorldPulseV1:
    now = datetime.now(timezone.utc)
    base = dict(
        run_id="run-1",
        date=now.date().isoformat(),
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="",
        created_at=now,
    )
    base.update(overrides)
    return DailyWorldPulseV1(**base)


def test_digest_defaults_to_empty_followups():
    digest = _minimal_digest()
    assert digest.curiosity_followups == []


def test_followup_round_trips_with_findings():
    followup = CuriosityFollowupV1(
        section="hardware_compute_gpu",
        driving_gap="missing",
        query="hardware compute gpu recent news coverage",
        articles=[
            CuriosityFindingV1(url="https://x/1", title="A", description="d", salience=0.5)
        ],
        action_id="fetch-abc",
        correlation_id="run-1",
    )
    digest = _minimal_digest(curiosity_followups=[followup])
    rehydrated = DailyWorldPulseV1.model_validate(digest.model_dump(mode="json"))
    assert rehydrated.curiosity_followups[0].section == "hardware_compute_gpu"
    assert rehydrated.curiosity_followups[0].articles[0].salience == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-world-pulse/tests/test_curiosity_followups_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'CuriosityFindingV1'`.

- [ ] **Step 3: Add the models and field**

In `orion/schemas/world_pulse.py`, immediately after the `SectionRollupV1` class (ends ~line 465), add:
```python
class CuriosityFindingV1(_WPBase):
    url: str
    title: str = ""
    description: str = ""
    salience: float = Field(default=0.0, ge=0.0, le=1.0)


class CuriosityFollowupV1(_WPBase):
    section: str
    driving_gap: SectionCoverageState = "missing"
    query: str = ""
    articles: list[CuriosityFindingV1] = Field(default_factory=list)
    action_id: str | None = None
    correlation_id: str | None = None
```

In `DailyWorldPulseV1`, after the `section_rollups` field (~line 483) add:
```python
    curiosity_followups: list[CuriosityFollowupV1] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-world-pulse/tests/test_curiosity_followups_schema.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify schema registry still validates**

Run: `python scripts/check_schema_registry.py`
Expected: exit 0 (models are registered by class reference in `orion/schemas/registry.py:1140,1143`; the added field needs no registry edit).

- [ ] **Step 6: Commit**

```bash
git add orion/schemas/world_pulse.py services/orion-world-pulse/tests/test_curiosity_followups_schema.py
git commit -m "feat(world-pulse): add curiosity followup schema to daily world pulse"
```

---

## Task 2: Public `tokenize_terms` helper in salience

**Files:**
- Modify: `orion/autonomy/salience.py`
- Test: `orion/autonomy/tests/test_salience_tokenize.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_salience_tokenize.py`:
```python
from orion.autonomy.salience import tokenize_terms


def test_tokenize_terms_splits_and_lowercases():
    assert tokenize_terms("Hardware Compute GPU") == {"hardware", "compute", "gpu"}


def test_tokenize_terms_handles_underscores_as_nonword():
    assert tokenize_terms("hardware_compute_gpu") == {"hardware", "compute", "gpu"}


def test_tokenize_terms_empty():
    assert tokenize_terms("") == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/autonomy/tests/test_salience_tokenize.py -v`
Expected: FAIL with `ImportError: cannot import name 'tokenize_terms'`.

- [ ] **Step 3: Add the public wrapper**

In `orion/autonomy/salience.py`, after `_tokenize` (ends ~line 20) add:
```python
def tokenize_terms(text: str) -> set[str]:
    """Public term tokenizer (lowercase alphanumeric). Reused by producers that
    need gap terms from a plain section label without a signal object."""
    return _tokenize(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/autonomy/tests/test_salience_tokenize.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/salience.py orion/autonomy/tests/test_salience_tokenize.py
git commit -m "feat(autonomy): expose public tokenize_terms for gap-term building"
```

---

## Task 3: World-pulse settings + env for the inline fetch

**Files:**
- Modify: `services/orion-world-pulse/app/settings.py` (after `world_pulse_fetch_enabled` ~line 23)
- Modify: `services/orion-world-pulse/.env_example`
- Test: `services/orion-world-pulse/tests/test_curiosity_settings.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-world-pulse/tests/test_curiosity_settings.py`:
```python
from app.settings import Settings


def test_curiosity_defaults_off():
    s = Settings()
    assert s.world_pulse_curiosity_fetch_enabled is False
    assert s.world_pulse_curiosity_max_articles_per_section == 5
    assert s.world_pulse_curiosity_max_sections == 9


def test_curiosity_env_override(monkeypatch):
    monkeypatch.setenv("WORLD_PULSE_CURIOSITY_FETCH_ENABLED", "true")
    monkeypatch.setenv("WORLD_PULSE_CURIOSITY_MAX_ARTICLES_PER_SECTION", "3")
    s = Settings()
    assert s.world_pulse_curiosity_fetch_enabled is True
    assert s.world_pulse_curiosity_max_articles_per_section == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-world-pulse/tests/test_curiosity_settings.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'world_pulse_curiosity_fetch_enabled'`.

- [ ] **Step 3: Add the settings fields**

In `services/orion-world-pulse/app/settings.py`, directly after the `world_pulse_fetch_enabled` line (~line 23) add:
```python
    world_pulse_curiosity_fetch_enabled: bool = Field(
        False, alias="WORLD_PULSE_CURIOSITY_FETCH_ENABLED"
    )
    world_pulse_curiosity_max_articles_per_section: int = Field(
        5, alias="WORLD_PULSE_CURIOSITY_MAX_ARTICLES_PER_SECTION"
    )
    world_pulse_curiosity_max_sections: int = Field(
        9, alias="WORLD_PULSE_CURIOSITY_MAX_SECTIONS"
    )
```

- [ ] **Step 4: Add the keys to `.env_example`**

In `services/orion-world-pulse/.env_example`, near the other `WORLD_PULSE_*` keys, add:
```bash
# Inline "Orion went looking" gap-fill fetch (billable; off by default).
# Effective on/off = this AND ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED.
WORLD_PULSE_CURIOSITY_FETCH_ENABLED=false
WORLD_PULSE_CURIOSITY_MAX_ARTICLES_PER_SECTION=5
WORLD_PULSE_CURIOSITY_MAX_SECTIONS=9
```

- [ ] **Step 5: Run test + env parity**

Run: `pytest services/orion-world-pulse/tests/test_curiosity_settings.py -v`
Expected: PASS (2 passed).

Run: `python scripts/sync_local_env_from_example.py && python scripts/check_env_template_parity.py`
Expected: local `.env` gains the three keys; parity check exits 0.

- [ ] **Step 6: Commit**

```bash
git add services/orion-world-pulse/app/settings.py services/orion-world-pulse/.env_example services/orion-world-pulse/tests/test_curiosity_settings.py
git commit -m "feat(world-pulse): add curiosity inline-fetch settings + env"
```

---

## Task 4: Producer — `build_curiosity_followups` (gate + fetch + map)

**Files:**
- Create: `services/orion-world-pulse/app/services/curiosity.py`
- Test: `services/orion-world-pulse/tests/test_curiosity.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-world-pulse/tests/test_curiosity.py`:
```python
import app.services.curiosity as curiosity
from orion.schemas.world_pulse import SectionCoverageV1


def _coverage(**status_by_section) -> dict[str, SectionCoverageV1]:
    out = {}
    for section, status in status_by_section.items():
        out[section] = SectionCoverageV1(status=status)
    return out


async def _fake_backend(query, *, max_articles):
    return {
        "success": True,
        "urls": ["https://ex/1"],
        "articles": [
            {"url": "https://ex/1", "title": "GPU news", "description": "compute gpu"},
        ],
    }


def test_disabled_returns_empty():
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=False,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_dry_run_skips_fetch():
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=True,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_no_gaps_returns_empty():
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="covered"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_capability_denied_returns_empty(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "false")
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert result == []


def test_fetches_under_covered_section(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")
    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(
            hardware_compute_gpu="missing", ai_technology="covered"
        ),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_fake_backend,
    )
    assert len(result) == 1
    f = result[0]
    assert f.section == "hardware_compute_gpu"
    assert f.driving_gap == "missing"
    assert f.query == "hardware compute gpu recent news coverage"
    assert f.articles[0].url == "https://ex/1"
    # gap terms {hardware, compute, gpu}; article text "GPU news compute gpu"
    # overlaps {compute, gpu} => 2/3.
    assert round(f.articles[0].salience, 2) == 0.67


def test_backend_error_degrades_to_no_followup(monkeypatch):
    monkeypatch.setenv("ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED", "true")

    async def _boom(query, *, max_articles):
        raise RuntimeError("firecrawl down")

    result = curiosity.build_curiosity_followups(
        run_id="r1",
        section_coverage=_coverage(hardware_compute_gpu="missing"),
        enabled=True,
        dry_run=False,
        max_articles_per_section=5,
        max_sections=9,
        fetch_backend=_boom,
    )
    # execute_readonly_fetch catches internally and returns a failed outcome
    # (success=False, no articles); we still record the followup with empty articles.
    assert len(result) == 1
    assert result[0].articles == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-world-pulse/tests/test_curiosity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.curiosity'`.

- [ ] **Step 3: Implement `build_curiosity_followups`**

Create `services/orion-world-pulse/app/services/curiosity.py`:
```python
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from orion.autonomy.capability_policy import CapabilityEvaluationContext, evaluate_capability
from orion.autonomy.episode_fetch import EpisodeFetchRequest, execute_readonly_fetch
from orion.autonomy.fetch_backend_resolve import resolve_fetch_backend
from orion.autonomy.salience import tokenize_terms
from orion.core.schemas.drives import GoalProposalV1
from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    SectionCoverageV1,
)

logger = logging.getLogger("orion-world-pulse.curiosity")

_READONLY_CAPABILITY = "web.fetch.readonly"


def _synthetic_goal(run_id: str) -> GoalProposalV1:
    return GoalProposalV1.model_validate(
        {
            "artifact_id": f"world-pulse-gap-{run_id}",
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "kind": "memory.goals.proposed.v1",
            "goal_statement": "World-pulse coverage-gap fetch (synthetic goal for capability policy).",
            "proposal_signature": f"world-pulse-gap-{run_id}",
            "drive_origin": "predictive",
            "proposal_status": "proposed",
            "provenance": {"intake_channel": "orion:world_pulse:run:result"},
        }
    )


def _gate_open(run_id: str) -> bool:
    ctx = CapabilityEvaluationContext(
        predictive_pressure=1.0,
        curiosity_strength=1.0,
        signal_kinds=["world_coverage_gap"],
        goal=_synthetic_goal(run_id),
        budget_used={},
    )
    decision = evaluate_capability(_READONLY_CAPABILITY, ctx)
    logger.info(
        "world_pulse_curiosity_gate outcome=%s reason=%s auto_execute=%s run_id=%s",
        decision.outcome,
        decision.reason_code,
        decision.auto_execute,
        run_id,
    )
    return decision.outcome == "allowed" and bool(decision.auto_execute)


def build_curiosity_followups(
    *,
    run_id: str,
    section_coverage: dict[str, SectionCoverageV1],
    enabled: bool,
    dry_run: bool,
    max_articles_per_section: int,
    max_sections: int,
    fetch_backend: Callable[..., Awaitable[dict]] | None = None,
) -> list[CuriosityFollowupV1]:
    """Inline gap-fill fetch for under-covered sections.

    Returns [] unless enabled, non-dry, there is at least one under-covered
    section, and the web.fetch.readonly capability gate allows. Each section is
    fetched independently; a fetch error degrades to an empty followup and never
    raises (so it can never fail the world-pulse run).
    """
    if not enabled or dry_run:
        return []
    under_covered = [
        section
        for section, coverage in section_coverage.items()
        if coverage.status != "covered"
    ][: max(0, int(max_sections))]
    if not under_covered:
        return []
    if not _gate_open(run_id):
        return []

    backend = fetch_backend or resolve_fetch_backend()
    followups: list[CuriosityFollowupV1] = []
    for section in under_covered:
        label = section.replace("_", " ")
        query = f"{label} recent news coverage"
        gap_terms = tuple(sorted(tokenize_terms(label)))
        req = EpisodeFetchRequest(
            subject="orion",
            goal_artifact_id=f"world-pulse-gap-{run_id}-{section}",
            spawned_correlation_id=run_id,
            query=query,
            max_articles=max(1, int(max_articles_per_section)),
            gap_terms=gap_terms,
        )
        try:
            outcome = asyncio.run(execute_readonly_fetch(req, fetch_backend=backend))
        except Exception:
            logger.warning(
                "world_pulse_curiosity_fetch_failed section=%s run_id=%s",
                section,
                run_id,
                exc_info=True,
            )
            continue
        followups.append(
            CuriosityFollowupV1(
                section=section,
                driving_gap=section_coverage[section].status,
                query=query,
                articles=[
                    CuriosityFindingV1(
                        url=a.url,
                        title=a.title,
                        description=a.description,
                        salience=a.salience,
                    )
                    for a in outcome.articles
                ],
                action_id=outcome.action_id,
                correlation_id=run_id,
            )
        )
    logger.info(
        "world_pulse_curiosity_followups run_id=%s sections=%s articles=%s",
        run_id,
        len(followups),
        sum(len(f.articles) for f in followups),
    )
    return followups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-world-pulse/tests/test_curiosity.py -v`
Expected: PASS (6 passed). If `test_fetches_under_covered_section` fails on dict ordering, note `section_coverage` is iterated in insertion order — the test only provides one under-covered section so order is deterministic.

- [ ] **Step 5: Commit**

```bash
git add services/orion-world-pulse/app/services/curiosity.py services/orion-world-pulse/tests/test_curiosity.py
git commit -m "feat(world-pulse): inline gap-fill fetch builder with capability gate"
```

---

## Task 5: `build_digest` passthrough + pipeline wiring

**Files:**
- Modify: `services/orion-world-pulse/app/services/digest.py:16-57` (`build_digest`)
- Modify: `services/orion-world-pulse/app/services/pipeline.py` (after `section_rollups`, ~line 454)
- Test: `services/orion-world-pulse/tests/test_world_pulse_pipeline.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `services/orion-world-pulse/tests/test_world_pulse_pipeline.py`:
```python
def test_curiosity_followups_skipped_on_dry_run(monkeypatch):
    import app.services.pipeline as pipeline

    called = {"count": 0}

    def _spy(**kwargs):
        called["count"] += 1
        return []

    monkeypatch.setattr(pipeline, "build_curiosity_followups", _spy)
    result = pipeline.run_world_pulse(run_id="cur-dry", requested_by="test", dry_run=True)
    assert result.digest is not None
    assert result.digest.curiosity_followups == []
    # dry run must never call the fetch builder's real path; builder returns []
    assert called["count"] == 1  # called, but with dry_run=True it self-skips


def test_curiosity_followups_attached_when_builder_returns(monkeypatch):
    import app.services.pipeline as pipeline
    from orion.schemas.world_pulse import CuriosityFindingV1, CuriosityFollowupV1

    def _fake(**kwargs):
        return [
            CuriosityFollowupV1(
                section="hardware_compute_gpu",
                driving_gap="missing",
                query="hardware compute gpu recent news coverage",
                articles=[CuriosityFindingV1(url="https://ex/1", title="t", salience=0.4)],
                action_id="fetch-x",
                correlation_id="cur-attached",
            )
        ]

    monkeypatch.setattr(pipeline, "build_curiosity_followups", _fake)
    result = pipeline.run_world_pulse(run_id="cur-attached", requested_by="test", dry_run=False)
    assert result.digest is not None
    assert len(result.digest.curiosity_followups) == 1
    assert result.digest.curiosity_followups[0].section == "hardware_compute_gpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-world-pulse/tests/test_world_pulse_pipeline.py -k curiosity -v`
Expected: FAIL with `AttributeError: <module 'app.services.pipeline'> does not have the attribute 'build_curiosity_followups'`.

- [ ] **Step 3: Add the `build_digest` optional param**

In `services/orion-world-pulse/app/services/digest.py`, update the imports (top of file) to include the new types:
```python
from orion.schemas.world_pulse import (
    CuriosityFollowupV1,
    DailyWorldPulseItemV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
    SectionCoverageV1,
    SectionRollupV1,
    WorthReadingItemV1,
    WorthWatchingItemV1,
)
```
Change the `build_digest` signature (line 16-21) to accept followups:
```python
def build_digest(
    run_id: str,
    items: list[DailyWorldPulseItemV1],
    worth_reading: list[WorthReadingItemV1],
    worth_watching: list[WorthWatchingItemV1],
    curiosity_followups: list[CuriosityFollowupV1] | None = None,
) -> DailyWorldPulseV1:
```
Add `curiosity_followups=curiosity_followups or []` into the `DailyWorldPulseV1(...)` constructor (inside the `return`, e.g. right before `created_at=now,`):
```python
        curiosity_followups=curiosity_followups or [],
        created_at=now,
```

- [ ] **Step 4: Wire the pipeline**

In `services/orion-world-pulse/app/services/pipeline.py`, add the import near the other `app.services` imports (~line 27):
```python
from app.services.curiosity import build_curiosity_followups
```
Then immediately after the `section_rollups = build_section_rollups(...)` call (~line 454) and before `_finalize_digest_aggregates(` (~line 455), insert:
```python
    digest.curiosity_followups = build_curiosity_followups(
        run_id=rid,
        section_coverage=section_coverage,
        enabled=settings.world_pulse_curiosity_fetch_enabled,
        dry_run=dry,
        max_articles_per_section=settings.world_pulse_curiosity_max_articles_per_section,
        max_sections=settings.world_pulse_curiosity_max_sections,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest services/orion-world-pulse/tests/test_world_pulse_pipeline.py -k curiosity -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the full pipeline suite for regressions**

Run: `pytest services/orion-world-pulse/tests/test_world_pulse_pipeline.py -q`
Expected: all pass (existing tests unaffected; `build_digest` new param is defaulted).

- [ ] **Step 7: Commit**

```bash
git add services/orion-world-pulse/app/services/digest.py services/orion-world-pulse/app/services/pipeline.py services/orion-world-pulse/tests/test_world_pulse_pipeline.py
git commit -m "feat(world-pulse): attach curiosity followups to digest in pipeline"
```

---

## Task 6: Render the "Orion went looking" block

**Files:**
- Modify: `services/orion-world-pulse/app/services/renderers.py:62-66` (inside `render_plaintext_digest`)
- Test: `services/orion-world-pulse/tests/test_renderers_curiosity.py`

- [ ] **Step 1: Write the failing test**

Create `services/orion-world-pulse/tests/test_renderers_curiosity.py`:
```python
from datetime import datetime, timezone

from app.services.renderers import render_hub_digest, render_plaintext_digest
from orion.schemas.world_pulse import (
    CuriosityFindingV1,
    CuriosityFollowupV1,
    DailyWorldPulseSectionsV1,
    DailyWorldPulseV1,
)


def _digest(followups):
    now = datetime.now(timezone.utc)
    return DailyWorldPulseV1(
        run_id="run-1",
        date=now.date().isoformat(),
        generated_at=now,
        title="Daily World Pulse",
        executive_summary="summary",
        sections=DailyWorldPulseSectionsV1(),
        orion_analysis_layer="",
        created_at=now,
        curiosity_followups=followups,
    )


def test_block_absent_when_no_followups():
    text = render_plaintext_digest(_digest([]))
    assert "Orion went looking" not in text


def test_block_renders_section_query_and_article():
    followup = CuriosityFollowupV1(
        section="hardware_compute_gpu",
        driving_gap="missing",
        query="hardware compute gpu recent news coverage",
        articles=[CuriosityFindingV1(url="https://ex/1", title="GPU news", salience=0.67)],
        action_id="fetch-x",
        correlation_id="run-1",
    )
    text = render_plaintext_digest(_digest([followup]))
    assert "Orion went looking" in text
    assert "hardware compute gpu" in text
    assert "hardware compute gpu recent news coverage" in text
    assert "GPU news" in text
    assert "0.67" in text


def test_hub_structured_payload_carries_followups():
    followup = CuriosityFollowupV1(
        section="hardware_compute_gpu",
        driving_gap="missing",
        query="q",
        articles=[CuriosityFindingV1(url="https://ex/1", title="t", salience=0.1)],
    )
    hub = render_hub_digest(_digest([followup]))
    assert hub.structured_payload["curiosity_followups"][0]["section"] == "hardware_compute_gpu"
    assert "Orion went looking" in hub.rendered_markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/orion-world-pulse/tests/test_renderers_curiosity.py -v`
Expected: FAIL — `test_block_renders_section_query_and_article` and the hub test fail (`"Orion went looking" not in text`).

- [ ] **Step 3: Add the block to `render_plaintext_digest`**

In `services/orion-world-pulse/app/services/renderers.py`, inside `render_plaintext_digest`, after the `things_worth_watching` block (~line 65) and before `return "\n".join(lines).strip()`, insert:
```python
    if digest.curiosity_followups:
        lines.append("Orion went looking (gaps our sources missed):")
        for followup in digest.curiosity_followups:
            label = followup.section.replace("_", " ")
            lines.append(f"- {label} — \"{followup.query}\"")
            if not followup.articles:
                lines.append("  (looked, found nothing)")
            for art in followup.articles:
                title = art.title or "(untitled)"
                lines.append(f"  • [{art.salience:.2f}] {title} — {art.url}")
        lines.append("")
```
`render_hub_digest` needs no change: `_normalized_structured_payload` already calls `digest.model_dump(mode="json")`, so `curiosity_followups` flows into `structured_payload`, and `rendered_markdown` reuses `render_plaintext_digest`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest services/orion-world-pulse/tests/test_renderers_curiosity.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add services/orion-world-pulse/app/services/renderers.py services/orion-world-pulse/tests/test_renderers_curiosity.py
git commit -m "feat(world-pulse): render Orion-went-looking block in digest"
```

---

## Task 7: Consumer helpers — select + rebuild outcome from followup

**Files:**
- Create: `orion/autonomy/curiosity_reuse.py`
- Test: `orion/autonomy/tests/test_curiosity_reuse.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_curiosity_reuse.py`:
```python
from orion.autonomy.curiosity_reuse import outcome_from_followup, select_reusable_followup
from orion.core.schemas.cognitive_substrate import SubstrateAnchorScopeV1
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.core.schemas.frontier_expansion import FrontierTargetZoneV1, FrontierTaskTypeV1
from orion.schemas.world_pulse import CuriosityFindingV1, CuriosityFollowupV1


def _gap_signal(section: str) -> FrontierInvocationSignalV1:
    return FrontierInvocationSignalV1(
        signal_type="world_coverage_gap",
        anchor_scope=SubstrateAnchorScopeV1(),
        target_zone=FrontierTargetZoneV1(),
        task_type_candidate=FrontierTaskTypeV1(),
        focal_node_refs=[f"section:{section}"],
        signal_strength=0.9,
        confidence=0.8,
    )


def _followup(section: str, articles=None) -> CuriosityFollowupV1:
    return CuriosityFollowupV1(
        section=section,
        driving_gap="missing",
        query=f"{section.replace('_', ' ')} recent news coverage",
        articles=articles
        or [CuriosityFindingV1(url="https://ex/1", title="t", description="d", salience=0.6)],
        action_id="fetch-abc",
        correlation_id="run-1",
    )


def test_selects_matching_section():
    followups = [_followup("ai_technology"), _followup("hardware_compute_gpu")]
    signals = [_gap_signal("hardware_compute_gpu")]
    chosen = select_reusable_followup(followups, signals)
    assert chosen is not None
    assert chosen.section == "hardware_compute_gpu"


def test_no_match_returns_none():
    followups = [_followup("ai_technology")]
    signals = [_gap_signal("hardware_compute_gpu")]
    assert select_reusable_followup(followups, signals) is None


def test_empty_articles_not_reused():
    followups = [_followup("hardware_compute_gpu", articles=[])]
    signals = [_gap_signal("hardware_compute_gpu")]
    assert select_reusable_followup(followups, signals) is None


def test_outcome_from_followup_maps_fields():
    outcome = outcome_from_followup(_followup("hardware_compute_gpu"), run_id="run-1")
    assert outcome.kind == "web.fetch.readonly"
    assert outcome.success is True
    assert outcome.action_id == "fetch-abc"
    assert outcome.query == "hardware compute gpu recent news coverage"
    assert len(outcome.articles) == 1
    assert outcome.articles[0].url == "https://ex/1"
    assert round(outcome.salience, 2) == 0.6
```

> Note: If `SubstrateAnchorScopeV1`, `FrontierTargetZoneV1`, or `FrontierTaskTypeV1` require constructor args, the implementer must supply the minimal valid instances during Step 2 (inspect each model). The helpers under test only read `signal_type` + `focal_node_refs`, so any valid signal instance works.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/autonomy/tests/test_curiosity_reuse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orion.autonomy.curiosity_reuse'`.

- [ ] **Step 3: Implement the helpers**

Create `orion/autonomy/curiosity_reuse.py`:
```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1
from orion.autonomy.salience import iter_gap_section_labels
from orion.core.schemas.frontier_curiosity import FrontierInvocationSignalV1
from orion.schemas.world_pulse import CuriosityFollowupV1

_FETCH_KIND = "web.fetch.readonly"


def select_reusable_followup(
    followups: Sequence[CuriosityFollowupV1],
    curiosity_signals: Sequence[FrontierInvocationSignalV1],
) -> CuriosityFollowupV1 | None:
    """Return the followup whose section matches the first gap-section label the
    reactive loop would act on, and only if it actually carries articles.

    Matching mirrors `build_readonly_fetch_query` (first `iter_gap_section_labels`).
    Section keys use underscores; labels use spaces — normalize both.
    """
    labels = list(iter_gap_section_labels(curiosity_signals))
    if not labels or not followups:
        return None
    wanted = labels[0].strip().lower()
    for followup in followups:
        if followup.section.replace("_", " ").strip().lower() == wanted and followup.articles:
            return followup
    return None


def outcome_from_followup(followup: CuriosityFollowupV1, *, run_id: str) -> ActionOutcomeRefV1:
    """Rebuild an ActionOutcomeRefV1 from a world-pulse curiosity followup so the
    reactive episode-journal path can reuse it without a second fetch."""
    articles = [
        FetchedArticleRefV1(
            url=a.url, title=a.title, description=a.description, salience=a.salience
        )
        for a in followup.articles
    ]
    return ActionOutcomeRefV1(
        action_id=followup.action_id or f"wp-followup-{run_id}",
        kind=_FETCH_KIND,
        summary=f"reused {len(articles)} article(s) from world-pulse gap fetch",
        success=True,
        surprise=0.0,
        observed_at=datetime.now(timezone.utc),
        query=followup.query,
        articles=articles,
        salience=max((a.salience for a in articles), default=0.0),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/autonomy/tests/test_curiosity_reuse.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/curiosity_reuse.py orion/autonomy/tests/test_curiosity_reuse.py
git commit -m "feat(autonomy): helpers to reuse world-pulse curiosity followups"
```

---

## Task 8: `prefetched_outcome` in `maybe_execute_substrate_act_after_metabolism`

**Files:**
- Modify: `orion/autonomy/policy_act.py:264-297`
- Test: `orion/autonomy/tests/test_policy_act_prefetched.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_policy_act_prefetched.py`:
```python
import asyncio
from datetime import datetime, timezone

from orion.autonomy.models import ActionOutcomeRefV1, FetchedArticleRefV1, SubstrateEpisodeIntentV1
from orion.autonomy.policy_act import maybe_execute_substrate_act_after_metabolism
from orion.core.schemas.drives import DriveStateV1


def _drive_state() -> DriveStateV1:
    return DriveStateV1.model_validate(
        {
            "subject": "orion",
            "model_layer": "self-model",
            "entity_id": "self:orion",
            "ts": datetime.now(timezone.utc).isoformat(),
            "pressures": {"predictive": 0.9},
            "activations": {},
            "confidence": 0.7,
        }
    )


def _prefetched() -> ActionOutcomeRefV1:
    return ActionOutcomeRefV1(
        action_id="fetch-prefetched",
        kind="web.fetch.readonly",
        summary="reused",
        success=True,
        observed_at=datetime.now(timezone.utc),
        query="q",
        articles=[FetchedArticleRefV1(url="https://ex/1", title="t", salience=0.5)],
        salience=0.5,
    )


def test_prefetched_outcome_skips_live_fetch():
    calls = {"count": 0}

    async def _backend(query, *, max_articles):
        calls["count"] += 1
        return {"success": True, "urls": ["https://x"], "articles": []}

    intent = SubstrateEpisodeIntentV1(
        goal_artifact_id="goal-1",
        drive_origin="predictive",
        spawned_correlation_id="run-1",
        subject="orion",
    )
    result = asyncio.run(
        maybe_execute_substrate_act_after_metabolism(
            episode_intent=intent,
            drive_state=_drive_state(),
            curiosity_signals=[],
            spawned_correlation_id="run-1",
            fetch_backend=_backend,
            journal_dispatch=None,
            budget_used={},
            episode_journal_enabled=False,
            prefetched_outcome=_prefetched(),
        )
    )
    assert calls["count"] == 0
    assert result.fetch_attempted is True
    assert result.fetch_outcome is not None
    assert result.fetch_outcome.action_id == "fetch-prefetched"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest orion/autonomy/tests/test_policy_act_prefetched.py -v`
Expected: FAIL with `TypeError: maybe_execute_substrate_act_after_metabolism() got an unexpected keyword argument 'prefetched_outcome'`.

- [ ] **Step 3: Add the `prefetched_outcome` short-circuit**

In `orion/autonomy/policy_act.py`, update the signature of `maybe_execute_substrate_act_after_metabolism` (line 264) to add a parameter before `episode_journal_enabled`:
```python
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
    prefetched_outcome: ActionOutcomeRefV1 | None = None,
) -> SubstrateActResultV1:
```
Then replace the fetch block (lines 277-294, from `result = SubstrateActResultV1()` through the `if fetch_decision.outcome ...` update) with:
```python
    result = SubstrateActResultV1()

    if prefetched_outcome is not None:
        # Single shared fetch: world-pulse already fetched this gap section and put
        # the findings on the run result. Reuse them; do not call the backend again.
        fetch_outcome = prefetched_outcome
        result = result.model_copy(
            update={
                "fetch_attempted": True,
                "fetch_outcome_id": fetch_outcome.action_id,
                "fetch_outcome": fetch_outcome,
            }
        )
    else:
        fetch_decision, fetch_outcome = await maybe_execute_readonly_fetch_after_goal(
            goal=synthetic_goal,
            drive_state=drive_state,
            curiosity_signals=curiosity_signals,
            spawned_correlation_id=run_id,
            fetch_backend=fetch_backend,
            budget_used=budget_used,
        )
        if fetch_decision.outcome == "allowed" and fetch_outcome is not None:
            result = result.model_copy(
                update={
                    "fetch_attempted": True,
                    "fetch_outcome_id": fetch_outcome.action_id,
                    "fetch_outcome": fetch_outcome,
                }
            )
```
The rest of the function (the `if not episode_journal_enabled or fetch_outcome is None: return result` guard and the journal compose block, lines 296-325) is unchanged and now works for both paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest orion/autonomy/tests/test_policy_act_prefetched.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Run existing policy_act tests for regressions**

Run: `pytest orion/autonomy/tests/ -k policy_act -v`
Expected: all pass (the `None` path is byte-identical to before).

- [ ] **Step 6: Commit**

```bash
git add orion/autonomy/policy_act.py orion/autonomy/tests/test_policy_act_prefetched.py
git commit -m "feat(autonomy): support prefetched_outcome to skip duplicate fetch"
```

---

## Task 9: Wire followup reuse into the concept-induction worker

**Files:**
- Modify: `orion/spark/concept_induction/bus_worker.py` (imports near line 33; `world.pulse.run.result.v1` branch ~line 601-603 and the act call ~line 771-784)

- [ ] **Step 1: Add imports**

In `orion/spark/concept_induction/bus_worker.py`, in the `from orion.autonomy.policy_act import (...)` block (line 33) confirm `maybe_execute_substrate_act_after_metabolism` is imported (it is). Add a new import line after it:
```python
from orion.autonomy.curiosity_reuse import outcome_from_followup, select_reusable_followup
```

- [ ] **Step 2: Keep `wp_result` in scope**

Near line 600, the `wp_result` local is only assigned inside the metabolism branch. Before the `if env.kind == "world.pulse.run.result.v1":` block (line 601), add an explicit initializer so it is always defined:
```python
        wp_result = None
```
(Place it right after the existing `spawned_correlation_id: str | None = None` line at 600.)

- [ ] **Step 3: Compute and pass `prefetched_outcome`**

In the act branch (the `elif env.kind == "world.pulse.run.result.v1" and ...:` block starting ~line 759), inside the `try:` (after `policy_budget: dict[str, int] = {}` at line 765, before the `async def _journal_dispatch` at 768), insert:
```python
                prefetched_outcome = None
                if (
                    wp_result is not None
                    and wp_result.digest is not None
                    and wp_result.digest.curiosity_followups
                ):
                    reusable = select_reusable_followup(
                        wp_result.digest.curiosity_followups,
                        metabolism_curiosity_signals,
                    )
                    if reusable is not None:
                        prefetched_outcome = outcome_from_followup(
                            reusable, run_id=spawned_correlation_id
                        )
                        logger.info(
                            "wp_curiosity_followup_reused run_id=%s section=%s action_id=%s articles=%s",
                            spawned_correlation_id,
                            reusable.section,
                            prefetched_outcome.action_id,
                            len(prefetched_outcome.articles),
                        )
```
Then add `prefetched_outcome=prefetched_outcome,` to the `maybe_execute_substrate_act_after_metabolism(...)` call (after `episode_journal_enabled=self.cfg.autonomy_episode_journal_enabled,` at line 783):
```python
                    episode_journal_enabled=self.cfg.autonomy_episode_journal_enabled,
                    prefetched_outcome=prefetched_outcome,
                )
```

- [ ] **Step 4: Verify import + syntax**

Run: `python -c "import orion.spark.concept_induction.bus_worker"`
Expected: no error (imports resolve, module compiles).

- [ ] **Step 5: Run concept-induction unit tests for regressions**

Run: `pytest services/orion-spark-concept-induction/tests -q`
Expected: existing tests pass (reuse is additive; `prefetched_outcome` defaults to `None` → live-fetch fallback path unchanged).

- [ ] **Step 6: Commit**

```bash
git add orion/spark/concept_induction/bus_worker.py
git commit -m "feat(concept-induction): reuse world-pulse curiosity followups in episode loop"
```

---

## Task 10: Full gate checks + Docker build validation

**Files:** none (verification only)

- [ ] **Step 1: Run all touched-package tests**

Run:
```bash
pytest services/orion-world-pulse/tests -q
pytest orion/autonomy/tests -q
pytest services/orion-spark-concept-induction/tests -q
```
Expected: all pass.

- [ ] **Step 2: Run repo contract gates**

Run:
```bash
git diff --check
python scripts/sync_local_env_from_example.py
python scripts/check_env_template_parity.py
python scripts/check_schema_registry.py
python scripts/check_bus_channels.py
```
Expected: all exit 0. `sync_local_env` reports the three new `WORLD_PULSE_CURIOSITY_*` keys added to local `.env` (report any skipped keys).

- [ ] **Step 3: Validate Docker compose for both services**

Run:
```bash
docker compose --env-file .env --env-file services/orion-world-pulse/.env -f services/orion-world-pulse/docker-compose.yml config >/dev/null && echo world-pulse-config-ok
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env -f services/orion-spark-concept-induction/docker-compose.yml config >/dev/null && echo concept-induction-config-ok
```
Expected: both print `*-config-ok`.

- [ ] **Step 4: Build the world-pulse image (new module imports must resolve at runtime)**

Run:
```bash
docker compose --env-file .env --env-file services/orion-world-pulse/.env -f services/orion-world-pulse/docker-compose.yml build
```
Expected: build succeeds. If `orion.autonomy` submodules are missing from the world-pulse image context, add them to the Dockerfile copy scope / `requirements` and re-run (verify `httpx`/`pyyaml` are present for `fetch_backends`/`capability_policy`).

- [ ] **Step 5: Commit any build fixes**

```bash
git add -A
git commit -m "chore(world-pulse): ensure autonomy deps available for inline fetch"
```
(Skip if nothing changed.)

---

## Task 11: Live smoke — single shared fetch end-to-end

**Files:** none (runtime verification; requires `ORION_CAPABILITY_POLICY_AUTO_READONLY_ENABLED=true`, `WORLD_PULSE_CURIOSITY_FETCH_ENABLED=true`, `FIRECRAWL_API_KEY` present, `ORION_AUTONOMY_EPISODE_JOURNAL_ENABLED=true`)

- [ ] **Step 1: Restart both services with the new flags**

Print for Juniper (do not run sudo yourself):
```bash
docker compose --env-file .env --env-file services/orion-world-pulse/.env -f services/orion-world-pulse/docker-compose.yml up -d --build
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env -f services/orion-spark-concept-induction/docker-compose.yml up -d --build
```

- [ ] **Step 2: Trigger a non-dry run**

Run:
```bash
curl -sS -X POST http://localhost:8628/api/world-pulse/run -H 'content-type: application/json' -d '{"dry_run": false, "requested_by":"manual"}' -o /tmp/wp_cur.json -w 'HTTP %{http_code}\n'
python3 -c "import json;d=json.load(open('/tmp/wp_cur.json'));fu=d['run'] and d.get('digest',{}) or {}; import sys; dg=json.load(open('/tmp/wp_cur.json'))['digest']; print('followups:', len(dg.get('curiosity_followups') or [])); [print(' -', f['section'], len(f['articles']), 'articles', 'action_id=', f['action_id']) for f in (dg.get('curiosity_followups') or [])]"
```
Expected: `HTTP 200`, ≥1 followup with ≥1 article for an under-covered section, each with an `action_id`.

- [ ] **Step 3: Confirm the reactive loop reused (one fetch, matching action_id)**

Run:
```bash
sleep 90
docker logs --tail 80 orion-athena-spark-concept-induction 2>&1 | egrep -i "wp_curiosity_followup_reused|substrate_policy_act|episode_journal_dispatched|action_outcome_emitted" | tail -20
```
Expected: a `wp_curiosity_followup_reused ... action_id=<X>` line whose `action_id` matches a followup from Step 2, followed by `substrate_policy_act journal.compose.episode ... allowed`, `substrate_episode_journal_dispatched`, and `action_outcome_emitted`. There must be **no** `substrate_policy_act web.fetch.readonly` execution for that reused section (fetch was skipped).

- [ ] **Step 4: Record smoke evidence in the PR report** (log lines + digest followup summary).

---

## Task 12: Review + PR

- [ ] **Step 1: Run the code-review skill in a subagent** over the diff; fix all material findings; re-run affected tests.

- [ ] **Step 2: Assemble the PR report** using the AGENTS.md §18 template. Include: schema/config changes, env keys added + `sync_local_env` run, tests/evals/Docker checks, live smoke evidence, review findings fixed, and exact restart commands (Task 11 Step 1).

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/world-pulse-curiosity-followups
gh pr create --title "feat(world-pulse): Orion went looking — curiosity followups in the digest" --body "$(cat <<'EOF'
<PR report per AGENTS.md §18>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Same-run inline fetch → Task 4/5. Single shared fetch (reuse) → Tasks 7–9. Dedicated block → Task 6. All 9 under-covered sections → Task 4 (`status != "covered"`). Generous/no-salience-floor → Task 4 (no salience filter; all articles kept). Capability policy + new flag → Tasks 3/4 (`_gate_open` + `WORLD_PULSE_CURIOSITY_FETCH_ENABLED`). Skip on dry runs → Task 4 (`dry_run` guard) + Task 5 test. Findings ride run result → Task 1 (field on `DailyWorldPulseV1`, which `WorldPulseRunResultV1.digest` carries) + Task 9 (consumer reads it). Non-goal (coverage unchanged) → `_compute_coverage` untouched.
- Schema/registry: no registry edit needed (class-reference map); Task 1 Step 5 verifies. Structured payload carries followups automatically (Task 6 Step 3 note).

**Placeholder scan:** No TBD/"handle errors" placeholders — every code step shows full code; error degradation is concrete (`build_curiosity_followups` try/except, `prefetched_outcome` guard).

**Type consistency:** `CuriosityFindingV1`/`CuriosityFollowupV1` fields (`url,title,description,salience`; `section,driving_gap,query,articles,action_id,correlation_id`) are consistent across schema (Task 1), producer (Task 4), renderer (Task 6), and reuse helpers (Task 7). `build_curiosity_followups` kwargs match the pipeline call (Task 5) and the tests (Task 4). `maybe_execute_substrate_act_after_metabolism(prefetched_outcome=...)` signature (Task 8) matches the worker call (Task 9) and test (Task 8). `select_reusable_followup`/`outcome_from_followup` signatures match across Task 7 and Task 9.

**Known trade-offs (from spec):** `execute_readonly_fetch` calls `append_action_outcome`, so world-pulse writes to the local action-outcome store — acceptable audit side effect; ensure `ORION_ACTION_OUTCOME_STORE_PATH` is writable in the world-pulse container. `asyncio.run` in the sync pipeline is safe because `/api/world-pulse/run` calls `run_world_pulse` synchronously (threadpool); Task 10/11 re-verify at runtime.
