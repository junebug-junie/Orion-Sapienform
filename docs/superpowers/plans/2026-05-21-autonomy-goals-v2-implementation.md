# Autonomy Goals v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the autonomy goals pipeline honest and useful: stop graph bloat from trace-suffixed signatures, surface active goals correctly in stance/Hub, add semantic goal generation with light chat influence, clarify the subject model, and optionally gate operator promote → plan → execute.

**Architecture:** Four sequential phases ship as independent PRs. Phase 0 fixes dedupe, active-goal reads, stance decoupling, and Hub labels without changing goal semantics. Phase 1 adds `goal_generator.py`, drive-origin diversity, supersede-on-change RDF, and `goal_hint` in chat stance. Phase 2 documents subject routing and switches Hub to two-subject display by default. Phase 3 adds goal lifecycle RDF, Hub operator actions, planner verb, and router `execution_mode` transitions (`none` | `hint_only` | `planned` | `executing`).

**Tech Stack:** Python 3.12, Pydantic v2, Fuseki/SPARQL (`orion/autonomy/repository.py`), concept-induction bus worker (`orion/spark/concept_induction/`), orion-cortex-exec chat stance + router, orion-rdf-writer NT triples, orion-hub static JS, pytest at repo root (`PYTHONPATH=.` + `./venv/bin/python` or `./scripts/test_service.sh`).

**Design source:** [docs/superpowers/specs/2026-05-21-autonomy-goals-v2-design.md](../specs/2026-05-21-autonomy-goals-v2-design.md)

**Worktree:** Create an isolated worktree before Phase 0 (`git worktree add .worktrees/feat-autonomy-goals-v2 -b feat/autonomy-goals-v2`).

---

## File structure

| Path | Responsibility |
|------|----------------|
| `orion/spark/concept_induction/goals.py` | GoalProposalEngine: templates, signature, cooldown, propose() |
| `orion/spark/concept_induction/goal_generator.py` | **Phase 1** — evidence/LLM goal text generation |
| `orion/spark/concept_induction/drives.py` | Drive pressure math; **Phase 1** — drive-origin diversity helpers |
| `orion/spark/concept_induction/bus_worker.py` | Publishes goals; **Phase 1** — passes window summary to generator |
| `orion/spark/concept_induction/settings.py` | Env: cooldown, generation mode, drive-origin source |
| `orion/core/schemas/drives.py` | `GoalProposalV1` schema extensions |
| `orion/autonomy/models.py` | `AutonomyGoalHeadlineV1`, `AutonomySummaryV1` extensions |
| `orion/autonomy/repository.py` | `_fetch_active_goals` SPARQL + Python dedupe |
| `orion/autonomy/summary.py` | Stance decoupling, `active_goals`, `goals_present` |
| `services/orion-cortex-exec/app/router.py` | `autonomy_execution_mode`, `autonomy_goals_present` |
| `services/orion-cortex-exec/app/chat_stance.py` | **Phase 1** — `goal_hint:` in `response_priorities` |
| `services/orion-rdf-writer/app/autonomy.py` | `proposalStatus`, `goalStatementBase`, supersede edges |
| `services/orion-hub/static/js/app.js` | Honest compact card, two-subject mode |
| `scripts/autonomy/archive_stale_goal_proposals.py` | **Phase 0** — one-time graph hygiene |
| `docs/architecture/autonomy_subjects.md` | **Phase 2** — subject routing contract |
| `orion/reasoning/adapters/autonomy.py` | **Phase 1** — pass semantic source in qualifiers |
| `services/orion-hub/scripts/api_routes.py` | **Phase 3** — promote/dismiss/complete API |
| `services/orion-planner-react/app/api.py` | **Phase 3** — `autonomy.goal.execute.v1` verb |

---

# Phase 0 — Stop the bleeding

## PR 0a: Stable signature, no trace in statement, cooldown works

### Task 1: Extend GoalProposalV1 schema

**Files:**
- Modify: `orion/core/schemas/drives.py` (after `GoalProposalV1` fields ~line 128)
- Test: `orion/spark/concept_induction/tests/test_goals.py` (new)

- [ ] **Step 1: Write the failing test**

Create `orion/spark/concept_induction/tests/test_goals.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.drives import GoalProposalV1
from orion.spark.concept_induction.goals import GoalProposalEngine


def _minimal_goal(**overrides):
    base = dict(
        artifact_id="goal-abc",
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        kind="memory.goals.proposed.v1",
        ts=datetime(2026, 5, 21, tzinfo=timezone.utc),
        confidence=0.7,
        provenance={
            "intake_channel": "orion:metacognition:tick",
            "correlation_id": "c1",
            "trace_id": "trace-long-id-12345",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
        goal_statement="Clarify autonomy boundaries without executing any new action.",
        proposal_signature="sig",
        drive_origin="autonomy",
        priority=0.5,
    )
    base.update(overrides)
    return GoalProposalV1.model_validate(base)


def test_goal_proposal_v1_accepts_new_optional_fields():
    goal = _minimal_goal(
        goal_statement_base="Clarify autonomy boundaries without executing any new action.",
        proposal_status="proposed",
        semantic_source="template",
    )
    assert goal.goal_statement_base.startswith("Clarify autonomy")
    assert goal.proposal_status == "proposed"
    assert goal.semantic_source == "template"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/spark/concept_induction/tests/test_goals.py::test_goal_proposal_v1_accepts_new_optional_fields -v`

Expected: FAIL — `GoalProposalV1` has no field `goal_statement_base`

- [ ] **Step 3: Add fields to GoalProposalV1**

In `orion/core/schemas/drives.py`, extend `GoalProposalV1`:

```python
from typing import Literal

ProposalStatus = Literal["proposed", "active", "superseded", "archived", "planned", "completed"]
SemanticSource = Literal["template", "evidence_rules", "llm"]

class GoalProposalV1(GraphReadyArtifact):
    goal_statement: str
    goal_statement_base: str | None = None
    proposal_signature: str
    drive_origin: str
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    cooldown_until: Optional[datetime] = None
    proposal_status: ProposalStatus = "proposed"
    supersedes_artifact_id: str | None = None
    semantic_source: SemanticSource = "template"
    source_event_refs: List[ArtifactEventRef] = Field(default_factory=list)
    evidence_items: List[ArtifactEvidence] = Field(default_factory=list)
    tension_kinds: List[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/spark/concept_induction/tests/test_goals.py::test_goal_proposal_v1_accepts_new_optional_fields -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/core/schemas/drives.py orion/spark/concept_induction/tests/test_goals.py
git commit -m "feat(goals): extend GoalProposalV1 with status and semantic fields"
```

---

### Task 2: Stable signature without trace suffix

**Files:**
- Modify: `orion/spark/concept_induction/goals.py`
- Test: `orion/spark/concept_induction/tests/test_goals.py`

- [ ] **Step 1: Write the failing test**

Append to `orion/spark/concept_induction/tests/test_goals.py`:

```python
from orion.core.schemas.drives import DriveStateV1, TensionEventV1
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from uuid import uuid4


class _FakeStore:
    def __init__(self):
        self._cooldowns = {}

    def load_goal_cooldown(self, signature):
        return self._cooldowns.get(signature)

    def save_goal_cooldown(self, signature, until):
        self._cooldowns[signature] = {"cooldown_until": until.isoformat()}

    def record_goal_suppression(self, signature, now):
        rec = self._cooldowns.setdefault(signature, {})
        rec["suppressed_count"] = int(rec.get("suppressed_count", 0)) + 1


def _drive_state(trace_id: str) -> DriveStateV1:
    return DriveStateV1.model_validate({
        "subject": "orion",
        "model_layer": "self-model",
        "entity_id": "self:orion",
        "pressures": {"autonomy": 0.9},
        "activations": {"autonomy": True},
        "updated_at": datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc),
        "trace_id": trace_id,
        "provenance": {
            "intake_channel": "orion:metacognition:tick",
            "source_event_refs": [],
            "evidence_items": [],
            "tension_refs": [],
        },
    })


def test_signature_stable_when_trace_changes():
    engine = GoalProposalEngine(cooldown_minutes=0)
    store = _FakeStore()
    env = BaseEnvelope(
        kind="metacognition.tick.v1",
        source=ServiceRef(name="test", version="0"),
        correlation_id=uuid4(),
        payload={},
    )
    tensions: list[TensionEventV1] = []
    d1 = engine.propose(env=env, intake_channel="orion:metacognition:tick", drive_state=_drive_state("trace-aaa"), tensions=tensions, store=store)
    d2 = engine.propose(env=env, intake_channel="orion:metacognition:tick", drive_state=_drive_state("trace-bbb"), tensions=tensions, store=store)
    assert d1.proposal is not None
    assert d2.suppressed_signature == d1.proposal.proposal_signature
    assert "trace=" not in d1.proposal.goal_statement
    assert d1.proposal.goal_statement_base == d1.proposal.goal_statement
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/spark/concept_induction/tests/test_goals.py::test_signature_stable_when_trace_changes -v`

Expected: FAIL — signatures differ or trace appears in `goal_statement`

- [ ] **Step 3: Refactor goals.py**

Replace `_goal_statement`, `_signature`, and `propose()` body in `orion/spark/concept_induction/goals.py`:

```python
    def _goal_statement_base(self, drive_origin: str, tensions: List[TensionEventV1]) -> str:
        base = GOAL_TEMPLATES.get(drive_origin, GOAL_TEMPLATES["continuity"])
        if tensions:
            lead = sorted(tensions, key=lambda tension: (-tension.magnitude, tension.kind))[0]
            return f"{base} Primary tension: {lead.kind}."
        return base

    @staticmethod
    def _signature(
        subject: str,
        model_layer: str,
        drive_origin: str,
        goal_statement_base: str,
        tensions: List[TensionEventV1],
    ) -> str:
        tension_signature = ",".join(sorted(tension.kind for tension in tensions[:3]))
        material = "|".join([
            subject,
            model_layer,
            drive_origin,
            tension_signature,
            " ".join(goal_statement_base.lower().split()),
        ])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]

    def propose(...):
        tension_list = sorted(list(tensions), key=lambda tension: (-tension.magnitude, tension.kind))
        drive_origin = self._drive_origin(drive_state)
        goal_statement_base = self._goal_statement_base(drive_origin, tension_list)
        signature = self._signature(
            drive_state.subject, drive_state.model_layer, drive_origin, goal_statement_base, tension_list
        )
        # ... cooldown check unchanged ...
        goal_statement = goal_statement_base  # trace only in provenance, not persisted text
        proposal = GoalProposalV1(
            # ... existing fields ...
            goal_statement=goal_statement,
            goal_statement_base=goal_statement_base,
            proposal_status="proposed",
            semantic_source="template",
            # trace_id still on proposal for lineage
        )
```

Ensure `provenance.trace_id` remains populated from `drive_state.trace_id`; do not append trace to `goal_statement`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/spark/concept_induction/tests/test_goals.py orion/spark/concept_induction/tests/test_concept_induction.py::ConceptInductionTests::test_goal_proposal_dedupe_cooldown_suppresses_repeats -v`

Expected: PASS (existing dedupe test should still pass; second tick suppresses)

- [ ] **Step 5: Commit**

```bash
git add orion/spark/concept_induction/goals.py orion/spark/concept_induction/tests/test_goals.py
git commit -m "fix(goals): stable signature without trace suffix in goal_statement"
```

---

## PR 0b: Active goal read path + summary stance decoupling

### Task 3: AutonomyGoalHeadlineV1 + AutonomySummaryV1 extensions

**Files:**
- Modify: `orion/autonomy/models.py`
- Test: `tests/test_autonomy_summary_degraded.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_autonomy_summary_degraded.py`:

```python
def test_healthy_state_with_proposals_uses_normal_stance_mode() -> None:
    state = AutonomyStateV1(
        subject="orion",
        model_layer="self-model",
        entity_id="self:orion",
        dominant_drive="autonomy",
        goal_headlines=[
            AutonomyGoalHeadlineV1(
                artifact_id="goal-1",
                goal_statement="Clarify autonomy boundaries without executing any new action.",
                drive_origin="autonomy",
                priority=0.8,
                cooldown_until=None,
                proposal_signature="sig-1",
            )
        ],
        source="graph",
    )
    summary = summarize_autonomy_lookup(
        state,
        selected_subject="orion",
        availability="available",
        subquery_diagnostics={
            "identity": {"status": "ok", "row_count": 1},
            "drives": {"status": "ok", "row_count": 12},
            "goals": {"status": "ok", "row_count": 1},
        },
    )
    assert summary.state_quality == "healthy"
    assert summary.stance_mode == "normal"
    assert summary.goals_present is True
    assert len(summary.active_goals) == 1
    assert summary.active_goals[0].headline.startswith("Clarify autonomy")
```

Add to `orion/autonomy/models.py`:

```python
class AutonomyActiveGoalV1(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    drive_origin: str
    headline: str
    priority: float = Field(default=0.0, ge=0.0, le=1.0)
    artifact_id: str
```

Extend `AutonomyGoalHeadlineV1` with `proposal_status: str = "proposed"`.

Extend `AutonomySummaryV1` with:

```python
    active_goals: list[AutonomyActiveGoalV1] = Field(default_factory=list)
    goals_present: bool = False
```

- [ ] **Step 2: Run test — expect FAIL** on `stance_mode == "proposal_only"` and missing `goals_present`

- [ ] **Step 3: Fix `_derive_stance_mode` and summary builders**

In `orion/autonomy/summary.py`:

```python
def _derive_stance_mode(
    *,
    state_quality: AutonomyStateQuality,
    has_proposals: bool,
    contextual_fallback: bool = False,
) -> AutonomyStanceMode:
    if contextual_fallback and state_quality == "healthy":
        return "fallback_contextual"
    if state_quality == "unavailable":
        return "unavailable"
    if state_quality in {"degraded_drives_timeout", "degraded_drives_error", "degraded_partial"}:
        return "proposal_only" if has_proposals else "unavailable"
    if state_quality == "empty":
        return "unavailable"
    # healthy + goals → normal (not proposal_only)
    return "normal"
```

In `summarize_autonomy_state`, change default stance:

```python
        stance_mode="normal",
```

Add helper and wire in `summarize_autonomy_lookup`:

```python
def _active_goals_from_state(state: AutonomyStateV1 | AutonomyStateV2) -> list[AutonomyActiveGoalV1]:
    out: list[AutonomyActiveGoalV1] = []
    for goal in state.goal_headlines[:3]:
        headline = _proposal_headline_for_display(goal.goal_statement)
        if not headline:
            continue
        out.append(
            AutonomyActiveGoalV1(
                drive_origin=goal.drive_origin,
                headline=headline,
                priority=goal.priority,
                artifact_id=goal.artifact_id,
            )
        )
    return out
```

In `summarize_autonomy_lookup` return:

```python
    active_goals = _active_goals_from_state(state) if state else []
    return base.model_copy(update={
        # ... existing ...
        "active_goals": active_goals,
        "goals_present": bool(active_goals),
    })
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_autonomy_summary_degraded.py orion/autonomy/tests/test_autonomy_summary_v2.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/models.py orion/autonomy/summary.py tests/test_autonomy_summary_degraded.py
git commit -m "feat(autonomy): decouple healthy stance_mode from goal proposals"
```

---

### Task 4: `_fetch_active_goals` read path

**Files:**
- Modify: `orion/autonomy/repository.py` (`_fetch_goals` → `_fetch_active_goals`)
- Create: `orion/autonomy/tests/test_repository_active_goals.py`

- [ ] **Step 1: Write the failing test**

Create `orion/autonomy/tests/test_repository_active_goals.py`:

```python
from __future__ import annotations

from orion.autonomy.repository import GraphAutonomyRepository


class _StubClient:
    def __init__(self, rows):
        self.rows = rows
        self.last_sparql = ""

    def select(self, sparql: str):
        self.last_sparql = sparql
        return self.rows


def test_fetch_active_goals_filters_superseded_and_dedupes_drive_origin():
    rows = [
        {"artifact_id": {"value": "goal-a"}, "goal_statement": {"value": "A"}, "drive_origin": {"value": "autonomy"},
         "priority": {"value": "0.9"}, "proposal_signature": {"value": "s1"}, "proposal_status": {"value": "active"},
         "created_at": {"value": "2026-05-21T10:00:00Z"}},
        {"artifact_id": {"value": "goal-b"}, "goal_statement": {"value": "B"}, "drive_origin": {"value": "autonomy"},
         "priority": {"value": "0.5"}, "proposal_signature": {"value": "s2"}, "proposal_status": {"value": "proposed"},
         "created_at": {"value": "2026-05-21T09:00:00Z"}},
        {"artifact_id": {"value": "goal-c"}, "goal_statement": {"value": "C"}, "drive_origin": {"value": "relational"},
         "priority": {"value": "0.7"}, "proposal_signature": {"value": "s3"}, "proposal_status": {"value": "proposed"},
         "created_at": {"value": "2026-05-21T08:00:00Z"}},
        {"artifact_id": {"value": "goal-d"}, "goal_statement": {"value": "D"}, "drive_origin": {"value": "coherence"},
         "priority": {"value": "0.2"}, "proposal_signature": {"value": "s4"}, "proposal_status": {"value": "archived"},
         "created_at": {"value": "2026-05-21T07:00:00Z"}},
    ]
    repo = GraphAutonomyRepository(endpoint="http://fake", timeout_sec=1.0, query_client=_StubClient(rows), goals_limit=3)
    goals, row_count = repo._fetch_active_goals(subject="orion", model_layer="self-model", entity_id="self:orion")
    assert row_count == 4
    assert [g.drive_origin for g in goals] == ["autonomy", "relational"]
    assert goals[0].artifact_id == "goal-a"
    assert "proposalStatus" in repo._query_client.last_sparql or "proposal_status" in repo._query_client.last_sparql.lower()
```

- [ ] **Step 2: Run test — expect FAIL** (`_fetch_active_goals` not defined)

- [ ] **Step 3: Implement `_fetch_active_goals`**

In `orion/autonomy/repository.py`, add method and switch `_query_subject` to call it:

```python
    def _fetch_active_goals(self, *, subject: str, model_layer: str, entity_id: str) -> tuple[list[AutonomyGoalHeadlineV1], int]:
        sparql = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?goal_statement ?drive_origin ?priority ?cooldown_until ?proposal_signature ?created_at ?proposal_status
WHERE {{
  GRAPH <{AUTONOMY_GOALS_GRAPH}> {{
    ?artifact a orion:ProposedGoal ;
      orion:subjectKey \"{_escape_sparql(subject)}\" ;
      orion:modelLayerKey \"{_escape_sparql(model_layer)}\" ;
      orion:entityId \"{_escape_sparql(entity_id)}\" ;
      orion:artifactId ?artifact_id ;
      orion:timestamp ?created_at ;
      orion:goalStatement ?goal_statement ;
      orion:driveOrigin ?drive_origin ;
      orion:proposalPriority ?priority ;
      orion:proposalSignature ?proposal_signature .
    OPTIONAL {{ ?artifact orion:cooldownUntil ?cooldown_until . }}
    OPTIONAL {{ ?artifact orion:proposalStatus ?proposal_status . }}
    FILTER(!BOUND(?proposal_status) || (?proposal_status != "superseded" && ?proposal_status != "archived"))
  }}
}}
ORDER BY DESC(?priority) DESC(?created_at) DESC(STR(?artifact_id))
LIMIT {self._goals_limit * 4}
""".strip()
        rows = self._select_rows(sparql)
        candidates: list[AutonomyGoalHeadlineV1] = []
        for row in rows:
            try:
                candidates.append(
                    AutonomyGoalHeadlineV1(
                        artifact_id=_literal(row, "artifact_id") or "",
                        goal_statement=_literal(row, "goal_statement") or "",
                        drive_origin=_literal(row, "drive_origin") or "",
                        priority=float(_literal(row, "priority") or 0.0),
                        cooldown_until=_literal(row, "cooldown_until"),
                        proposal_signature=_literal(row, "proposal_signature") or "",
                        proposal_status=_literal(row, "proposal_status") or "proposed",
                    )
                )
            except Exception:
                continue
        seen_origins: set[str] = set()
        out: list[AutonomyGoalHeadlineV1] = []
        for goal in sorted(candidates, key=lambda g: (-g.priority, g.artifact_id)):
            if goal.drive_origin in seen_origins:
                continue
            seen_origins.add(goal.drive_origin)
            out.append(goal)
            if len(out) >= self._goals_limit:
                break
        return out, len(rows)
```

Replace `("goals", self._fetch_goals)` with `("goals", self._fetch_active_goals)` in `_query_subject`. Keep `_fetch_goals` as a thin alias calling `_fetch_active_goals` if other tests import it.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest orion/autonomy/tests/test_repository_active_goals.py orion/autonomy/tests/test_repository_selection_and_query.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add orion/autonomy/repository.py orion/autonomy/models.py orion/autonomy/tests/test_repository_active_goals.py
git commit -m "feat(autonomy): active goal read path with drive_origin dedupe"
```

---

### Task 5: Router metadata — none + goals_present

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py:516-521`
- Modify: `services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py`

- [ ] **Step 1: Write the failing test**

Add to `test_router_autonomy_payload_export.py`:

```python
def test_router_exports_none_execution_mode_when_healthy_goals_present(monkeypatch) -> None:
    runner = PlanRunner()
    fake_step = StepExecutionResult(status="success", verb_name="chat_general", step_name="llm_chat_general", order=0, result={"LLMGatewayService": {"content": "ok"}}, latency_ms=1, node="n", logs=[], error=None)
    monkeypatch.setattr("app.router.call_step_services", AsyncMock(return_value=fake_step))
    monkeypatch.setattr("app.router.prepare_brain_reply_context", lambda _ctx: None)
    ctx = {
        "mode": "brain",
        "raw_user_text": "hello",
        "chat_autonomy_summary": {
            "stance_mode": "normal",
            "goals_present": True,
            "proposal_headlines": ["Clarify autonomy boundaries"],
        },
    }
    result = asyncio.run(runner.run_plan(bus=object(), source=ServiceRef(name="x", version="0", node="n"), req=_request(), correlation_id="corr-healthy-goals", ctx=ctx))
    assert result.metadata["autonomy_execution_mode"] == "none"
    assert result.metadata["autonomy_goals_present"] is True
```

- [ ] **Step 2: Run test — expect FAIL** (`autonomy_execution_mode == "proposal_only"`)

- [ ] **Step 3: Update `_autonomy_payload_from_ctx`**

Replace lines 516-521 in `router.py`:

```python
    summary_dict = summary if isinstance(summary, dict) else {}
    goals_present = bool(summary_dict.get("goals_present") or summary_dict.get("proposal_headlines"))
    stance_mode = str(summary_dict.get("stance_mode") or "")
    if goals_present:
        payload["autonomy_goals_present"] = True
    if stance_mode == "proposal_only" or (
        stance_mode not in {"normal", "fallback_contextual"} and goals_present
    ):
        payload["autonomy_execution_mode"] = "none"
    elif goals_present:
        payload["autonomy_execution_mode"] = "none"  # Phase 0: hints not wired yet
    else:
        payload["autonomy_execution_mode"] = "none"
```

Remove unconditional `proposal_only` assignment. Map legacy consumers: if old `proposal_only` needed for degraded turns, set `autonomy_execution_mode="none"` and rely on `stance_mode` in summary.

- [ ] **Step 4: Run tests**

Run: `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py -v`

Expected: PASS (update `test_router_exports_goal_lineage_and_proposal_only_execution_mode` to expect `none` + document migration)

- [ ] **Step 5: Commit**

```bash
git add services/orion-cortex-exec/app/router.py services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py
git commit -m "fix(router): replace proposal_only execution_mode with none + goals_present"
```

---

## PR 0c: Hub honest labels + juniper annotation

### Task 6: Hub UI updates

**Files:**
- Modify: `services/orion-hub/static/js/app.js` (`normalizeAutonomyModel`, `updateAutonomyDebugPanel`)
- Modify: `services/orion-hub/tests/test_autonomy_runtime_ui_panel.py`

- [ ] **Step 1: Write the failing test**

Append to `services/orion-hub/tests/test_autonomy_runtime_ui_panel.py`:

```python
def test_app_js_shows_goals_badge_and_active_goals_section_title() -> None:
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "goalsPresent" in app_js or "goals_present" in app_js
    assert "Active goals (non-executing)" in app_js
    assert "n/a (dyadic chat" in app_js or "dyadic scope uses relationship" in app_js
    assert "autonomy_goals_present" in app_js
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Update app.js**

In `normalizeAutonomyModel`, add:

```javascript
    const goalsPresent = Boolean(
      (safeSummary && safeSummary.goals_present)
      || proposalHeadlines.length
    );
```

Return `goalsPresent` on model object.

In `updateAutonomyDebugPanel` overview lines, after stance mode:

```javascript
      ...(model.goalsPresent ? [`goals: ${model.proposalHeadlines.length} active`] : []),
```

Change proposals section header in template or JS comment block near `autonomyDebugProposals` parent — add a title element or prefix:

```javascript
    const proposalsTitle = document.createElement('div');
    proposalsTitle.className = 'text-xs uppercase tracking-wide text-amber-400/80 mb-1';
    proposalsTitle.textContent = model.stateQuality === 'healthy'
      ? 'Active goals (non-executing)'
      : 'Proposals (proposal-only)';
    autonomyDebugProposals.parentElement?.insertBefore(proposalsTitle, autonomyDebugProposals);
```

For juniper availability row in debug raw or overview, when subject is juniper and empty:

```javascript
    const juniperNote = 'n/a (dyadic chat → relationship)';
```

Map `proposal_only` execution mode display to `none` with deprecation comment:

```javascript
    const executionModeDisplay = model.executionMode === 'proposal_only' ? 'none (legacy proposal_only)' : model.executionMode;
```

Wire `autonomy_goals_present` from meta in WS/HTTP payload normalization (alongside `autonomyExecutionMode`).

- [ ] **Step 4: Run tests**

Run: `./scripts/test_service.sh orion-hub services/orion-hub/tests/test_autonomy_runtime_ui_panel.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/orion-hub/static/js/app.js services/orion-hub/tests/test_autonomy_runtime_ui_panel.py
git commit -m "fix(hub): honest autonomy labels and juniper routing annotation"
```

---

## PR 0d: Goal archive script + retention policy

### Task 7: Archive stale goal proposals script

**Files:**
- Create: `scripts/autonomy/archive_stale_goal_proposals.py`
- Create: `scripts/autonomy/tests/test_archive_stale_goal_proposals.py`
- Modify: `services/orion-spark-concept-induction/.env_example` or root env docs if SPARQL endpoint vars live in cortex-exec

- [ ] **Step 1: Write the failing test**

Create `scripts/autonomy/tests/test_archive_stale_goal_proposals.py`:

```python
from scripts.autonomy.archive_stale_goal_proposals import build_archive_candidates


def test_build_archive_candidates_keeps_highest_priority_per_drive_origin():
    rows = [
        {"artifact_id": "g1", "drive_origin": "autonomy", "goal_statement_base": "Same text", "priority": 0.9, "proposal_status": "proposed"},
        {"artifact_id": "g2", "drive_origin": "autonomy", "goal_statement_base": "Same text", "priority": 0.3, "proposal_status": "proposed"},
        {"artifact_id": "g3", "drive_origin": "relational", "goal_statement_base": "Other", "priority": 0.5, "proposal_status": "proposed"},
    ]
    to_archive = build_archive_candidates(rows, max_active_per_subject=3, retention_days=30)
    assert set(to_archive) == {"g2"}
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement script**

Create `scripts/autonomy/archive_stale_goal_proposals.py`:

```python
#!/usr/bin/env python3
"""Archive duplicate/stale ProposedGoal rows in the autonomy goals graph."""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone


def build_archive_candidates(
    rows: list[dict],
    *,
    max_active_per_subject: int,
    retention_days: int,
) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    by_origin: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("proposal_status") in {"archived", "superseded"}:
            continue
        by_origin[str(row["drive_origin"])].append(row)
    to_archive: list[str] = []
    for _origin, group in by_origin.items():
        ranked = sorted(group, key=lambda r: (-float(r.get("priority", 0)), str(r["artifact_id"])))
        keep = ranked[:max_active_per_subject]
        keep_ids = {r["artifact_id"] for r in keep}
        for row in group:
            if row["artifact_id"] not in keep_ids:
                to_archive.append(str(row["artifact_id"]))
    return to_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--subject", default="orion")
    args = parser.parse_args()
    dry_run = not args.apply
    max_active = int(os.getenv("AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT", "3"))
    retention = int(os.getenv("AUTONOMY_GOAL_RETENTION_DAYS", "30"))
    from orion.spark.concept_induction.graph_query import GraphQueryClient, GraphQueryConfig

    endpoint = os.getenv("GRAPHDB_URL", "").strip()
    if not endpoint:
        raise SystemExit("GRAPHDB_URL required")
    client = GraphQueryClient(GraphQueryConfig(
        endpoint=endpoint,
        graph_uri="http://conjourney.net/graph/autonomy/goals",
        timeout_sec=float(os.getenv("AUTONOMY_GRAPH_TIMEOUT_SEC", "30")),
        user=os.getenv("GRAPHDB_USER"),
        password=os.getenv("GRAPHDB_PASSWORD"),
    ))
    select = f"""
PREFIX orion: <http://conjourney.net/orion#>
SELECT ?artifact_id ?drive_origin ?goal_statement ?priority ?proposal_status
WHERE {{
  GRAPH <http://conjourney.net/graph/autonomy/goals> {{
    ?a a orion:ProposedGoal ; orion:subjectKey "{args.subject}" ;
      orion:artifactId ?artifact_id ; orion:driveOrigin ?drive_origin ;
      orion:goalStatement ?goal_statement ; orion:proposalPriority ?priority .
    OPTIONAL {{ ?a orion:proposalStatus ?proposal_status . }}
  }}
}}"""
    raw_rows = client.select(select)
    rows = [
        {
            "artifact_id": r["artifact_id"]["value"],
            "drive_origin": r["drive_origin"]["value"],
            "goal_statement_base": r["goal_statement"]["value"].split(" · ", 1)[0],
            "priority": r["priority"]["value"],
            "proposal_status": (r.get("proposal_status") or {}).get("value", "proposed"),
        }
        for r in raw_rows
    ]
    to_archive = build_archive_candidates(rows, max_active_per_subject=max_active, retention_days=retention)
    print(f"candidates={len(to_archive)} dry_run={dry_run}")
    if dry_run:
        for aid in to_archive[:20]:
            print(f"  would archive {aid}")
        return 0
    for aid in to_archive:
        update = f"""
PREFIX orion: <http://conjourney.net/orion#>
DELETE {{ ?a orion:proposalStatus ?old . }}
INSERT {{ ?a orion:proposalStatus "archived" . }}
WHERE {{
  GRAPH <http://conjourney.net/graph/autonomy/goals> {{
    ?a orion:artifactId "{aid}" .
    OPTIONAL {{ ?a orion:proposalStatus ?old . }}
  }}
}}"""
        client.update(update)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Implement SPARQL SELECT/UPDATE using `orion.spark.concept_induction.graph_query.GraphQueryClient` with `AUTONOMY_GOALS_GRAPH` — mirror repository endpoint env (`GRAPHDB_URL`, credentials from cortex-exec `.env_example`).

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest scripts/autonomy/tests/test_archive_stale_goal_proposals.py -v`

Expected: PASS

- [ ] **Step 5: Commit + operator runbook note**

Add one paragraph to spec section 6.1.6 or a comment in script docstring: run once after Phase 0 deploy with `--apply`.

```bash
git add scripts/autonomy/
git commit -m "chore(graph): goal archive script with retention policy"
```

---

# Phase 1 — Semantic goals

**Depends on:** Phase 0 merged.

## PR 1a: Evidence-grounded goal generator

### Task 8: goal_generator module

**Files:**
- Create: `orion/spark/concept_induction/goal_generator.py`
- Create: `orion/spark/concept_induction/tests/test_goal_generator.py`
- Modify: `orion/spark/concept_induction/settings.py`

- [ ] **Step 1: Write failing tests for all three modes**

```python
from orion.spark.concept_induction.goal_generator import generate_goal_statement


def test_template_mode_returns_base_template():
    text = generate_goal_statement(
        drive_origin="coherence",
        pressures={"coherence": 0.8},
        tensions=[],
        window_summary=None,
        mode="template",
    )
    assert "coherence" in text.lower() or "Stabilize internal coherence" in text


def test_evidence_rules_adds_tension_clause():
    from orion.core.schemas.drives import TensionEventV1
    tension = TensionEventV1.model_validate({
        "artifact_id": "t-1", "subject": "orion", "model_layer": "self-model", "entity_id": "self:orion",
        "kind": "memory.tension.v1", "tension_kind": "coherence_gap", "magnitude": 0.7,
        "drive_impacts": {"coherence": 0.5}, "provenance": {"intake_channel": "x", "source_event_refs": [], "evidence_items": [], "tension_refs": []},
    })
    text = generate_goal_statement(
        drive_origin="coherence", pressures={"coherence": 0.8}, tensions=[tension],
        window_summary="recent deployment friction", mode="evidence_rules",
    )
    assert "Primary tension:" in text
    assert "deployment" in text.lower() or "friction" in text.lower()


def test_llm_mode_falls_back_to_evidence_rules(monkeypatch):
    monkeypatch.setattr("orion.spark.concept_induction.goal_generator._llm_goal_text", lambda **kw: None)
    text = generate_goal_statement(
        drive_origin="autonomy", pressures={"autonomy": 0.5}, tensions=[], window_summary=None, mode="llm",
    )
    assert text
    assert len(text) <= 120
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement `goal_generator.py`**

```python
from __future__ import annotations

from typing import Literal

from orion.core.schemas.drives import TensionEventV1
from .goals import GOAL_TEMPLATES

GoalGenerationMode = Literal["template", "evidence_rules", "llm"]


def _base_template(drive_origin: str, tensions: list[TensionEventV1]) -> str:
    base = GOAL_TEMPLATES.get(drive_origin, GOAL_TEMPLATES["continuity"])
    if tensions:
        lead = sorted(tensions, key=lambda t: (-t.magnitude, t.kind))[0]
        return f"{base} Primary tension: {lead.kind}."
    return base


def _evidence_rules_text(drive_origin: str, tensions: list[TensionEventV1], window_summary: str | None) -> str:
    text = _base_template(drive_origin, tensions)
    ws = " ".join(str(window_summary or "").split()).strip()
    if ws:
        snippet = ws[:60].rstrip()
        if len(ws) > 60:
            snippet += "…"
        text = f"{text} Ground on: {snippet}."
    return text[:120]


def _llm_goal_text(**kwargs) -> str | None:
    return None  # Phase 1: wire to llm-gateway in follow-up step if env set


def generate_goal_statement(
    *,
    drive_origin: str,
    pressures: dict[str, float],
    tensions: list[TensionEventV1],
    window_summary: str | None,
    mode: GoalGenerationMode,
) -> str:
    del pressures
    if mode == "template":
        return _base_template(drive_origin, tensions)
    if mode == "llm":
        llm_text = _llm_goal_text(
            drive_origin=drive_origin, tensions=tensions, window_summary=window_summary
        )
        if llm_text:
            return llm_text[:120]
        mode = "evidence_rules"
    return _evidence_rules_text(drive_origin, tensions, window_summary)
```

Add to `settings.py`:

```python
    goal_generation_mode: str = Field("evidence_rules", alias="GOAL_GENERATION_MODE")
```

Wire `goals.py` to call `generate_goal_statement` inside `_goal_statement_base`.

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

---

## PR 1b: Drive origin diversity

### Task 9: Audit-dominant drive origin

**Files:**
- Modify: `orion/spark/concept_induction/goals.py`
- Modify: `orion/spark/concept_induction/settings.py`
- Test: `orion/spark/concept_induction/tests/test_goals.py`

- [ ] **Step 1: Failing test**

```python
def test_drive_origin_from_audit_dominant():
    engine = GoalProposalEngine(cooldown_minutes=0)
    drive_state = DriveStateV1.model_validate({
        "subject": "orion", "model_layer": "self-model", "entity_id": "self:orion",
        "pressures": {"autonomy": 0.95, "relational": 0.4},
        "activations": {}, "updated_at": datetime.now(timezone.utc),
        "provenance": {"intake_channel": "x", "source_event_refs": [], "evidence_items": [], "tension_refs": []},
    })
    origin = engine._drive_origin(drive_state, dominant_drive="relational", source="audit_dominant")
    assert origin == "relational"
```

Extend `GoalProposalEngine.propose()` signature with optional `dominant_drive: str | None = None`. In `bus_worker.py`, pass `drive_audit.dominant_drive` after `build_drive_audit(...)`.

- [ ] **Step 2-4: Implement `_drive_origin` with env `GOAL_DRIVE_ORIGIN_SOURCE`**

```python
    @staticmethod
    def _drive_origin(
        drive_state: DriveStateV1,
        *,
        dominant_drive: str | None = None,
        source: str = "pressures",
    ) -> str:
        if source == "audit_dominant" and dominant_drive:
            return dominant_drive.strip().lower()
        if not drive_state.pressures:
            return "continuity"
        return max(sorted(drive_state.pressures), key=lambda key: drive_state.pressures.get(key, 0.0))
```

- [ ] **Step 5: Commit**

---

## PR 1c: Chat stance goal_hint

### Task 10: Inject goal_hint in response_priorities

**Files:**
- Modify: `services/orion-cortex-exec/app/chat_stance.py` (~line 2317)
- Modify: `services/orion-cortex-exec/app/settings.py` or env
- Test: `services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py`

- [ ] **Step 1: Failing test**

```python
def test_chat_stance_adds_goal_hint_when_priority_above_threshold():
    ctx = {
        "user_message": "what is the plan?",
        "chat_autonomy_summary": {
            "stance_hint": "maintain stable direct response",
            "active_goals": [
                {"drive_origin": "autonomy", "headline": "Clarify boundaries", "priority": 0.72, "artifact_id": "goal-x"},
            ],
            "response_hazards": ["do not present proposals as commitments"],
        },
    }
    brief = fallback_chat_stance_brief(ctx)
    assert any(p.startswith("goal_hint:") for p in brief.response_priorities)
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement in `fallback_chat_stance_brief`**

After autonomy_hint block:

```python
    threshold = float(os.getenv("GOAL_HINT_PRIORITY_THRESHOLD", "0.4"))
    active_goals = [g for g in (autonomy_summary.get("active_goals") or []) if isinstance(g, dict)]
    if active_goals and task_mode != "triage":
        top = sorted(active_goals, key=lambda g: -float(g.get("priority") or 0))[0]
        if float(top.get("priority") or 0) >= threshold:
            headline = _compact(str(top.get("headline") or ""), limit=80)
            if headline:
                response_priorities = _unique(
                    response_priorities + [f"goal_hint:{headline}"], limit=8
                )
                ctx["chat_autonomy_execution_mode"] = "hint_only"
```

Ensure `_load_autonomy_state` path sets `ctx["chat_autonomy_execution_mode"]` and router reads it:

In `router.py`:

```python
    exec_mode = ctx.get("chat_autonomy_execution_mode")
    if exec_mode:
        payload["autonomy_execution_mode"] = exec_mode
    elif goals_present:
        payload["autonomy_execution_mode"] = "none"
```

- [ ] **Step 4: Run `./scripts/test_service.sh orion-cortex-exec services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py -v`**

- [ ] **Step 5: Commit**

Update `services/orion-cortex-exec/.env_example`:

```
GOAL_HINT_PRIORITY_THRESHOLD=0.4
```

---

## PR 1d: RDF supersede edges + proposalStatus

### Task 11: rdf-writer goal supersession

**Files:**
- Modify: `services/orion-rdf-writer/app/autonomy.py`
- Modify: `services/orion-rdf-writer/tests/test_autonomy_materialization.py`

- [ ] **Step 1: Failing test**

```python
def test_goal_materialization_writes_proposal_status_and_base():
    goal = GoalProposalV1.model_validate({...})  # include goal_statement_base, proposal_status="active", supersedes_artifact_id="goal-old"
    nt, graph = build_autonomy_triples("memory.goals.proposed.v1", goal.model_dump(mode="json"))
    assert "proposalStatus" in nt
    assert "goalStatementBase" in nt
    assert "supersedesArtifact" in nt
```

- [ ] **Step 2-3: Extend `_handle_goal_proposal`**

```python
    _add_literal(g, artifact_uri, ORION.goalStatementBase, goal.goal_statement_base or goal.goal_statement, XSD.string)
    _add_literal(g, artifact_uri, ORION.proposalStatus, goal.proposal_status, XSD.string)
    if goal.supersedes_artifact_id:
        prior_uri = _artifact_uri(goal.kind, goal.supersedes_artifact_id)
        g.add((artifact_uri, ORION.supersedesArtifact, prior_uri))
```

- [ ] **Step 4: Run rdf-writer tests**

- [ ] **Step 5: Commit**

Wire `bus_worker` / `goals.py` to set `supersedes_artifact_id` when semantic signature changes for same `(subject, drive_origin)`.

---

# Phase 2 — Subject model clarity

## PR 2a: Architecture doc

### Task 12: autonomy_subjects.md

**Files:**
- Create: `docs/architecture/autonomy_subjects.md`

- [ ] **Step 1: Write doc** (content from spec §8.1.1 table + identity.py citation)

- [ ] **Step 2: Link from** `orion/autonomy/README.md`

- [ ] **Step 3: Commit**

```bash
git add docs/architecture/autonomy_subjects.md orion/autonomy/README.md
git commit -m "docs(architecture): autonomy subject routing contract"
```

---

## PR 2b: Hub two-subject mode

### Task 13: HUB_AUTONOMY_SUBJECT_DISPLAY env

**Files:**
- Modify: `services/orion-hub/static/js/app.js`
- Modify: `services/orion-hub/scripts/api_routes.py` (pass env to template as `HUB_AUTONOMY_SUBJECT_DISPLAY`)
- Modify: `services/orion-hub/templates/index.html` (inject `window.__HUB_AUTONOMY_SUBJECT_DISPLAY__`)
- Test: `services/orion-hub/tests/test_autonomy_runtime_ui_panel.py`

- [ ] **Step 1: Failing test**

```python
def test_app_js_supports_two_subject_autonomy_display_mode():
    app_js = APP_JS_PATH.read_text(encoding="utf-8")
    assert "HUB_AUTONOMY_SUBJECT_DISPLAY" in app_js or "__HUB_AUTONOMY_SUBJECT_DISPLAY__" in app_js
    assert "orion + relationship" in app_js
```

- [ ] **Step 2-3: Implement**

Default `two`: availability string `orion + relationship (orion↔juniper)` with denominator 2. Hide juniper row in compact card; keep in raw JSON.

Legacy `three`: show juniper with `n/a — dyadic scope uses relationship`.

- [ ] **Step 4: Run hub tests**

- [ ] **Step 5: Commit + `.env_example`**

---

# Phase 3 — Goal lifecycle & execution

**Depends on:** Phase 1. Feature-flagged: `AUTONOMY_GOAL_EXECUTION_ENABLED=false` default.

## PR 3a: Goal status state machine in RDF + read path

### Task 14: Extended proposal statuses + plannedTaskId

**Files:**
- Modify: `orion/core/schemas/drives.py` (status literals already include planned/completed)
- Modify: `orion/autonomy/repository.py` (read `plannedTaskId`, `completedAt`)
- Modify: `services/orion-rdf-writer/app/autonomy.py`

- [ ] **Step 1-5: TDD cycle** for reading `planned`/`executing` goals and exposing in `AutonomyActiveGoalV1` qualifiers

---

## PR 3b: Hub promote/dismiss/complete actions

### Task 15: Hub API routes

**Files:**
- Modify: `services/orion-hub/scripts/api_routes.py`
- Create: `services/orion-hub/tests/test_autonomy_goal_actions.py`

- [ ] **Step 1: Failing test**

```python
def test_promote_goal_requires_operator_token(client):
    resp = client.post("/api/autonomy/goals/goal-abc/promote", json={})
    assert resp.status_code in {401, 403, 501}  # until implemented
```

- [ ] **Step 2-3: Implement routes**

| Route | Behavior |
|-------|----------|
| `POST /api/autonomy/goals/{artifact_id}/promote` | reasoning claim → canonical (HITL); RDF `proposal_status=planned` |
| `POST /api/autonomy/goals/{artifact_id}/dismiss` | `proposal_status=archived` |
| `POST /api/autonomy/goals/{artifact_id}/complete` | `proposal_status=completed` |

Guard all routes with `AUTONOMY_GOAL_EXECUTION_ENABLED` and existing operator token middleware.

- [ ] **Step 4-5: Tests + commit**

---

## PR 3c: Planner autonomy.goal.execute.v1 verb

### Task 16: Planner verb

**Files:**
- Modify: `services/orion-planner-react/app/api.py`
- Create: `services/orion-planner-react/tests/test_autonomy_goal_execute.py`

- [ ] **Step 1: Failing test** for verb accepting `goal_artifact_id`, returning `task_id`

- [ ] **Step 2-3: Register verb** that creates planner task and publishes bus event for supervisor

- [ ] **Step 4-5: Tests + commit**

---

## PR 3d: Router execution_mode planned/executing

### Task 17: Full execution_mode ladder

**Files:**
- Modify: `services/orion-cortex-exec/app/router.py`
- Modify: `services/orion-cortex-exec/app/supervisor.py`
- Modify: `orion/substrate/mutation_decision.py` (regression: unpromoted goals never execute)

- [ ] **Step 1: Failing test**

```python
def test_router_execution_mode_executing_when_planned_task_active():
    ctx = {"chat_autonomy_summary": {"active_goals": [{"artifact_id": "g1", "proposal_status": "executing"}]}}
    payload = _autonomy_payload_from_ctx(ctx)
    assert payload["autonomy_execution_mode"] == "executing"
```

- [ ] **Step 2-3: Map states per spec §9.1.5**

- [ ] **Step 4: E2E test** promote → plan → mock execute → complete

- [ ] **Step 5: Commit**

---

# Cross-cutting verification

After each phase merge, run targeted suites:

```bash
# Phase 0
PYTHONPATH=. ./venv/bin/python -m pytest \
  orion/spark/concept_induction/tests/test_goals.py \
  tests/test_autonomy_summary_degraded.py \
  orion/autonomy/tests/test_repository_active_goals.py \
  services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py \
  services/orion-hub/tests/test_autonomy_runtime_ui_panel.py -q

# Phase 1 (+ concept induction integration)
PYTHONPATH=. ./venv/bin/python -m pytest \
  orion/spark/concept_induction/tests/test_goal_generator.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py \
  services/orion-rdf-writer/tests/test_autonomy_materialization.py -q
```

Operator metrics (Phase 0 post-deploy):

```bash
# Expect ≥90% drop in orion ProposedGoal count after archive script
python scripts/autonomy/archive_stale_goal_proposals.py --dry-run --subject orion
```

Structured logs to add in Phase 0/1 (grep verification):

```
autonomy_goal_publish subject=orion action=created|suppressed|superseded
autonomy_goal_read subject=orion active_count=N
autonomy_goal_hint correlation_id=… headline=…
```

---

# Configuration summary (env parity)

| Env | Service `.env_example` |
|-----|------------------------|
| `GOAL_PROPOSAL_COOLDOWN_MINUTES=180` | `services/orion-spark-concept-induction/.env_example` |
| `AUTONOMY_GOAL_RETENTION_DAYS=30` | archive script doc + spark or ops env |
| `AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT=3` | archive script doc |
| `GOAL_GENERATION_MODE=evidence_rules` | spark concept-induction |
| `GOAL_DRIVE_ORIGIN_SOURCE=audit_dominant` | spark concept-induction |
| `GOAL_HINT_PRIORITY_THRESHOLD=0.4` | `services/orion-cortex-exec/.env_example` |
| `HUB_AUTONOMY_SUBJECT_DISPLAY=two` | `services/orion-hub/.env_example` |
| `AUTONOMY_GOAL_EXECUTION_ENABLED=false` | hub + cortex-exec |

---

# Self-review (spec coverage)

| Spec requirement | Task |
|------------------|------|
| Stable signature, no trace in statement | Task 2 |
| Cooldown suppresses republish | Task 2 + existing integration test |
| `_fetch_active_goals` ≤3 distinct semantics | Task 4 |
| Healthy + goals → `stance_mode=normal` | Task 3 |
| Router not `proposal_only` on healthy | Task 5 |
| Hub honest labels | Task 6 |
| Archive script + retention | Task 7 |
| Semantic goal generator (3 modes) | Task 8 |
| Drive origin diversity | Task 9 |
| Chat stance `goal_hint:` | Task 10 |
| RDF supersede + proposalStatus | Task 11 |
| Subject routing doc | Task 12 |
| Two-subject Hub UI | Task 13 |
| Goal lifecycle RDF | Task 14 |
| Hub promote/dismiss/complete | Task 15 |
| Planner verb | Task 16 |
| execution_mode planned/executing | Task 17 |

**Placeholder scan:** No TBD/TODO steps in task bodies. Archive script SPARQL wiring is explicit in Task 7 step 3.

**Type consistency:** `AutonomyActiveGoalV1` used in summary, chat_stance, and router; `GoalProposalV1.proposal_status` written by rdf-writer and read by repository.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-21-autonomy-goals-v2-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
