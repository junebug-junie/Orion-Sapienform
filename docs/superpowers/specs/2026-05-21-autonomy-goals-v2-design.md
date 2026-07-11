# Autonomy Goals v2 — Design Spec

**Status:** Draft for review  
**Date:** 2026-05-21  
**Authors:** Operator + agent (brainstorming session)  
**Related:** PR #601 (autonomy compact degraded state), Phase 3 materialization report, concept-induction goals pipeline

---

## 1. Executive summary

The autonomy goals pipeline is **live but semantically stubbed**. It materializes ~47k `ProposedGoal` artifacts into Fuseki, displays a static template headline in the Hub compact card, and labels every turn `proposal_only` without changing chat behavior. Juniper's subject slot is always empty by routing design.

This spec defines **four in-scope phases** to make goals honest, useful, and eventually executable:

| Phase | Name | Outcome |
|-------|------|---------|
| **0** | Stop the bleeding | Dedupe, honest UI, graph hygiene, faster reads |
| **1** | Semantic goals | Context-aware goal text, drive diversity, light chat wiring |
| **2** | Subject model clarity | Fix juniper/relationship confusion in UI and docs |
| **3** | Goal lifecycle & execution | Promote → plan → execute with operator gates |

Phases are **sequential dependencies**: each phase ships independently as one or more PRs, but Phase N assumes Phase N−1 contracts are stable.

---

## 2. Problem statement

### 2.1 Observed failures (2026-05-21 live stack)

| Symptom | Root cause |
|---------|------------|
| Goal headline never changes | Six hardcoded `GOAL_TEMPLATES`; `drive_origin` always `autonomy` |
| 47k+ identical semantic goals in graph | Signature includes trace suffix → cooldown bypassed every tick |
| `stance_mode: proposal_only` on healthy turns | Any `proposal_headlines` forces proposal_only (not degradation) |
| `execution: proposal_only` confusing | Router metadata flag; does not gate chat or tools |
| Juniper always `empty` | Chat turns materialize to `relationship`, not `user:juniper` |
| Goals don't affect replies | Only `response_hazards`; no goal→stance brief injection |
| Orion drives query ~17s | Graph bloat + heavy artifact history |

### 2.2 Current architecture (as-is)

```
Bus events (metacog, chat, telemetry)
  → spark-concept-induction (ConceptInductionBusWorker.handle_envelope)
    → DriveEngine.update() → drive pressures
    → GoalProposalEngine.propose() → GOAL_TEMPLATES + signature
    → bus: orion:memory:goals:proposed
  → orion-rdf-writer (_handle_goal_proposal, executionMode=proposal-only)
    → Fuseki goals graph (orion:ProposedGoal)

Chat turn
  → cortex-exec chat_stance (GraphAutonomyRepository._fetch_goals)
    → summarize_autonomy_lookup → proposal_headlines
  → router (_autonomy_payload_from_ctx) → Hub metadata
  → reasoning adapter (goal_proposal_headline, status=proposed)
  → substrate adapter (GoalNodeV1, non-executing)
```

Key files:

| Area | Path |
|------|------|
| Goal generation | `orion/spark/concept_induction/goals.py` |
| Drive math | `orion/spark/concept_induction/drives.py` |
| Subject routing | `orion/spark/concept_induction/identity.py` |
| Graph read | `orion/autonomy/repository.py` (`_fetch_goals`) |
| Summary / stance_mode | `orion/autonomy/summary.py` |
| Chat stance | `services/orion-cortex-exec/app/chat_stance.py` |
| Hub UI | `services/orion-hub/static/js/app.js` |
| RDF write | `services/orion-rdf-writer/app/autonomy.py` |
| Router export | `services/orion-cortex-exec/app/router.py` |
| Reasoning claims | `orion/reasoning/adapters/autonomy.py` |
| Promotion policy | `orion/reasoning/promotion.py` |

---

## 3. Goals & non-goals

### 3.1 Goals

1. **Honest compact card** — labels reflect actual state (healthy vs degraded vs goals-present).
2. **Bounded graph growth** — ≤1 active semantic goal per `(subject, drive_origin)`; historical artifacts archived or superseded.
3. **Meaningful goal text** — goals vary with drive context, tensions, and recent evidence (Phase 1+).
4. **Light behavioral influence** — active goals surface in chat stance as hints, not action plans (Phase 1).
5. **Clear subject model** — operators understand orion vs relationship vs juniper (Phase 2).
6. **Optional execution path** — operator-gated promote → plan → execute (Phase 3).

### 3.2 Non-goals (this spec)

- Autonomous tool execution without operator approval
- Replacing the planner or agent-chain architecture
- LLM-generated goals on every metacog tick (Phase 1 uses bounded triggers)
- Removing the cognitive substrate `proposal_only` mutation gate
- Migrating off Fuseki or redesigning the full autonomy v2 reducer

---

## 4. Target architecture (to-be)

```
Bus events
  → concept-induction (deduped goal publish)
    → GoalProposalEngine v2 (semantic text, stable signature)
    → optional GoalGenerator (LLM/rules on window, Phase 1)
  → rdf-writer (ProposedGoal + proposalStatus + supersedes edge)
  → Fuseki (bounded active goals + archived history)

Chat turn
  → repository._fetch_active_goals()  # not raw latest-N
  → summarize_autonomy_lookup (stance_mode decoupled from goals when healthy)
  → chat_stance (goal_hint in response_priorities when priority ≥ threshold)
  → router (autonomy_execution_mode: none | hint_only | planned | executing)
  → Hub (honest labels, 2-subject or annotated 3-subject UI)

Phase 3 only:
  → Hub promote action → reasoning canonical claim → planner task → supervisor
```

---

## 5. Data model changes

### 5.1 GoalProposalV1 (Phase 0–1, backward compatible)

Add optional fields (extra allowed on GraphReadyArtifact if needed):

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `goal_statement_base` | string | 0 | Template/semantic text without trace suffix |
| `proposal_status` | enum | 0 | `proposed` \| `active` \| `superseded` \| `archived` |
| `supersedes_artifact_id` | string? | 0 | Lineage when updating active goal |
| `semantic_source` | enum | 1 | `template` \| `evidence_rules` \| `llm` |

RDF (`orion-rdf-writer`): materialize `orion:proposalStatus`, `orion:supersedesArtifact`, `orion:goalStatementBase`.

### 5.2 AutonomySummaryV1 (Phase 0–1)

| Field | Change |
|-------|--------|
| `stance_mode` | `proposal_only` only when degraded + proposals, OR explicit execution lane engaged |
| `active_goals` | New: list of `{drive_origin, headline, priority, artifact_id}` (max 3) |
| `goals_present` | New bool for UI badge |

### 5.3 Router / Hub metadata (Phase 0–3)

Replace overloaded `autonomy_execution_mode=proposal_only` with:

| Value | Meaning |
|-------|---------|
| `none` | No goals or goals ignored |
| `hint_only` | Goal injected in stance brief (Phase 1 default when goals exist) |
| `planned` | Promoted goal linked to planner task (Phase 3) |
| `executing` | Approved plan in progress (Phase 3) |

Deprecated: bare `proposal_only` → map to `hint_only` in Hub for one release, then remove.

---

## 6. Phase 0 — Stop the bleeding

**Objective:** Fix dedupe, read path, UI honesty, and graph volume without changing goal semantics.

### 6.1 Scope

#### 6.1.1 Goal signature & publish dedupe

**File:** `orion/spark/concept_induction/goals.py`

1. **`_goal_statement()`** — store trace in provenance only; do not append `· trace=…` to persisted `goal_statement`.
2. **`_signature()`** — hash `(subject, model_layer, drive_origin, tension_signature, goal_statement_base)` where `goal_statement_base` is template text without tension suffix unless tension changes semantic content.
3. **`propose()`** — when signature matches existing active goal within cooldown, return `GoalDecision(proposal=None, suppressed_signature=signature)` (already partially implemented; fix inputs so it actually fires).
4. **Artifact ID** — keep `goal-{signature}` stable per semantic goal (same signature → same artifact ID → rdf-writer upsert or skip).

**Env:** `GOAL_PROPOSAL_COOLDOWN_MINUTES` (default 180) — document that cooldown is per semantic signature.

#### 6.1.2 Active goal read path

**File:** `orion/autonomy/repository.py`

Replace `_fetch_goals` "latest N by timestamp" with `_fetch_active_goals`:

```sparql
# Pseudologic: per drive_origin, pick highest priority ProposedGoal
# where proposalStatus != superseded|archived (default proposed|active)
# ORDER BY priority DESC, created_at DESC
# LIMIT goals_limit (default 3)
```

Post-process in Python: dedupe by `drive_origin`, keep highest priority.

**File:** `orion/autonomy/summary.py`

- `_proposal_headline_for_display()` — unchanged (still strips operational suffixes for legacy rows).

#### 6.1.3 Stance mode decoupling (healthy + goals)

**File:** `orion/autonomy/summary.py` — `_derive_stance_mode()`

| state_quality | has_proposals | stance_mode |
|---------------|---------------|-------------|
| healthy | yes | `normal` (not `proposal_only`) |
| healthy | no | `normal` |
| degraded_* | yes | `proposal_only` |
| degraded_* | no | `unavailable` |
| contextual_fallback | yes | `fallback_contextual` |

Add `goals_present: bool` to summary for Hub badge.

#### 6.1.4 Router metadata

**File:** `services/orion-cortex-exec/app/router.py`

- When goals exist: `autonomy_execution_mode: "hint_only"` (Phase 0 interim) or `"none"` until Phase 1 wiring confirmed.
- Decision: Phase 0 sets `"none"` + new field `autonomy_goals_present: true` to avoid lying about hints not yet wired.

#### 6.1.5 Hub UI honesty

**File:** `services/orion-hub/static/js/app.js`

| Line | Before | After |
|------|--------|-------|
| Overview state | `proposal_only` when goals | `healthy` + badge `goals: N active` |
| execution | `proposal_only` | `none` or `hint_only` |
| Juniper empty | blank row | `n/a (dyadic chat → relationship)` |
| Proposals section title | `Proposals (proposal-only)` | `Active goals (non-executing)` when healthy |

#### 6.1.6 Graph hygiene (one-time + retention)

**New script:** `scripts/autonomy/archive_stale_goal_proposals.py`

- Target: goals graph, subject=orion, `drive_origin=autonomy`, duplicate base text
- Mark `proposalStatus=archived` via SPARQL UPDATE or rdf-writer batch
- Retention policy env: `AUTONOMY_GOAL_RETENTION_DAYS=30`, `AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT=3`

**Operator runbook:** execute once after Phase 0 deploy; expect graph size reduction and drives query latency improvement.

### 6.2 Out of scope (Phase 0)

- LLM goal generation
- Juniper materialization
- Planner / execution hooks

### 6.3 Acceptance criteria (Phase 0)

- [ ] New metacog ticks do not create new Fuseki goal rows when `(subject, drive_origin, base_text, tensions)` unchanged for 180 min
- [ ] Compact card shows `stance_mode: normal` when `state_quality=healthy` and goals exist
- [ ] `autonomy_execution_mode` is not `proposal_only` on healthy turns
- [ ] `_fetch_active_goals` returns ≤3 rows with distinct semantics per subject
- [ ] After archive job, Orion `ProposedGoal` count drops by ≥90% (operator metric)
- [ ] Orion drives SPARQL p95 latency improves measurably (target: <10s at 20s timeout budget)
- [ ] All existing autonomy tests pass; new tests for signature dedupe and stance_mode

### 6.4 Tests (Phase 0)

| Test file | Cases |
|-----------|-------|
| `orion/spark/concept_induction/tests/test_goals.py` (new) | signature stable without trace; cooldown suppresses republish |
| `tests/test_autonomy_summary_degraded.py` | healthy + proposals → `stance_mode=normal` |
| `orion/autonomy/tests/test_repository_active_goals.py` (new) | active goal selection, dedupe by drive_origin |
| `services/orion-hub/tests/test_autonomy_runtime_ui_panel.py` | updated labels |

### 6.5 PRs (Phase 0)

| PR | Title |
|----|-------|
| 0a | fix(goal): stable signature, no trace in statement, cooldown works |
| 0b | feat(autonomy): active goal read path + summary stance decoupling |
| 0c | fix(hub): honest autonomy labels + juniper annotation |
| 0d | chore(graph): goal archive script + retention policy |

---

## 7. Phase 1 — Semantic goals

**Objective:** Goals vary with context; one active goal influences chat stance lightly.

**Depends on:** Phase 0 dedupe and active-goal read path.

### 7.1 Scope

#### 7.1.1 Evidence-grounded goal generation

**New module:** `orion/spark/concept_induction/goal_generator.py`

Interface:

```python
def generate_goal_statement(
    *,
    drive_origin: str,
    pressures: dict[str, float],
    tensions: list[TensionEventV1],
    window_summary: str | None,  # from concept-induction window
    mode: Literal["template", "evidence_rules", "llm"],
) -> str: ...
```

**Modes (env `GOAL_GENERATION_MODE`):**

| Mode | Behavior |
|------|----------|
| `template` | Current GOAL_TEMPLATES (fallback) |
| `evidence_rules` | Template + tension clause + optional window noun phrase (no LLM) |
| `llm` | Single LLM call, ≤120 chars, structured output; fallback to evidence_rules |

**Trigger:** Only on `run_for_subject()` completion or chat turn concept induction — **not** every metacog tick unless window materially changed.

**File changes:** `goals.py` delegates to `goal_generator`; `bus_worker.py` passes window summary.

#### 7.1.2 Drive origin diversity

**File:** `orion/spark/concept_induction/drives.py` + tension extractors

Problem: `autonomy` always wins `_drive_origin()` because metacog ticks inflate autonomy pressure.

Mitigations (implement at least one):

1. **Event-type priors** — metacog tick applies +0.05 autonomy, chat turn applies +0.1 relational, etc.
2. **Dominant drive from audit** — goal `drive_origin` follows `dominant_drive` on latest DriveAudit, not raw max pressure
3. **Rotation guard** — if same drive_origin &gt; N consecutive proposals, force secondary drive by pressure

**Env:** `GOAL_DRIVE_ORIGIN_SOURCE=pressures|audit_dominant` (default `audit_dominant` in Phase 1).

#### 7.1.3 Supersede-on-change publish

When semantic goal changes for `(subject, drive_origin)`:

1. Publish new `GoalProposalV1` with `proposal_status=active`
2. Publish supersession metadata linking old `artifact_id`
3. rdf-writer marks prior goal `superseded`

#### 7.1.4 Chat stance goal hint

**File:** `services/orion-cortex-exec/app/chat_stance.py`

When `summary.active_goals[0].priority >= GOAL_HINT_PRIORITY_THRESHOLD` (default 0.4):

```python
response_priorities += [f"goal_hint:{headline[:80]}"]
```

And set `autonomy_execution_mode: "hint_only"`.

**Constraint:** Hint is contextual, not imperative — hazard remains `"do not present proposals as commitments"`.

#### 7.1.5 Mind / reasoning (read-only)

**Files:** `orion/reasoning/adapters/autonomy.py`, `services/orion-mind/app/evidence.py`

- Pass `active_goals` with semantic source in claim qualifiers
- No change to promotion policy (still `proposed` only)

### 7.2 Out of scope (Phase 1)

- Operator promote UI
- Planner task creation
- Removing template fallback entirely

### 7.3 Acceptance criteria (Phase 1)

- [ ] Goal headline changes when drive_origin or lead tension changes (live verification)
- [ ] Same context within cooldown does not republish (Phase 0 dedupe still holds)
- [ ] At least two distinct `drive_origin` values appear over 24h normal operation (orion + relationship)
- [ ] Chat stance logs show `goal_hint:` in `response_priorities` when priority ≥ threshold
- [ ] `autonomy_execution_mode=hint_only` when hint wired; `none` when no goals
- [ ] LLM mode fails closed to evidence_rules without empty goals

### 7.4 Tests (Phase 1)

| Test file | Cases |
|-----------|-------|
| `orion/spark/concept_induction/tests/test_goal_generator.py` | all modes, fallback |
| `orion/spark/concept_induction/tests/test_goals.py` | drive_origin from audit |
| `services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py` | goal_hint in priorities |

### 7.5 PRs (Phase 1)

| PR | Title |
|----|-------|
| 1a | feat(goals): evidence-grounded goal generator + env modes |
| 1b | fix(drives): drive_origin diversity for goal proposals |
| 1c | feat(chat-stance): goal_hint in response_priorities |
| 1d | feat(rdf-writer): supersede edges + proposalStatus |

---

## 8. Phase 2 — Subject model clarity

**Objective:** Remove juniper confusion; align UI with materialization routing.

**Depends on:** Phase 0 UI patterns; can parallelize with Phase 1.

### 8.1 Scope

#### 8.1.1 Document routing contract

**New doc section:** `docs/architecture/autonomy_subjects.md`

| Subject | Entity ID | Materialized when | Used for |
|---------|-----------|-------------------|----------|
| orion | `self:orion` | metacog, assistant turns, telemetry | self-model drives/goals |
| relationship | `relationship:orion\|juniper` | dyadic chat turns | dyadic drives/goals |
| juniper | `user:juniper` | explicit user-model events (future) | user-model layer (not chat) |

Source: `orion/spark/concept_induction/identity.py` lines 73–78 (chat → relationship by design).

#### 8.1.2 Hub UI — two-subject mode (recommended)

**Env:** `HUB_AUTONOMY_SUBJECT_DISPLAY=three|two` (default `two`)

**two mode:**
- Availability: `orion + relationship (orion↔juniper)` — 2/2 not 2/3
- Remove juniper debug row from compact card
- Raw debug still includes juniper if backend returns it (for engineers)

**three mode (legacy):**
- Juniper row shows `n/a — dyadic scope uses relationship` when empty

#### 8.1.3 Optional: juniper materialization path (stretch)

If operator wants juniper populated:

- New trigger: `source_kind in {user_profile, biometrics, journal_write}` with `subject=juniper`
- **Do not** route chat turns to juniper (preserve relationship dyad)
- Behind env: `CONCEPT_MATERIALIZE_JUNIPER_USER_MODEL=false` (default)

### 8.2 Acceptance criteria (Phase 2)

- [ ] Default Hub UI does not imply broken juniper fetch
- [ ] Architecture doc merged and linked from PR report
- [ ] Tests cover two-subject and three-subject display modes

### 8.3 PRs (Phase 2)

| PR | Title |
|----|-------|
| 2a | docs(architecture): autonomy subject routing |
| 2b | feat(hub): two-subject autonomy compact card mode |

---

## 9. Phase 3 — Goal lifecycle & execution

**Objective:** Operator can promote a goal to a planned/executed action with existing mutation/reasoning gates.

**Depends on:** Phase 1 semantic goals; Phase 0 stable artifact IDs.

### 9.1 Scope

#### 9.1.1 Goal status state machine

```
proposed → active → completed
         ↘ superseded / archived
active → planned (operator promote)
planned → executing (planner accepts)
executing → completed | failed
```

**RDF:** extend `orion:proposalStatus` literals; add `orion:plannedTaskId`, `orion:completedAt`.

#### 9.1.2 Hub operator actions

**File:** `services/orion-hub/scripts/api_routes.py` (or new autonomy routes)

| Action | Effect |
|--------|--------|
| **Promote goal** | reasoning claim → `canonical` (HITL); goal `proposal_status=planned` |
| **Dismiss goal** | `proposal_status=archived` |
| **Complete goal** | `proposal_status=completed` |

Uses existing `orion/reasoning/promotion.py` gate: `goal_proposal_headline` → canonical requires HITL.

#### 9.1.3 Planner integration

**File:** `services/orion-planner-react/app/api.py` (or cortex-orch verb)

- New verb or extension: `autonomy.goal.execute.v1`
- Input: `goal_artifact_id`, `goal_statement`, `drive_origin`
- Output: planner task ID stored on goal artifact

#### 9.1.4 Supervisor / agent-chain gate

**Files:** `services/orion-cortex-exec/app/supervisor.py`, `orion/substrate/mutation_decision.py`

- `autonomy_execution_mode=executing` only when planned task is active
- Cognitive lane mutations remain `require_review` / `cognitive_lane_proposal_only`
- Goals do not bypass capability selector or bound-capability contracts

#### 9.1.5 Router metadata (final)

| State | `autonomy_execution_mode` |
|-------|---------------------------|
| No goals | `none` |
| Goals, no promote | `hint_only` |
| Operator promoted, planner queued | `planned` |
| Task running | `executing` |

### 9.2 Out of scope (Phase 3)

- Fully autonomous goal execution without operator promote
- Auto-promote goals to canonical claims
- Replacing agent-chain bound-capability model

### 9.3 Acceptance criteria (Phase 3)

- [ ] Operator can promote active goal from Hub; status transitions logged
- [ ] Promoted goal creates planner task; `autonomy_execution_mode=planned`
- [ ] Reasoning promotion to canonical requires HITL (existing policy preserved)
- [ ] No autonomous tool calls from unpromoted goals (regression test)
- [ ] Complete/dismiss updates graph status

### 9.4 Tests (Phase 3)

| Area | Cases |
|------|-------|
| Hub API | promote, dismiss, complete |
| Router | execution_mode transitions |
| Promotion | goal canonical still HITL |
| E2E | promote → plan → mock execute → complete |

### 9.5 PRs (Phase 3)

| PR | Title |
|----|-------|
| 3a | feat(autonomy): goal status state machine in RDF + read path |
| 3b | feat(hub): goal promote/dismiss/complete actions |
| 3c | feat(planner): autonomy goal execute verb |
| 3d | feat(router): execution_mode planned/executing |

---

## 10. Cross-cutting concerns

### 10.1 Performance

| Metric | Current | Phase 0 target | Phase 1+ target |
|--------|---------|----------------|-----------------|
| Orion ProposedGoal count | ~47k | <500 active | <200 active |
| Orion drives query p95 | ~17s | <10s | <5s |
| Goals SPARQL | unbounded scan | indexed active query | same |

Consider Fuseki index on `(subjectKey, entityId, proposalStatus, driveOrigin)` if SPARQL remains slow post-archive.

### 10.2 Observability

Structured log events (all phases):

```
autonomy_goal_publish subject=orion drive_origin=autonomy signature=… action=created|suppressed|superseded
autonomy_goal_read subject=orion active_count=1
autonomy_goal_hint correlation_id=… headline=… priority=0.72
autonomy_goal_promote artifact_id=… operator=…
```

### 10.3 Configuration summary

| Env | Default | Phase |
|-----|---------|-------|
| `GOAL_PROPOSAL_COOLDOWN_MINUTES` | 180 | 0 |
| `AUTONOMY_GOAL_RETENTION_DAYS` | 30 | 0 |
| `AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT` | 3 | 0 |
| `GOAL_GENERATION_MODE` | `evidence_rules` | 1 |
| `GOAL_DRIVE_ORIGIN_SOURCE` | `audit_dominant` | 1 |
| `GOAL_HINT_PRIORITY_THRESHOLD` | 0.4 | 1 |
| `HUB_AUTONOMY_SUBJECT_DISPLAY` | `two` | 2 |
| `CONCEPT_MATERIALIZE_JUNIPER_USER_MODEL` | false | 2 |
| `AUTONOMY_GOAL_EXECUTION_ENABLED` | false | 3 |

### 10.4 Migration & rollback

| Phase | Migration | Rollback |
|-------|-----------|----------|
| 0 | Run archive script; deploy dedupe | Revert read path; goals still in graph |
| 1 | Enable generator mode gradually | `GOAL_GENERATION_MODE=template` |
| 2 | Hub display env | `HUB_AUTONOMY_SUBJECT_DISPLAY=three` |
| 3 | Feature flag `AUTONOMY_GOAL_EXECUTION_ENABLED` | Disable flag; hints still work |

### 10.5 Security & safety

- Goals remain **non-executing** until Phase 3 promote + HITL
- LLM goal generation must not include raw user PII in persisted statement (use window summary redaction)
- Archive script requires operator confirmation (dry-run mode default)

---

## 11. Success metrics (overall)

After all phases (or Phase 1 for partial success):

| Metric | Success |
|--------|---------|
| Goal headline static over 10 turns | **Fail** — must vary with context |
| Fuseki active goals per subject | ≤3 |
| Healthy turn shows `proposal_only` | **Never** |
| Operator understands juniper empty | **Yes** (survey or doc) |
| Goal affects chat priorities | **Yes** (log evidence) |
| Unpromoted goal triggers tool call | **Never** |

---

## 12. Implementation order & estimates

| Phase | Effort | Risk |
|-------|--------|------|
| 0 | 2–3 days | Low |
| 1 | 4–6 days | Medium (LLM mode) |
| 2 | 1–2 days | Low |
| 3 | 8–12 days | High (cross-service) |

**Recommended merge order:** 0a→0d, then 1a→1d, then 2a→2b, then 3a→3d.

---

## 13. Open questions

1. **Archive vs delete** — Should archived goals remain in Fuseki forever or move to cold storage?
2. **LLM provider for Phase 1** — Reuse mind/llm-gateway or local rules-only first?
3. **Relationship goals in UI** — Show relationship's active goal when orion selected, or only on fallback?
4. **Phase 3 planner verb** — New verb vs extend existing planner-react write verbs?

---

## 14. References

- `orion/spark/concept_induction/goals.py` — GOAL_TEMPLATES, GoalProposalEngine
- `orion/spark/concept_induction/identity.py` — chat → relationship routing
- `docs/reports/orion_autonomy_phase3_materialization.md` — proposal-only encoding
- `docs/superpowers/pr-reports/2026-05-20-autonomy-compact-degraded-state.md` — recent read-path fixes
- Live Fuseki counts (2026-05-21): orion 47,689 ProposedGoal; juniper 0; relationship 22

---

## 15. Approval checklist

- [ ] Operator approves four-phase scope
- [ ] Phase 0 archive job runbook accepted
- [ ] Phase 1 LLM mode scope accepted (or rules-only first)
- [ ] Phase 2 two-subject UI default accepted
- [ ] Phase 3 execution gated behind feature flag accepted

**Next step after approval:** Invoke `writing-plans` skill to produce `docs/superpowers/plans/2026-05-21-autonomy-goals-v2-implementation.md` with task-level breakdown per PR.
