# PR: Autonomy Goals v2 — honest pipeline, semantic goals, lifecycle execution

**Branch:** `feat/autonomy-goals-v2`  
**Worktree:** `.worktrees/feat-autonomy-goals-v2`  
**Design:** [docs/superpowers/specs/2026-05-21-autonomy-goals-v2-design.md](../../specs/2026-05-21-autonomy-goals-v2-design.md)  
**Plan:** [docs/superpowers/plans/2026-05-21-autonomy-goals-v2-implementation.md](../../plans/2026-05-21-autonomy-goals-v2-implementation.md)

## Summary

Four-phase implementation makes the autonomy goals pipeline honest and useful:

1. **Phase 0 — Stop the bleeding:** Stable signatures (no trace suffix), active-goal SPARQL read path with `drive_origin` dedupe, healthy stance decoupled from `proposal_only`, router exports `autonomy_execution_mode=none` + `autonomy_goals_present`, Hub honest labels, archive script for graph hygiene.
2. **Phase 1 — Semantic goals:** Evidence-grounded `goal_generator` (template / evidence_rules / llm stub), audit-dominant drive origin, `goal_hint:` in chat stance priorities, RDF supersession + `proposalStatus` materialization.
3. **Phase 2 — Subject clarity:** `docs/architecture/autonomy_subjects.md`, Hub `HUB_AUTONOMY_SUBJECT_DISPLAY=two` (orion + relationship, juniper annotated).
4. **Phase 3 — Lifecycle & execution (feature-flagged):** `plannedTaskId` / `completedAt` RDF fields, Hub promote/dismiss/complete API, planner `autonomy.goal.execute.v1`, full `execution_mode` ladder (`none` | `hint_only` | `planned` | `executing`), promote→planner chain, secured planner verb.

**20 commits** · **47 files** · **+2986 / −60 lines** (vs `main`)

## Key changes by area

| Area | Change |
|------|--------|
| `orion/core/schemas/drives.py` | `GoalProposalV1` lifecycle fields (`goal_statement_base`, `proposal_status`, `semantic_source`, `planned_task_id`, …) |
| `orion/spark/concept_induction/goals.py` | Stable signature, active status on supersede, window summary → generator, supersede artifact tracking |
| `orion/spark/concept_induction/goal_generator.py` | **New** — `GOAL_GENERATION_MODE` (default `evidence_rules`) |
| `orion/autonomy/repository.py` | `_fetch_active_goals` filters superseded/archived/completed, dedupes by `drive_origin` |
| `orion/autonomy/summary.py` | `active_goals`, `goals_present`, healthy → `stance_mode=normal` |
| `orion/autonomy/goal_actions.py` | **New** — promote/dismiss/complete + promote→planner task allocation |
| `services/orion-cortex-exec` | Router execution ladder; `goal_hint:` in chat stance; supervisor unpromoted-goal gate |
| `services/orion-rdf-writer` | `goalStatementBase`, `proposalStatus`, `supersedesArtifact`, prior marked `superseded` |
| `services/orion-hub` | Honest compact labels; two-subject display; goal action API routes |
| `services/orion-planner-react` | `autonomy.goal.execute.v1` verb (gated + operator auth) |
| `scripts/autonomy/archive_stale_goal_proposals.py` | **New** — retention + max-active archive (dry-run default) |
| `docs/architecture/autonomy_subjects.md` | **New** — subject routing contract |
| `orion/schemas/registry.py` | `GoalProposalV1`, `AutonomyGoalPlannedV1` registered |
| `orion/bus/channels.yaml` | `orion:memory:goals:proposed`, `orion:autonomy:goal:planned` catalog entries |

## Configuration (env parity)

| Env | Default | Service `.env_example` |
|-----|---------|------------------------|
| `GOAL_PROPOSAL_COOLDOWN_MINUTES` | 180 | spark-concept-induction |
| `GOAL_GENERATION_MODE` | `evidence_rules` | spark-concept-induction |
| `GOAL_DRIVE_ORIGIN_SOURCE` | `audit_dominant` | spark-concept-induction |
| `AUTONOMY_GOAL_RETENTION_DAYS` | 30 | cortex-exec |
| `AUTONOMY_GOAL_MAX_ACTIVE_PER_SUBJECT` | 3 | cortex-exec |
| `GOAL_HINT_PRIORITY_THRESHOLD` | 0.4 | cortex-exec |
| `HUB_AUTONOMY_SUBJECT_DISPLAY` | `two` | orion-hub |
| `AUTONOMY_GOAL_EXECUTION_ENABLED` | **false** | hub + cortex-exec + planner |
| `BUS_AUTONOMY_GOAL_PLANNED_OUT` | `orion:autonomy:goal:planned` | planner + root `.env_example` |

Local `.env` files mirror `.env_example` for cortex-exec, hub, spark-concept-induction, and planner (operator machine copies).

## Verification

Run from worktree (symlink or use main `venv`):

```bash
cd .worktrees/feat-autonomy-goals-v2
ln -sf ../../venv venv  # if worktree has no venv

# Core + cortex + hub (repo root pytest)
PYTHONPATH=. ./venv/bin/python -m pytest \
  orion/spark/concept_induction/tests/test_goals.py \
  orion/spark/concept_induction/tests/test_goal_generator.py \
  tests/test_autonomy_summary_degraded.py \
  orion/autonomy/tests/ \
  services/orion-cortex-exec/tests/test_router_autonomy_payload_export.py \
  services/orion-cortex-exec/tests/test_chat_stance_autonomy_plumbing.py \
  services/orion-hub/tests/test_autonomy_runtime_ui_panel.py \
  -q

# Service-scoped
ORION_PYTHON=../../venv/bin/python ./scripts/test_service.sh orion-hub services/orion-hub/tests/test_autonomy_goal_actions.py -q
ORION_PYTHON=../../venv/bin/python ./scripts/test_service.sh orion-rdf-writer services/orion-rdf-writer/tests/test_autonomy_materialization.py -q
ORION_PYTHON=../../venv/bin/python ./scripts/test_service.sh orion-planner-react services/orion-planner-react/tests/test_autonomy_goal_execute.py -q
```

| Suite | Result |
|-------|--------|
| Core autonomy + concept-induction + cortex router/stance | **114+ passed** |
| Hub goal actions | **6 passed** |
| RDF materialization | **7 passed** |
| Planner execute verb | **3 passed** |

**Live Fuseki / E2E promote flow:** UNVERIFIED — deploy stack and run operator promote with `AUTONOMY_GOAL_EXECUTION_ENABLED=true`.

## Post-deploy operator runbook (Phase 0)

```bash
# Dry-run stale goal archive (expect large candidate list on orion)
PYTHONPATH=. python scripts/autonomy/archive_stale_goal_proposals.py --dry-run --subject orion

# After review
PYTHONPATH=. python scripts/autonomy/archive_stale_goal_proposals.py --apply --subject orion
```

## Code review follow-ups (addressed in branch)

- Promote now chains to planner task + `plannedTaskId`
- Planner execute verb gated (`AUTONOMY_GOAL_EXECUTION_ENABLED`, operator/internal token, `planned` status)
- Prior goal marked `superseded` in RDF on supersede edge
- `window_summary` wired from bus worker
- Archive script honors `AUTONOMY_GOAL_RETENTION_DAYS`
- `semantic_source` in reasoning qualifiers

## Remaining risks / out of scope

- **LLM goal mode** — `_llm_goal_text()` stub; falls back to `evidence_rules` (by design for Phase 1)
- **Hub UI buttons** — promote/dismiss/complete are API-only; no compact-card buttons yet
- **Fuseki index** — if active-goal query still slow post-archive, add index on `(subjectKey, proposalStatus, driveOrigin)`
- **Phase 3 default off** — execution requires `AUTONOMY_GOAL_EXECUTION_ENABLED=true` on hub, cortex-exec, and planner

## Test plan (operator)

- [ ] Merge branch; rebuild `orion-spark-concept-induction`, `orion-cortex-exec`, `orion-hub`, `orion-rdf-writer`, `orion-planner-react`
- [ ] Copy new env vars from `.env_example` → local `.env`
- [ ] Run archive script dry-run on orion; review candidate count
- [ ] Confirm healthy chat turn: `stance_mode=normal`, `autonomy_execution_mode=none` or `hint_only`, goals in compact card
- [ ] Confirm two-subject Hub: `orion + relationship (orion↔juniper)`, no broken juniper row
- [ ] With execution flag **off**: promote API returns 503
- [ ] With execution flag **on** + operator token: promote → planned + task_id; execute verb requires auth
- [ ] Grep logs: `autonomy_goal_publish`, `autonomy_goal_read`, `autonomy_goal_hint`, `autonomy_goal_promote`

## Commits (newest first)

```
a4761d91 test(cortex-exec): align autonomy debug worker caps with lane defaults
fd0aa481 fix(autonomy): goals pipeline hardening for supersession, retention, and semantics
fa7394ad fix(autonomy): chain promote to planner task allocation and secure execute verb
5164d71c feat(router): execution_mode planned/executing ladder for autonomy goals
b0c8d1ea feat(planner): add autonomy.goal.execute.v1 verb for promoted goals
1198f7ce feat(hub): add operator goal promote/dismiss/complete API routes
b9f1323a feat(autonomy): read planned task lifecycle fields from goals graph
bb767f59 feat(hub): two-subject autonomy display via HUB_AUTONOMY_SUBJECT_DISPLAY
03eb5403 docs(architecture): autonomy subject routing contract
fcb57b86 feat(rdf-writer): materialize goal supersession and proposal status
228704ab feat(cortex-exec): inject goal_hint in chat stance priorities
6b228904 feat(concept-induction): use audit-dominant drive for goal origin
b96a2510 feat(concept-induction): add evidence-grounded goal generator
88d1b30f chore(graph): goal archive script with retention policy
2eccd687 fix(hub): honest autonomy labels and juniper routing annotation
324b2a59 fix(router): replace proposal_only execution_mode with none + goals_present
e7606d6e feat(autonomy): active goal read path with drive_origin dedupe
9baba12b feat(autonomy): decouple healthy stance_mode from goal proposals
5e6bcd82 fix(goals): stable signature without trace suffix in goal_statement
2a095db5 feat(goals): extend GoalProposalV1 with status and semantic fields
```
