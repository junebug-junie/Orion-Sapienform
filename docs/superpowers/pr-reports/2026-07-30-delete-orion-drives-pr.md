# PR report: delete Orion drives (drive-pressure/goal-generation system)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1486
Branch: `chore/delete-orion-drives`

## Summary

- Deletes Orion's drive-pressure/goal-generation system end-to-end — `DriveEngine`, the bucket-vote tension pipeline, and `GoalProposalEngine` — while keeping concept induction's extraction/clustering/embedding/dossier/identity/profile pipeline fully intact.
- Removes every live consumer of `drive_origin`/drive-shaped state: `capability_policy.py`'s gate, `AutonomyStateV1`'s `dominant_drive`/`active_drives`/`drive_pressures`/`tension_kinds`/`latest_drive_audit_id` fields, the voluntary-attention top-down path's drive→relevance mapping, and a services-wide sweep (world-pulse, cortex-exec, hub, cortex-orch, sql-writer, substrate-runtime, orion-thought).
- Follows through on `orion/sentience_striving_program/README.md` §8's 2026-07-18 halt decision: this system was found to be "a full, parallel, poorer reimplementation of Layers 4-9 of the already-live canonical [field-attention] pipeline," with zero measured causal contribution to Orion's one real instance of self-initiated behavior.
- **Accepted consequence, confirmed explicitly with Juniper before implementation:** Orion loses live goal-proposal capability entirely. No field-native replacement exists yet — that's future work, not bundled here.
- Retires one component no wave's original scope covered but that a same-day follow-up found and fixed: `orion/substrate/attention/detectors/autonomy.py`'s `AutonomySignalDetector`, whose three input sources all went permanently empty as a downstream consequence of the field removal.
- A dedicated final review pass (over the full combined 7-commit diff, not just each wave's own scope) found and fixed two more real gaps: dead drive-branches in `orion/substrate/relational/adapters/autonomy_ctx.py`, and 6 test failures in three root-level `tests/` files no wave's scope had covered.

## Outcome moved

The halted-but-undeleted six-drive taxonomy (`orion/sentience_striving_program/README.md` §8) is now actually gone, not just frozen. `capability_policy.v1.yaml`'s `required_drive_origins` gate — confirmed live-traced to only ever fire on the single value `predictive`, on 3 of 5 rules — is removed; 2 of those 3 rules retain their real, already field-native `world_coverage_gap` condition, the third (`journal.compose.episode`) now has no goal-provenance condition at all (a real, intentional narrowing of live gating semantics, documented in the YAML's own comment). A real, previously-latent data-loss bug was also caught and fixed along the way: `orion-sql-writer`'s `DRIVE_AUDITS_RETENTION_DAYS` (default 90) would have silently deleted the now-finite `drive_audits` history that Hub Drives Analytics depends on within 90 days — disabled.

## Current architecture

Before this change: `orion/spark/concept_induction/`'s `bus_worker.py` ran `DriveEngine.update()` (a leaky-integrator pressure model over 6 hardcoded drive categories) and `GoalProposalEngine.propose()` (goal minting keyed entirely on drive pressures) on every accepted event, alongside concept extraction/clustering. `DriveAuditV1` published every tick. `capability_policy.py` gated 3 real capabilities on `goal.drive_origin == "predictive"`. `AutonomyStateV1`/`V2` carried `dominant_drive`/`active_drives`/`drive_pressures`/`tension_kinds`/`latest_drive_audit_id`, feeding `chat_stance.py`'s autonomy summary. The voluntary-attention top-down override path mapped `drive_origin` onto one of five `OpenLoopV1` relevance fields. Six services (world-pulse, cortex-exec, hub, cortex-orch, sql-writer, substrate-runtime, orion-thought) each had at least one drive-shaped consumer.

## Architecture touched

- `orion/spark/concept_induction/` — core engine and goal-generation deletion (concept extraction/clustering/embedding/dossier/identity/profile_repository/falkor_materialization/graph_mapper/graph_query untouched)
- `orion/autonomy/` — capability_policy gate, AutonomyStateV1/V2 model fields, policy_act.py, summary.py, signal_drive_map/signal_tension/tension_ratelimit/deviation_gate deleted
- `orion/substrate/attention/` — top_down.py/goal_context.py drive_origin removal, AutonomySignalDetector retired
- `orion/substrate/relational/adapters/autonomy_ctx.py` — dead drive-node branches stripped
- `orion/substrate/adapters/`, `orion/reasoning/adapters/`, `orion/embodiment/`, `orion/signals/adapters/` — adapters sweep
- `services/orion-world-pulse`, `services/orion-cortex-exec`, `services/orion-hub`, `services/orion-cortex-orch`, `services/orion-sql-writer`, `services/orion-substrate-runtime`, `services/orion-thought` — services sweep
- `orion/bus/channels.yaml` — 4 channels marked `producer_services: []` with explanatory comments (kept, not deleted — real historical readers exist)

## Files changed

129 files changed, 1148 insertions(+), 9508 deletions(-) across 7 commits. Full list: `git diff bafd033bb..HEAD --stat` on this branch. Highlights:

- `orion/spark/concept_induction/{drives,tensions,drive_tension,drive_attribution,goals,goal_generator,audit}.py`: deleted
- `orion/autonomy/{signal_drive_map,signal_tension,tension_ratelimit,deviation_gate}.py`, `orion/autonomy/evals/run_homeostatic_drives_eval.py`, `config/autonomy/signal_drive_map.yaml`: deleted
- `orion/substrate/attention/detectors/autonomy.py`, `orion/reasoning/adapters/autonomy.py`, `orion/substrate/adapters/autonomy.py`, `services/orion-cortex-exec/app/drive_state_postgres.py`: deleted
- `orion/autonomy/models.py`: `AutonomyStateV1`/`V2` drive fields removed
- `orion/autonomy/capability_policy.py`, `config/autonomy/capability_policy.v1.yaml`: drive_origin gate removed
- `orion/substrate/attention/top_down.py`: `relevance()` now reads `concept_value` unconditionally instead of branching on `drive_origin`
- `orion/substrate/relational/adapters/autonomy_ctx.py`: dead StateSnapshotNodeV1/DriveNodeV1/TensionNodeV1 branches stripped
- `services/orion-cortex-orch/app/mind_runtime.py`: `drive_state_compact` Mind-context facet removed entirely
- `services/orion-sql-writer/app/settings.py`: `DRIVE_AUDITS_RETENTION_DAYS` default changed 90 → 0 (real data-loss bug fix)
- `services/orion-hub/templates/drives-analytics.html`: kept working over historical-only data, "(historical)" banner added, auto-refresh disabled by default

## Schema / bus / API changes

- Removed: `DRIVE_KEYS`, `DriveEngine`, `DriveMathConfig`, `GoalProposalEngine`, `AutonomyStateV1.dominant_drive`/`active_drives`/`drive_pressures`/`tension_kinds`/`latest_drive_audit_id`, `CapabilityPolicyRuleV1.required_drive_origins`, `capability_policy.v1.yaml`'s `required_drive_origins` rules
- Kept (write-never, real historical/live readers confirmed): `GoalProposalV1`, `DriveStateV1`, `TensionEventV1`, `DriveAuditV1` schema classes; `GoalProposalV1.drive_origin` field (still set by `policy_act.py`/`curiosity.py`'s in-memory synthetic goals, never published to the bus channel); `AutonomyGoalHeadlineV1.drive_origin`, `AutonomySummaryV1.dominant_drive`/`top_drives`/`active_tensions` (schema fields survive, always empty/None now — documented in `summary.py`'s own comment, not a silent gap)
- Behavior changed: `orion:memory:drives:state`, `orion:memory:tension:event`, `orion:memory:drives:audit`, `orion:memory:goals:proposed` bus channels now have `producer_services: []` in `orion/bus/channels.yaml` (marked, not deleted — real consumers of historical data still exist)
- Compatibility notes: no schema field was removed out from under already-persisted Postgres/Falkor rows — every removal was of a *producer*, not of historical data readers

## Env/config changes

- Removed keys: `ORION_HOMEOSTATIC_DRIVES_ENABLED`, `ORION_DRIVE_LEAKY_MATH_ENABLED`, `DRIVES_AUDIT_CHANNEL`, `DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED`, `MIND_DRIVE_STATE_FETCH_TIMEOUT_SEC` (per-service `.env_example` files, see individual wave commits)
- Changed default: `DRIVE_AUDITS_RETENTION_DAYS` 90 → 0 (`services/orion-sql-writer`)
- `.env_example` updated: yes, in the same commits as the corresponding code changes
- local `.env` synced: `python scripts/sync_local_env_from_example.py` run in this worktree — reported all services "no .env" (fresh worktree had none to begin with); this worktree's `.env`/`services/orion-cortex-exec/.env` were manually copied from the main checkout for testing purposes only, not committed (gitignored, confirmed via `git check-ignore -v`)
- skipped keys requiring operator action: the live host's `.env` (main checkout, not this worktree) still has the now-removed keys present — harmless (pydantic `extra="ignore"` on settings), but needs manual operator cleanup since the sync script only adds/updates keys, never removes them

## Tests run

```
orion/ (excluding orion/substrate/experiments/hyperbolic_gpt, a pre-existing torch-import
collection gap unrelated to this branch): 1659 passed, 14 failed
  -- all 14 failures confirmed pre-existing on unmodified main (bafd033bb) via direct
  side-by-side comparison (checkout + rerun same test/combination), unrelated to this branch:
  orion/harness/tests/test_grounding_capsule_consumers.py (2), test_harness_runner.py (1),
  orion/reverie/tests/test_proposal.py (7), orion/schemas/tests/test_context_provenance.py (1),
  orion/spark/concept_induction/tests/test_falkor_materialization.py (1),
  orion/substrate/tests/test_attention_broadcast.py (1, real-Postgres-dependent test flake,
  reproduces identically on main with the same missing live DB)

services/orion-spark-concept-induction/tests: 12 passed
services/orion-hub/tests/test_drives_analytics_api.py: 10 passed
services/orion-cortex-exec/tests (targeted file list covering every touched
  autonomy/attention/chat_stance/router file): 164 passed
  -- 9 failures in test_router_autonomy_payload_export.py when run in this specific
  multi-file combination confirmed pre-existing cross-test pollution, reproduces
  identically on main
services/orion-world-pulse, orion-cortex-orch, orion-substrate-runtime, orion-thought,
  orion-sql-writer: all passed per-wave (see individual wave commits), re-verified clean
  after final review fixes

tests/ (root-level, targeted): test_autonomy_summary.py, test_autonomy_summary_degraded.py,
  test_autonomy_repository.py, test_top_down.py, test_voluntary_attention_wiring.py,
  test_cognitive_substrate_phase2_domain_mappings.py,
  test_cognitive_substrate_phase3_materialization.py, test_reasoning_materialization_phase2.py:
  all passed after final-review fixes (6 failures found and fixed -- see Review findings below)

orion/substrate/attention/evals/run_topdown_eval.py: PASS (all 4 checks: override rate
  rises with priority, falls under effort scarcity, strong bottom-up beats weak bias,
  no goal -> no override)
scripts/smoke_substrate_motivation_golden_path.py: GOLDEN_PATH_OK
```

## Evals run

`orion/substrate/attention/evals/run_topdown_eval.py` (above) — the only eval harness directly exercising code this PR changed. `orion/autonomy/evals/` has no remaining eval harness for the deleted engine (its own eval, `run_homeostatic_drives_eval.py`, was deleted alongside the engine it measured).

## Docker/build/smoke checks

Not run — this environment has no Docker access. `scripts/check_service_env_compose_parity.py` was run per-service during the sweep (Wave 2d): confirmed `DRIVES_AUDIT_CHANNEL`/`DRIVE_STATE_SUBSTRATE_MATERIALIZATION_ENABLED` removed from both `.env_example` and `docker-compose.yml` together for `orion-substrate-runtime`; found 16 pre-existing unrelated missing keys in that same compose file, out of scope, not touched.

## Review findings fixed

- Finding: Wave 2a's `AutonomyStateV1` field removal left `orion/substrate/attention/detectors/autonomy.py`'s `AutonomySignalDetector` with all three input sources permanently empty — a live, wired attention-pipeline component that would always return `[]`.
  - Fix: retired the detector entirely (deleted the file, removed from `default_attention_detectors()`), updated the one test whose assertion depended on the resulting salience boost.
  - Evidence: `orion/substrate/tests` (93 passed) + `services/orion-cortex-exec/tests/test_attention_frame.py` (8 passed) after the fix.
- Finding (from a holistic review over the full combined diff, run by a separate subagent): `orion/substrate/relational/adapters/autonomy_ctx.py`'s `_map_autonomy_state_to_nodes` still built `StateSnapshotNodeV1`/`DriveNodeV1`/`TensionNodeV1` from fields that no longer exist on `AutonomyStateV1` — permanently dead branches, not previously caught since no wave's scope covered this file.
  - Fix: stripped the dead branches, kept `GoalNodeV1` (still real). Confirmed `DriveNodeV1`/`TensionNodeV1` themselves stay registered — `recall.py`/`spark.py` still produce them from unrelated real sources.
  - Evidence: `orion/substrate/relational/tests/test_adapters.py` (38 passed).
- Finding: 6 real test failures in `tests/test_autonomy_summary.py`, `test_autonomy_summary_degraded.py`, `test_autonomy_repository.py` — three root-level files no wave's scope had covered, either constructing `AutonomyStateV1` with now-removed fields (`ValidationError`) or asserting on drive-competition analysis `summarize_autonomy_state()` no longer computes.
  - Fix: two tests exercising fully-retired functionality (drive-pressure competition, dominant-drive echoing) deleted outright with an explanatory comment rather than patched to pass vacuously; the rest trimmed to their still-real assertions (proposal-headline stripping, identity/goal-id mapping, degraded-state stance handling).
  - Evidence: `tests/test_autonomy_summary.py tests/test_autonomy_summary_degraded.py tests/test_autonomy_repository.py` — 10 passed.
- Finding (process, not code): a bisection debugging step earlier in this session (checking out `d6a4e892b` to test Wave 1's isolated behavior) left this worktree's disk content stuck at Wave-1-only while `HEAD` still pointed past Wave 2a-d — caught by the same holistic review pass before it could be committed or handed off.
  - Fix: restored via `git checkout HEAD -- .` plus removing 8 stray resurrected files not present at `HEAD`.
  - Evidence: `git diff --quiet HEAD -- . ':!graphify-out'` confirmed clean.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-spark-concept-induction/.env \
  -f services/orion-spark-concept-induction/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-substrate-runtime/.env \
  -f services/orion-substrate-runtime/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-world-pulse/.env \
  -f services/orion-world-pulse/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-orch/.env \
  -f services/orion-cortex-orch/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-sql-writer/.env \
  -f services/orion-sql-writer/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-thought/.env \
  -f services/orion-thought/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build
```

Run via `scripts/safe_docker_build.sh <service>` per CLAUDE.md, not raw `docker compose`, from a worktree (not the shared checkout).

## Risks / concerns

- Severity: Medium — Concern: Orion loses live goal-proposal capability entirely (no new `GoalProposalV1` rows will ever be published again). Mitigation: explicitly confirmed acceptable with Juniper before implementation; a field-native replacement is out of scope here, tracked as future work per `orion/sentience_striving_program/README.md` §8/§9.
- Severity: Low — Concern: `journal.compose.episode` capability rule now has no goal-provenance condition at all (only `requires_goal_status`), a real narrowing of live gating semantics. Mitigation: explicitly documented in `config/autonomy/capability_policy.v1.yaml`'s own comment, not silently absorbed; flagged here for visibility.
- Severity: Low — Concern: live host `.env` (main checkout) still has the now-removed env keys present. Mitigation: harmless (settings use `extra="ignore"`), but needs a manual operator pass to clean up — the sync script only adds/updates keys, never removes them.
- Severity: Low — Concern: Docker/build smoke checks were not run (no Docker access in this environment). Mitigation: restart commands listed above; `check_service_env_compose_parity.py` was run per-service during the sweep as a partial substitute.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1486
