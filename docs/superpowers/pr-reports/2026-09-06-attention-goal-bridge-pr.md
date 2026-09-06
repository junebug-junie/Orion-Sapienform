# PR report: the one bridge -- the goal producer reads the substrate competition

**Date:** 2026-09-06
**Branch:** `feat/attention-goal-bridge`
**Design:** `docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md`, "The read side" / "The one bridge (shipped 2026-09-06)"
**Follows:** PR #2124 (attention schema surface, merged + deployed)

## Summary

- Orion has two attention systems that both key on the same `node:substrate.*` ids and never read each other: the goal producer (`orion-attention-runtime`) picks what Orion should aim at, and the substrate workspace competition (`orion-substrate-runtime`) picks what actually wins. The substrate honours a goal only when its target is exactly one of the competing loops' source refs, so a goal about a node that is not competing cannot be acted on. That was 37% of all self-model ticks in the 24h before this shipped.
- This patch is the design doc's "one bridge": before choosing a target, the goal producer reads which node ids are currently competing and prefers, among its qualified candidates, the highest-salience one the competition can see. Nothing else: no reconciler, no router, no shared taxonomy.
- It never fires less than before: no candidate competing, or a missing/stale projection, falls back to the old top-1. `ORION_GOAL_PROVENANCE_READS_COMPETITION=false` is the kill switch and restores the pre-bridge behaviour exactly.
- Each emitted goal carries a typed receipt (`competition_read`, `competing_refs`) of what the producer saw.
- Deployed to `orion-attention-runtime` at 10:25Z from this worktree; live within a minute (see below). The falsifiable number is the self-model's `goal_matched_no_loop` share, banked before deploy.

## Outcome moved

The design doc's acceptance number for this bridge: `goal_matched_no_loop` falls and `top_down_override` rises, off instruments that were already running with a control window banked.

Control, 24h before deploy (self-model rows, n=2734): `goal_matched_no_loop` 36.9%, `goal_target_already_winning` 29.2%, `no_open_loops` 21.0%, `top_down_override` 12.9%.

Post-deploy measurement: see "Measurement" below.

## Current architecture

- `orion-attention-runtime` `_maybe_build_goal`: `top_node_substrate_target(frame)` = highest-salience qualified `node:substrate.*` field target; a dominance streak of 3 real ticks gates emission; publishes `FieldGoalProvenanceV1` on `orion:memory:goals:proposed`.
- `orion-substrate-runtime` consumes that into `GoalContext`; `top_down.py::relevance(goal, loop)` is an exact match of `goal.target_id` against a competing loop's `source_refs`; the winner is recorded per tick in `substrate_attention_self_model` with `voluntary_override_absent_reason`.
- The competition itself is readable: `substrate_attention_broadcast_projection` (singleton, ~30s) carries the frame's open loops and their `source_refs`.

## Architecture touched

- `orion/attention/field_attention/goal_provenance.py`: `top_node_substrate_target(frame, competing=None)` -- restrict to in-competition candidates when any qualify; else unchanged.
- `services/orion-attention-runtime/app/store.py`: `load_competing_loop_refs(max_age_sec)` -- one read of the projection's loop refs; `None` on absent/stale (unknown is not empty).
- `services/orion-attention-runtime/app/worker.py`: `_load_competition()` (kill switch, fail-open) feeding the selector; typed receipt on the goal.
- `orion/schemas/field_goal.py`: `competition_read`, `competing_refs` (additive, defaults). Channel note on `orion:memory:goals:proposed`.
- Settings + `.env_example`: `ORION_GOAL_PROVENANCE_READS_COMPETITION=true`, `ORION_GOAL_PROVENANCE_COMPETITION_MAX_AGE_SEC=120.0`.

## Files changed

- `orion/attention/field_attention/goal_provenance.py`: competition-aware selection.
- `orion/schemas/field_goal.py`: two typed receipt fields.
- `orion/bus/channels.yaml`: description note on the goal channel (additive contract change).
- `services/orion-attention-runtime/app/{worker,store,settings}.py`, `.env_example`, `README.md`.
- `tests/test_attention_field_goal_provenance.py`, `services/orion-attention-runtime/tests/test_goal_provenance_producer.py`: selector, worker, store-reader, kill-switch, fail-open.
- `docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md`: "The one bridge" section with the banked control window.
- `config/metrics/metric_definitions.lock.json`: re-locked (branch-cut drift, no definition change).

## Schema / bus / API changes

- Added: `FieldGoalProvenanceV1.competition_read: Literal[in_competition|not_in_competition|unavailable] | None`, `competing_refs: list[str]` (max 16). Additive with defaults; the only consumer (`orion-substrate-runtime`'s goal_context_listener) validates older and newer payloads alike.
- Removed / Renamed: none.
- Behavior changed: which target a goal names when a qualified candidate is competing. Nothing downstream of the goal changed.

## Env/config changes

- Added keys: `ORION_GOAL_PROVENANCE_READS_COMPETITION`, `ORION_GOAL_PROVENANCE_COMPETITION_MAX_AGE_SEC` (orion-attention-runtime).
- `.env_example` updated: yes. Local `.env` synced: `sync_local_env_from_example.py orion-attention-runtime --all-keys` -> both keys added to the primary checkout; copied into the worktree for the deploy.
- Skipped keys: none.

## Tests run

```text
tests/test_attention_field_goal_provenance.py + services/orion-attention-runtime/tests   57 passed
  (2 failures in the same run are pre-existing on main and unrelated: test_autonomy_goals_bus_catalog
   schema registry, test_channel_prefix_guardrail)
check_env_key_single_source OK, check_inner_state_registry OK, check_definition_drift --gate PASS,
check_service_env_compose_parity orion-attention-runtime N/A (env_file), git diff --check OK
```

## Evals run

```text
No eval harness for this seam. The eval is the live measurement below, against the instrument
the design doc pre-registered (`voluntary_override_absent_reason` shares, matched windows).
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-attention-runtime up -d --build   10:25:20Z -> 10:25:24Z
code-in-container: worker.py has _load_competition (grep 2)
first goals after restart: field_goal_provenance_published target=node:substrate.execution streak=3, 4 ...
projection freshness at that moment: 1.3s
75s bus sample of orion:memory:goals:proposed: 36 goals; competition_read in_competition 18 /
  not_in_competition 18 (those 18 saw competing_refs=[] -- the competition held no loops that tick);
  every goal targeted node:substrate.execution, which the competition was holding.
```

## Measurement

(filled after the 40-minute window)

## Review findings fixed

(filled after review)

## Restart required

Already restarted from this worktree:

```bash
scripts/safe_docker_build.sh orion-attention-runtime up -d --build
```

## Risks / concerns

- Severity: medium
  Concern: part of the 37% is structural, not a selection problem. The goal producer can only target five domains (`PREDICTION_ERROR_NATIVE_TARGETS`: biometrics, execution, chat, route, bus_synaptic) while the competition also holds `node:substrate.codebase` and others. The bridge closes only the slice where a targetable domain *was* competing and the producer picked a different one.
  Mitigation: measure first; widening the producer's target set is a separate decision with its own gate.
- Severity: low
  Concern: a fresh projection with zero loops reads as "empty competition" (fallback + `not_in_competition`), which is correct but indistinguishable in the receipt from "no qualified candidate competing".
  Mitigation: `competing_refs` on the goal disambiguates (empty list vs non-empty), and the self-model's `no_open_loops` is the downstream truth.
- Severity: low
  Concern: preferring an in-competition candidate can change the winner mid-streak and reset the dominance streak, delaying emission by up to 3 ticks (~6s).
  Mitigation: acceptable and bounded; visible in `orion:debug:attention:streak_tick`.

## PR link

(filled on open)
