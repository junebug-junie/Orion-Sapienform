# PR report: the one bridge -- the goal producer reads the substrate competition

**Date:** 2026-09-06
**Branch:** `feat/attention-goal-bridge`
**Design:** `docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md`, "The read side" / "The one bridge (shipped 2026-09-06)"
**Follows:** PR #2124 (attention schema surface, merged + deployed)

## Summary

- Orion has two attention systems that both key on the same `node:substrate.*` ids and never read each other: the goal producer (`orion-attention-runtime`) picks what Orion should aim at, and the substrate workspace competition (`orion-substrate-runtime`) picks what actually wins. The substrate honours a goal only when its target is exactly one of the competing loops' source refs, so a goal about a node that is not competing cannot be acted on. That was 37% of all self-model ticks in the 24h before this shipped.
- This patch is the design doc's "one bridge": before choosing a target, the goal producer reads which node ids are currently competing and prefers, among its qualified candidates, the highest-salience one the competition can see. Nothing else: no reconciler, no router, no shared taxonomy.
- It never fires less than before: no candidate competing, or a missing/stale projection, falls back to the old top-1. `ORION_GOAL_PROVENANCE_READS_COMPETITION=false` is the kill switch and restores the pre-bridge behaviour exactly.
- The producer logs a receipt of what it saw on every emission (`competition_read=in_competition|not_in_competition|unavailable` on its `field_goal_provenance_published` line). Deliberately not a schema field: see Review round 1.
- Deployed to `orion-attention-runtime` at 10:25Z from this worktree; a review finding (strict consumer schemas) made that first deploy drop goals for nine minutes; hot-fix live 10:34:32Z. The falsifiable number is the self-model's `goal_matched_no_loop` share, banked before deploy.

## Outcome moved

The design doc's acceptance number for this bridge: `goal_matched_no_loop` falls and `top_down_override` rises, off instruments that were already running with a control window banked.

Control, 24h before deploy (self-model rows, n=2734): `goal_matched_no_loop` 36.9%, `goal_target_already_winning` 29.2%, `no_open_loops` 21.0%, `top_down_override` 12.9%.

Post-deploy measurement: see "Measurement" below.

## Current architecture

- `orion-attention-runtime` `_maybe_build_goal`: `top_node_substrate_target(frame)` = highest-salience qualified `node:substrate.*` field target; a dominance streak of 3 real ticks gates emission; publishes `FieldGoalProvenanceV1` on `orion:memory:goals:proposed`.
- `orion-substrate-runtime` consumes that into `GoalContext`; `top_down.py::relevance(goal, loop)` is an exact match of `goal.target_id` against a competing loop's `source_refs`; the winner is recorded per tick in `substrate_attention_self_model` with `voluntary_override_absent_reason`.
- The competition itself is readable: `substrate_attention_broadcast_projection` (singleton, ~30s) carries the frame's open loops and their `source_refs`.

## Architecture touched

- `orion/attention/field_attention/goal_provenance.py`: `top_node_substrate_target(frame, competing=None, current=None)` -- restrict to in-competition candidates when any qualify; on an unknown/empty read keep the current streak target (hysteresis); else the raw top-1.
- `services/orion-attention-runtime/app/store.py`: `load_competing_loop_refs(max_age_sec)` -- one read of the projection's loop `node:` refs; `None` on absent/stale (unknown is not empty).
- `services/orion-attention-runtime/app/worker.py`: `_load_competition()` (kill switch, fail-open) feeding the selector; receipt on the log line.
- `orion/schemas/field_goal.py`: comments only (the target is no longer always the raw field top-1). No field changes.
- `services/orion-attention-runtime/docker-compose.yml`: forwards the two new keys plus two pre-existing streak-telemetry keys that were never reaching the container.
- Settings + `.env_example`: `ORION_GOAL_PROVENANCE_READS_COMPETITION=true`, `ORION_GOAL_PROVENANCE_COMPETITION_MAX_AGE_SEC=120.0`.

## Files changed

- `orion/attention/field_attention/goal_provenance.py`: competition-aware selection.
- `orion/schemas/field_goal.py`: comment corrections only.
- `services/orion-attention-runtime/docker-compose.yml`: env passthrough.
- `services/orion-attention-runtime/app/{worker,store,settings}.py`, `.env_example`, `README.md`.
- `tests/test_attention_field_goal_provenance.py`, `services/orion-attention-runtime/tests/test_goal_provenance_producer.py`: selector, worker, store-reader, kill-switch, fail-open.
- `docs/superpowers/specs/2026-09-04-attention-schema-surface-design.md`: "The one bridge" section with the banked control window.
- `config/metrics/metric_definitions.lock.json`: re-locked (branch-cut drift, no definition change).

## Schema / bus / API changes

- Added: none. (Two receipt fields were added in the first commit and removed in the hot-fix: `FieldGoalProvenanceV1` is `extra="forbid"` on three consumers, so any new field is a consumer-first migration, not an additive change -- see Review round 1.)
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
check_service_env_compose_parity orion-attention-runtime OK (27/27 keys exposed), git diff --check OK
after hot-fix: 60 passed (same two pre-existing failures)
```

## Evals run

```text
No eval harness for this seam. The eval is the live measurement below, against the instrument
the design doc pre-registered (`voluntary_override_absent_reason` shares, matched windows).
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-attention-runtime up -d --build   10:25:20Z -> 10:25:24Z  (first deploy)
                                                                     10:34:28Z -> 10:34:32Z  (hot-fix, review round 1)
                                                                     15:14:03Z -> 15:14:07Z  (review round 2)
after round 2: 18/18 receipts in 40s read in_competition with competing=node:substrate.execution; 0 tracebacks; 0 rejected goals.
code-in-container: worker.py has _load_competition (grep 2)
first goals after restart: field_goal_provenance_published target=node:substrate.execution streak=3, 4 ...
projection freshness at that moment: 1.3s
75s bus sample of orion:memory:goals:proposed: 36 goals; competition_read in_competition 18 /
  not_in_competition 18 (those 18 saw competing_refs=[] -- the competition held no loops that tick);
  every goal targeted node:substrate.execution, which the competition was holding.
```

## Measurement

Instrument: the substrate self-model's `voluntary_override_absent_reason` (plus `attention_reason`
= `top_down_override` when one fired), one row per ~30s tick. Control = the same number of ticks
immediately before the first bridge deploy (10:25:24Z). Treatment = ticks after the hot-fix
(10:34:32Z), so the nine minutes of rejected goals are excluded from both arms.

**4.7 hours, n = 530 ticks per arm (10:34:32Z -> 15:14:03Z vs the 530 ticks before 10:25:24Z):**

| tick outcome | control | bridge | change |
|---|---|---|---|
| `goal_matched_no_loop` (goal about nothing competing) | 42.1% | **14.3%** | -66% relative |
| `top_down_override` (goal flipped the winner) | 4.2% | **10.2%** | 2.4x |
| `goal_target_already_winning` | 20.8% | 50.2% | the expected conversion |
| `no_open_loops` (competition empty; not the bridge's) | 33.0% | 22.8% | drifted on its own |
| `bias_did_not_flip_winner` (goal pushed a competing loop and lost) | 0.0% | 2.5% | new, and real |

The pre-registered number moved in the predicted direction, and the second effect moved too. The
attention-surface table (`substrate_attention_schema`, `process='substrate_attention'`, PR #2124)
reports the identical counts for the same windows through its own bus -> sql-writer path.

The first 40-minute window (n = 78 per arm) showed the override rise (5.1% -> 41.0%) but not the
`goal_matched_no_loop` fall; it is superseded by the 4.7h read above and kept only as a reminder that
78 ticks is too few to call.

Honest reading: most of what the bridge does is convert "goal about nothing competing" into "goal
names the loop that is already winning" -- that is the reviewer's altitude note, confirmed. The
overrides are the minority case where the competing target the goal names is *not* the bottom-up
winner and the bias flips it. `bias_did_not_flip_winner` appearing for the first time is the same
mechanism losing, which is what a real competition looks like.

Confounds not controlled: time of day (the control arm is the preceding morning), and the
producer's own hysteresis changing which target it names. Re-run the same query after 24h before
treating the shares as settled.

```sql
-- the bridge's number, any window
SELECT coalesce(self_model_json->>'voluntary_override_absent_reason', self_model_json->>'attention_reason') r,
       count(*), round(100.0*count(*)/sum(count(*)) over (),1)
FROM substrate_attention_self_model WHERE created_at > '2026-09-06 10:34:32+00' GROUP BY 1 ORDER BY 2 DESC;
```

## Review findings fixed

Round 1 (six findings, all acted on; one was a live incident):

- Finding: the two typed receipt fields I added to `FieldGoalProvenanceV1` are rejected by all three consumers, which validate with `extra="forbid"` -- a producer-first deploy drops every goal until the consumers are rebuilt. **This happened live**: 186 goals rejected by `orion-substrate-runtime` (`goal_context invalid payload`) between 10:25Z and 10:34Z.
  - Fix: fields removed; the schema is byte-for-byte what the consumers expect. The receipt (`competition_read=`) is now on the producer's own `field_goal_provenance_published` log line, which is inspectable and needs no contract change. Hot-fix redeployed 10:34:32Z; rejected goals since: 0.
  - Evidence: substrate log count 0 after redeploy; the contaminated 10:25-10:34 window was discarded and measurement restarted from 10:34:32Z.
- Finding: the competition read changes every ~30s and is None on any error, while the frame ticks every ~2s; alternating winners would reset the 3-tick dominance streak and the producer could fire *less* than before.
  - Fix: hysteresis in `top_node_substrate_target(..., current=)` -- an unknown/empty read keeps the current streak target while it is still qualified; only a competing read naming a different qualified target moves it.
  - Evidence: `test_bridge_unknown_reads_do_not_flap_the_streak` (reads: competing, None, empty, error, competing -> one target, streak 5, three goals) and `test_bridge_hysteresis_keeps_the_current_target_on_unknown_or_empty_reads`.
- Finding: the two new env keys were in `.env_example`/settings but not in this service's explicit compose `environment:` list (no `env_file`), so the kill switch never reached the container. The parity gate had in fact printed this and I read only its last line.
  - Fix: both keys forwarded, plus two pre-existing streak-telemetry keys the same gate had been flagging. `check_service_env_compose_parity orion-attention-runtime`: OK, 27/27.
  - Evidence: `docker exec ... env | grep READS_COMPETITION` = 1 in the redeployed container; `test_bridge_compose_forwards_the_kill_switch`.
- Finding: `source_refs` also carry up to 20 redis stream ids per loop; a sorted, truncated receipt would have been all stream ids.
  - Fix: the store reader keeps only `node:` ids (the only refs a goal target can ever match). `test_bridge_store_reader_keeps_only_node_ids`.
- Finding: the goal's `salience_score`/`priority` are now the *chosen* target's own salience, which may be lower than the field's raw top-1, and two schema comments still said "the real field winner".
  - Fix: comments corrected on `FieldGoalProvenanceV1.field_target_id` and `DominanceStreakTickV1.target_id`; the README paragraph and design section say the target is competition-aware.
- Finding: no test could observe a streak reset caused by the bridge.
  - Fix: the flap test above drives a constant frame through five alternating reads with `min_streak=3`.

Reviewer's altitude note, agreed and recorded under Risks: the overlap is real (1,839 of 2,735 recent broadcasts held a targetable id, so the bridge is not a no-op), but its main effect may convert `goal_matched_no_loop` into `goal_target_already_winning` rather than into overrides. The pre-registered number is `goal_matched_no_loop` falling; `voluntary_override` rising is the hoped-for second effect, not the gate.

Round 2 (simplification + efficiency angles completed; the correctness angles were killed by the API
rate limit twice, so those I re-read myself against the round-1 list):

- Finding: the receipt was hidden instance state (`_last_competition_read`) read back through a `getattr` default, and the README claimed it lived on `provenance.competition_read`, which an `extra="forbid"` model cannot hold.
  - Fix: computed and logged where the goal is built (`field_goal_provenance_competition_read ... competition_read=... competing=...`); README says log line. Tests read it via caplog.
- Finding: the reader fetched and decoded the full 3.5-8.7KB projection every ~2s tick for a ~55-byte answer, checked staleness client-side after the transfer, and had dead defensive branches that would turn a schema drift into a silent "nothing competing".
  - Fix: node refs are extracted in SQL, the age bound is a `WHERE` on the DB clock, and a row whose `frame.open_loops` is not an array reads as unknown (`None`), never empty. Verified live: `array | {node:substrate.execution} | age 1.7s`.
- Finding: the read ran even when 0 or 1 candidates qualified, where no competition set can change the answer.
  - Fix: `qualified_node_targets()` exposed; the read is skipped below two. `test_bridge_skips_the_read_when_it_cannot_matter`.
- Finding: three-tier precedence written as nested early returns; a 24-line comment restating the design doc.
  - Fix: `pool = seen or held or candidates`; two-sentence comment with a pointer.
- Finding: duplicated hand-rolled SQLAlchemy fakes and a mis-named test helper; a per-key compose test re-implementing the parity gate.
  - Fix: one `_store_with(row)` fake, `_bridge_worker`/`_emit_twice`, and the compose test now runs `scripts/check_service_env_compose_parity.py` itself.
- Not taken: a `generated_at` memo to skip the ~2s read when the ~30s projection has not moved. With the SQL-side extraction the read is ~55 bytes plus a pool checkout; adding cache state for that was judged not worth it. Recorded here so it is not re-derived.

Scope note the reviewer flagged: `docker-compose.yml` also forwards `ORION_GOAL_PROVENANCE_STREAK_TICK_TELEMETRY_ENABLED` and `CHANNEL_GOAL_PROVENANCE_STREAK_TICK`, which predate this branch and were never reaching the container -- the parity gate had been reporting them. Deliberate, small, and it makes that gate green for this service.

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

https://github.com/junebug-junie/Orion-Sapienform/pull/2126
