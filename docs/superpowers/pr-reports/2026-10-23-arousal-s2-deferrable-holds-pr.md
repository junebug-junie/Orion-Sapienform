# fix(regulation): strain no longer latches on Orion's own queued self-work

## Summary

- Orion's GPU-queue strain input (arousal S2) now leaves out Orion's own deferrable work. That work is queued background holds with no parent hold: readings, curiosity compaction. Waiting self-work is hunger, not strain (body-economy design, "S2 decision", 10-23).
- The rule uses lease fields only, never holder names: `status=queued`, `priority='background'`, `kind='hold'`, `hold_lease_id IS NULL`. Requests at every priority still count. Background *requests* also still count.
- It follows the 10-20 heat-shed pattern. A pure helper (`orion/gpu_pool/shed.py:deferrable_queued`) counts the leases. sql-writer saves the count in a new column, `gpu_pool_state_history.deferrable_depth`. The durable-runs regulate node reads the column. `gpu_queue_evidence` subtracts the count. A lease the heat shed already holds is never counted twice.
- The reading says how much self-work is waiting: reason `gpu_queue_self_deferrable:<n>`, plus `ArousalInputsV1.gpu_queue_self_deferrable`.
- The replay script can rebuild the count for older rows from the pool's own hold events (`--reconstruct-deferrable`). It can also turn the new rule off for a before/after comparison (`--no-gpu-deferrable`).
- This is a stopgap. S2 is retired when Patch D moves the holds onto `bid()`. The design doc records this decision as question 11 and adds a row to the Retirements table.

## Outcome moved

Live replay of the last 24 h (10-22 16:40 to 10-23 16:40 UTC), stepping the shipped reducer at the regulate node's cadence:

| rule | strained | idle | strained spans |
|---|---|---|---|
| old (`--no-gpu-deferrable`) | 1410 min | 30 min | 16:40 to 11:24 (heat, then queue); 11:32 to 12:50; 13:00 to 13:46; 13:58 to the end |
| new (`--reconstruct-deferrable`) | 616 min | 824 min | 16:40 to 19:08 (heat); 19:18 to 02:20 (heat, then a short real queue at 02:00); 03:46 to 04:02; 10:12 to 10:26; 11:54 to 12:10. Idle since 12:10. |

The old-rule replay matches the live `arousal_transition` rows: strained all day, with idle only at 11:24 and 13:48. That match is evidence that the replay is faithful. Under the new rule, strain clears 10 min after the heat clears, at 02:20. After that, Orion is strained only for short spans where 2 or more *non*-self-work leases waited for 5 min or longer: system holds, metacog requests.

The new numbers are a proxy. Older rows have no `deferrable_depth`, so the count is rebuilt from the pool's `admitted`/`queued` -> `granted`/`backlogged`/... events for background holds. Exact counts start when sql-writer writes the column.

## Current architecture

S2 = sum of `gpu_pool_state_history.queue_depth` minus the heat-shed-held leases (`shed_held_depth`, 10-20) at or above 2 for 300 s -> `gpu_queue_sustained` -> strained. Strain latches until every strain input reads clear for 600 s. Live 10-23: after the heat cleared, the agent lane held 8 to 14 queued background holds that never drained. S2 stayed true, and curiosity, outreach, dream sleep, self-naming and the retro goal loop stayed held.

## Architecture touched

- `orion/gpu_pool/shed.py`: new pure helpers `is_deferrable_self_work`, `deferrable_queued`.
- orion-sql-writer: one new nullable json column on an existing table, written on the existing `gpu_pool.state.v1` history path. No new channel, route or env key.
- orion-durable-runs: the regulate node's S2 SELECT reads the new column.
- `orion/regulation/arousal.py`: the load arithmetic, and one annotation reason.
- `orion/schemas/regulation.py`: an additive optional field on `ArousalInputsV1`. Its only writer and reader is durable-runs (its own checkpoint; `orion/schema_skew_discovery.py` declares it).

## Files changed

- `orion/gpu_pool/shed.py`: `is_deferrable_self_work`, `deferrable_queued` (skips leases `shed_held_queue` counts).
- `orion/regulation/arousal.py`: `gpu_queue_evidence` reads an optional 5th element and returns a 6-tuple `(..., deferrable)`. In `classify_arousal`, deferrable work is left out of the queue before the shed check, so a queue of only self-work reads fresh and clear (not stale). The fault check still uses the raw depth: a lane-controller fault with only self-work queued still voids S2, because that work may be waiting *because of* the fault (review finding). New annotation `gpu_queue_self_deferrable:<n>`, appended last.
- `orion/schemas/regulation.py`: `ArousalInputsV1.gpu_queue_self_deferrable`.
- `services/orion-sql-writer/app/models/gpu_pool_state.py`: `deferrable_depth` column and value. None (unknown) when the snapshot has a queue but no lease rows. `{}` when the queue is empty.
- `services/orion-sql-db/manual_migration_regulation_gpu_deferrable_depth_v1.sql`: `ADD COLUMN IF NOT EXISTS`, `REQUIRED-BY: orion-sql-writer, orion-durable-runs`.
- `services/orion-durable-runs/app/regulation_store.py`: selects and passes the column.
- `scripts/analysis/measure_arousal_reducer_replay.py`: column export, event-based reconstruction, flags.
- Tests: `tests/test_regulation_arousal.py`, `orion/gpu_pool/tests/test_scheduler_shed.py`, `services/orion-sql-writer/tests/test_gpu_pool_state_history.py`, `services/orion-durable-runs/tests/test_temporal_self_regulate.py`, `scripts/analysis/tests/test_measure_arousal_reducer_replay.py`.
- `docs/superpowers/specs/2026-10-23-organism-body-economy-design.md`: the S2 decision (question 11) and a retirement row.

## Schema / bus / API changes

- Added: column `gpu_pool_state_history.deferrable_depth json NULL`; field `ArousalInputsV1.gpu_queue_self_deferrable: Optional[int]`; reading reason `gpu_queue_self_deferrable:<n>` (annotation only, never first).
- Removed / renamed: none.
- Behavior changed: S2 load = queue - shed-held - deferrable, never below 0. A queue made only of self-work reads clear, not stale, unless a lane-controller fault is open (then S2 is voided as before). `ArousalReadingV1.gpu_queue_depth` still reports the full queue; a display showing "queue 8" next to "idle" is explained only by the `gpu_queue_self_deferrable:8` reason.
- Compatibility: `ArousalReadingV1` and `RegulationStateV1` (read by Hub, dream, thought, substrate-runtime) have **no** new field. Only `reasons` gains a string. `arousal_view` reads `reasons[0]`, and the annotation is always last. Consumers of `STRAIN_CAUSES` (strain arc in `temporal_self_graph`, `temporal_self/sources.py`, the stance cue's `STRAIN_CAUSE_WORDS`, `strain_news` / `node:substrate.strain`, `replay_strain_news.py`) all filter `reasons` to the cause set. `STRAIN_CAUSES` is unchanged. Old rows (NULL column, 4-tuples) read as 0 deferrable.

## Env/config changes

- Added / removed / renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced: not needed (no template changed).
- Skipped keys: none. The sql-writer `.env` does not need a hand edit: no new channel or route.

## Tests run

```text
pytest tests/test_regulation_arousal.py tests/test_regulation_immune.py scripts/analysis/tests/test_measure_arousal_reducer_replay.py \
  tests/test_regulation_strain_news.py scripts/analysis/tests/test_replay_strain_news.py tests/test_regulation_arousal_read.py \
  tests/test_regulation_arousal_view.py tests/scripts/test_schema_skew_discovery.py tests/test_sql_migration_drift_gate.py \
  tests/test_deploy_migration_gate.py tests/test_post_pull_apply_migrations.py orion/physiology/tests orion/temporal_self/tests orion/gpu_pool/tests
  -> 1231 passed
services/orion-sql-writer: tests/test_gpu_pool_state_history.py -> 10 passed, 1 skipped (Postgres-only)
services/orion-durable-runs: tests/test_temporal_self_regulate.py -> 36 passed, 3 skipped (Postgres-only); test_physiology_node + chronicle_postgres -> 10 passed, 14 skipped
services/orion-substrate-runtime: test_strain_news_tick.py test_cabinet_heat_tick.py -> 13 passed
services/orion-hub: test_regulation_arousal_readers.py -> 57 passed
services/orion-dream: test_dream_reads_arousal.py -> 20 passed
scripts/check_migration_required_by.py --base origin/main -> OK
make check-body-economy-contract -> OK (66 state-reading modules, all registered; no new reader module)
git diff --check -> clean
```

The full sql-writer suite has 14 failures. They also fail on clean `main` (15 there): they depend on the local environment and on test order, and none is in a touched file.

New tests: only-background-holds -> no strain; system requests at the floor for 300 s -> strain; a mix; shed-held plus deferrable overlap counted once (helper and reducer); the live 10-23 regression (`{"agent": 8}`, all background holds -> strain clears 600 s after the heat); a fault with only self-work queued; back-compat for None/malformed values; the sql-writer None-vs-{} rule; the store reads the column; replay reconstruction.

## Evals run

```text
scripts/analysis/measure_arousal_reducer_replay.py over live 24 h (read-only export), old vs new rule: see "Outcome moved".
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-sql-writer build    -> OK (image built; in-image check: `GpuPoolStateSQL.__table__.c` has `deferrable_depth` -> True)
scripts/safe_docker_build.sh orion-durable-runs build  -> OK (image built; in-image check: `GPU_SNAPSHOTS_SQL` selects `deferrable_depth` -> True)
scripts/apply_post_pull_migrations.py --dry-run <migration> -> "NOT APPLIED ... would apply (single transaction, lock_timeout=10s)"
```

## Review findings fixed

Review by an orion-repo-agent subagent (read-only). It found no must-fix issues.

- Finding (should-fix): a fault with only self-work queued could let a latched strain clear during the fault. Those holds may be waiting because of the fault.
  - Fix: fault-voiding uses the raw queue depth again (`orion/regulation/arousal.py`), which is the 10-09 rule.
  - Evidence: `test_a_fault_still_voids_s2_when_only_self_work_is_queued` (fresh -> unknown; latched -> unknown, still latched).
- Finding (should-fix): the shed-held-then-deferrable ordering was not pinned by a test.
  - Fix: `test_shed_then_deferrable_ordering_is_pinned`. Held 2 + deferrable 1 of 3 -> `gpu_shed:*` (stale). Held 1 + deferrable 2 of 3 -> fresh and clear.
  - Evidence: passes.
- Finding (nit): a Juniper-path durable run that keeps the default `background` priority would read as self-work.
  - Fix: the `is_deferrable_self_work` docstring says a Juniper-facing hold must ask for `system` or `urgent`. Live check: every current top-level background hold is a durable-runs self-work holder.
  - Evidence: docstring.
- Finding (nit): the replay proxy under-counts in two cases (re-queue after `aborted` with no `queued` event; rows with anything shed-held). It also lacked a type guard on `shed_held_depth`.
  - Fix: both cases are documented in the PROXY docstring. Added an `isinstance(held, dict)` guard.
  - Evidence: replay tests pass, and the 24 h replay numbers are unchanged.
- Finding (nit): a rollback drops the checkpoint inputs once, and the reading's `gpu_queue_depth` still shows the full queue.
  - Fix: both are documented under Risks and Compatibility. No code change.

## Restart required

Order: migration first, then sql-writer, then durable-runs. After the merge, the post-pull rebuild on the primary `main` checkout does all three: it applies the additive migration before the rebuilds. To run it by hand, from the clean primary `main` checkout:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_regulation_gpu_deferrable_depth_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

## Proposal mode (§0A: regulation-loop change)

- Capability change: strain no longer latches on self-queued work. Waiting self-work is reported as `gpu_queue_self_deferrable:<n>` instead.
- Data touched: one new nullable column on `gpu_pool_state_history`, plus one optional field in the regulate node's own checkpoint.
- Privacy boundary: none crossed. The column holds counts per work class only, with no lease bodies or holder names.
- Trace that proves it worked: after deploy, `SELECT generated_at, queue_depth, deferrable_depth FROM gpu_pool_state_history ORDER BY generated_at DESC LIMIT 5` shows non-NULL values. `orion:regulation:latest` reasons carry `gpu_queue_self_deferrable:<n>`. `temporal_self_event` gets an `arousal_transition` out of `strained` when no other strain cause holds.
- Dangerous failure: a real jam made only of background holds no longer reads as strain. Mitigations: the count stays visible in the reading's reasons and in history. Lane prices take over in Patch B/D. The body organ's `lane:<role>` reading shows slot-seconds *held*, not queue wait, so it does **not** show a jam of waiting holds by itself (see Risks).
- Disable / roll back: revert the PR. Or ignore the column: NULL reads as 0 deferrable, which is the old behavior. The column can stay.

## Risks / concerns

- Severity: medium. Concern: no surface today alarms on a long jam made only of self-work holds. The brief named `lane:<role>` as the mitigation, but that reading measures slot-seconds held, not waiting. Mitigation: `gpu_queue_self_deferrable:<n>` is in every reading. Patch B/D's lane price is the real successor. Follow-up: the body organ could read `deferrable_depth` as hunger.
- Severity: low. Concern: a rollback of durable-runs reads a checkpoint holding the new `ArousalInputsV1` field once and drops it (`extra="forbid"` -> `_prev_inputs` returns None). Effect: the cabinet reflex hysteresis seed is lost for one step, and physiology reads heat as not-read for one tick. Mitigation: none needed; the same happened with prior additive fields.
- Severity: low. Concern: the 24 h replay is a proxy built from pool events. Mitigation: exact from deploy on; the old-rule replay matches the live transitions.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
