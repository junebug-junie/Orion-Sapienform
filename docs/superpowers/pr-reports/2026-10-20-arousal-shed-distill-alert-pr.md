# Heat-held GPU queue is not arousal load; memory_distill gets gpu1; painting-gap alert names the arousal deferral

## Summary

- Orion's arousal no longer counts GPU jobs that a heat-reflex shed is holding back as GPU load. The heat-reflex reasons are `cabinet_hot`, `cabinet_unknown` and `cooling_incident`. Orion's own `orion_self_shed` is not excluded. Those jobs are waiting on heat, which arousal already reads directly. A queue that only the shed lifts to the floor now reads as "no evidence" (`gpu_shed:<reason>`), the same way a controller fault already does. A shed that holds only part of the queue is annotated `gpu_shed_partial:<reason>:<n>`.
- sql-writer now records, for every saved GPU pool snapshot, how many queued jobs the shed is holding and why (two new nullable columns, one additive migration).
- The memory distiller now co-owns gpu1 (`config/gpu_pool.yaml`). Before this, it was a borrower there, so every queued background agent hold got gpu1 first, and a running distill run was recalled when one of them arrived.
- When the waking baseline painting is deferred because arousal reads strained, the scheduler now records it in the checkpoint (`deferral: {reason, since, last_at}`). The painting-gap alert reads that record and arousal's live reasons, and names this cause first.

## Outcome moved

- The heat no longer counts twice. Each hot minute strains arousal through `cabinet_hot` only. Shed time no longer adds to the 5-minute queue-sustain window, so when the heat lifts the queue must stay backed up for 5 minutes on its own before it strains arousal.
- Two system distill holds waited 10+ hours behind background reading holds. With this change they are served in priority order. Replayed through the shipped scheduler (tests below).
- A painting gap caused by arousal deferral no longer sends Juniper to debug the GPU lane controller.

## Metric quality gate (changed signal: arousal S2, the GPU queue depth)

1. **Provenance.** Queue depth: `services/orion-gpu-pool/app/runtime.py` `snapshot()` counts `status == "queued"` leases per work class. The shed-held count is `orion/gpu_pool/shed.py` `shed_held_queue()`. It applies the scheduler's own U4 test (`scheduler.schedule`: no `hold_lease_id`, priority in `shed.blocked`) to the same broadcast. `test_shed_held_queue_agrees_with_the_scheduler_on_queued_leases` checks that both agree. sql-writer stores the count (`state_history_row`), and the regulate node reads it (`regulation_store._gpu`).
2. **Independence.** Before this patch, S2 was not independent of S1 while the cabinet was hot. The cabinet reflex (S1) asserts the `cabinet_hot` shed, the shed holds every background and system lease, and those held leases are the queue. That is a short, direct chain, so the same heat was counted twice. After this patch, S2 counts only the leases the shed is not holding, which removes that chain.
3. **Theory anchor.** S2 is meant to read "GPU demand exceeds supply". A job the shed is holding is not waiting for a GPU. It is waiting for the heat to drop, and S1 already measures heat. This is the same argument the existing `gpu_fault` exclusion uses.
4. **Live data.** Live broadcast at 2026-10-20 18:13Z: queue `{"memory_distill": 2, "agent": 9}`, shed `blocked = {background, system} -> cabinet_hot`. All 11 queued leases were held by the shed. Replay of the shipped reducer, 10-18 06:00 to 10-20 18:00 (`scripts/analysis/measure_arousal_reducer_replay.py --reconstruct-shed`; older snapshots have no shed columns, so a minute counts as fully shed-held when the pool logged a `shed:*` refusal in that minute):

   | | 10-18 | 10-19 | 10-20 (to 18:00) |
   |---|---|---|---|
   | before (`--no-gpu-shed`), strained min | 1006 | 1440 | 1080 |
   | after (`--reconstruct-shed`), strained / unknown min | 1004 / 2 | 1428 / 12 | 1074 / 6 |

   **The gate check "can the corrected signal read calm in today's cool windows" FAILS.** Of 1,659 cool minutes (cabinet reflex not hot) from 10-18 12:00 onward, arousal read strained in 1,632, unknown in 21 and idle in 6. The raw queue fell below the floor in only 1 of them. The remaining queue is real. On gpu1, background `agent` reading and compaction holds (one hold at a time, about 10 to 20 minutes each) arrive faster than they are served, and memory_distill was stuck behind them (fix 2). The signal can still return to calm: it read idle on 10-17 and 10-18 when the queue drained. **So this fix removes the double count, but it will not by itself end today's strained reading.** The residual is the background-hold backlog. See Risks.
5. **Existing mechanism.** The shed lever and the `gpu_fault` exclusion already existed. This patch reuses the scheduler's U4 rule and copies the fault exclusion's shape. It adds no new metric.
6. **Reversibility.** Low cost. The columns are nullable and additive. NULL means "nothing held", which is the old behaviour, so removing the reader restores today's S2 exactly.

## Fix 2 verification (memory_distill starvation)

- Live evidence: on 10-18 at 11:39:25 a distill hold was granted gpu1. At 11:39:55 it was recalled with `reason=owner_waiting`. That was the last distill grant. On 10-20, holds `9837acb7` (04:52, system) and `b7ab08ff` (06:05, system) were still queued at 18:10, while background `agent` holds got gpu1 in every cool window.
- Replayed before changing anything (`orion/gpu_pool/tests/test_memory_distill_priority.py`). The old config gives gpu1 to the background reading hold and never to the distill hold (10-20 state). It recalls the running distill hold `owner_waiting` (10-18 state, gpu2 seat not loaded). The new config grants the oldest distill hold first and recalls nothing.
- Blast radius: `memory_distill` becomes an owner of the `agent` role only. Its class still lists only `[agent]`, and the test asserts it owns no other role. As an owner, it now counts for rule 5 like any agent owner: it can recall a metacog or fast borrower on gpu1, and it can drain the experiment seat. Agent `system` one-off calls still win by age (queue order). A system distill hold's gaps are not open to agent system calls (H3 needs strictly higher priority). Those calls fall to agent-gpu2, agent-deep or the lent chat card. **Hub reading, curiosity and self-study calls on the `agent` route can now wait behind one distill run while gpu2 is not loaded.** Distill runs live are 27 to 97 s. This is pinned by `test_agent_system_call_waits_for_a_running_distill_hold_or_takes_its_gap`. The owner list is not part of `launch_digest`, so circe's lane controller needs no lockstep change (`check_gpu_pool_config.py` ok, all 455 `orion/gpu_pool/tests` pass).

## Current architecture

- S2 summed `GpuPoolStateV1.queue_depth`, so shed-held leases counted the same as leases waiting for a GPU.
- `gpu_pool_state_history` stored only depth maps, not which leases the shed was holding.
- gpu1's `agent` role had owner `agent` only. memory_distill's only role is gpu1, so it was a borrower there.
- The baseline checkpoint recorded no deferral reason. The gap alert listed only the GPU controller, thermal refusals and a wedged worker.

## Architecture touched

orion-gpu-pool (config only), orion-sql-writer (history row), orion-durable-runs (regulate node read), orion-proposal-runtime (via `orion/reverie/baseline.py`), orion-thought (gap alert). DB: `gpu_pool_state_history` +2 columns.

## Files changed

- `orion/regulation/arousal.py`: `gpu_queue_evidence` takes optional held/reason per snapshot, runs the sustain window on load depth and returns a 5-tuple. `classify_arousal` adds the `s2_shed` no-evidence branch.
- `orion/schemas/regulation.py`: `ArousalInputsV1.gpu_queue_shed_held`, `gpu_shed_reason`.
- `orion/gpu_pool/shed.py`: `shed_held_queue()`.
- `services/orion-sql-writer/app/models/gpu_pool_state.py`: `shed_held_depth`, `shed_reason` columns and row derivation.
- `services/orion-sql-db/manual_migration_regulation_gpu_shed_held_v1.sql`: additive DDL, `REQUIRED-BY: orion-sql-writer, orion-durable-runs`.
- `services/orion-durable-runs/app/regulation_store.py`: selects and passes the new columns.
- `scripts/analysis/measure_arousal_reducer_replay.py`: `--no-gpu-shed`, `--reconstruct-shed`.
- `config/gpu_pool.yaml`: `agent` owner `[agent, memory_distill]`.
- `orion/reverie/baseline.py`: `DEFERRAL_KEY` record.
- `services/orion-thought/app/store.py`: `visual_baseline_deferral()`.
- `services/orion-thought/app/visual_chain_health_monitor.py`: `DeferralEvidence`; arousal-deferral cause first when recorded.
- Tests: `tests/test_regulation_arousal.py`, `orion/gpu_pool/tests/test_memory_distill_priority.py`, `orion/gpu_pool/tests/test_scheduler_shed.py`, `services/orion-sql-writer/tests/test_gpu_pool_state_history.py`, `services/orion-durable-runs/tests/test_temporal_self_regulate.py`, `tests/test_visual_baseline_arousal_reader.py`, `services/orion-thought/tests/test_visual_painting_gap_monitor.py`.
- Workflows: `regulation-history.yml` and `orion-durable-runs-tests.yml` add the migration to their path triggers. `visual-baseline.yml` now runs the baseline-arousal and gap-monitor tests, which no workflow ran before.

## Schema / bus / API changes

- Added: `ArousalInputsV1.gpu_queue_shed_held`, `gpu_shed_reason`. These are internal to durable-runs: only its own regulate node and checkpoint use them (declared in `schema_skew_discovery`). `gpu_pool_state_history.shed_held_depth` (json), `shed_reason` (varchar). Baseline checkpoint key `deferral`.
- Removed / renamed: none.
- Behavior changed: S2 load excludes shed-held queue entries. A new arousal reason string, `gpu_shed:<reason>`. `ArousalReadingV1` and `GpuPoolStateV1` are unchanged, so no Redis or bus reader changes.
- Compatibility: rows with NULL columns read as nothing held. The migration must be applied before sql-writer and durable-runs restart. The deploy wrapper enforces this through the REQUIRED-BY header, and the post-pull applier auto-applies it (`auto_apply_refusals` = `[]`).

## Env/config changes

- Added / removed / renamed keys: none. No `.env_example` changed, so no `.env` sync was needed and no keys were skipped. No new flags.

## Tests run

```text
orion/gpu_pool/tests + tests/test_regulation_arousal.py + tests/test_visual_baseline_arousal_reader.py + scripts/analysis/tests: 1030 passed
services/orion-durable-runs/tests: 454 passed, 91 skipped
services/orion-gpu-pool/tests (schema drift gate etc.): pass after the migration rename
services/orion-sql-writer/tests: test_gpu_pool_state_history 9 passed, incl. the real-Postgres persist test on a throwaway postgres:15 with both migrations applied.
  Full suite: same local-env failures as main (14 on branch vs 15 on main), none new.
services/orion-proposal-runtime tests+evals: 14 passed
services/orion-thought tests+evals: 745 passed; 1 failure (test_mind_enrichment_defaults_off) also fails on main (local .env)
Regression tests that fail on old code:
  - arousal: gpu_queue_evidence 5-tuple/held runs; 10-20 replay reads cabinet_hot only
  - scheduler: old-config replays reproduce starvation and owner_waiting recall
  - thought: message names arousal deferral first
check_migration_required_by.py OK; check_gpu_pool_config.py ok; check_env_template_parity PASS; git diff --check clean
```

## Evals run

```text
scripts/analysis/measure_arousal_reducer_replay.py over live 10-18 06:00 -> 10-20 18:00 (43,279 exported rows),
--no-gpu-shed vs --reconstruct-shed: see the metric-gate table above.
services/orion-proposal-runtime/evals/test_baseline_replay.py, services/orion-thought/evals/test_painting_gap_replay.py: pass.
```

## Docker/build/smoke checks

```text
Not run. No Docker build or deploy (no deploy was requested, and another session is deploying on circe).
Live path UNVERIFIED until the migration is applied and the services restart.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. It found no blockers.

- Finding (should): the alert could blame arousal on a stale deferral record, or when arousal was no longer strained.
  - Fix: the cause is named only when the latest deferral is within `DEFERRAL_FRESH_SEC` (2×retry_sec + 600 s) AND arousal reads `strained` now. Otherwise the alert keeps the GPU-first wording.
  - Evidence: `test_stale_deferral_record_is_not_named_as_the_cause`, `test_deferral_is_not_named_when_arousal_is_not_strained_now[idle|engaged|unknown]`.
- Finding (should): `orion_self_shed` fires below the reflex, where S1 is not hot, so excluding it would turn idle/engaged into unknown.
  - Fix: sql-writer counts only `REFLEX_REASONS` holds (`shed_held_queue(..., reasons=REFLEX_REASONS)`), and the replay proxy uses only reflex reasons.
  - Evidence: `test_shed_held_queue_reflex_filter_drops_orion_self_shed`, plus the sql-writer self-shed row test.
- Finding (should): the owner-change blast radius on running distill holds was untested.
  - Fix: added a test. A granted distill hold is no longer recalled by an agent system call, and that call waits. A background hold's gap stays open to it. Noted in the Fix 2 section above.
  - Evidence: `test_agent_system_call_waits_for_a_running_distill_hold_or_takes_its_gap`.
- Finding (should): deploy order. Before the migration runs, sql-writer inserts fail and the regulate node's GPU read goes stale.
  - Fix: no code change. The REQUIRED-BY header plus post-pull auto-apply cover the normal path, and the restart block applies the migration first.
  - Evidence: `auto_apply_refusals` = `[]`. Do not use `ORION_ALLOW_UNAPPLIED_MIGRATION=1` for these two services.
- Finding (nit): the `model_fields_set` guard was dead code.
  - Fix: removed. Only `shed == {}` (no lever) reads as unknown.
- Finding (nit): the reading did not show partial shed holds.
  - Fix: annotate-only reason `gpu_shed_partial:<reason>:<n>`.
- Finding (nit): string literal and flag comment.
  - Fix: `AROUSAL_STRAINED_REASON` is imported. A comment explains why the diagnostic arousal read is not behind a flag.
- Finding (nit): the replay SQL lost one reason when two sheds fired in the same minute.
  - Fix: `string_agg(DISTINCT ...)` per minute.
- Also caught while fixing: a local variable shadowed the new `reasons` parameter in `shed_held_queue`. Caught by the scheduler-agreement test and renamed.

## Restart required

```bash
# 1. migration FIRST (sql-writer inserts and the regulate node's GPU read fail without it)
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_regulation_gpu_shed_held_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
scripts/safe_docker_build.sh orion-gpu-pool up -d --build
scripts/safe_docker_build.sh orion-proposal-runtime up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
```

Run these from the clean primary `main` checkout after merge. The post-pull rebuild applies the migration itself. All five services run on athena.

## Risks / concerns

- Severity: high (expectation). Concern: this PR will not end the strained reading that has held since 10-18 by itself. In cool windows the queue that is not shed-held is a real backlog: background `agent` holds wait for gpu1, one hold at a time. Mitigation: fix 2 drains the 2 distill holds. Whether S2 should count patient background durable-run holds as "strain" at all is a separate design question. It needs its own metric gate, and this PR does not decide it.
- Severity: medium. Concern: the cabinet sat at 32 to 34.8 °C for two days and nothing alerted. hardware-watch opens an incident only for an AC/cooling failure, and none has opened since 10-15. Mitigation: proposed follow-up, a "cabinet hot for more than N hours" alert in hardware-watch. Not built here.
- Severity: low. Concern: the alert names the deferral only after proposal-runtime has restarted and written a `deferral` record. Until then it keeps the old wording. Mitigation: restart order above.
- Severity: low. Concern: `--reconstruct-shed` is a proxy for snapshots saved before the shed columns existed. Mitigation: it is documented as a proxy, and from now on the real columns replace it.

## PR link

PR_LINK

🤖 Generated with [Claude Code](https://claude.com/claude-code)
