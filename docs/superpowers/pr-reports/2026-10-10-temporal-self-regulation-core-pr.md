# Temporal Self rev 4, order 3: R2/R3 core (arousal reading + dream idle fix)

## Summary

- Orion now has one slow reading of its own situation, **arousal**: `engaged` (Juniper talked in the last 45 min), `idle` (nobody and nothing needs Orion), `strained` (the cabinet is hot, or GPU work has queued up for 5+ minutes), or `unknown` (an input is stale; never treated as idle). It is computed by a pure, I/O-free reducer (`orion/regulation/arousal.py`).
- It runs in a new self-driven durable-runs thread, `temporal_self:orion:<local date>`, whose `regulate` node steps every 120 s and immediately on each Juniper turn. It writes Redis `orion:regulation:latest` (360 s TTL) and `GET /regulation/state`, and records every level change as an `arousal_transition` row in `temporal_self_event`.
- **The dream's idle clock now counts only Juniper's turns.** Orion's own outreach messages no longer reset it (spec R2 repair 1). The rule lives once in `orion/regulation/juniper_turns.py` and is shared with arousal.
- GPU strain reads `queue_depth`, not `backlog_depth`. `queue_depth` passed the metric gate on live snapshots; `backlog_depth` is a ~10 s transient status and was empty in every saved snapshot (#2576).
- **No dial reads arousal yet.** Spec order 6 wires each reader in its own PR. Rest-drive readers are unchanged.

## Outcome moved

- Failure mode closed: on 10-09/10-10, 37 of 217 saved dream checks were "not idle" only because Orion had just sent outreach (old query: 47 not-idle; fixed query: 10). In that window none of those 37 checks also had pressure at or above threshold, so no sleep would have fired earlier; the fix removes the contamination before it matters.
- New capability: a live, inspectable arousal state with history, which later dials (curiosity pacing, outreach cooldown, reverie, self-modification) can read for their own gain.

## Current architecture

- About 30 "when do I act" dials each read their own clock or counter; there was no shared state (Spark `arousal` has been dead since 2026-07-28).
- The dream idle gate read `max(created_at)` over all of `chat_history_log`, including Orion's promptless outreach rows (`services/orion-dream/app/cycle_store.py`).
- Rest drive (#2584) publishes `DriveReadingV1` to Redis `orion:drive:rest:latest`; Hub curiosity/outreach read it.
- orion-durable-runs hosts one self-driven thread, the Situation Graph (`situation.update`).

## Architecture touched

- `orion/regulation/` (pure): new `arousal.py`, `juniper_turns.py`.
- `orion/schemas/regulation.py` (new): `ArousalInputsV1`, `ArousalReadingV1`, `RegulationStateV1`; registered in both schema registries and the inner-state registry.
- `services/orion-durable-runs`: new `temporal_self.update` thread (graph, driver, store), route, health, settings, env.
- `services/orion-dream`: idle SQL only.
- `services/orion-sql-db`: one new hand-applied migration.

## Files changed

- `orion/regulation/arousal.py`: the reducer (`classify_arousal`), S2 evidence builder (`gpu_queue_evidence`), `level_seconds`.
- `orion/regulation/juniper_turns.py`: "Juniper spoke" rule (SQL predicate + bus-payload check), shared by dream and arousal.
- `orion/schemas/regulation.py`: schemas, Redis key, workflow/thread constants, tolerant parser.
- `orion/schemas/registry.py`: `RegulationStateV1` in `_REGISTRY` and `SCHEMA_REGISTRY`.
- `orion/inner_state_registry.py`: `regulation.state.v1` entry with rest/absent semantics (SHADOW: no reader yet).
- `config/metrics/metric_definitions.lock.json`: re-locked (1 added URN).
- `services/orion-dream/app/cycle_store.py`: `IDLE_MINUTES_SQL = JUNIPER_IDLE_MINUTES_SQL`.
- `services/orion-durable-runs/app/temporal_self_graph.py`: `ingest -> regulate -> done`.
- `services/orion-durable-runs/app/temporal_self_driver.py`: tick + Juniper-turn wake, local-day threads, seeding, retention.
- `services/orion-durable-runs/app/regulation_store.py`: E1/S1/S2 reads, transition write.
- `services/orion-durable-runs/app/main.py`: build/start/close, chat-turn routing, `GET /regulation/state`, `/health.temporal_self`.
- `services/orion-durable-runs/app/runner.py`: `temporal_self.update` added to `SELF_DRIVEN_WORKFLOWS`.
- `services/orion-durable-runs/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: keys and docs.
- `services/orion-sql-db/manual_migration_temporal_self_event_v1.sql`: `temporal_self_event` (spec columns).
- `scripts/check_env_key_single_source.py`: `DREAM_IDLE_MINUTES` owned by orion-dream; durable-runs' copy is gated.
- `scripts/sync_local_env_from_example.py`: `TEMPORAL_SELF_`, `ORION_REGULATION_` prefixes; `ORION_SITUATION_TIMEZONE`, `DREAM_IDLE_MINUTES` exact.
- `scripts/analysis/measure_arousal_reducer_replay.py` + test: eval replaying the shipped reducer over saved history.
- `docs/superpowers/evidence/2026-10-10-temporal-self-regulation-core/`: replay + gate evidence.
- Tests: `tests/test_regulation_arousal.py`, `services/orion-durable-runs/tests/test_temporal_self_regulate.py`, `services/orion-dream/tests/test_idle_counts_only_juniper.py`.
- CI: new `.github/workflows/regulation-arousal.yml`; path filters/steps in `orion-durable-runs-tests.yml`, `regulation-history.yml`.

## Schema / bus / API changes

- Added: `RegulationStateV1` (+ nested `ArousalReadingV1`, `ArousalInputsV1`); Redis key `orion:regulation:latest`; route `GET /regulation/state`; table `temporal_self_event`; workflow `temporal_self.update` (checkpoint-only).
- Removed / renamed: none.
- Behavior changed: dream idle counts only Juniper's turns.
- Compatibility notes:
  - **No bus channel.** The regulation state has no bus consumer (spec: "not published on a bus channel in v1").
  - **Rest drive is read from Redis, not `orion:drive:reading`.** The key already exists with a TTL equal to the staleness bound, and the regulation state only embeds it for the trace (arousal must not read it: that would be a loop). A bus channel would add a second copy and a publisher change in orion-dream for no extra information. `channels.yaml` is unchanged.
  - **GPU state is read from `gpu_pool_state_history`, not a subscription to `orion:gpu_pool:state`.** The same rows the eval replays, restart-proof (a 5-minute sustain survives a durable-runs restart), and no 5 s envelope stream into durable-runs. sql-writer lands each snapshot in about 10 ms (live p99 18 ms). If sql-writer stops, S2 goes stale and arousal reads `unknown`, which is honest.
  - **No `DurableRunStateV1` rows for this workflow.** That model's `workflow` is a closed Literal read by sql-writer and Hub; widening it needs a consumer-first rollout for a trace `temporal_self_event` already holds. Transitions are also logged (`arousal_transition ...`).
  - **Juniper-turn rule deviates from the spec's `source='hub_orion'`.** Live (every row ever): every prompted row is Juniper-initiated (hub_orion 317, hub_ws 33, hub_http 3, hub 2, her dream button and collapse-mirror entries 14), and every promptless row (236) is Orion's outreach stamped `unsolicited`. A source-label list would drop her hub_ws/hub_http turns and need editing for each new surface.

## Metric gate: S2 GPU `queue_depth` (spec R3 S2)

1. **Provenance.** `services/orion-gpu-pool/app/runtime.py:1459-1465`: per `work_class`, the count of live leases with status `queued`, in every `GpuPoolStateV1` (5 s). Persisted by sql-writer to `gpu_pool_state_history.queue_depth` (`manual_migration_regulation_history.sql`).
2. **Independence.** S1 is a different sensor (cabinet thermometer). `backlog_depth` comes from the same snapshot but a different lease status, and is excluded. Biometrics `strain` is excluded (a blend of GPU and heat). Named loop kept: Orion's own GPU work fills the queue. That is negative feedback (readers will slow optional work, the queue drains), guarded by the 10-minute clear time.
3. **Theory.** Queueing / Little's law: a queue that holds at depth >= 2 for minutes means arrivals exceed service capacity (utilisation at or above 1) over that window, so compute is over capacity. Limit, stated: a queue jammed by a fault (no service at all) reads the same as overload.
4. **Live sanity** (`gpu_pool_state_history`, host circe, 21,629 snapshots, 10-09 06:46 to 10-10 18:24 UTC). Summed depth ranges 0..21. It reaches the rest value 0 in 5,839 snapshots (27%), and whole hours read 0 on 10-10 (07:00, 15:00-17:00). 10-09 06:46 to 10-10 ~02:00 it never went below 8: every hour logged `actuate_refused config_unloadable:ValidationError` (the stale lane-controller config), so the agent queue jammed for about 18 h. A single stuck lease is also real: `{"diffusion": 1}` held for 1,509 snapshots (~2 h). **Floor 2, not 1**, so one stuck lease cannot hold strain. Sustain simulation after recovery (from 10-10 03:00, 15.4 h): floor 1 gives 3.98 h strained over 11 entries; floor 2 gives 0.96 h over 6; floor 3 gives 0.39 h over 3 (`queue-depth-gate-sustain-sim.txt`).
5. **Existing mechanism.** None. `backlog_depth` cannot sustain by design. The lease-event reconstruction was shown unreliable in #2576.
6. **Reversibility.** Two env keys, no schema field baked in. `ArousalReadingV1.gpu_queue_*` are evidence fields only.

**Verdict: PASS for strain at floor 2, sustained 300 s.** `backlog_depth` is not a dead metric: it is a ~10 s transient status (#2576), which is why it never sustains. It is just the wrong instrument for sustained strain.

## Tests run

```text
tests/test_regulation_arousal.py                                  36 passed (after review fixes: 40 incl. eval)
scripts/analysis/tests/test_measure_arousal_reducer_replay.py      4 passed
  (both also green in a fresh venv with only CI's deps: sql-writer requirements + pytest)
services/orion-dream/tests (REGULATION_HISTORY_TEST_POSTGRES_URI on throwaway Postgres 16)   198 passed
  incl. test_idle_counts_only_juniper.py::test_outreach_row_does_not_reset_idle_postgres PASSED (real Postgres)
services/orion-durable-runs/tests (ORION_ADMISSION_TEST_DSN on throwaway Postgres 16)        435 passed, 1 skipped
  incl. test_temporal_self_regulate.py 14 passed after review fixes (store SQL on real Postgres)
scripts/check_metric_lineage.py --gate             PASS (orphans unchanged: bus 17, inner_state 11)
scripts/check_metric_lineage.py --prompt-semantics PASS
scripts/check_definition_drift.py --gate           PASS (after --update; 1 added)
scripts/check_inner_state_registry.py              OK (21 entries)
scripts/check_env_template_parity.py               PASS
scripts/check_env_key_single_source.py             OK (5 owned keys)
check_metric_lineage.py --metric regulation.state.v1   lineage card resolves (inner_state, SHADOW, rest/absent semantics)
check_metric_lineage.py --generic-consumers        no generic reader of inner-state entries
```

## Evals run

`scripts/analysis/measure_arousal_reducer_replay.py` steps the shipped reducer at the node's cadence (120 s plus each Juniper turn) over the saved history. It builds the inputs exactly as `regulation_store.py` does. Window: 2026-10-09 07:00 to 2026-10-10 18:00 UTC, the whole span that has saved GPU snapshots. Input sha256 is in `floor-comparison.json`.

| UTC day | engaged | idle | strained | unknown |
|---|---|---|---|---|
| 2026-10-09 (17 h) | 0 min | 0 min | 1020 min | 0 min |
| 2026-10-10 (18 h) | 0 min | 770 min | 310 min | 0 min |

- Strained exits: 6 on 10-10, under the spec's 12/day oscillation bar. Transitions: 11.
- 10-09 is strained throughout, from the config-fault queue jam above. 10-10's strained spans run 14-60 min: agent work queued 2-4 deep.
- **Engaged is 0 minutes in this window.** Juniper made 3 turns. One (10-10 00:08) landed inside the jam, and strain outranks engaged by spec rule order. So the "every level between 0% and 100%" bar is **not met on this window**. That is a fact about this window, not the instrument: engaged is reachable (tests), and #2576's 14-day lane showed 5% engaged. A 14-day replay of S2 is impossible because GPU snapshots only start 10-09 06:46.
- Floor sensitivity (`floor-comparison.json`): floor 1 gives 518 min strained on 10-10, floor 3 gives 250.

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-durable-runs build          Image built
docker run (throwaway, built image): ZoneInfo('America/Denver') ok; app.main imports; settings
  defaults True/120.0/45.0/2; resolve('RegulationStateV1') ok
Production image tag restored afterwards: orion-durable-runs-durable-runs:latest -> d40302c233f7
  (the running container's image), so a later `up` without --build cannot pick up this branch.
Read-only live smoke (the node's own reads, default_transaction_read_only=on, 18:49 UTC 10-10):
  E1 1120.7 min, cabinet 28.38 C normal, GPU queue 0 (snapshot age 6.8 s) -> idle ['no_juniper_turn_45m']
```

Live path after deploy: **UNVERIFIED** (not deployed, per instructions).

## Review findings fixed

Review: orion-repo-agent subagent, read-only, target `origin/main...feat/temporal-self-regulation-core`. No blockers.

- Finding (should-fix): the cabinet reflex's hysteresis seed (last step's hot/critical state) was carried across an outage of any length, including day-rollover seeding. That could read `strained` from a past nobody observed.
  - Fix: `temporal_self_graph.regulate` drops the stored inputs under the same gap rule as the previous reading (more than 3 ticks). The replay eval does the same.
  - Evidence: `test_old_cabinet_seed_is_dropped_after_an_outage`.
- Finding (nit): a GPU host that stopped publishing could outrank a fresh host and turn S2 stale.
  - Fix: fresh hosts are ranked first (store and eval).
  - Evidence: `test_stale_gpu_host_does_not_outrank_a_fresh_one`.
- Finding (nit): a 15 s GPU staleness bound is tight against live snapshot gaps (max 39.8 s over 6 h). A step landing in a gap would read `unknown` and restart a latched strain's clear clock.
  - Fix: `GPU_STATE_STALE_SEC = 30`. This deviates from the spec's 3x-cadence rule; the rationale is in `arousal.py`.
  - Evidence: replay numbers unchanged; tests updated.
- Finding (nit): the docs said "each level change is one row", but a restart after a gap also writes one.
  - Fix: reworded in the README, `.env_example`, schema docstring, and driver docstring.
- Finding (nit, not changed): each boot issues about 30 idempotent `adelete_thread` calls (the retention sweep's memory is in-process). It is harmless and cheap, and noted here so it is not mistaken for a leak.

## Restart required

Deploy order: migration first, then orion-durable-runs, then orion-dream. Either service order is safe: the dream fix stands alone, and durable-runs without the migration still writes Redis (it only loses transition history and logs `regulation_transition_write_failed`). From the primary checkout on main after merge (the wrapper needs its escape hatch there):

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker exec -i orion-athena-sql-db psql -U postgres -d conjourney -v ON_ERROR_STOP=1 < services/orion-sql-db/manual_migration_temporal_self_event_v1.sql && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-durable-runs up -d --build && ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-dream up -d --build
```

Live checks after deploy (spec order 3 exit evidence):

- `python3 scripts/check_sql_migrations_applied.py --file manual_migration_temporal_self_event_v1.sql` reports applied.
- Within 2 minutes: `curl -fsS localhost:8124/health` shows `temporal_self.steps >= 1`; `curl -fsS localhost:8124/regulation/state` returns a state; `redis-cli -u "$ORION_BUS_URL" TTL orion:regulation:latest` is between 0 and 360.
- `docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select occurred_at, label, payload_json->>'from' from temporal_self_event where source_kind='arousal_transition' order by occurred_at desc limit 5"` shows the boot transition.
- `engaged` within one step of a real Juniper turn (log line `arousal_transition ... to=engaged`), and `idle` holding overnight.
- Delete the key (`redis-cli -u "$ORION_BUS_URL" DEL orion:regulation:latest`): the dream keeps sleeping on its own Juniper-filtered idle query (no reader of arousal exists yet, so this holds by construction), and the key reappears on the next step.
- Dream: after the next outreach, `dream_pressure_observation` rows keep `idle_minutes` counting from Juniper's last turn, not the outreach.

## Risks / concerns

- Severity: medium. Concern: S2 cannot tell a fault-jammed queue from overload, so the 10-09 jam would read as 17 h strained. Mitigation: once readers land, strained only slows optional work, which is harmless during a fault. The `reasons` field and `gpu_queue_depth` show which it was.
- Severity: medium. Concern: only ~36 h of GPU history backs the floor and sustain choice, and most of it is the fault. Mitigation: env-tunable; re-fit with the arousal gains after two weeks (spec Missing question 12; noted, not built).
- Severity: low. Concern: `GPU_STATE_STALE_SEC` is 30 s, not the spec's 3x cadence (15 s), because of live snapshot gaps. Mitigation: still 10x shorter than the 300 s sustain window.
- Severity: low. Concern: thread retention keeps ~720 checkpoints a day for 2 days, which the runner's resume sweep walks (self-driven, so skipped). Mitigation: `TEMPORAL_SELF_RETENTION_DAYS`; same pattern as the Situation Graph.
- Severity: low. Concern: `temporal_self_event` has no retention job yet (a handful of rows a day). Mitigation: patch 3 adds retention (spec: events 30 days).

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
