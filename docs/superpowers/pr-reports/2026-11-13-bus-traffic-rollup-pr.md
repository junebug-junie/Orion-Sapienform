## Summary

The bus mirror now keeps a durable 5-minute history of bus traffic. Before this, the raw log was deleted after 24 h and the graph kept only running averages, so no history existed. This is Patch 1 of `docs/superpowers/specs/2026-11-11-causal-geometry-redesign.md`.

- `orion-bus-mirror` counts every message per channel, counts decode failures, and counts every cross-organ hop with its mean latency. It does this in wall-clock 5-minute buckets (`app/rate_rollup.py`).
- Each closed bucket is published as `BusTrafficRollupV1` on `orion:bus:traffic:rollup`. Publishing is fail-open and has a bounded 1-hour retry queue.
- `orion-sql-writer` stores one insert-only row per bucket in `bus_traffic_rollup_5m`.
- A bucket the mirror did not hear in full is flagged `partial`. This covers process start and silences longer than 30 s, such as a Redis drop. Readers exclude these rows, so an outage does not read as a traffic dip.
- `MIRROR_TRAFFIC_ROLLUP_ENABLED` defaults to on. The ChainTracker now also runs when the graph is off, so the rollup still gets hops.

## Outcome moved

Bus history now survives past 24 h. Every day without this patch lost data the Patch 2 replay needs. Nothing in cognition reads this table. Its only reader is the Patch 2 replay eval.

Metric gate note: this patch stores raw counts and does not wire anything into a model, so the metric quality gate does not apply yet. The Patch 2 replay is where each derived feature must pass that gate.

## Current architecture

`orion-bus-mirror` does three things:
- It subscribes to `orion:*`.
- It writes each message to a SQLite log, which is pruned at 24 h.
- When `MIRROR_GRAPH_ENABLED` is on, it keeps EWMA (running-average) edges in FalkorDB.

No per-interval counts were persisted anywhere.

## Architecture touched

- **orion-bus-mirror:** a new accumulator, a flush/publish background loop, and wiring in the message loop. `catalog_names` is now always loaded. The ChainTracker runs when either the graph or the rollup is on.
- **Contract:** a new schema, a registry entry, and a channel.
- **orion-sql-writer:** a new model, plus entries in the route map, the subscribe list, and INSERT_ONLY.
- **orion-sql-db:** an additive migration.
- **CI:** a new `orion-bus-mirror-tests` workflow. Until now, no CI workflow ran the bus-mirror tests.

## Files changed

- `orion/schemas/bus_traffic.py`: `BusTrafficRollupV1` and `BusTrafficHopV1`, the channel, kind and column constants.
- `orion/schemas/registry.py`, `orion/bus/channels.yaml`: register the schema and the channel.
- `services/orion-bus-mirror/app/rate_rollup.py`: the pure accumulator. It handles bucketing, the channel/hop caps, `partial` flags, the flush grace, and late-message handling.
- `services/orion-bus-mirror/app/main.py`: feeds messages and hops into the accumulator, and runs the flush/publish loop.
- `services/orion-bus-mirror/app/settings.py`, `.env_example`, `docker-compose.yml`: add `MIRROR_TRAFFIC_ROLLUP_ENABLED=true`.
- `services/orion-bus-mirror/README.md`: new "Traffic rollup" section.
- `services/orion-sql-writer/app/models/bus_traffic_rollup.py`, `app/models/__init__.py`, `app/worker.py`, `app/settings.py`, `.env_example`, `README.md`: write path for `bus_traffic_rollup_5m`.
- `services/orion-sql-db/manual_migration_bus_traffic_rollup_5m_v1.sql`: additive table and `bucket_start` index, with the `REQUIRED-BY: orion-sql-writer` header.
- `scripts/sync_local_env_from_example.py`: adds `orion-bus-mirror` to the default services and adds the `MIRROR_TRAFFIC_` prefix. Without these, the default sync visited nothing and added nothing.
- `config/metrics/metric_definitions.lock.json`: re-locked for the new bus channel metric.
- `.github/workflows/orion-bus-mirror-tests.yml`: new CI workflow.
- Tests: `services/orion-bus-mirror/tests/test_rate_rollup.py`, `test_main_traffic_rollup.py`, `services/orion-sql-writer/tests/test_bus_traffic_rollup_sql_shape.py`, `tests/test_bus_traffic_rollup_bus_catalog.py`.

## Schema / bus / API changes

- **Added:** schema `BusTrafficRollupV1` (kind `bus.traffic.rollup.v1`) on channel `orion:bus:traffic:rollup`. The producer is `orion-bus-mirror` and the consumer is `orion-sql-writer`. The new table is `bus_traffic_rollup_5m`.
- **Removed / Renamed:** none.
- **Behavior changed:** the mirror loads the channel catalog even when the graph is off, and it runs the ChainTracker when the rollup is on.
- **Compatibility notes:**
  - The spec's earlier draft proposed two tables. The Decisions section replaced them with one jsonb row per bucket, and this patch follows that.
  - The mirror counts its own rollup message: one per bucket.

## Env/config changes

- **Added keys:** `MIRROR_TRAFFIC_ROLLUP_ENABLED=true` in orion-bus-mirror.
- **Changed values:** in orion-sql-writer, `SQL_WRITER_SUBSCRIBE_CHANNELS` gains `orion:bus:traffic:rollup`, and `SQL_WRITER_ROUTE_MAP_JSON` gains `bus.traffic.rollup.v1 -> BusTrafficRollupSQL`.
- **`.env_example` updated:** yes, for both services.
- **Local `.env` synced:** yes.
  - The default run of `python scripts/sync_local_env_from_example.py` added `MIRROR_TRAFFIC_ROLLUP_ENABLED='true'` to the bus mirror.
  - Running `... orion-sql-writer --all-keys --force` updated the two sql-writer list keys. A diff first confirmed the new entries were the only difference between the live `.env` and the template.
- **Skipped keys requiring operator action:** none. The sync's other "diverged" lines are pre-existing and belong to unrelated services.

## Tests run

```text
services/orion-bus-mirror/tests                                    124 passed
  (also 118 passed in a clean venv with the service's pinned requirements.txt + requirements-dev.txt)
services/orion-sql-writer/tests (whole folder) + migration gates     1386 passed, 214 skipped
tests/test_bus_traffic_rollup_bus_catalog.py                       3 passed
CI static-gate pytest set (env sync, settings defaults, schema registry, journald,
  deploy migration gate, sql migration drift, grammar catalog, metric semantics, ...)  317 passed, 1 failed
  - failed: test_single_consumer_channels_gate::test_every_request_kind_single_consumer_channel_is_annotated
    pre-existing on origin/main (gpu_pool / reading request channels), unrelated
check_metric_lineage --gate / --prompt-semantics                   PASS / PASS
check_definition_drift --gate                                      PASS (after re-lock)
check_migration_required_by --base origin/main                     OK
check_env_template_parity / env_key_single_source / bus_reply_channels / system_health_producers /
  compose_no_relative_mounts / service_hostname_refs / inner_state_registry / sentience_instruments /
  async_routes_not_blocking / scripts_dir_no_stdlib_shadow          all OK
migration auto-apply allowlist (orion.sql_migration_drift.auto_apply_refusals)   []
```

## Evals run

```text
None. This patch is storage only. Its eval is the Patch 2 replay, which reads this table.
Acceptance check 1 (the hourly total matches the SQLite raw log within 1%) needs live
data after deploy: UNVERIFIED.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-bus-mirror build  -> exited 0 but skipped the build
  (no root .env in the worktree)
docker build -t orion-bus-mirror-rollup-check:pr -f services/orion-bus-mirror/Dockerfile .  -> OK
in-image smoke: rollup_enabled True, catalog 358, rollup channel in catalog,
  rollup envelope passes OrionBusAsync._validate_payload. Test image removed.
The live bus-mirror container was not touched.
```

## Review findings fixed

The review found no MUST items.

- **Finding:** a Redis-outage gap, or a slow first connect, would be stored as a normal bucket that reads as a traffic dip.
  - **Fix:** added the `partial` flag to the schema, the model and the migration. It is set for the startup bucket and for both sides of any silence longer than 30 s. The startup bucket is now flagged instead of dropped.
  - **Evidence:** `test_silence_gap_flags_both_sides_and_skips_blind_buckets`, `test_slow_first_connect_flags_the_first_bucket_partial`.
- **Finding:** a message stamped before the SQLite await could recreate an already-published bucket. The sql-writer would drop it as a duplicate, so the message was lost.
  - **Fix:** a bucket is now flushed 5 s after it ends, and a late message goes into the next open bucket.
  - **Evidence:** `test_grace_holds_a_just_closed_bucket`, `test_late_message_after_flush_goes_to_the_next_open_bucket`.
- **Finding:** the sql-writer test did not go through the real `_write_row`.
  - **Fix:** added a test that runs the bus JSON payload through `worker._write_row` twice and asserts one row with tz-aware time, jsonb and `partial`.
  - **Evidence:** `test_real_write_row_path_from_a_bus_json_payload`.
- **Finding (NIT):** the README claimed a missing row could mean zero traffic.
  - **Fix:** reworded. A healthy mirror always counts its own heartbeat, so a missing row always means no data.
- **Finding (NIT):** no test covered hop overflow together with the negative-latency clamp.
  - **Fix:** added `test_hop_overflow_key_and_negative_latency_clamp_together`.
- **Finding (NIT):** the 1% comment might be wrong.
  - **Result:** checked, no change needed. Undecodable messages are stored in SQLite as rows by `_serialize_envelope`, so the rollup and the raw log count the same messages.

## Restart required

Deploy after merge from the clean primary `main` checkout. The post-pull rebuild auto-applies the migration first, because it is additive and has a REQUIRED-BY header.

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-bus-mirror up -d --build
```

## Risks / concerns

- **Severity: low**
  - **Concern:** the live result is UNVERIFIED. No rows will exist until both services are deployed.
  - **Mitigation:** the README has a check query. Expect one row every 5 minutes, and the first row flagged `partial`.
- **Severity: low**
  - **Concern:** uncataloged dynamic channels keep their literal names, up to 2000 per bucket. Past that they count under `__overflow__`.
  - **Mitigation:** the cap bounds memory and row size.
- **Severity: low**
  - **Concern:** after more than 1 h of failed publishes, the oldest buckets are dropped and a warning is logged.
  - **Mitigation:** this is fail-open by design and never blocks the mirror loop.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3282

🤖 Generated with [Claude Code](https://claude.com/claude-code)
