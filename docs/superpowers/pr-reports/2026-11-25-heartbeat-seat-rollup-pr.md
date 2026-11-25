## Summary

The heartbeat now keeps a lasting record of how busy each of its 5 seats was in every 5-minute window. Before, that history lived only in `grammar_events`, which keeps about 3 days, so the field-demon replay had only 311 training rows. Refs #3886.

- orion-heartbeat adds up, per seat and per 5-minute bucket, the gate level of every atom it routes (`route_atom`), plus how many atoms it routed and how many its input rules skipped. It publishes one row per seat when the bucket closes.
- orion-sql-writer stores the rows in a new table, `heartbeat_seat_rollup_5m`. It keeps them; nothing deletes them.
- Missing reads as unknown, never as zero. A window the heartbeat did not hear has no rows. The start window and any window touched by a silence of more than 60 s are marked `partial`; the replay treats them as missing.
- `scripts/demon_network_replay.py` reads the rollup first and uses `grammar_events` only for windows with no rollup rows. It labels each window's source in `results.json`.
- A new eval proves the rollup and the replay give exactly the same number (`==`, not "close") for every window and seat.

## Outcome moved

The replay's training data stops being capped at ~3 days. After deploy the rollup grows by 288 windows a day, so the replay reaches the spec's 14 days about 2 weeks after deploy, and the 4 weeks (~8,000 rows) the rev 3 design plans after about 4 weeks. Issue #3886's check ("the replay can read 14+ days of seat rows") is met only then, so this PR does not close it.

## Current architecture

- orion-heartbeat subscribes to `orion:grammar:event`, routes each atom to a seat with `route_atom` (`app/substrate/routing.py`), and absorbs it into its tensor network. It kept no per-window history.
- The replay rebuilt per-seat 5-minute sums from `grammar_events` every run. sql-writer deletes `grammar_events` rows older than 3 days (`GRAMMAR_EVENTS_RETENTION_DAYS=3`).
- Pattern reused: PR #3282's bus traffic rollup (`services/orion-bus-mirror/app/rate_rollup.py` -> `orion:bus:traffic:rollup` -> sql-writer -> `bus_traffic_rollup_5m`).

## Architecture touched

- orion-heartbeat: new in-memory accumulator `app/seat_rollup.py`; `app/service.py` feeds it from the existing routing path and runs a 10 s flush loop that publishes closed windows; `/health` gains a `seat_rollup` block (published, failed, pending, late/future/duplicate atoms dropped).
- New bus channel `orion:heartbeat:seat:rollup` (producer orion-heartbeat, consumer orion-sql-writer).
- orion-sql-writer: new insert-only model `HeartbeatSeatRollupSQL`, routed by the generic `_write_row` path.
- Replay read path: rollup first, ledger fallback, source labels.

How the value matches the replay exactly:

- Both bucket an atom by the grammar event's `emitted_at`.
- Both sum with `math.fsum`, which is exactly rounded, so the sum does not depend on arrival order. (The replay used `+=` before; this changes its numbers by at most one rounding step.)
- The rollup counts an atom before the absorb queue, so a full queue does not make it disagree with the ledger.
- The rollup ignores a repeated `event_id` in a window; the ledger dedupes on `event_id` too.

## Files changed

- `orion/schemas/heartbeat_seat_rollup.py`: new `HeartbeatSeatRollupV1` contract, channel/kind constants, SQL column list.
- `orion/schemas/registry.py`: registers the schema.
- `orion/bus/channels.yaml`: new channel entry.
- `config/integration/organs.yaml`: `exempt_channels` entry with a reason (telemetry, not experience).
- `config/metrics/metric_definitions.lock.json`: re-locked (one added bus-channel metric).
- `services/orion-heartbeat/app/seat_rollup.py`: the accumulator.
- `services/orion-heartbeat/app/service.py`: feeds the accumulator, flush/publish loop, `/health` block.
- `services/orion-heartbeat/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: three `HEARTBEAT_SEAT_ROLLUP_*` keys and the "Seat rollup" section.
- `services/orion-heartbeat/tests/test_seat_rollup.py`: accumulator, service path, publish/retry, replay merge.
- `services/orion-heartbeat/evals/seat_rollup_match.py`: the exact-match eval.
- `services/orion-sql-writer/app/models/heartbeat_seat_rollup.py`, `models/__init__.py`, `worker.py`, `settings.py`, `.env_example`, `README.md`: model, route, subscription, insert-only.
- `services/orion-sql-writer/tests/test_heartbeat_seat_rollup_sql_shape.py`: write-path shape test.
- `services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v1.sql`: the table.
- `scripts/demon_network_replay.py`: rollup-first read path, source labels, `fsum`.
- `scripts/sync_local_env_from_example.py`: visits orion-heartbeat, syncs `HEARTBEAT_SEAT_ROLLUP_`.
- `tests/test_heartbeat_seat_rollup_bus_catalog.py`, `.github/workflows/orion-heartbeat-tests.yml`: catalog/registry test, run by the heartbeat workflow.

## Schema / bus / API changes

- Added: `HeartbeatSeatRollupV1` (kind `heartbeat.seat.rollup.v1`) on `orion:heartbeat:seat:rollup`; table `heartbeat_seat_rollup_5m`; `seat_rollup` block on heartbeat `/health`.
- Removed: none.
- Renamed: none.
- Behavior changed: the replay reads the rollup first; its `load()` now returns a source label per window.
- Compatibility notes: additive. Before deploy the table does not exist and the replay falls back to `grammar_events` for every window (checked: `rollup_rows_read=0`, all 865 windows labelled `grammar_events`).

Migration: `services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v1.sql`, header `-- ORION-MIGRATION-REQUIRED-BY: orion-sql-writer`. Statements: `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` only (no `DO $$`, no `GRANT`), so post-pull auto-apply accepts it. sql-writer's `create_all` also makes the table at boot.

## Metric quality gate

1. Provenance: `level_sum` = `math.fsum` of `route_atom(...).level` (`services/orion-heartbeat/app/substrate/routing.py::route_atom`) over the seat's atoms in the window, computed in `app/seat_rollup.py::_to_events`; fed from `HeartbeatService._handle_atom_payload`.
2. Independence: not a new signal. It is a durable copy of the value the replay already computes from `grammar_events`. Within the 5 seats, the independence questions are the rev 3 spec's (bus vs. biometrics etc.), unchanged here.
3. Theory anchor: the rev 3 spec's seat value (summed gate level per 5-minute window), which the network replay models.
4. Live-data sanity (824 complete windows, 2026-11-22..25): hub 56% of windows at 0, cortex-exec 22%, harness 19% (event-driven seats read calm by going silent). Bus and biometrics never read 0: bus p05 3.35, p50 3.82; biometrics p05 5.27. For the bus seat the calm expectation per window is about 2.96 (0.110 per tick x ~27 ticks; per tick E[max(0, |z| - 0.798)] / 2.202 for a calm z). The replay subtracts a time-of-day profile, so this floor is not new to this PR, but it is real: filed #3895.
5. Existing mechanism: none kept seat history; `grammar_events` retention is 3 days by design (16 GB table). Followed #3282's rollup pattern.
6. Reversibility: cheap. Turn off `HEARTBEAT_SEAT_ROLLUP_ENABLED`; the replay falls back on its own; `DROP TABLE heartbeat_seat_rollup_5m` removes the history.

## Env/config changes

- Added keys: `HEARTBEAT_SEAT_ROLLUP_ENABLED=true`, `HEARTBEAT_SEAT_ROLLUP_SILENCE_GAP_SEC=60.0`, `HEARTBEAT_SEAT_ROLLUP_FLUSH_GRACE_SEC=30.0` (orion-heartbeat). sql-writer: `orion:heartbeat:seat:rollup` added to `SQL_WRITER_SUBSCRIBE_CHANNELS`; `"heartbeat.seat.rollup.v1":"HeartbeatSeatRollupSQL"` added to `SQL_WRITER_ROUTE_MAP_JSON`.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (orion-heartbeat, orion-sql-writer).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes; it added the 3 heartbeat keys. The two sql-writer list keys are outside the script's default set; local values equalled main's template, so I set them to the new template values by hand. A sql-writer running main's code ignores the unknown route.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-heartbeat: pytest -q tests                                  196 passed
services/orion-sql-writer: pytest -q tests/test_heartbeat_seat_rollup_sql_shape.py tests/test_bus_traffic_rollup_sql_shape.py   19 passed
pytest -q tests/test_heartbeat_seat_rollup_bus_catalog.py                   4 passed
pytest -q tests/test_sql_migration_drift_gate.py tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py tests/test_grammar_event_producer_catalog.py tests/test_compose_logging_journald.py tests/test_compose_restart_policy.py tests/test_bus_traffic_rollup_bus_catalog.py   137 passed
scripts/check_integration_seam.py                                           OK (1 new exemption listed)
scripts/check_definition_drift.py --gate                                    PASS
scripts/check_metric_lineage.py --gate / --prompt-semantics                 PASS / PASS
scripts/check_env_template_parity.py                                        PASS
scripts/check_env_key_single_source.py, check_compose_no_relative_mounts.py, check_service_hostname_refs.py, check_system_health_producers.py, check_inner_state_registry.py, check_scripts_dir_no_stdlib_shadow.py   OK
```

## Evals run

`services/orion-heartbeat/evals/seat_rollup_match.py --mode replay` pushes every routed-service atom of a window from the live `grammar_events` through the real `HeartbeatService._handle_grammar_message` path and compares each window x seat with the replay's own query and `aggregate_grammar_rows`, using `==`.

```text
window 2026-11-25 01:30..02:30 UTC: 7,126 atoms fed, 60/60 cells compared, 0 mismatches -> EXACT MATCH
window 2026-11-24 15:00..21:00 UTC (spans the 15:55-19:05 ledger gap): 17,936 atoms fed,
  160 cells compared, 0 mismatches; 10 partial rows (the 2 windows at the gap edges) and
  the windows inside the gap have no rows -> missing, not zero -> EXACT MATCH
```

`--mode table` (stored rows vs. ledger) runs after deploy: UNVERIFIED.

## Docker/build/smoke checks

```text
Replay smoke before deploy: scripts/demon_network_replay.py --n-null 2 ran to completion;
  rollup_rows_read=0, source_buckets {grammar_events: 865}, scored 455 rows.
No Docker build run in this session; no deploy (per task).
Live publish/store path: UNVERIFIED until deploy.
```

## Review findings fixed

REVIEW_PLACEHOLDER

## Restart required

Deploy from the clean primary `main` checkout after merge. The post-pull rebuild applies the migration and rebuilds both services. By hand:

```bash
docker exec -i orion-athena-sql-db psql -U postgres -d conjourney < services/orion-sql-db/manual_migration_heartbeat_seat_rollup_5m_v1.sql
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-heartbeat up -d --build
```

Deploy sql-writer before the heartbeat, so the first windows have a consumer.

Live check after about 1 hour:

```bash
curl -fsS http://localhost:7251/health | python3 -c "import json,sys; print(json.load(sys.stdin)['seat_rollup'])"
docker exec orion-athena-sql-db psql -U postgres -d conjourney -c "select seat, count(*), sum(partial::int) from heartbeat_seat_rollup_5m group by 1"
/mnt/scripts/Orion-Sapienform/.venv/bin/python services/orion-heartbeat/evals/seat_rollup_match.py --mode table
```

## Risks / concerns

- Severity: medium. Concern: the stored rows and the ledger can differ a little when sql-writer drops a `grammar_events` row on a persist timeout that the heartbeat counted. Mitigation: `--mode table` reports every mismatching cell; the rollup is then the more complete of the two.
- Severity: low. Concern: an atom stamped in a window that was already published (more than 30 s late) is dropped, not added. Mitigation: counted on `/health` as `late_atoms_dropped`. The real receipt lag is not measured yet (the ledger has no receive-time column): read that counter after deploy.
- Severity: low. Concern: if the bus refuses publishes for over 1 hour, the oldest pending rows are dropped. Mitigation: dropped windows have no rows, so they read as missing; counted as `dropped_pending_full`.
- Issues opened: #3895 (clockwork seat sums have a calm floor). Issues closed: none (#3886 stays open until the replay reads 14+ days of rows).

## PR link

PR_LINK_PLACEHOLDER

🤖 Generated with [Claude Code](https://claude.com/claude-code)
