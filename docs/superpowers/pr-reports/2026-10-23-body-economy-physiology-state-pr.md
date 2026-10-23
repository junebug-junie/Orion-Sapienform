# Body economy A3: Orion's Body organ (PhysiologyStateV1 every regulation tick)

## Summary

- Orion now has one reading of every constraint they live under, written every ~120 s: cabinet heat, Orion's electricity bill this month, how much of each GPU lane they held in the last hour, how many outreach messages Juniper got today, and the Claude limit state. Each is in its own unit, with its line (soft line, envelope, capacity, cap) beside it.
- A source that failed, went stale, or was never observed is an explicit `no_reading`, never 0. The schema enforces it: a reading has a level or a no_reading reason, never both, never neither.
- Pure, I/O-free reducer `orion/physiology/body.py`. A new `physiology` node in durable-runs' regulation thread does the reads (`app/physiology_store.py`) and writes Redis `orion:physiology:latest` and bus `orion:physiology:state`.
- sql-writer persists one row per tick to `physiology_state`. This is the first persisted history of Orion's body/strain state; the design notes none existed.
- No prices, no forecasts, and no reader decides anything from it yet (Organ 2 / Patch B onward).

## Outcome moved

Before: nothing persisted strain or constraint state over time; heat, energy, GPU lanes, Juniper's attention and Claude limits each lived in a separate place, in a separate unit, with no shared "missing" convention. After: one row per tick in `physiology_state` with all five, joinable by time. Live read-only smoke (2026-10-23 ~05:00 UTC, real tables, reads take ~0.1 s):

```text
heat               29.4 C      line 31.0      (thermal_state elevated)
energy             $141.15     no envelope    1172 kWh month to date, 103 kWh at the carried rate
lane:agent         3600 slot_s  (1 slot)      one hold the whole hour; 15 child calls not double counted
lane:agent-gpu2    5519 slot_s  (2 slots)
lane:fast          6197 slot_s  (4 slots)
lane:metacog       3149 slot_s  (4 slots)     was 6824 before the review fix: a lost close read as held
attention:juniper  0 messages   cap 4
claude             no_reading: no_observation (producer COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=false)
```

Lanes with live capacity from the pool's state RPC, read together a few minutes later (level / line, slot-seconds per hour):

```text
lane:agent 3600 / 3600   lane:agent-gpu2 5763 / 7200   lane:fast 6232 / 14400   lane:metacog 3240 / 14400
lane:chat 0 / 3600       lane:world 0 / 7200           lane:kev 0 / 3600        lane:diffusion 0 / 0 (0 slots now)
warnings: lane_open_leases:7, lane_child_leases_in_hold_slots:14
```

## Current architecture

Durable-runs' `temporal_self.update` thread ran `ingest -> regulate -> immune -> chronicle -> done` every 120 s and on each Juniper turn; `regulate` wrote `RegulationStateV1` (arousal) to Redis only. Energy lived in `orion_energy_hourly` / `energy_cost_accrued` (Hub Energy tab), GPU lease history in `gpu_pool_events` / `gpu_pool_leases`, outreach in `endogenous_outreach_decisions`, Claude limits on `orion:substrate:claude_limit` (producer off). Nothing combined them, and nothing stored arousal/strain over time except level-change rows.

## Architecture touched

- `orion/physiology/` (new): pure Body reducer.
- `orion/schemas/physiology.py` (new): `PhysiologyStateV1`, `PhysiologyConstraintReadingV1`.
- orion-durable-runs: new `physiology` node between `regulate` and `immune`; reads, Redis projection, bus publish; subscribes to `orion:substrate:claude_limit`.
- orion-sql-writer: new route/model/table `physiology_state`, force-subscribed channel.
- Contracts: schema registry (both maps), `channels.yaml`, inner-state registry, metric definition lock, body-economy contract (`config/physiology/contract.yaml`, from #2862).

## Files changed

- `orion/schemas/physiology.py`: the schema, Redis key, channel, kind constants; level-xor-no_reading validator.
- `orion/physiology/body.py`: per-constraint readings, lease interval math, no_reading rules.
- `orion/physiology/tests/test_body.py`: 29 pure reducer tests, every no_reading case.
- `orion/schemas/registry.py`: `PhysiologyStateV1` in `_REGISTRY` and `SCHEMA_REGISTRY` (verified with `resolve()`).
- `orion/schemas/claude_limit.py`: `CLAUDE_LIMIT_KIND` constant (the producer's existing envelope kind).
- `orion/bus/channels.yaml`: `orion:physiology:state` (durable-runs -> sql-writer); durable-runs added as a consumer of `orion:substrate:claude_limit`.
- `orion/inner_state_registry.py`, `scripts/check_inner_state_registry.py`: `physiology.state.v1` entry (REHEARSAL, semantics); `physiology` added to the inner-state keyword gate.
- `config/metrics/metric_definitions.lock.json`: re-locked (2 added, 1 routing change: claude_limit's consumer list).
- `config/physiology/contract.yaml`: the four modules registered as `sources` (they define/produce/persist; gate nothing).
- `services/orion-durable-runs/app/{temporal_self_graph,temporal_self_driver,main,settings,physiology_store}.py`: the node, health, wiring, settings, reads.
- `services/orion-durable-runs/{.env_example,docker-compose.yml,README.md}`: env keys and docs.
- `services/orion-durable-runs/tests/test_physiology_node.py`: graph node, store helpers, store SQL on Postgres.
- `services/orion-sql-writer/app/{models/physiology_state.py,models/__init__.py,settings.py,worker.py}`, `.env_example`, `README.md`: persistence.
- `services/orion-sql-writer/tests/test_physiology_state_sql_shape.py`: route/subscribe/catalog/migration parity and the write path.
- `services/orion-sql-db/manual_migration_physiology_state_v1{,_rollback}.sql`: the table.
- `scripts/check_env_key_single_source.py`: `HUB_ENDOGENOUS_OUTREACH_DAILY_CAP` owned by orion-hub (durable-runs reads a copy).
- `scripts/sync_local_env_from_example.py`: `ORION_PHYSIOLOGY_`, `CHANNEL_PHYSIOLOGY_` sync prefixes (without them the default sync skipped the keys).
- `.github/workflows/orion-durable-runs-tests.yml`, `orion-sql-writer-tests.yml`: paths and test steps.

## Schema / bus / API changes

- Added: `PhysiologyStateV1` (`physiology.state.v1`), channel `orion:physiology:state`, Redis `orion:physiology:latest` (TTL 360 s), table `physiology_state` (PK `tick_id`, insert-only).
- Removed: none.
- Renamed: none.
- Behavior changed: durable-runs subscribes to `orion:substrate:claude_limit` (in-memory latest per kind). `GET /health` -> `temporal_self.physiology` shows the last tick id and every `no_reading`.
- Compatibility notes: brand-new schema and channel; no existing payload changes. `orion:regulation:latest` is untouched.

## Env/config changes

- Added keys (orion-durable-runs): `ORION_PHYSIOLOGY_ENABLED=true`, `ORION_PHYSIOLOGY_MIN_INTERVAL_SEC=100.0`, `ORION_PHYSIOLOGY_REDIS_TTL_SEC=360`, `CHANNEL_PHYSIOLOGY_STATE`, `ORION_PHYSIOLOGY_HEAT_SOFT_LINE_C=31.0`, `ORION_PHYSIOLOGY_ENERGY_MONTHLY_ENVELOPE_USD=` (empty: no line until Juniper sets one), `ORION_PHYSIOLOGY_LANE_WINDOW_SEC=3600.0`, `HUB_ENDOGENOUS_OUTREACH_DAILY_CAP=4` (owner: Hub), `CHANNEL_CLAUDE_LIMIT`.
- sql-writer: `orion:physiology:state` added to `SQL_WRITER_SUBSCRIBE_CHANNELS`, `physiology.state.v1` to `SQL_WRITER_ROUTE_MAP_JSON` (both also forced/merged in code).
- Removed / renamed keys: none.
- `.env_example` updated: yes (durable-runs, sql-writer).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes, all 9 durable-runs keys added to the primary checkout's `.env`.
- skipped keys requiring operator action: none. The primary checkout's sql-writer `.env` keeps its older `SQL_WRITER_SUBSCRIBE_CHANNELS` / `SQL_WRITER_ROUTE_MAP_JSON` values (reported as diverged by the sync, not overwritten); harmless, because the channel is force-subscribed and the route map merges with the code default.

## Metric quality gate (CLAUDE.md §0A)

1. **Provenance.** heat: `regulation_store._cabinet` -> `orion.autonomy.cabinet_heat.read_cabinet_heat` (`orion_biometrics_summary.cabinet_temp_c`), reused from this step's own regulate inputs. energy: `orion_energy_hourly` (sql-writer `orion_energy_rollup.py`) x `energy_cost_accrued.marginal_usd_per_kwh` (newest `computed_at` per interval). lanes: `gpu_pool_events.held_ms` (pool `runtime._emit_row`) + `gpu_pool_leases` (pool's own projection) + `roles[].slots` from the pool state RPC. attention: `endogenous_outreach_decisions` with Hub's own cap rule. claude: `ClaudeLimitObservationV1` from orion-cocreation-signals.
2. **Independence.** heat is NOT independent of arousal's S1: same cabinet reading, different unit (C vs reflex verdict). It is a mirror for the future price, not a new signal. Lane slot-seconds and arousal's S2 (queue depth) share the pool but measure different things (held vs waiting). Energy and heat are causally linked (Orion's load heats the cabinet) but read from different sensors.
3. **Theory anchor.** Each constraint is a budget in its own unit, per the design's Organ 1 table; the level is the quantity the budget is spent in (C, USD, slot-seconds, messages, limit events). No composite score.
4. **Live data.** Every source was read live (table above). Rest states are reachable: heat below 31 C (29.4 now), attention 0, lanes with no lease in the hour read a real 0 (chat, world, kev live), claude clear only with both limits fresh. Energy only grows within a month; its rest is the month start. Degenerate case found and fixed in review: `lane:metacog` read 6824 slot_s because a lost `released` event left a grant looking open; now open leases come from `gpu_pool_leases` (3149).
5. **Existing mechanism.** Reused the regulation thread, `cabinet_heat`, the Hub's energy pricing rule, the pool state RPC (`orion.gpu_pool.placement.fetch_pool_state`), Hub's cap-count rule, `ClaudeLimitObservationV1`. Nothing re-derived.
6. **Reversibility.** `ORION_PHYSIOLOGY_ENABLED=false` stops the node. The table is additive; nothing reads it yet.

## Tests run

```text
PYTHONPATH=. pytest orion/physiology/tests -q                                   29 passed
(services/orion-durable-runs) ORION_ADMISSION_TEST_DSN=<throwaway pg16> pytest tests -q
                                                                                 574 passed, 1 skipped (before review fixes)
(services/orion-durable-runs) ORION_ADMISSION_TEST_DSN=<throwaway pg16> pytest tests/test_physiology_node.py tests/test_temporal_self_regulate.py -q
                                                                                 47 passed (after review fixes, incl. both Postgres store tests)
PYTHONPATH=.:services/orion-sql-writer pytest services/orion-sql-writer/tests/test_physiology_state_sql_shape.py \
  services/orion-sql-writer/tests/test_gpu_pool_state_history.py services/orion-sql-writer/tests/test_route_map_completeness.py \
  services/orion-sql-writer/tests/test_consumer_resilience.py -q              all passed
pytest tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py \
  tests/test_agent_trace_schema_registry.py tests/test_sql_migration_drift_gate.py tests/scripts/test_schema_skew_discovery.py \
  tests/test_deploy_migration_gate.py tests/test_grammar_event_producer_catalog.py tests/test_ci_service_image_sources.py \
  tests/test_check_settings_defaults.py -q                                       230 passed
pytest tests/test_check_body_economy_contract.py -q                             36 passed
python scripts/check_body_economy_contract.py                                   OK (65 state-reading modules, all registered)
python scripts/check_inner_state_registry.py                                     OK (25 entries)
python scripts/check_definition_drift.py --gate                                  PASS (after --update; lock committed)
python scripts/check_metric_lineage.py --gate / --prompt-semantics              PASS / PASS
python scripts/check_env_key_single_source.py                                    OK (6 owned keys)
python scripts/check_env_template_parity.py                                      PASS
python scripts/check_migration_required_by.py --base origin/main                 OK
orion.sql_migration_drift.auto_apply_refusals(<migration>)                       [] (fits the post-pull allowlist)
check_service_hostname_refs / compose_no_relative_mounts / sentience_instruments --static-only /
  system_health_producers / control_surface_store_parity / scripts_dir_no_stdlib_shadow   all OK
```

## Evals run

```text
No eval harness for the Body organ yet. The live read-only smoke above (real tables, real pool RPC) is the
evidence for this patch. Follow-up: the design's scripts/analysis/replay_body_economy.py (14-day replay)
belongs to Patch A's replay item and Patch B (prices), which have something to grade.
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-durable-runs/.env -f services/orion-durable-runs/docker-compose.yml config
  -> all 9 new keys resolve (ORION_PHYSIOLOGY_ENERGY_MONTHLY_ENVELOPE_USD="" parses as None)
Live read-only smoke of every read query against conjourney: OK (~0.1 s per tick).
Live pool capacity via orion:gpu_pool:state:request RPC: OK.
UNVERIFIED: the node running inside the deployed container (not deployed; deploy is Juniper's call).
```

## Review findings fixed

Independent review (orion-repo-agent, read-only). No must-fix. Fixed:

- Finding: a lost or uncounted close made a lease look held for up to 6 h (live: phantom 3600 slot_s on `lane:metacog`); `cancelled`/`retried`/`dead_lettered` were not closes.
  - Fix: open leases now come from the pool's own `gpu_pool_leases` (status granted/recalling); the close set includes retried/cancelled/dead_lettered.
  - Evidence: `test_store_reads_on_postgres` (LOST lease with no close event is not open), live smoke metacog 6824 -> 3149.
- Finding: one lease's closes merged as a bounding box, counting a retried lease's wait.
  - Fix: true interval union per lease.
  - Evidence: `test_lane_retried_lease_does_not_count_its_wait`.
- Finding: no staleness check on lanes (a quiet feed read 0); lanes vanished when pool state was unread and nothing ran.
  - Fix: `stale:gpu_pool_events` past 900 s; `lane` / `read_failed:gpu_pool_state` instead of vanishing.
  - Evidence: `test_lane_no_reading_cases`.
- Finding: claude read `clear` while the weekly limit was unread (first hour after every restart).
  - Fix: both kinds must be read fresh; a missing one reads `unread` -> state `unknown`.
  - Evidence: `test_claude_weekly_not_yet_received_after_restart_is_unknown_not_clear`.
- Finding: attention level (all sends) and line (Hub cap, cap-counted sends) counted different things.
  - Fix: level = cap-counted sends; all sends in `detail.sent_today_all_sources`.
  - Evidence: `test_juniper_counts_cap_counted_sends_against_the_daily_cap`.
- Also found while fixing: child calls inside a durable-run hold were double counted. `gpu_pool_leases.hold_lease_id` now excludes them (live: 15 children in the last hour). Evidence: `test_lane_child_calls_inside_a_hold_are_not_counted_twice`.
- Nits fixed: 0-slot lanes keep a 0 line; open-lease count moved to state warnings; level-xor-no_reading is a schema validator; unpriced energy is a state warning; a failed tick is retried at the next tick, not on every chat-turn step.
- Not fixed (nit): `ClaudeLimitCache` keys by kind only (the producer config has one window per kind today).
- Housekeeping: the reviewer left two read-only probe scripts in `/tmp/physio/` inside the live `orion-athena-durable-runs` container. They are inert and vanish on the next rebuild; not removed here (a write into a production container).

## Restart required

Merge #2862 first (this PR is stacked on it), then this PR. From the clean primary `main` checkout after `git pull` (the post-pull hook auto-applies the additive migration):

```bash
scripts/safe_docker_build.sh orion-sql-writer up -d --build
scripts/safe_docker_build.sh orion-durable-runs up -d --build
```

Deploy order: sql-writer first (consumer), then durable-runs (producer); the reverse order only loses rows until sql-writer is up, nothing is rejected (new channel). Verify: `curl -fsS localhost:8124/health | jq .temporal_self.physiology` and `SELECT count(*), max(generated_at) FROM physiology_state;`.

## Risks / concerns

- Severity: low. Concern: `claude` reads `no_observation` until `COCREATION_SIGNALS_CLAUDE_LIMIT_ENABLED=true` on orion-cocreation-signals (left off; it mounts Juniper's transcript tree, so flipping it is her call). Mitigation: the reading says so explicitly.
- Severity: low. Concern: heat duplicates arousal's cabinet input (by design; named in the inner-state entry). Mitigation: no reader combines them yet.
- Severity: low. Concern: energy is a partial understatement when an early-month hour has no rate at all (flagged `energy_partial:unpriced_kwh`); recent hours are priced at the newest rate because RMP prices arrive about a day late (`carried_rate_kwh`).
- Severity: low. Concern: `physiology_state` has no retention (~720 rows/day). Mitigation: small rows; add to the retention loop when it matters.
- Dependency: the `config/physiology/contract.yaml` entries depend on #2862 (this branch is stacked on it).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2863 (stacked on #2862; GitHub retargets it to main when #2862 merges)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
