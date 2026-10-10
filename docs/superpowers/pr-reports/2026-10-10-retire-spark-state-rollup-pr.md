# Retire the zombie spark-state rollup writer (orion-state-journaler)

## Summary

- The state journaler was still averaging Orion's old "spark" mood readings (valence, arousal, coherence, novelty) every 30 seconds and saving them to Postgres. The thing that produced those readings was deleted on 2026-07-28, so for ten weeks every saved row has been all zeros with "100% missing". This removes that writer.
- Removed: the rollup loop, both bus subscriptions (`orion:spark:state:snapshot`, `orion:equilibrium:snapshot`), the Postgres writer, and their settings, env keys and compose wiring. No fallback.
- `GET /rollups` now answers `410 Gone` with an explanation instead of serving the frozen table as if it were current.
- The container keeps only its standard bus heartbeat, so equilibrium's expected-services check does not suddenly count it as down.
- `spark_state_rollups` is **not** dropped and no rows are deleted. It is now a frozen historical table.
- Spec: `docs/superpowers/specs/2026-10-07-orion-self-calibration-design.md` (PR #2528), "Real bugs found" item 4.

## Outcome moved

Stops ~1,750 new rows/day (1,440 + 288 + 24 buckets for the 60s/300s/3600s windows, via ~8,640 upserts/day; live: 12,160 rows in the last 7 days) of fake-zero "mood" from landing in Postgres, and removes an endpoint that would have reported those zeros as Orion's current state.

## Current architecture

- `services/orion-state-journaler/app/service.py`: `StateJournaler(BaseChassis)` subscribed to the spark snapshot and equilibrium snapshot channels, buffered events in memory, and every 30s upserted one row per window (60/300/3600s) into `spark_state_rollups`.
- `app/main.py`: `GET /rollups` read that table back.

Live evidence before the patch (read-only, 2026-10-10 01:44 UTC):

```text
SELECT ... FROM spark_state_rollups ORDER BY bucket_ts DESC LIMIT 12;
2026-10-10 01:44:00+00|60|0|0|0|0|1|0.024691358024691357   (valence|arousal|coherence|novelty|pct_missing|distress)
... every recent row: spark columns 0, pct_missing 1, still written each minute
last 7 days: 12,160 rows, 0 with non-zero valence
last row with pct_missing < 1.0 ever: 2026-07-28 07:09:00+00
```

Bus check (read-only, 30s `SUBSCRIBE` against `ORION_BUS_URL`): 0 messages on `orion:spark:state:snapshot`, 2 on `orion:equilibrium:snapshot`. `PUBSUB NUMSUB orion:spark:state:snapshot` = 3 (subscribers only, nobody publishing). Code search: no publisher of `spark.state.snapshot.v1` exists; `orion-equilibrium-service` publishes `spark.signal.v1`, not this kind.

**Note on `avg_distress`:** this column was genuinely live (equilibrium distress, ~0.025). It was the only real value in the table, and nothing in the repo read it (no hub route, SQL view, MCP tool, prompt or script queries `spark_state_rollups` or `/rollups`). Equilibrium distress remains available live on `orion:equilibrium:snapshot` (consumed by mesh-guardian and others). It goes with the rollup.

## Architecture touched

- `orion-state-journaler`: now a heartbeat-only service (`HeartbeatOnly` chassis + FastAPI with a `410` `/rollups`).
- `orion/signals/registry.py`: `state_journaler` organ's `bus_channels` emptied (the adapter never matched those channels anyway; metric lock does not encode `bus_channels`, so no re-lock).

## Files changed

- `services/orion-state-journaler/app/service.py`: rollup/writer/subscriptions removed; `build_chassis()` returns `HeartbeatOnly`.
- `services/orion-state-journaler/app/main.py`: `/rollups` returns 410; lifespan drives the heartbeat chassis.
- `services/orion-state-journaler/app/settings.py`: retired fields removed.
- `services/orion-state-journaler/.env_example`, `docker-compose.yml`: retired keys removed.
- `services/orion-state-journaler/requirements.txt`: `asyncpg` dropped (no longer imported).
- `services/orion-state-journaler/README.md`: new; documents retirement and the frozen table.
- `services/orion-state-journaler/tests/test_spark_rollup_retired.py`: regression tests.
- `.github/workflows/orion-state-journaler-tests.yml`: runs them in CI.
- `tests/test_state_journaler_semantics.py`: deleted (tested the removed rollup math).
- `.env_example` (root): `SPARK_ROLLUP_TABLE`, `ROLLUP_WINDOWS_SEC`, `ROLLUP_INTERVAL_SEC`, `ROLLUP_RETENTION_HOURS` removed (only the journaler read them). `CHANNEL_SPARK_STATE_SNAPSHOT` kept: `orion-state-service` compose still reads it.
- `docs/equilibrium_service.md`: diagram edge and `/rollups` step updated.
- `orion/signals/registry.py`: see above.

## Schema / bus / API changes

- Added: none.
- Removed: journaler subscriptions to `orion:spark:state:snapshot` and `orion:equilibrium:snapshot` (neither listed the journaler in `orion/bus/channels.yaml` consumers, so no catalog edit).
- Renamed: none.
- Behavior changed: `GET /rollups` → `410 Gone` (`{"retired": true, "detail": ...}`); no more writes to `spark_state_rollups`.
- Compatibility notes: no in-repo caller of `/rollups`. Channel catalog entry for `orion:spark:state:snapshot` left alone (state-service and sql-writer still subscribe); see Risks for its stale `producer_services`.

## Env/config changes

- Removed keys (service `.env_example`): `CHANNEL_SPARK_STATE_SNAPSHOT`, `CHANNEL_EQUILIBRIUM_SNAPSHOT`, `POSTGRES_URI`, `SPARK_ROLLUP_TABLE`, `ROLLUP_WINDOWS_SEC`, `ROLLUP_INTERVAL_SEC`, `ROLLUP_RETENTION_HOURS`.
- Removed keys (root `.env_example`): `SPARK_ROLLUP_TABLE`, `ROLLUP_WINDOWS_SEC`, `ROLLUP_INTERVAL_SEC`, `ROLLUP_RETENTION_HOURS`.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys orion-state-journaler`: yes ("No changes needed"; sync never removes keys). The 7 retired keys were removed by hand from the primary checkout's `services/orion-state-journaler/.env`, backup at `services/orion-state-journaler/.env.bak-2026-10-10`.
- skipped keys requiring operator action: root `.env` still carries the 4 stale `SPARK_ROLLUP_TABLE`/`ROLLUP_*` lines (harmless, nothing reads them).

## Tests run

```text
PYTHONPATH=.:services/orion-state-journaler python -m pytest -q services/orion-state-journaler/tests
5 passed   (also 5 passed in a clean venv with only requirements.txt + pytest httpx PyYAML)

Mutation check (scratch copy, not the working tree):
  original origin/main code (+ build_chassis shim)      -> 5 failed
  heartbeat chassis that re-subscribes to spark channel -> 2 failed (first pass)
  after review fix, real start_background() path:
    _run re-subscribes to orion:spark:state:snapshot      -> 1 failed
    heartbeat loop removed (service publishes nothing)    -> 1 failed
  /rollups back to 200                                  -> 1 failed
  SPARK_ROLLUP_TABLE back in .env_example               -> 1 failed

pytest orion/signals tests/test_world_pulse_reflective_journal_bus_catalog.py tests/test_report_dead_env_keys.py
       tests/scripts/test_sync_local_env_from_example.py scripts/tests/test_check_env_template_parity.py
133 passed

check_env_template_parity PASS; check_system_health_producers OK; check_metric_lineage --gate PASS;
check_definition_drift --gate PASS; check_inner_state_registry OK; check_service_hostname_refs OK;
check_compose_no_relative_mounts PASS; check_async_routes_not_blocking OK; check_sentience_instruments --static-only OK
```

## Evals run

```text
No eval harness for orion-state-journaler; the service no longer computes anything to evaluate.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-state-journaler build   -> Image orion-state-journaler-state-journaler Built
docker run --rm --network none <image> python -c "import app.main ..."  -> HeartbeatOnly ['/openapi.json', ..., '/rollups']
Not deployed.
```

## Review findings fixed

Code review ran in a subagent against `origin/main...HEAD`. No material findings; it confirmed no remaining reader of `spark_state_rollups`, `/rollups`, port 8380 or the removed keys anywhere in the repo, the branch merges cleanly, and the metric lock is unaffected.

- Finding (minor): the "never subscribes" test drove `_run()` directly, so it never exercised the heartbeat (the one remaining job) and mostly passed regardless of the code.
  - Fix: replaced with `test_running_service_heartbeats_and_never_subscribes`, which runs the real `start_background()`/`stop()` path against a recording fake bus and asserts only `system.health.v1` from `state-journaler` is published on the health channel and `subscribe` is never called.
  - Evidence: mutations "re-subscribe in `_run`" and "heartbeat loop removed" each fail it; 5 passed in repo venv and in a clean venv.
- Finding (minor): workflow path filter missed chassis dependencies.
  - Fix: added `orion/core/bus/async_service.py` and `orion/schemas/telemetry/system_health.py`.
- Finding (minor, not fixed here): `orion/bus/channels.yaml` claims `orion-equilibrium-service` produces `orion:spark:state:snapshot`; it does not. Changing `producer_services` changes the locked `metric://bus_channel/orion-equilibrium-service/orion:spark:state:snapshot` entry, so it is left for the channel-retirement follow-up (see Risks).
- Finding (minor, not fixed here): `state_journaler` organ still declares `signal_kinds` and locked metric URNs. Documented as the container-retirement follow-up.
- Finding (minor): stale keys in local `.env`. Handled: service `.env` cleaned by hand with backup; root `.env` stale lines listed under Env/config.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only && docker compose --env-file .env --env-file services/orion-state-journaler/.env -f services/orion-state-journaler/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low. Concern: the container is now a heartbeat with no job. Mitigation: kept on purpose so equilibrium's expected-services list does not register an outage. Follow-up: retire it outright (drop `state-journaler` from `EQUILIBRIUM_EXPECTED_SERVICES`, the `state_journaler` organ in `orion/signals/registry.py` and its metric-lock entries).
- Severity: low. Concern: `orion/bus/channels.yaml` lists `orion-equilibrium-service` as producer of `orion:spark:state:snapshot`, but it does not publish that kind; state-service and sql-writer still subscribe to a silent channel. Mitigation: out of scope here; follow-up to retire the channel and its remaining consumers.
- Severity: low. Concern: `avg_distress` rollup history stops. Mitigation: it had no reader; live distress is still on the bus.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2575

🤖 Generated with [Claude Code](https://claude.com/claude-code)
