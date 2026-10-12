## Summary

- The shared bus listener (`Hunter` in `orion/core/bus/bus_service_chassis.py`) no longer writes one log line per received message unless the service sets `LOG_LEVEL=DEBUG`.
- One shared fix covers all 19 services that use `Hunter`; no per-service edits.
- The first-message-after-subscribe line and the once-a-minute intake summary still log at INFO.
- New gate test `orion/core/bus/tests/test_hunter_intake_log_quiet.py`, wired into the `orion-sql-writer-tests` workflow (which already triggers on `orion/core/bus/**`).

## Outcome moved

Per-message intake lines no longer fill the shared journald cap (PR #2602) and push out other services' history.

Why it still printed after #2607: #2607 moved the line to `logger.debug`, but the chassis logs through **loguru**, whose default stderr sink is DEBUG. Most services only configure stdlib `logging` (`orion-sql-writer`: `logging.basicConfig(INFO)`) and never touch loguru, so the "DEBUG" line printed anyway. Only services that call `loguru.logger.remove()` and re-add a leveled sink (only signal-gateway among Hunter services, after #2607) were quiet. The fix makes the decision in the chassis from the service's own `LOG_LEVEL` env var, so it no longer depends on how each service set up logging.

Live measurement, 10-minute window, 2026-10-11 ~23:50 UTC (`journalctl CONTAINER_NAME=<c> --since -10min`):

| container | total lines | Hunter per-message lines | expected total after |
|---|---|---|---|
| orion-athena-sql-writer | 4,748 | 3,190 DEBUG (+27 INFO first/summary) | ~1,560 (-67%) |
| orion-athena-state-service | 116 | 116 INFO (still on pre-#2607 image, started 10-11 00:33) | ~10 |
| orion-athena-biometrics | 88 | 77 DEBUG | ~11 |
| orion-athena-gpu-pool | 360 | 0 (summary only) | unchanged |
| orion-athena-cortex-exec | 35 | 2 DEBUG | ~33 |

No live container sets `LOG_LEVEL=DEBUG` (checked `docker inspect` env on every running container), so nothing is opted in by accident.

Not addressed here (separate sources, reported for follow-up):
- `orion-signal-gateway-otel-collector-1`: ~53,500 lines / 10 min of `system.cpu.utilization{...}` metric dumps. Container started 2026-10-11 00:32, before #2607 switched its debug exporter to `basic`; it needs a restart, not code.
- Other high-volume INFO lines are service-owned, not the listener: substrate-runtime biometrics sweeps, hub/vector-host uvicorn access logs, mesh-guardian httpx health polls, execution-dispatch/feedback frame-saved lines.
- `Rabbit` (RPC request handler) still logs each request at INFO; RPC volume is low and not in the top patterns.

## Current architecture

`Hunter._log_intake` logged first-after-subscribe at INFO, every other message at `logger.debug` via loguru, plus a 60 s INFO summary. Whether the debug line printed depended on each service's loguru configuration, which most services never set.

## Architecture touched

- `orion/core/bus/bus_service_chassis.py` only (shared chassis). No bus channels, schemas, or env keys changed.

## Files changed

- `orion/core/bus/bus_service_chassis.py`: `intake_log_each_enabled()` (true only when `LOG_LEVEL=DEBUG`); `Hunter` skips the per-message line otherwise.
- `orion/core/bus/tests/test_hunter_intake_log_quiet.py`: gate test.
- `.github/workflows/orion-sql-writer-tests.yml`: runs the gate test.
- `docs/superpowers/pr-reports/2026-10-12-bus-listener-log-volume-fleet-pr.md`: this report.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: per-message `Hunter intake channel=...` lines need `LOG_LEVEL=DEBUG` in every Hunter service (previously in signal-gateway only).
- Compatibility notes: runbooks that grep `Hunter intake channel=` will see only the first-after-subscribe line and `Hunter intake summary` counts unless the service runs with `LOG_LEVEL=DEBUG`.

## Env/config changes

- Added keys: none (reuses existing `LOG_LEVEL`)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/core/bus/tests/test_hunter_intake_log_quiet.py            -> 15 passed
pytest orion/core/bus/tests/ services/orion-sql-writer/tests/test_consumer_resilience.py tests/test_hunter_reconnect.py -> 82 passed
services/orion-signal-gateway: pytest -q                                -> 38 passed
Mutation: per-message line unconditional again (pre-fix behavior)       -> 4 gate cases FAIL (as intended)
```

The gate attaches a loguru sink at DEBUG on purpose (that is what an unconfigured service looks like) and checks: unset/INFO/info/WARNING `LOG_LEVEL` -> exactly one intake line (first-after-subscribe, INFO) for 40 messages; `LOG_LEVEL=DEBUG` -> 40. A third case fails if any Hunter service's `.env_example`/`docker-compose.yml` (or the root `.env_example`) defaults `LOG_LEVEL` to DEBUG.

## Evals run

```text
None. Log-volume change only; no cognition or quality behavior moved. Live before-numbers above; after-numbers are expected, UNVERIFIED until services restart.
```

## Docker/build/smoke checks

```text
Not run (task: do not deploy). Chassis code is baked into each image, so services need a rebuild to pick it up.
```

## Review findings fixed

Review subagent: no must-fix findings.

- Finding: restart loop would be refused by `safe_docker_build.sh` in the primary checkout.
  - Fix: added `ORION_ALLOW_SHARED_CHECKOUT_WRITE=1` prefix.
  - Evidence: `scripts/safe_docker_build.sh:48`.
- Finding: DEBUG-default regex missed quoted list items, `${X-DEBUG}`, other-var indirection, `export`; only scanned `docker-compose.yml`.
  - Fix: looser comment-aware pattern, scans `docker-compose*.y*ml`, added shape test cases.
  - Evidence: `test_debug_default_pattern_shapes` (9 cases pass).
- Finding: report credited vision services, which do not use Hunter.
  - Fix: corrected to signal-gateway only.

## Restart required

From the primary checkout on main after merge and `git pull --ff-only`:

```bash
for s in orion-sql-writer orion-state-service orion-biometrics orion-cortex-exec orion-gpu-pool orion-signal-gateway; do ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh $s up -d --build; done
```

Remaining Hunter services pick it up on their next rebuild. Separately, restart the otel collector to load #2607's `basic` exporter: `docker restart orion-signal-gateway-otel-collector-1`.

## Risks / concerns

- Severity: low
- Concern: services that name their level key differently (e.g. no `LOG_LEVEL`) cannot opt back into per-message lines without setting `LOG_LEVEL=DEBUG`.
- Mitigation: per-minute summary still gives counts by channel and kind; setting `LOG_LEVEL=DEBUG` restores per-message lines.

## PR link

TBD
