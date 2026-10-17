## Summary

- On shutdown the governor now stops taking new requests, then waits for turns already running to finish before it cancels the rest.
- Chat lane waits up to 90 s. Agent lane (curiosity, ~40 min runs) waits up to 900 s.
- The cancel listener now stays up until both lanes have drained, so durable-runs can still cancel a draining turn.
- Docker waits `stop_grace_period` 1000 s before SIGKILL (default was 10 s, which would have voided any drain).

## Outcome moved

A governor redeploy killed a 229-step run at once (2026-10-16 22:37Z; trace `116458b1`, grounding `interrupted_resumable`). Its retry was then refused by the resume window gate, so the run's unwritten work was lost. Shutdown cancelled every in-flight turn immediately (`run_bus_worker` `finally`), and Docker's default 10 s grace period would have SIGKILLed anything left. Turns now finish instead of being killed; resume stays as the net for what outlasts the budget.

## Current architecture

`app/main.py` `lifespan` set `bus_stop_event`, then `task.cancel()` on both lane workers and the cancel listener. `run_bus_worker`'s `finally` cancelled all in-flight turn tasks. Compose had no `stop_grace_period`.

## Architecture touched

`services/orion-harness-governor` only: `app/bus_listener.py`, `app/main.py`, `app/settings.py`, `docker-compose.yml`, `.env_example`.

## Files changed

- `app/bus_listener.py`: `run_bus_worker(drain_sec=0.0)`; waits for in-flight tasks up to `drain_sec` before cancelling. Logs `governor_drain_begin` / `governor_drain_end`.
- `app/main.py`: passes per-lane budgets; waits for both lanes concurrently; separate `cancel_stop_event` so the cancel listener stops last.
- `app/settings.py`: `HARNESS_SHUTDOWN_DRAIN_SEC_CHAT` (90), `HARNESS_SHUTDOWN_DRAIN_SEC_AGENT` (900).
- `docker-compose.yml`: `stop_grace_period: ${HARNESS_STOP_GRACE_SEC:-1000}s`, drain env passthrough.
- `.env_example`: three keys with the coupling rule (grace must exceed drain + ~60).
- `tests/test_shutdown_drain.py`: 6 tests.

## Schema / bus / API changes

None. No channel or schema change.

## Env/config changes

- Added keys: `HARNESS_SHUTDOWN_DRAIN_SEC_CHAT=90`, `HARNESS_SHUTDOWN_DRAIN_SEC_AGENT=900`, `HARNESS_STOP_GRACE_SEC=1000`.
- `.env_example` updated: yes. Local `.env` synced with `python3 scripts/sync_local_env_from_example.py --all-keys orion-harness-governor` (verified present). Note: the script's default service list does NOT include the governor, so the plain command silently adds nothing for it. Parity PASS.
- Skipped keys: none.

## Tests run

```text
services/orion-harness-governor: 91 passed (was 85; +6 new)
```

New tests: running turn finishes during drain; turn still running after the budget is cancelled; zero budget cancels at once; no new request taken while draining; lifespan keeps the cancel listener until both lanes finish; compose grace period defaults exceed the drain defaults and every key matches across settings, `.env_example` and compose.

## Evals run

No eval harness for shutdown. Compose render check: `docker compose config` renders `stop_grace_period: 16m40s` and both drain keys.

## Docker/build/smoke checks

UNVERIFIED live. The deploy that installs this still stops the OLD container with the old code. The first restart after this one shows `governor_drain_begin lane=agent in_flight=N` then `governor_drain_end ... finished=N cancelled=0` in the governor log. SIGTERM reaching the Python process (not a shell wrapper) is assumed from the Dockerfile CMD; the post-restart log line confirms it.

## Review findings fixed

(filled after review)

## Restart required

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

## Risks / concerns

- Severity: medium. While draining, the lane does not consume new requests. Redis pub/sub does not queue, so chat requests that arrive in that window time out, and the agent lane's durable runs rely on their own retry. A redeploy can now take up to ~15 min if a long run is in progress; the compose recreate waits for it.
- Mitigation: drain budgets are env-tunable (0 restores the old immediate cancel), and chat drains for only 90 s.
- Not covered: crash or host reboot. Those still rely on the resume mechanism (#2676), which refuses transcripts too big for the new lane window.

## PR link

(see PR)
