## Summary

- #3033 taught one bus reader (`iter_messages`) to notice a connection that died silently. Four other reader loops still could not notice it. This patch fixes all four.
- The check now lives in one small helper, `PubSubWatchdog` (`orion/core/bus/async_service.py`). After 30 s of silence it sends PING. If nothing comes back in 15 s, it raises `PubSubSilentError`, and each loop's existing reconnect code runs.
- `iter_messages` now uses the same helper. Its behavior does not change.
- Fixed loops:
  1. The shared RPC reply worker (`_run_rpc_only`). 9 services use it: actions, chat-memory, cocreation-signals, cortex-exec, cortex-gateway, cortex-orch, hub, llm-gateway, vision-council.
  2. `run_subscription` (`orion/core/bus/resilience.py`). About 25 services and 8 hub caches/notifiers use it.
  3. orion-thought's two hand-written loops (`bus_listener.py`, `reasoning_activity.py`).

## Outcome moved

A dead connection looks alive: every read returns "nothing yet" on time. The old guards fired only when a read ran late, so they never fired. A dead RPC reply line made every RPC call in that service time out until a restart, with no log of the cause. Now each loop logs `pubsub silent ...` and reconnects within about 45 s.

## Current architecture

- RPC reply worker: reads in 1 s slices and reconnects on a read error. A silent line never raised an error.
- `run_subscription`: reconnects after 5 reads in a row overrun 1.2 s. A silent line never overruns.
- orion-thought: checks the subscriber count with `PUBSUB NUMSUB`, but only after 30 overrunning reads. A silent line never overruns.

## Architecture touched

- `orion/core/bus/async_service.py`: `PubSubWatchdog`; `iter_messages` and `_run_rpc_only` use it.
- `orion/core/bus/resilience.py`: `run_subscription` uses it.
- `services/orion-thought/app/bus_listener.py`, `services/orion-thought/app/reasoning_activity.py`: use it.

## Files changed

- `orion/core/bus/async_service.py`: helper, and the iter_messages and RPC worker changes.
- `orion/core/bus/resilience.py`: run_subscription change.
- `services/orion-thought/app/bus_listener.py`, `reasoning_activity.py`: loop changes.
- `orion/core/bus/tests/test_pubsub_watchdog_listeners.py`: helper unit tests; dead line reopens `run_subscription` and the RPC worker; a quiet live line stays open.
- `services/orion-thought/tests/test_pubsub_silent_line_reconnects.py`: both orion-thought loops leave a dead line.

## Schema / bus / API changes

- Added: `PubSubWatchdog` (Python helper, not a schema).
- Behavior changed: every listener on these 4 paths sends one PING per 30 s of quiet on its pub/sub connection.
- Compatibility notes: no payload or channel changes.

## Env/config changes

None.

## Tests run

```text
pytest orion/core/bus/tests tests/test_bus_async_rpc_worker.py      148 passed
pytest services/orion-thought/tests                                  787 passed, 32 skipped
pytest orion/core/sql_router/tests/test_sql_router_worker_resubscribe.py tests/test_bus_async_rpc_worker.py   18 passed
services: vision-frame-router 123, llm-gateway 470, actions 281, vision-scribe 12, bus-mirror 99, vision-edge 25 passed
Mutation check: replacing the watchdog call with `msg = None` fails the 3 loop tests in
test_pubsub_watchdog_listeners.py and both orion-thought tests.
```

Not run locally (environment only, same failure on main): world-model (needs torch), cortex-exec (verb registry collision at collection), substrate-runtime `test_grammar_consumer_integration.py` (import path).

## Evals run

No eval harness covers bus transport. The live check is the one below.

## Docker/build/smoke checks

Not run. The live proof is a dead-line reconnect in a running service's log: `pubsub silent` followed by a resubscribe. UNVERIFIED until a line actually dies after deploy.

## Review findings fixed

- Finding: the RPC worker took `_rpc_lock` after every quiet read. A caller that held the lock for a long time (for example `_rpc_subscribe` waiting on a 10 s connect) would stall reply reads for other pending RPCs.
  - Fix: `PubSubWatchdog.io_due()`. The worker takes the lock only when a PING or drain is due.
  - Evidence: bus test suites pass (below).
- Finding: the orion-thought tests proved a dead line is left, but not that a quiet live line is kept.
  - Fix: `test_worker_keeps_a_quiet_live_line` for both loops (one line, at least 2 pings).
  - Evidence: 4 passed.

## Restart required

Every service that imports `orion/core/bus` picks this up on its next rebuild. The post-pull rebuild covers it after merge.

## Risks / concerns

- Severity: low. Concern: one PING per 30 s of quiet per listener connection (roughly 60 listeners). Mitigation: negligible load on Redis.
- Severity: low. Concern: Redis stream readers (XREADGROUP) were not checked. They use the command connection, which has a 60 s socket timeout. UNVERIFIED.

## PR link

TBD

🤖 Generated with [Claude Code](https://claude.com/claude-code)
