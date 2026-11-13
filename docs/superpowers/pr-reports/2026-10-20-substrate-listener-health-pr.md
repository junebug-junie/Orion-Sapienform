## Summary

- substrate-runtime's `/health` now shows whether each bus listener is subscribed. If any listener is down, `/health` reports degraded.
- The existing alert monitor raises a critical `substrate_listeners_down` card in the hub's attention box when a listener is still down after its recheck delay. Before this change, nothing read `/health`, so a dead listener paged no one.
- The shared helper `run_subscription` records each listener's state: subscribed, message count, failure count, last failure, and how long ago it last got a message. Any service that uses the helper can report the same thing.
- `run_subscription` now also takes several channels at once and glob patterns. The follow-up PRs need this to move the remaining 41 bare listener loops onto the helper.

## Outcome moved

Before this change, a listener that died left the service looking healthy. On 2026-10-20 chat lost reflection for hours, and nothing showed it until someone read the logs. With this change, a listener that cannot reconnect shows on `/health` within seconds. It raises an alert if it stays down past the recheck delay.

## Current architecture

`/health` returned only the grammar-truth status. `HealthMonitor` alerted on grammar truth and on node availability. Neither checked whether the bus listeners were alive.

## Architecture touched

- `orion/core/bus/resilience.py`: listener state registry, `subscription_health()`, and multi-channel/pattern support.
- orion-substrate-runtime: the `/health` route, `HealthMonitor.run_checks`, and the README.

## Files changed

- `orion/core/bus/resilience.py`: adds `SubscriptionState`, `subscription_health()` and multi-channel/pattern support. It also warns when a name is registered twice while the first is still running.
- `orion/core/bus/tests/test_resilience.py`: tests for health through a drop and recovery, backoff, and multi-channel patterns.
- `services/orion-substrate-runtime/app/main.py`: `/health` adds `grammar_ok`, `listeners_ok`, `listeners_down` and `listeners`. `ok`/`status` now count listeners.
- `services/orion-substrate-runtime/app/health_monitor.py`: new `substrate_listeners_down` critical check.
- `services/orion-substrate-runtime/tests/test_health_listeners.py`, `tests/test_health_monitor.py`: route and monitor tests.
- `services/orion-substrate-runtime/README.md`: documents the fields and what they do not prove.

## Schema / bus / API changes

- Added: `/health` fields `grammar_ok`, `listeners_ok`, `listeners_down`, `listeners`.
- Removed: none
- Renamed: none
- Behavior changed: `/health` `ok`/`status`/`degraded` now include listener state. I searched for consumers (compose healthcheck, mesh roster, hub, scripts) and found none.
- Compatibility notes: `run_subscription` signature is unchanged for existing callers. The new `patterns` argument defaults to off.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/core/bus/tests                     -> 112 passed
pytest services/orion-substrate-runtime/tests   -> 17 failed, 480 passed, 1 collection error
  (the same 17 + 1 fail on origin/main; nothing new)
mutation: removing `state.subscribed = False` on failure -> the drop test fails
```

## Evals run

```text
No eval harness covers listener liveness. Live check after deploy, below.
```

## Docker/build/smoke checks

```text
After deploy: curl -s localhost:8115/health | jq '.listeners_ok, .listeners_down, (.listeners|length)'
expect: true, [], 11
```
UNVERIFIED until this is deployed.

## Review findings fixed

- Finding: the health test passed even with the "mark down on failure" line deleted.
  - Fix: a new test covers subscribed, then a drop, then down, then back up.
  - Evidence: the mutation check above.
- Finding: nothing reads `/health`, so a dead listener still paged no one.
  - Fix: `HealthMonitor` now has a `substrate_listeners_down` critical check, with the existing recheck debounce and edge trigger.
  - Evidence: `test_health_monitor_flags_a_down_listener`.
- Finding: two live listeners with the same name would hide one of them.
  - Fix: a warning is logged when that happens.
- Finding: the monitor reads the registry from a thread.
  - Fix: the registry is copied before iterating.
- Finding: a hung handler and a stopped listener are easy to misread.
  - Fix: documented in the docstring and README.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```
Run this from the primary main checkout after merge.

## Risks / concerns

- Severity: low
  - Concern: a listener stopped on purpose at runtime would keep health degraded. Today nothing stops a listener at runtime, so this cannot happen yet.
  - Mitigation: documented.
- Severity: low
  - Concern: `subscribed=True` does not prove messages are being processed. A hung handler is not detected.
  - Mitigation: documented. If the event loop is blocked, `/health` itself times out, and that timeout is the signal.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2786
