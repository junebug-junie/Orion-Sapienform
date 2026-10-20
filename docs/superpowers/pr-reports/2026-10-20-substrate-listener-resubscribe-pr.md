## Summary

- Each Redis error ended the bus listener it hit in orion-substrate-runtime, and the listener did not come back. The process stayed up and looked healthy.
- Every startup blocks the event loop for about 3 minutes (Falkor hydrate). This times out every subscription, so 10 of 11 listeners died on each boot, finalize_appraisal included.
- Chat then shipped every turn as "Response delivered without reflection (substrate appraisal unavailable (RPC timeout))".
- New `orion.core.bus.resilience.run_subscription` resubscribes with backoff. All 11 substrate-runtime listeners now use it.

## Outcome moved

A listener now recovers on its own after a Redis drop or a startup freeze. Before, recovery needed a code change, because a restart freezes again and the listeners die again. Verified live on 2026-10-20: a restart at 05:09 reproduced the same 8 deaths at 05:12.

## Current architecture

Each listener had its own copy of the subscribe loop. Only `_rpc_health_listener_loop` had a resubscribe wrapper. The other loops caught only `asyncio.TimeoutError`. `redis.exceptions.TimeoutError` / `ConnectionError` are different classes, so they went past that handler and ended the task silently.

Live evidence (2026-10-20, container `orion-athena-substrate-runtime`):
- 04:44:10 → 04:46:38: no log lines at all (event loop blocked). The gap ends with `falkor_substrate_hydrate_stats`.
- 04:46:38–42: `system_health_heartbeat_start_failed`, then `listener stopped` for finalize_appraisal, post_turn_closure, goal_context, ask_answered, field_channel_anomaly, cabinet_ambient_spike, perception_prediction_error and codebase_delta. Only rpc_health resubscribed.
- Governor: 2 of 2 finalize appraisal RPCs after that point timed out at 5 s. The substrate's own RPC delivery monitor reported `worst_hop=orion:substrate:finalize_appraisal:request worst=2/2`.

## Architecture touched

- `orion/core/bus/resilience.py`: new `run_subscription` helper, next to `publish_with_reconnect`.
- orion-substrate-runtime: the 4 module listeners and the 7 worker listener loops now call that helper.

## Files changed

- `orion/core/bus/resilience.py`: `run_subscription` (resubscribe with backoff; backoff resets only after a message arrives; a session whose reads hang is reopened after 5 overruns).
- `orion/core/bus/tests/test_resilience.py`: tests for a mid-stream drop, a subscribe failure, handler errors, hanging reads and cancellation.
- `services/orion-substrate-runtime/app/{finalize_appraisal,post_turn_closure,goal_context,ask_answered}_listener.py`: these now use the helper.
- `services/orion-substrate-runtime/app/worker.py`: these 7 listener loops now use the helper: rpc_health, perception PE, drive state, field anomaly, cabinet spike, codebase delta and perception ingest.
- `services/orion-substrate-runtime/tests/test_listener_resubscribe.py`: regression test. It fails on origin/main and passes here.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: listeners resubscribe instead of exiting. Log lines now come from the `orion.bus.resilience` logger, as `<name> subscribing|stopped|handle_failed|subscription_failed channel=...`. Old labels such as `substrate_rpc_health_listener_failed` are gone. A repo search found nothing that reads them.
- Compatibility notes: other copies of this loop exist elsewhere, for example `orion/autonomy/goal_state_listener.py`. They are not touched here.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed
- skipped keys requiring operator action: none

## Tests run

```text
pytest orion/core/bus/tests                                   -> 109 passed
pytest services/orion-substrate-runtime/tests (PYTHONPATH=.:services/orion-substrate-runtime)
  branch: 17 failed, 471 passed, 1 collection error
  main  : 17 failed, 470 passed, 1 collection error   (same pre-existing failures; +1 = new test)
test_listener_resubscribe.py against the origin/main listener -> FAILED (reproduces the bug)
```

## Evals run

```text
No eval harness covers bus listener liveness. Gap: no /health field for listener state (see Risks).
```

## Docker/build/smoke checks

```text
Live diagnosis only (logs above). Post-deploy check: after startup completes, `docker logs orion-athena-substrate-runtime | grep "finalize_appraisal listener subscribing"` should show a resubscribe after the freeze, and the next chat turn should not show "delivered without reflection".
```
UNVERIFIED until this is deployed.

## Review findings fixed

- Finding: the first retry wait ignored `max_backoff_sec`.
  - Fix: backoff starts at `min(1.0, max_backoff_sec)`.
  - Evidence: the backoff tests now take about 0.02 s, down from 1 s.
- Finding: the backoff reset on every successful SUBSCRIBE, so an instant-drop loop retried every second and flooded the log.
  - Fix: the backoff resets only after a real message arrives.
  - Evidence: `test_run_subscription_resubscribes_after_a_mid_stream_drop`.
- Finding: a session whose reads always overran the 1.2 s guard was polled forever with no log line.
  - Fix: after 5 overruns in a row, the session is reopened through the logged backoff path.
  - Evidence: `test_run_subscription_reopens_a_session_whose_reads_hang`.
- Finding: nothing tested that cancellation propagates.
  - Fix: `test_run_subscription_propagates_cancellation`.
  - Evidence: passes.

## Restart required

```bash
scripts/safe_docker_build.sh orion-substrate-runtime up -d --build
```
Run this from the primary main checkout after merge.

## Risks / concerns

- Severity: medium
  - Concern: the root freeze is not fixed. Startup still blocks the event loop for about 3 minutes, from a synchronous Falkor hydrate that runs once per store build. Chat turns during that window still degrade.
  - Mitigation: follow-up to move the hydrate off the loop thread.
- Severity: low
  - Concern: `/health` does not report listener liveness. A listener that keeps retrying shows up only in the logs.
  - Mitigation: follow-up.
- Severity: low
  - Concern: the 6 other converted worker loops have no test of their own. They share the helper, which is tested.
  - Mitigation: shared code path.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2779
