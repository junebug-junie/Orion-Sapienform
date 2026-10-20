# Listeners in six more services survive a Redis drop

**Depends on the `feat/substrate-listener-health` PR.** That PR adds `run_subscription` (in `orion/core/bus/resilience.py`), and this branch is stacked on it. Merge that PR first.

## Summary

- Before this patch, seven long-lived bus listeners ended for good on one Redis error. The process stayed up and looked healthy. Each listener now reconnects and resubscribes, waiting 1 s first, then doubling up to 60 s.
- Harness governor (the chat path): the chat lane, the agent lane, and the cancel listener now reconnect. The two lanes still start each turn as its own task. The legacy lock, the in-flight task set, the duplicate-turn guard, and the shutdown drain did not change, and they carry across a reconnect.
- Embodiment intents, social-memory turns, and PSU commands now reconnect. Each handler still runs to completion before the next message is read, the same as before.
- Mesh guardian: `/ready` now reports whether the equilibrium watch is subscribed right now. Before, one failure set it to "not alive" until the next restart.
- Bus mirror: after a Redis drop it now reconnects inside the process. Before, the process exited and waited for a container restart.
- Each service has one regression test. In it, the first subscribe fails like a dropped Redis and the second one delivers messages. Every new test fails against the old code.

## Outcome moved

Failure mode: "a listener dies on one Redis error and the service still looks up". This is gone for seven listeners in six services. The live case was on 2026-10-20 in orion-substrate-runtime. There, every subscription timed out at once and stayed dead until a code fix.

## Current architecture

Each site had its own loop: `async with bus.subscribe(ch) as pubsub:` followed by `get_message` polling or `async for msg in bus.iter_messages(pubsub)`. Nothing retried when the subscribe or a read raised. The task ended, and the process kept running:

- governor lanes and cancel listener: the task ended, and chat turns stopped arriving.
- embodiment and social-memory: the background task ended without a sound.
- mesh-guardian: the error was logged, `alive=False` was set, and nothing retried.
- bus-mirror: the process exited (`restart: unless-stopped` brought it back).
- PSU: the error was logged in an outer `except`, and the listener stopped.

## Architecture touched

Service code only, in the six services below. There are no contract, schema, channel, or env changes.

## Files changed

- `services/orion-harness-governor/app/bus_listener.py`: `run_bus_worker` uses `run_subscription`, with the name `harness-governor-{lane}-lane` (one per lane). The per-message logic moved into `start_turn`. It creates the task and returns None, so turns are never awaited in the read loop.
- `services/orion-harness-governor/app/cancel_listener.py`: `run_cancel_worker` uses `run_subscription`, with the name `harness-governor-cancel`.
- `services/orion-embodiment/app/worker.py`: `_consume_loop` uses `run_subscription`, with the name `embodiment-intent` and the stop event `self._stop`.
- `services/orion-social-memory/app/service.py`: `_consume_social_turns` uses `run_subscription`, with the name `social-memory-turns`. The decode and kind filter moved into `_handle_social_turn_message`.
- `services/orion-mesh-guardian/app/equilibrium_watch.py`: `watch_equilibrium` uses `run_subscription`, with the name `mesh-guardian-equilibrium`. New `equilibrium_subscribed()` reads the listener's live state.
- `services/orion-mesh-guardian/app/service.py`: `equilibrium_subscriber_alive()` now returns `equilibrium_subscribed()`. The `_equilibrium_task_alive` flag is removed.
- `services/orion-bus-mirror/app/main.py`: `mirror_bus` uses `run_subscription(patterns=True)`, with the name `bus-mirror-pattern`. The per-message insert moved into `mirror_one`.
- `services/orion-gpu-cluster-power/app/bus.py`: the PSU listener uses `run_subscription`, with the name `gpu-cluster-power-psu-command`. The decode and action check stay as they were.
- `services/*/tests/test_*resubscribe*.py` (one per service, six in total): regression tests.
- `services/orion-gpu-cluster-power/tests/conftest.py`: new. This service had no tests before.
- `services/orion-bus-mirror/tests/test_heartbeat_chassis.py`: these two tests used a subscribe that always fails to end `mirror_bus`. That subscribe is now retried, so the tests stub `run_subscription` instead. They still check the heartbeat start/stop order.

## Schema / bus / API changes

- Added: none
- Removed: none
- Renamed: none
- Behavior changed: mesh-guardian `/ready` returns 503 while the equilibrium watch is reconnecting. Before, it returned 503 for good after one failure. It can also return 503 for the moment before the first subscribe.
- Compatibility notes: channels, payloads, and handlers are unchanged.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: no
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed (no template changed)
- skipped keys requiring operator action: none

## Tests run

All commands used `/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest tests -q -p no:cacheprovider`. Each ran from `services/<svc>`, with `PYTHONPATH=<worktree root>`.

```text
service                 before (this branch, pre-patch)   after              origin/main 6b48ef9aa
orion-harness-governor  93 passed                         96 passed          -
orion-embodiment        14 failed, 80 passed              14 failed, 81 passed   14 failed, 80 passed
orion-social-memory     2 failed, 76 passed               2 failed, 77 passed    2 failed, 76 passed
orion-mesh-guardian     100 passed                        101 passed         -
orion-bus-mirror        97 passed                         99 passed          -
orion-gpu-cluster-power (no tests)                        1 passed           -
```

The 14 embodiment failures and the 2 social-memory failures are the same tests on `origin/main`. They are not touched by this patch (embodiment: `'EmbodimentWorker' object has no attribute '_abandon_own_utterances'` in `test_worker_conversation_memory.py`; social-memory: two tests in `test_social_memory_service.py`).

Mutation check: I put the old app files back and ran each new test. All failed. Five failed on the first `ConnectionError`. mesh-guardian failed at import, because `equilibrium_subscribed` did not exist yet.

## Evals run

```text
None. These services have no eval harness for listener resilience. This is transport plumbing,
so the regression tests are the check.
```

## Docker/build/smoke checks

```text
Not run. There are no dependency, Dockerfile, or compose changes; orion/core is already copied into
each image. Live reconnect after a real Redis drop is UNVERIFIED until these services are redeployed.
```

## Review findings fixed

A subagent reviewed the diff. It found no blockers.

- Finding (should): bus-mirror no longer exited on errors. A store that stays broken (disk full, corrupt db) would log once per message, write nothing, and keep heartbeating as healthy. Nothing would restart it.
  - Fix: `mirror_one` counts failed writes in a row. At `MIRROR_MAX_CONSECUTIVE_WRITE_FAILURES` (50) it raises `SystemExit`, and the container restarts it as before. A Redis drop still reconnects inside the process.
  - Evidence: `test_mirror_exits_for_a_restart_when_writes_keep_failing` passes. Without the counter it would time out.
- Finding (should): the change to mesh-guardian `/ready` was not stated.
  - Fix: stated under "Behavior changed" above. The only consumer in this repo is `scripts/smoke_mesh_guardian.sh` (with `|| true`). The compose file has no healthcheck.
  - Evidence: `rg` over the repo.
- Finding (nit): the tests covered a drop only at subscribe time, not in the middle of a session.
  - Fix: added `test_lane_worker_keeps_inflight_turns_across_a_mid_stream_drop`. A turn that is running when the socket drops keeps running. After the resubscribe, its duplicate is still refused.
  - Evidence: governor suite, 96 passed.
- Finding (nit): the governor logged "subscribed" before it had subscribed.
  - Fix: the line now says "starting lane=... / starting cancel ...". `run_subscription` logs the real subscribe.
  - Evidence: the diff.
- Finding (nit): `equilibrium_subscribed()` is keyed by one fixed listener name. Two guardian instances in one process would share it.
  - Fix: not changed. Production runs one guardian per process. The name is a parameter, and tests pass a unique one.
  - Evidence: `test_equilibrium_resubscribe.py`.

## Restart required

Each service must be rebuilt and redeployed from the clean primary `main` checkout after merge. Run one command at a time:

```bash
scripts/safe_docker_build.sh orion-harness-governor up -d --build
```

```bash
scripts/safe_docker_build.sh orion-embodiment up -d --build
```

```bash
scripts/safe_docker_build.sh orion-social-memory up -d --build
```

```bash
scripts/safe_docker_build.sh orion-mesh-guardian up -d --build
```

```bash
scripts/safe_docker_build.sh orion-bus-mirror up -d --build
```

```bash
scripts/safe_docker_build.sh orion-gpu-cluster-power up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: bus-mirror no longer exits on a single error. One failed write is logged, and that message is skipped. 50 failed writes in a row still exit the process for a restart. A Redis slow-consumer kill (the 2026-10-02 crash loop) now leads to an in-process reconnect, not a container restart. The guardian's crash-loop check will no longer see it. The guardian's slow-consumer check reads the Redis counter `client_output_buffer_limit_disconnections`, so it still alerts.
- Mitigation: the listener's state is recorded by `run_subscription` (`subscription_health()`). These services do not show it on `/health` yet. That is a follow-up.

- Severity: low
- Concern: the governor, embodiment, social-memory, bus-mirror, and PSU services do not show `subscription_health()` on their HTTP health endpoints. A listener that is reconnecting is visible in the logs (`subscription_failed`), but not on `/health`.
- Mitigation: follow-up patch to add `subscription_health()` to each service's `/health`, the same way the substrate-runtime PR does.

- Severity: low
- Concern: the PSU listener and the social-memory consumer have no stop event. They end only when cancelled, the same as before.
- Mitigation: none needed. This matches the old behavior.

## PR link

Not opened (per task instructions). Branch: `fix/listener-resubscribe-services-a`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
