# PR: enable concurrent bus handler processing in orion-actions

Branch: `fix/actions-concurrent-handlers`

## Summary

`orion-actions`' bus consumer (`Hunter`, `orion/core/bus/bus_service_chassis.py`) was the only one of six sibling services' consumers not explicitly setting `concurrent_handlers` — silently falling back to the class default of `False` (sequential). In sequential mode, the consumer directly `await`s each handler to completion before reading the next message off **any** subscribed channel.

Discovered live 2026-07-13 while investigating a failed chat turn: `orion-actions` restarted, immediately fired two scheduled startup catch-up jobs (`daily_pulse_v1`, a daily journal trigger) each RPC-bound with a 420s timeout, and a live chat message routed to `orion:actions:manage:workflow.v1` arrived in the middle of that window. It sat completely unprocessed for ~4 minutes — until both catch-up jobs finally resolved — before `orion-actions`' loop advanced far enough to even read it, well past the caller's own (shorter) timeout, causing a visible 500 error in Hub.

## Outcome moved

Live messages on any channel no longer queue behind a long-running scheduled job's RPC wait. `orion-actions` now runs handlers concurrently, matching `orion-recall`, `orion-llm-gateway`, `orion-cortex-exec`, `orion-spark-introspector`, and `orion-cortex-orch`, all of which already do this.

## Current architecture

`Hunter._run()` (`orion/core/bus/bus_service_chassis.py:448-530`) has a `concurrent_handlers: bool = False` constructor default. When `True`, each incoming message's handler runs via `asyncio.create_task(...)`, letting the consumer loop immediately continue reading the next message. When `False` (the default, and what `orion-actions` was silently using), it directly `await`s the handler, blocking the entire consumer loop — across every subscribed channel — for that handler's full duration.

## Architecture touched

- `services/orion-actions/app/settings.py`: new `actions_concurrent_handlers` setting.
- `services/orion-actions/app/main.py`: one `Hunter(...)` call site, one new keyword argument.

## Files changed

- `services/orion-actions/app/settings.py`: `actions_concurrent_handlers: bool = Field(True, alias="ACTIONS_CONCURRENT_HANDLERS")`.
- `services/orion-actions/app/main.py`: `Hunter(..., concurrent_handlers=settings.actions_concurrent_handlers)`.
- `services/orion-actions/.env_example`, `.env`, `docker-compose.yml`: new key, default `true`.
- `services/orion-actions/README.md`: new subsection under "Daily scheduler and restarts" documenting the mechanism and the live-observed failure mode.
- `services/orion-actions/tests/test_hunter_concurrent_handlers.py` (new): settings default, direct `Hunter` construction proving the wiring contract for both `True`/`False`, and a source-inspection guard against the exact regression this PR fixes (the call site silently omitting the parameter again).

## Schema / bus / API changes

None.

## Env/config changes

- Added key: `ACTIONS_CONCURRENT_HANDLERS` (default `true`).
- `.env_example` updated: yes. Local `.env` synced by hand (confirmed key-for-key parity via diff).

## Tests run

```
$ .venv/bin/python -m pytest services/orion-actions/tests/test_hunter_concurrent_handlers.py -q
3 passed

$ .venv/bin/python -m pytest services/orion-actions/tests/ -q
1 failed, 86 passed
```

The one failure (`test_handle_envelope_world_pulse_journal.py::test_handle_envelope_routes_world_pulse_run_result_to_dispatch_journal`) is pre-existing — confirmed identical on a clean checkout via `git stash` before touching anything, unrelated to this change.

## Evals run

No eval harness exists for this service; this is a concurrency-configuration fix, not a model-quality question.

## Docker/build/smoke checks

```
$ docker compose -f services/orion-actions/docker-compose.yml config --quiet   # exit 0
$ docker compose -f services/orion-actions/docker-compose.yml build             # Image Built
$ docker compose -f services/orion-actions/docker-compose.yml up -d             # Started
$ curl -fsS http://localhost:7160/health                                       # {"ok":true,"service":"actions",...}
```

**Live verification, not just "it started":** confirmed concurrent execution directly in the post-restart logs — at `01:47:23.224` an RPC wait began for one handler (`corr_id=2f719116...`), and 135ms later, **while that RPC was still pending**, a second RPC wait began for a different handler (`corr_id=7f2482f7...`). Under the prior sequential mode, starting a second RPC wait before the first resolves is structurally impossible — the consumer loop would still be blocked inside `await self.handler(env)` for the first message. This is direct, positive proof `concurrent_handlers=True` is actually in effect, not just configured. Zero real errors in the post-restart logs (one pre-existing, unrelated cosmetic logging-format warning present on every startup, confirmed identical before this change).

**Safety check performed before deploying:** enabling concurrency across this service's entire handler set means multiple handlers can now interleave at `await` points. Checked both stateful, disk-persisted components (`WorkflowScheduleStore`, `SchedulerCursorStore`) — both guard their in-memory state with `threading.RLock()` around purely synchronous methods, with zero `await` inside any critical section (confirmed via grep across both files). Python's single-threaded event loop means no coroutine can interleave mid-critical-section regardless of `concurrent_handlers`, so this fix introduces no new data-race risk for either store. Also checked `main.py` for unprotected module-level mutable state — none found.

## Review findings fixed

No formal multi-angle review run for this PR (small, single-line, well-precedented change identical to a pattern already in production in five sibling services) — instead did a targeted, deliberate concurrency-safety check (above) before deploying, given the risk of enabling concurrency service-wide without checking for races is exactly the kind of thing that would matter here.

## Restart required

Already done as part of this change:

```bash
docker compose -f services/orion-actions/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: low
  - Concern: no other orion-actions handler beyond the two stateful stores was individually audited line-by-line for concurrency safety (only checked for unprotected module-level mutable state, which found none).
  - Mitigation: this is the identical configuration change already running successfully in five sibling services handling comparable message volume; the two components with actual persisted state were directly verified safe.

## PR link

Branch pushed to `origin/fix/actions-concurrent-handlers`. `gh` was not authenticated in this environment for the prior PRs either — open manually at:
https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/actions-concurrent-handlers
