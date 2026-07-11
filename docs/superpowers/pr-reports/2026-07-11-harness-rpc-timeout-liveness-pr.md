## Summary

- `HarnessGovernorClient.run()` (services/orion-hub/scripts/harness_governor_client.py) no longer gives up on a unified turn after one fixed `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` (960s) wait — it extends the wait in that same increment, up to a new `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC` ceiling (default 3600s), as long as the harness governor is still emitting steps for that turn.
- Liveness is read from `HarnessStepRelay` (services/orion-hub/scripts/harness_step_relay.py) via a `_last_seen` timestamp per correlation_id (`seen_recently()`/`forget()`), checked against a fixed `HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC` (600s, deliberately generous — a single slow tool call can legitimately go minutes without an intermediate step), and bounded by both a TTL sweep and a hard entry-count cap (`HUB_HARNESS_STEP_RELAY_LIVENESS_TTL_SEC` / `HUB_HARNESS_STEP_RELAY_LIVENESS_MAX_ENTRIES`), matching this service's existing `CognitionTraceCache`/`SignalsInspectCache` cap+TTL pattern.
- The retry loop reuses the bus's shared, pooled RPC-worker connection (`fork_rpc_client`/`start_rpc_worker=True`) when available, instead of opening a dedicated ad-hoc Redis pubsub connection per turn — avoiding connection exhaustion under concurrent long-running turns. When no worker is available it falls back to `pubsub.get_message(timeout=...)` polling, which avoids externally cancelling a live redis-py socket read on retry (a protocol-desync risk the first version of this fix had).
- Wired the HTTP `/api/chat` (`mode=orion`) entry point (services/orion-hub/scripts/api_routes.py) to the same `harness_step_relay`/`harness_rpc_bus` the WebSocket path already used — previously that entry point silently got zero benefit from this fix.

## Outcome moved

`orion-hub` turns that do heavy tool use (many Bash/Read/Agent/ToolSearch steps) no longer surface `Turn error (harness): harness_rpc_timeout` and drop the eventual real answer just because they cross the fixed RPC wait budget — as long as the governor keeps producing steps, the wait keeps extending. A genuinely stalled governor still fails fast. This was confirmed live: the bug that motivated this PR was caught red-handed in this session's own correlation_id (`docker logs orion-athena-hub`/`orion-athena-harness-governor` showed the hub's RPC wait timing out at 03:54:00 while the governor kept emitting `harness.run.step.v1` events for that same turn past 03:57:49).

## Current architecture

`orion-hub`'s `turn_orchestrator.execute_unified_turn()` published a `harness.run.request.v1` envelope and waited once, with a single fixed timeout (`HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC`), for a reply on `orion:harness:run:result:<corr_id>`. If the harness governor (a separate service, `orion-harness-governor`) hadn't replied by then, the hub gave up and returned a hard `turn_error`/`harness_rpc_timeout` frame — even if the governor was still actively running and would have produced a real answer minutes later. That answer, once produced, was silently dropped since nothing was listening for it anymore.

## Architecture touched

- `services/orion-hub/scripts/harness_governor_client.py` — RPC wait is now a bounded, liveness-gated retry loop with two wait strategies (`_run_via_worker` reusing the forked bus's shared connection, `_run_via_ad_hoc_subscribe` as a fallback).
- `services/orion-hub/scripts/harness_step_relay.py` — `_last_seen` liveness bookkeeping bounded by both a TTL sweep and a hard `OrderedDict` entry cap.
- `orion/hub/turn_orchestrator.py` — wires `harness_step_relay.seen_recently` into `HarnessGovernorClient.run()` as the liveness check; cleans up bookkeeping in the `finally` block.
- `services/orion-hub/scripts/api_routes.py` / `services/orion-hub/scripts/main.py` — HTTP chat entry point now gets the same wiring as the WebSocket path; `HarnessStepRelay` constructed with the new TTL + cap settings.
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example` — four new config keys (below).
- `scripts/sync_local_env_from_example.py` — new keys added to `SYNC_EXACT` so local `.env` stays in sync on future `.env_example` edits.
- `services/orion-hub/pytest.ini` — explicit `asyncio_mode`/`asyncio_default_fixture_loop_scope` (was silently relying on defaults, producing a deprecation warning on every async test).
- `requirements-dev.txt` — added `pytest-asyncio`, which the existing (and new) `@pytest.mark.asyncio` hub test suite required but this repo's root dev-deps file never declared; async hub tests could not run at all in a from-scratch environment before this.

## Files changed

- `services/orion-hub/scripts/harness_governor_client.py`: bounded liveness-extending retry loop, shared-worker-connection reuse with ad-hoc-subscribe fallback, ceiling clamp fix, liveness_check exception guard
- `services/orion-hub/scripts/harness_step_relay.py`: `_last_seen` liveness tracking with TTL sweep + hard entry-count cap (`OrderedDict`)
- `orion/hub/turn_orchestrator.py`: wires liveness_check through to `HarnessGovernorClient.run()`
- `services/orion-hub/scripts/api_routes.py`: HTTP chat entry point now passes `harness_step_relay`/`harness_rpc_bus`
- `services/orion-hub/scripts/main.py`: `HarnessStepRelay` constructed with the new TTL/cap settings
- `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`: new config keys
- `scripts/sync_local_env_from_example.py`: new keys added to `SYNC_EXACT`
- `services/orion-hub/pytest.ini`: explicit asyncio config
- `requirements-dev.txt`: added `pytest-asyncio`
- `services/orion-hub/tests/test_harness_governor_client_liveness.py` (new), `services/orion-hub/tests/test_hub_harness_step_relay.py`: regression tests, including dedicated coverage for the shared-worker-connection path and the entry-count cap

## Schema / bus / API changes

- Added: none (reuses existing `harness.run.request.v1` / `harness.run.v1` / `harness.run.step.v1` channels and schemas — no `orion/bus/channels.yaml` or `orion/schemas/registry.py` changes needed)
- Removed: none
- Renamed: none
- Behavior changed: `HarnessGovernorClient.run()` can now wait longer than `HUB_HARNESS_GOVERNOR_RPC_TIMEOUT_SEC` for a reply (up to `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC`) when the governor is observably still active; the HTTP chat entry point now also benefits from liveness-extended waits.
- Compatibility notes: `HarnessGovernorClient.run()`'s new `liveness_check` kwarg defaults to `None` (old single-wait behavior unchanged for any other caller). `OrionBusAsync.rpc_request()` and its ~130 other call sites repo-wide were **not** touched.

## Env/config changes

- Added keys: `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC` (default 3600), `HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC` (default 600), `HUB_HARNESS_STEP_RELAY_LIVENESS_TTL_SEC` (default 7200), `HUB_HARNESS_STEP_RELAY_LIVENESS_MAX_ENTRIES` (default 2000)
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes (all four keys present, confirmed via `grep HUB_HARNESS services/orion-hub/.env`)
- skipped keys requiring operator action: none

## Tests run

```text
./scripts/test_hub.sh services/orion-hub/tests/test_harness_governor_client_liveness.py \
  services/orion-hub/tests/test_hub_harness_step_relay.py -q --tb=short
13 passed, 1 warning
(6 in the liveness file: extend-while-alive, give-up-when-not-alive, no-liveness-check
 preserves old behavior, max-wait ceiling, fixed-window-not-shrinking-poll_sec, and the
 new shared-worker-connection path; 7 in the step-relay file including the new
 max-entries-cap regression test)

./scripts/test_hub.sh services/orion-hub/tests -q --tb=line   (full suite)
53 failed, 729 passed, 4 skipped
(diffed the failure set against a clean `main` baseline run: zero new failures
 introduced by this branch; all 53 are pre-existing, environment-specific failures
 in this sandbox unrelated to harness/turn_orchestrator code, e.g.
 test_memory_api.py::test_memory_cards_returns_503_when_pool_unconfigured expects an
 unconfigured DB pool but this container has a live one)
```

## Evals run

orion-hub has no eval harness (`find services/orion-hub -iname "*eval*"` returns nothing). Flagging per repo contract rather than claiming eval coverage. Follow-up: add one if/when this turn-orchestration path gets a quality-oriented eval surface (e.g. measuring how often long turns actually complete vs. time out, once this ships).

## Docker/build/smoke checks

Live root-cause confirmation was done against the running `orion-athena-hub` / `orion-athena-harness-governor` containers (`docker logs`, `docker ps`) before any code changed — see Outcome moved above. Did not rebuild/restart the containers as part of this PR (see Restart required below).

## Review findings fixed

Two independent 8-angle review passes ran on this branch (the second because a separate concurrent Claude Code session was found to be editing the same working directory during the first pass — see Risks/concerns).

- Finding: repeated external `asyncio.wait_for` cancellation of a live redis-py `listen()` read across retries risked corrupting the shared pubsub connection's protocol state
  - Fix: switched to `pubsub.get_message(ignore_subscribe_messages=True, timeout=...)` polling — the same primitive this codebase's own RPC worker (`_run_rpc_only`) already uses for repeated bounded waits
  - Evidence: `services/orion-hub/scripts/harness_governor_client.py::_run_via_ad_hoc_subscribe`
- Finding (raised independently 5 times across both review passes): the retry loop always opened a dedicated ad-hoc Redis pubsub connection instead of reusing the shared `fork_rpc_client` worker connection, risking connection exhaustion under concurrent long-running turns
  - Fix: added `_run_via_worker()`, which reuses `OrionBusAsync`'s existing `_pending_rpc`/`_rpc_subscribe`/`_rpc_lock` worker machinery (via `asyncio.shield` so a per-chunk timeout can't cancel the shared future) whenever the bus was forked with a live RPC worker; falls back to the ad-hoc path only when there's no worker
  - Evidence: `services/orion-hub/scripts/harness_governor_client.py::_run_via_worker`; `test_run_uses_shared_worker_connection_when_available`
- Finding: `HarnessStepRelay._last_seen` had no hard size cap, only a time-gated TTL sweep — a burst of unique correlation_ids within one sweep interval (or before any entry individually crosses the TTL) could still grow it unboundedly, and this codebase already has an established cap+TTL pattern (`CognitionTraceCache`, `SignalsInspectCache`) this didn't follow
  - Fix: `_last_seen` is now an `OrderedDict` with a hard `last_seen_max_entries` cap (evicting the least-recently-touched entry first via `move_to_end`), on top of the existing TTL sweep
  - Evidence: `services/orion-hub/scripts/harness_step_relay.py`; `test_harness_step_relay_last_seen_bounded_by_max_entries`
- Finding: the fixed liveness recency window (120s default) was too tight — a single slow tool call can legitimately exceed it without emitting an intermediate step, causing exactly the premature timeout this PR was meant to fix
  - Fix: raised the default to 600s and documented the tradeoff explicitly
  - Evidence: `services/orion-hub/app/settings.py`, `services/orion-hub/.env_example`
- Finding: liveness window (`within_sec`) was originally the same variable as the shrinking per-retry poll timeout, so a genuinely active governor could read as "not alive" near the ceiling
  - Fix: decoupled into a fixed `HUB_HARNESS_GOVERNOR_LIVENESS_WINDOW_SEC`
  - Evidence: `services/orion-hub/scripts/harness_governor_client.py`; `test_liveness_check_receives_fixed_window_not_shrinking_poll_sec`
- Finding: `liveness_check` callback had no exception guard, so a bug in a future consumer would crash `run()` instead of degrading to the old give-up behavior
  - Fix: wrapped in try/except (`_liveness_alive`), logs and treats as not-alive
  - Evidence: `services/orion-hub/scripts/harness_governor_client.py`
- Finding: `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC` ceiling could be silently exceeded by an oversized explicit `timeout_sec`
  - Fix: clamp poll size down to the ceiling instead of expanding the ceiling
  - Evidence: `services/orion-hub/scripts/harness_governor_client.py`
- Finding: HTTP `/api/chat` (`mode=orion`) never wired `harness_step_relay`/`harness_rpc_bus`, so that entry point got zero benefit from this fix
  - Fix: wired both through, matching the WebSocket path
  - Evidence: `services/orion-hub/scripts/api_routes.py`
- Finding: `services/orion-hub/pytest.ini` had no `asyncio_mode`/`asyncio_default_fixture_loop_scope`, producing a deprecation warning on every async test in the suite
  - Fix: added explicit config
  - Evidence: `services/orion-hub/pytest.ini`

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build hub-app
```

## Risks / concerns

- Severity: low (process hygiene, not a code defect)
  - Concern: partway through this task, a separate concurrent Claude Code session was found to be editing files in this same shared working directory (not an isolated worktree) — `ps aux` showed 3 active `claude` processes. Several edits I initially attributed to "my own prior work" (the `get_message`-based rewrite, the TTL sweep, new settings) were actually made by that other session. Everything landed in a single commit (`12c915e1`) on this branch; a full second independent 8-angle review was run on the final combined diff specifically because of this to make sure nothing was missed in the handoff.
  - Mitigation: none needed going forward for this PR (the combined result was independently re-reviewed and tests pass), but future parallel work on this repo should use separate worktrees per the repo's own contract (`git worktree add ...`) to avoid this recurring.
- Severity: low-medium
  - Concern: `HarnessStepRelay._run()` (unchanged by this diff) has no restart/watchdog — if its background subscribe loop dies (broad except, logs and returns), `_last_seen` stops updating fleet-wide, and every subsequent turn's liveness check reads `alive=False`, collapsing back to the old fixed-timeout behavior process-wide (not worse than before this PR, but silently loses the benefit with only one ERROR log line at the moment of the crash).
  - Mitigation: none in this PR (would need a supervisor/health-check, a separate concern). Follow-up if this is observed in practice.
- Severity: low
  - Concern: `AgentStepRelay` (the parallel relay for agent-chain steps) didn't get the same liveness/TTL/cap treatment; if the same RPC-timeout bug exists on that path it isn't fixed here.
  - Mitigation: no evidence yet that this bug has been reported for that path; follow-up if it surfaces.

## PR link

Branch pushed: `fix/harness-rpc-timeout-liveness` → https://github.com/junebug-junie/Orion-Sapienform/pull/new/fix/harness-rpc-timeout-liveness (PR not yet opened — `gh` CLI is not authenticated in this environment; run `gh auth login` or open the compare link above manually).
