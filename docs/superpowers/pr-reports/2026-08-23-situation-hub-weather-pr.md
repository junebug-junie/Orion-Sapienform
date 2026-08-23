## Summary

- Follow-up to merged PR #1845, which wired local time-of-day/day-phase/conversation-phase/presence context into the unified-turn chat prompt but explicitly left weather OFF for orion-hub, pending a vetted network dependency for that call site.
- Turns weather on: `hub_settings_to_runtime_namespace()` now reads real hub-configured weather settings instead of hardcoding it off.
- `services/orion-hub/app/settings.py` gains `ORION_SITUATION_WEATHER_ENABLED`/`LAT`/`LON`/`TTL_SECONDS` (`ORION_SITUATION_WEATHER_PROVIDER` already existed there, unused by the adapter until now).
- Fixed the same event-loop-blocking bug in `_build_environment_context` that `_build_runtime_context` already got in #1845: its blocking `urlopen` now runs via `asyncio.to_thread`, since this is awaited from orion-hub's single shared WebSocket event loop, not just cortex-exec's per-turn dispatch.
- Real weather coordinates (Utah/Ogden, openmeteo) added to `services/orion-hub/.env_example`, matching `services/orion-cortex-exec/.env_example`'s already-committed values for the same physical location — synced to the primary checkout's live `.env` in the same session (the sync script reads `.env_example` from the primary checkout, so it can't see worktree-local additions until merge; hand-edited to keep it from silently drifting until then).

## Outcome moved

Unified-turn chat prompts now carry real current weather (temperature, conditions, next-2h/6h/24h precipitation/wind, practical flags like "take umbrella"/"take jacket") when relevant, not just time/day-phase/conversation-phase/presence.

## Current architecture

`hub_settings_to_runtime_namespace()` (added in #1845) explicitly disabled weather/lab/perception for the hub call site as a documented follow-up, since none of those had a vetted runtime dependency for hub yet. `orion.situational.context._build_environment_context`/`_fetch_weather` already worked correctly for cortex-exec's call site (a plain sync `urlopen`, harmless there since that service dispatches per-turn), but had never been exercised from a single-shared-event-loop caller.

## Architecture touched

- `services/orion-hub/app/settings.py`: new Settings fields.
- `orion/situational/context.py`: `_build_environment_context` async conversion, adapter wiring.
- `services/orion-hub/.env_example`: new real config values.

## Files changed

- `services/orion-hub/app/settings.py`: added `ORION_SITUATION_WEATHER_ENABLED/LAT/LON/TTL_SECONDS` Fields + a `field_validator` turning blank-string env values into `None` for lat/lon (mirrors cortex-exec's identical validator).
- `orion/situational/context.py`: `_build_environment_context` converted sync→async; its blocking `_fetch_weather` call now runs via `await asyncio.to_thread(...)`; `hub_settings_to_runtime_namespace()` reads the four weather fields from real hub settings instead of hardcoding them off; docstring updated.
- `services/orion-hub/.env_example`: added the four new keys with real production values (lat=41.2230, lon=-111.9738, provider=openmeteo, ttl=600).
- `orion/situational/tests/test_hub_settings_adapter.py`: two new tests (`test_adapter_reads_hub_weather_config`, `test_adapter_weather_disabled_when_hub_sets_it_off`); renamed/updated the test that used to assert weather was off by default.
- `services/orion-cortex-exec/tests/test_situation_provider.py`: two new tests for the async `_build_environment_context` conversion (populated-result assertion, TTL-caching assertion) plus a `_clear_weather_cache` autouse fixture mirroring the existing `_clear_runtime_cache` one.

## Schema / bus / API changes

- Added: none (no schema/bus changes; this is config + one function's async-ness).
- Removed: none.
- Renamed: none.
- Behavior changed: unified-turn prompts can now include real weather content when the situation fragment resolves; previously always "Weather: unavailable or low-confidence; do not infer."
- Compatibility notes: cortex-exec's own call site is unaffected in behavior — same weather logic, just now awaited via `asyncio.to_thread` internally instead of blocking inline (functionally identical from that caller's perspective, since it already awaited `build_situation_for_ctx`).

## Env/config changes

- Added keys (`services/orion-hub`): `ORION_SITUATION_WEATHER_ENABLED=true`, `ORION_SITUATION_WEATHER_LAT=41.2230`, `ORION_SITUATION_WEATHER_LON=-111.9738`, `ORION_SITUATION_WEATHER_TTL_SECONDS=600`. (`ORION_SITUATION_WEATHER_PROVIDER` already existed; changed its intended default from `stub` to `openmeteo` in `.env_example`.)
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes (`services/orion-hub/.env_example`).
- local `.env` synced: hand-edited directly in the primary checkout (`/mnt/scripts/Orion-Sapienform/services/orion-hub/.env`) rather than via `scripts/sync_local_env_from_example.py`, because that script reads `.env_example` from the primary checkout and therefore cannot see this worktree's additions until merge. Confirmed live (read the file directly) that all five weather keys are present and match the new `.env_example` values.
- skipped keys requiring operator action: none.

## Tests run

```text
services/orion-cortex-exec/tests/test_situation_provider.py
services/orion-cortex-exec/tests/test_situation_conversation_phase.py
services/orion-cortex-exec/tests/test_situation_perception_context.py
services/orion-cortex-exec/tests/test_session_turn_phase.py -q
  75 passed (73 pre-existing + 2 new weather tests)

services/orion-hub/tests/test_turn_orchestrator_ws_frames.py
services/orion-hub/tests/test_handle_chat_request_orion_mode_degraded.py
services/orion-hub/tests/test_unified_orion_turn_pollution_firewall.py
services/orion-hub/tests/test_handle_chat_request_orion_mode_continuity.py
services/orion-hub/tests/test_chat_route_tagging.py
services/orion-hub/tests/test_pre_turn_appraisal_wiring.py
services/orion-hub/tests/test_chat_history_no_raw_publish.py
services/orion-hub/tests/test_endogenous_outreach.py
services/orion-hub/tests/test_presence_api.py
services/orion-hub/tests/test_situation_settings_env.py
orion/situational/tests/test_hub_settings_adapter.py
orion/harness/tests/test_harness_prefix.py
orion/harness/tests/test_harness_runner.py -q
  237 passed, 1 failed (test_harness_runner_surfaces_fcc_error_code:
  verified byte-identical failure on main HEAD in PR #1845, unrelated
  string-format assertion, pre-existing)

python scripts/sync_local_env_from_example.py
  ran clean; only pre-existing, unrelated "diverged" entries reported
  (none from this patch)
```

## Evals run

No dedicated eval harness exists for this module; covered by the gate tests above.

## Docker/build/smoke checks

Not run — Python config/async-conversion change only, no Dockerfile/compose/dependency changes. Restart required for the new env values and code to take effect.

## Review findings fixed

None material — the review pass found no correctness bugs, reuse issues, or efficiency issues (it could not execute the test suite in its own sandbox, but the test results above were independently produced in this session with the repo's real venv).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env -f services/orion-hub/docker-compose.yml up -d --build
```

(Use `scripts/safe_docker_build.sh` from a worktree, not a bare `docker compose`, per repo convention. Juniper should run this — not run myself.)

## Risks / concerns

- Severity: low
  Concern: adds a real outbound HTTPS call (open-meteo.com) from orion-hub's process on a situation-cache miss (TTL 300s) / weather-cache miss (TTL 600s), where none existed before.
  Mitigation: offloaded via `asyncio.to_thread`, bounded by a 4s `urlopen` timeout in `_fetch_weather`, wrapped in the existing fail-open try/except in `_build_environment_context` (any failure -> `available=False, source="error"`, never an exception into turn assembly).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1849

🤖 Generated with [Claude Code](https://claude.com/claude-code)
