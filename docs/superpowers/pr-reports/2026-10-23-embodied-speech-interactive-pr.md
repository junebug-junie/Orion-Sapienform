## Summary

- Orion's speech in AI Town moves to a new pool route, `quick_embodied`: the same fast lane, at GPU pool tier 1 (`interactive`) instead of tier 4 (`background`).
- `quick_embodied` is system-only, like `quick_turn`: it is hidden from the Hub picker, and Hub and orion-actions refuse it.
- One explicit exception, `orion.llm.routes.CORTEX_OVERRIDE_SYSTEM_ROUTES`, lets cortex-exec accept it as a caller override. Embodiment's speech arrives only that way.
- Embodiment's `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE` now defaults to `quick_embodied` (settings, `.env_example`, compose).

## Outcome moved

Failure mode: Orion never answered anyone in town. On 2026-10-23, every reply was a `chat_quick` call on `quick_background`. It queued in the fast lane behind `system` work (`orion-topic-foundry` leases), and the embodiment worker's 45 s RPC timed out (`embodiment_speech_rpc_failed`, then `embodiment_speech_empty_reply_skipped`). Gateway round trips for those calls took 19–175 s. The lane itself answered an Orion-sized prompt (3,853 tokens) in 2.2 s.

## Current architecture

Embodiment sends cortex-exec `ctx["llm_route"]`. `normalize_llm_route` accepts it unless it is unknown or in `SYSTEM_LLM_ROUTES`. The gateway maps the route to a pool class and priority from `config/gpu_pool.yaml`. The pool tiers, highest first: urgent, interactive, perception, system, background.

## Architecture touched

`config/gpu_pool.yaml`, `orion/llm/routes.py`, `services/orion-cortex-exec/app/executor.py`, and `services/orion-embodiment` config. No bus or schema change.

## Files changed

- `config/gpu_pool.yaml`: `quick_embodied: {class: fast, priority: interactive}`.
- `orion/llm/routes.py`: accepted, display order, system-only; `CORTEX_OVERRIDE_SYSTEM_ROUTES`; `normalize_llm_route(..., allow_system=)`; import-time check that the exception names only system routes.
- `services/orion-cortex-exec/app/executor.py`: the override passes `allow_system`.
- `services/orion-hub/scripts/cortex_request_builder.py`: a refused `options.llm_route` is dropped instead of forwarded.
- `services/orion-embodiment/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: default `quick_embodied`.
- Tests: `orion/llm/tests/test_routes.py`, `orion/gpu_pool/tests/test_route_priorities_explicit.py`, `orion/gpu_pool/tests/test_route_view.py`, `services/orion-llm-gateway/tests/test_anthropic_passthrough.py`, `services/orion-cortex-exec/tests/test_executor_llm_route_override.py`, `services/orion-embodiment/tests/test_worker_conversation_memory.py`.

## Schema / bus / API changes

- Added: route `quick_embodied`.
- Removed: none.
- Renamed: none.
- Behavior changed: `normalize_llm_route` takes an optional `allow_system`. Existing callers are unchanged.
- Compatibility notes: Hub's picker and request body, and orion-actions, still refuse every system route.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- Value changed: `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE` `quick_background` → `quick_embodied` (`.env_example`).
- local `.env` synced: the key already exists with the old value, which the sync script does not overwrite. The live `services/orion-embodiment/.env` is updated to `quick_embodied` after cortex-exec and the gateway are deployed. Until then, cortex-exec would refuse it.
- skipped keys requiring operator action: none.

## Tests run

```text
pytest orion/llm/tests orion/gpu_pool/tests -> 528 passed
cortex-exec tests/test_executor_llm_route_override.py tests/test_pre_turn_appraisal_rpc.py -> 19 passed
llm-gateway tests/test_anthropic_passthrough.py -> 25 passed
hub tests/test_cortex_request_builder.py tests/test_llm_route_selector.py -> 60 passed (leak test fails with the fix removed)
embodiment tests -> 82 passed, 14 failed; the same 14 fail on main (missing _abandon_* attributes in old tests), unrelated
scripts/check_env_template_parity.py -> PASS
git diff --check -> ok
```

## Evals run

None. The live check is acceptance check A3 in #2666 (20 NPC↔Orion turns with no echo), which needs Orion's replies to land.

## Docker/build/smoke checks

Deploy after merge: cortex-exec lanes, llm-gateway, gpu-pool (config), and embodiment, then the `.env` value. Live result UNVERIFIED until Orion's replies appear in town.

## Review findings fixed

- Finding (high): the live `services/orion-embodiment/.env` sets `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_background`, which overrides every new default, and the sync script won't overwrite a value that differs. So deploying alone changes nothing.
  - Fix: the deploy step edits that line, then checks two things: the container env, and a cortex-exec log line `llm_route_selected ... route=quick_embodied`.
  - Evidence: Restart section below.
- Finding (medium): Hub's builder copied `payload["options"]` and left a refused `options.llm_route` in place, so a hand-built POST /api/chat could reach `quick_embodied` through cortex-exec.
  - Fix: the builder now pops the refused value.
  - Evidence: `test_system_route_in_options_is_dropped_not_forwarded` covers every system route; it fails with the pop removed.
- Finding (medium, policy, kept): at tier 1, Orion's town speech is never shed by heat. During a lockdown it runs ahead of Hub turns on `quick`, which is tier 3 and shed. That includes Orion talking to NPCs, whose own lines are shed.
  - Decision: kept. Juniper chose tier 1 for Orion's town speech on 2026-10-23. Each line is about 2 s.
- Finding (low): deploy order. An old gateway image doesn't know the route and quietly rewrites it to `quick`.
  - Fix: the post-pull rebuild covers orion-llm-gateway and orion-gpu-pool. The `.env` flip comes after them.
  - Evidence: `rebuild_affected_services.py --list-only` lists both.
- Noted (low): `route_view` shows system-only routes as "system", the same as `quick_turn`. And a request granted after embodiment's 45 s timeout still runs. Neither is new.

## Restart required

After merge, from the clean primary main checkout on athena:

1. `git pull --ff-only`. The post-pull rebuild covers cortex-exec, llm-gateway, gpu-pool, and embodiment.
2. Set `EMBODIMENT_SPEECH_QUICK_LLM_ROUTE=quick_embodied` in `services/orion-embodiment/.env`.
3. `scripts/safe_docker_build.sh orion-embodiment up -d --force-recreate`
4. Verify: `docker exec orion-athena-embodiment env | grep SPEECH_QUICK`, then a cortex-exec-chat log line `llm_route_selected ... route=quick_embodied`.

## Risks / concerns

- Severity: low. Concern: Orion's town chat competes with Juniper's Hub turns at the same tier. Mitigation: each line is about 2 s on the lane.
- Severity: low. Concern: town speech is no longer shed by heat. Mitigation: it is short; the A0 perception load is far larger.

## PR link

(filled after push)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
