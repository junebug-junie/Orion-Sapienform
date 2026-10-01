# GPU pool stage 6.3: move every `GET /routes` reader to pool state

## Summary

- Everything that asked the gateway "which model is on which route" now asks the GPU pool directly. The gateway's answer was built from pool state anyway, and stage 6.5 deletes it.
- The code that turns pool state into a per-route view moved out of the gateway into `orion/gpu_pool/route_view.py`. Readers build it from one `orion:gpu_pool:state` RPC (remote procedure call over the bus) that asks for the pool's config too, so no reader needs its own copy of `config/gpu_pool.yaml`. The gateway's `/routes` calls the same function until 6.5.
- Readers moved: the Hub Compute picker and attach-image vision flag, the situation brief's "which model am I" line (Hub and cortex-exec), and the harness's pre-turn model line and context window (harness-governor). Also context-exec's route health check, which was deleted, and two manual scripts.
- When the pool can't be asked, every reader says so: all routes are `unknown`, and there is no model, context size or vision flag. A route default is never guessed. Each reader has a test for this.
- The gateway's `/routes` stays, but it now counts and logs every read by caller and User-Agent, so 6.5's "24 h of zero reads" can be measured.

## Outcome moved

- **Before:** about 140 `/routes` reads an hour. The Hub made about 236 an hour from the host network (live, 2 h window, 2026-09-30: 472 reads) and harness-governor about 3.5 an hour (7 reads). After this deploys, the expected rate is 0, and the counter proves it.
- **Held harness turns now size their context window from the role they were granted, not from the route's default role.** Before, a turn holding `agent-gpu2` was budgeted from whatever `/routes` said about `agent`. PR #2399 had already fixed the prompt line for held turns. This fixes the window the same way.
- **One pool read per harness turn instead of two `/routes` reads.** The prompt line and the motor's window share the state the runner read.

## Current architecture

- The gateway's `pool_placement.build_routes_compat` produced `/routes` from a cached pool-state RPC.
- The Hub proxied it (`llm_gateway_client.fetch_routes`, aiohttp, polled every 30 s per open tab) and returned 502 when the gateway was down.
- `orion/situational/context.py::_fetch_runtime_context` did a blocking `urlopen` of `/routes` on a thread.
- `fcc_motor.probe_route_runtime` did an httpx `GET /routes` twice per turn: once for the prompt and once for the window.
- context-exec's `fetch_route_status_map` did an httpx `GET /routes` per run. Its container is not running.

## Architecture touched

- `orion/gpu_pool` (a new shared view module, plus `include_config` on `fetch_pool_state`).
- orion-hub, orion-harness-governor, orion-cortex-exec (through the shared `orion/situational` and `orion/harness` code), orion-context-exec and orion-llm-gateway.
- Bus catalog: orion-cortex-exec is added as a requester of `orion:gpu_pool:state:request` and a consumer of its reply channel.

## Files changed

- `orion/gpu_pool/route_view.py` (new): the per-route view, moved from the gateway. It builds from state plus the config the pool sent. With no state or no config, every route is `unknown`. Entries gain `role`, the pool role the route lands on right now (only set when the route is up).
- `orion/gpu_pool/placement.py`: `fetch_pool_state(..., include_config=False)`.
- `orion/gpu_pool/tests/test_route_view.py` (new): the view built from the pool's own payload matches the gateway generator, down and unknown cases, and the RPC shape.
- `services/orion-llm-gateway/app/pool_placement.py`: `build_routes_compat` delegates to `route_view` (its 100-line copy is deleted).
- `services/orion-llm-gateway/app/routes_compat_reads.py` (new) and `app/main.py`: a per-read counter and WARNING log line on `/routes`, plus `GET /debug/routes-compat-reads`.
- `services/orion-hub/scripts/llm_gateway_client.py`: the catalog comes from pool state over the Hub's forked RPC bus. It is cached 10 s (2 s after a failure), one read at a time. It never raises and adds `source`.
- `services/orion-hub/scripts/api_routes.py`: `/api/llm-routes` no longer returns 502. An unreachable pool gives a 200 with every lane `unknown`.
- `services/orion-hub/scripts/main.py`: situation stores are bound after the RPC fork (`rpc_bus=rpc_bus`).
- `services/orion-hub/{app/settings.py,.env_example,docker-compose.yml,README.md}`: `HUB_LLM_GATEWAY_URL` is removed (its only reader was the `/routes` proxy).
- `orion/situational/runtime_route_view.py` (new) and `state_buses.py`: a pool-state source for the runtime line, bound once per process (on the RPC fork when there is one).
- `orion/situational/context.py`: `_fetch_runtime_context` is now async and reads the pool view (`source="gpu_pool"`). The `llm_gateway_base_url` setting is removed.
- `orion/harness/fcc_motor.py`: `probe_route_runtime` and `probe_current_served_model` take `pool_state` and make no HTTP call. New `held_role_window`. `run_fcc_turn(pool_state=...)` sizes a held turn from its grant. The httpx import is gone.
- `orion/harness/runner.py`: one pool-state read per turn (held, or a pool-backed route), passed to the prompt probe and the motor.
- `services/orion-context-exec/app/{llm_profile_resolver.py,runner.py,settings.py}`, `.env_example`, `docker-compose.yml`, README: `fetch_route_status_map`, the fallback path, `LLMProfileUnavailableError` and `CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED` are deleted. Under the pool, a refusal happens at dispatch time, and the old check treated "unreachable" as "all available" anyway.
- `scripts/smoke_llm_gateway_routes.py`: checks the pool route view over the bus (the `--gateway-url` option is removed).
- `services/orion-mind/scripts/verify_mind_llm_e2e.sh`: reads the Hub's `/api/llm-routes` and asserts `source=gpu_pool`.
- `orion/bus/channels.yaml`: orion-cortex-exec added on the pool state request and reply channels.
- Docs and comments: `orion/fcc/context_budget.py`, `orion/harness/prefix.py`, `orion/llm/routes.py`, harness-governor settings and README, and the cortex-exec and Hub READMEs.
- Tests: see below.

## Schema / bus / API changes

- **Added:**
  - `GET /debug/routes-compat-reads` (gateway).
  - `source` field on the Hub's `/api/llm-routes` payload.
  - `role` field on `/routes` and route-view entries.
- **Removed:** none on the wire. `GET /routes` stays until 6.5.
- **Renamed:** none.
- **Behavior changed:**
  - `/api/llm-routes` returns 200 with `unknown` lanes instead of 502 when the pool can't be asked. The picker already renders `unknown`.
  - `RuntimeContextV1.source` is `gpu_pool` (it was `orion-llm-gateway`).
  - Held harness turns use their granted role's `ctx_per_slot` as the window.
  - **context-exec:**
    - It no longer fails fast when the pool says a route is down. The call is dispatched, and the pool queues or refuses it at dispatch time. A queued call can wait up to `CONTEXT_EXEC_LLM_TIMEOUT_SEC` (120 s).
    - The smolagents path no longer forces `agent` on an unavailable route.
    - `fallback_used` / `fallback_reason` are still emitted in `runtime_debug`, but they are now always `False` / `None`.
    - The service is not running today (spec, 2026-09-30).
  - **Hub, pool unreachable:**
    - The attach-image button greys out, because vision is unknown.
    - The lend toggle keeps its last state.
- **Compatibility notes:**
  - `include_config` has been in `GpuPoolStateRequestV1` since stage 2, so the live pool already answers it.
  - The payloads are dicts and the only new key is additive. No schema-model change.
  - `orion:gpu_pool:state:request` is already cataloged. Only the service lists changed.

## Env/config changes

- **Added keys:** none.
- **Removed keys:**
  - `CORTEX_EXEC_LLM_GATEWAY_URL` (orion-cortex-exec). Its only reader was the situational `/routes` read.
  - `HUB_LLM_GATEWAY_URL` (orion-hub). `HUB_LLM_GATEWAY_TIMEOUT_SEC` stays, because the concept classifier reads it.
  - `CONTEXT_EXEC_LLM_PROFILE_FALLBACK_ENABLED` (orion-context-exec).
- **Renamed keys:** none.
- **`.env_example` updated:** yes (hub, context-exec, cortex-exec).
- **Local `.env` synced with `python scripts/sync_local_env_from_example.py`:** run. Parity check: PASS (94 services). The sync script adds keys but does not delete them, so all three removed keys are still sitting in the local `.env` files, where they do nothing (`extra="ignore"`). PR 6.6's `report_dead_env_keys.py --apply` removes them.
- **Skipped keys requiring operator action:** none.

## Tests run

```text
orion/gpu_pool/tests                                    337 passed
orion/harness/tests                                     383 passed
orion/situational/tests                                  83 passed
services/orion-llm-gateway/tests                        306 passed
services/orion-harness-governor/tests                    58 passed
services/orion-durable-runs/tests                       246 passed, 71 skipped
services/orion-cortex-exec/tests/test_situation_provider.py   22 passed
services/orion-hub/tests/{test_llm_gateway_client_routes,test_llm_route_selector,
  test_situation_state_buses_bound,test_unified_turn_gpu_placement,test_gpu_pool_routes,
  test_runtime_activity_routes}.py                      pass except 3 pre-existing selector
                                                         failures (identical on main)
services/orion-hub/tests (full)                         no new failures vs main. Two tests
                                                         (test_memory_graph_routes roundtrip,
                                                         test_substrate_effect_pipeline) fail
                                                         depending on test order, and fail the
                                                         same way on main when run alone
services/orion-context-exec/tests (full)                no new failures vs main (29 pre-existing,
                                                         missing smolagents etc. locally)
python scripts/check_env_template_parity.py             PASS
python scripts/check_bus_reply_channels.py              16 prefixes resolved, 0 uncovered
python scripts/check_definition_drift.py --gate         PASS (lock regenerated after merging
                                                         origin/main: orion-cortex-exec joins the
                                                         pool-state request/reply catalog entries)
After review fixes: orion/{gpu_pool,situational,harness}/tests 810 passed; gateway 306 passed;
cortex-exec situation + admission_cue 55 passed; Hub route client 23 passed
```

New tests, one per reader, each including the pool-unreachable case:

- **route view:** `orion/gpu_pool/tests/test_route_view.py`
- **Hub:** `TestCatalogFromPoolState`, `test_api_llm_routes_is_200_with_unknown_lanes...` and `test_hub_no_longer_reads_the_gateway_routes_endpoint`
- **situational:**
  - `test_runtime_context_reads_pool_state_not_the_gateway`
  - `test_runtime_context_degrades_when_pool_unreachable` (two cases)
  - `test_runtime_context_unbound_bus_is_unavailable`
  - `test_runtime_route_view_rides_the_rpc_fork_when_one_is_given`
- **harness:**
  - `test_probe_is_unknown_without_usable_pool_state`
  - `test_held_role_window_is_the_grants_ctx_not_the_route_default`
  - `test_a_held_turn_sizes_its_window_from_the_granted_role_not_the_route`
  - `test_no_pool_state_falls_back_to_the_env_ceiling`
  - `test_unheld_pool_turn_reads_pool_state_once...`
  - `test_pool_unreachable_unheld_turn_states_no_model...`
- **context-exec:** `test_gateway_routes_reader_is_gone` and `test_resolve_makes_no_network_call`
- **gateway counter:** `test_get_routes_is_counted_and_logged_per_caller` and `test_routes_compat_caller_table_is_bounded`

## Evals run

```text
None. This is a pure transport move with no quality-scored behavior; the touched services have
no eval harness for route catalogs. The acceptance check is the live 24 h zero-read window below.
```

## Docker/build/smoke checks

```text
Not deployed (per task). Pre-deploy read-only evidence of who reads /routes today:
$ docker logs --since 2h orion-llm-gateway 2>&1 | grep 'GET /routes' | <group by client IP>
    472 172.18.0.1     (host network = orion-athena-hub; gpu-cluster-power is the only other
                        host-network container and contains no /routes reference)
      7 172.18.0.68    (orion-athena-harness-governor)
```

## Review findings fixed

The code-review subagent found no blockers. It ran main's old `build_routes_compat` and the new `build_route_view` on 3,000 random pool states and the no-state case, and the output differed only by the new `role` key. It also searched `.py`, `.js`, `.sh`, compose and Makefiles and found no `/routes` reader left.

- **Finding:** `CORTEX_EXEC_LLM_GATEWAY_URL` became dead config. Its only reader was the situational `llm_gateway_client` URL. Its comments still said "probes GET /routes".
  - **Fix:** removed it from cortex-exec `settings.py`, `.env_example`, `docker-compose.yml`, README and the test fixtures, and rewrote the comments. Ran the env sync.
  - **Evidence:** env parity PASS. `rg CORTEX_EXEC_LLM_GATEWAY_URL` finds nothing.
- **Finding:** when the pool can't be reached, the Hub's lend toggle showed "closed". The unknown catalog has `gate_open: null`, and `=== true` read that as false, which is a guess.
  - **Fix:** `chatBurstGateFromCatalog` only acts on a real boolean. On `null` the toggle keeps its last known state.
  - **Evidence:** a node check (true -> open, false -> closed, null -> unchanged) and `test_lend_toggle_ignores_an_unknown_gate_instead_of_rendering_closed`. `node --check app.js` passes.
- **Finding:** the parity test compared the new generator with itself.
  - **Fix:** froze a golden fixture of main's (a005658db) `build_routes_compat` output over 8 states: all up with gpu0 lent and not lent, a spill of agent to agent-gpu2 and metacog to fast, chat only with and without the lend, chat down while lent, no roles, and no state. `test_view_matches_the_old_gateway_generator_output` asserts the new view matches it, ignoring `role`.
  - **Evidence:** 8 cases pass. A mutation (chat-burst `operator_closed` changed to `down`) fails 4 of them.
- **Finding:** the Hub's `default_route` always fell back to "quick" while looking as if the gateway had reported it.
  - **Fix:** it is now an explicit `HUB_DEFAULT_ROUTE = "quick"`, documented as the Hub's own constant.
  - **Evidence:** `test_default_route_is_the_hubs_own_constant_matching_the_composer` also checks it matches app.js's `HUB_COMPUTE_DEFAULT`.
- **Finding:** the context-exec behavior change was not stated in the report.
  - **Fix:** it is now stated under "Behavior changed" below.
- **Nits fixed:**
  - The stale "/admission + /routes" runtime-activity comments in Hub `.env_example` and `settings.py`.
  - The leftover `HUB_LLM_GATEWAY_URL` in `test_turn_orchestrator_ws_frames.py`.
- **Nits not fixed (noted):**
  - The situational pool read identifies itself as `orion-situational`, not the host service's name. It is trace metadata only.
  - `test_no_pool_state_falls_back_to_the_env_ceiling` only checks that the guard did not fire. It does not check the ceiling value it fell back to.
  - On the gateway-only path, an `unknown` entry still names its first role's `served_by`/`upstream`. That is inherited behavior, and it goes away with 6.5.

## Restart required

Deploy order. Every step is independent (the pool already answers `include_config`), but the gateway goes first so the counter is live and sees each reader drop off:

```bash
# 1. gateway: counter + log on /routes (and route_view delegation)
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
# 2. hub: Compute catalog, vision flag, situational runtime line
scripts/safe_docker_build.sh orion-hub up -d --build
# 3. harness-governor: pre-turn model line + motor window
scripts/safe_docker_build.sh orion-harness-governor up -d --build
# 4. cortex-exec (shared orion/situational runtime line; all cortex-exec containers)
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
# context-exec: not running, no deploy needed.
```

The 24 h zero-read window (acceptance check 4, before PR 6.5 merges):

```bash
# Exact count since the gateway process started, per caller (IP + User-Agent):
curl -s http://127.0.0.1:8210/debug/routes-compat-reads | jq '{counting_since, uptime_sec, reads_total, last_read_at, by_caller}'
# The window holds when reads_total == 0 and uptime_sec >= 86400.
# If the gateway restarted inside the window, the log line survives restarts:
docker logs orion-llm-gateway 2>&1 | grep -c routes_compat_read
# ...and the uvicorn access log is the independent cross-check:
docker logs orion-llm-gateway 2>&1 | grep -c '"GET /routes'
```

User-Agent tells readers apart even though every athena-host caller shares one source IP:

- `aiohttp`: the old Hub picker.
- `Python-urllib`: the old situational line.
- `python-httpx`: fcc_motor or context-exec.
- `curl`: a person or script.

## Risks / concerns

- **Severity: low.** A stale browser tab keeps working. The payload shape is unchanged (plus `source`), and `/api/llm-routes` never goes through the gateway any more.
- **Severity: low.** Each Hub and harness pool read now carries the pool's parsed config (about 5 KB). The Hub caches it for 10 s, so it is one read per 10 s however many tabs are open. The harness makes one read per turn.
- **Severity: low.** The smoke script no longer checks the gateway's `default_route`. `/routes` was the only place that value was visible, and nothing reads it: the Hub picker uses its own `HUB_COMPUTE_DEFAULT`.
- **Severity: info.** Readers moved: 5 (Hub catalog and vision flag, situational runtime line, fcc_motor prompt and window, context-exec resolver, and 2 manual scripts). durable-runs had no reader left (stage 4.5 removed it). Its frozen `http:llm-gateway:8210/routes` transport-baseline key is 6.7's cleanup.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2444

🤖 Generated with [Claude Code](https://claude.com/claude-code)
