# GPU pool stage 3: every LLM call takes a lease from the pool

## Summary

- The LLM gateway no longer decides on its own where a call goes. Each call asks `orion-gpu-pool` for a lease, runs on the URL the pool granted, and hands the lease back with its outcome (ok, upstream_error, timeout, cancelled). This covers bus calls and the `/v1/messages` and `/v1/chat/completions` passthroughs.
- The gateway's hand-rolled routing layer is gone: capacity, upstream admission, priority admission, the lane gate, the admission ledger, the route catalog, and roughly 25 env keys (the `LLM_ROUTE_*` URLs, served-by values and route table JSON). Routes map to pool classes in `config/gpu_pool.yaml`. A route not listed there is refused, never guessed.
- A recall or loss stops the in-flight call. When the pool claws a card back (for example chat reclaiming gpu0) or loses a lease, the gateway stops the upstream call and releases it as `cancelled`.
- The "am I backed up" signal (`queue_contention_score`) now reads real pool queue waits. It used to read `gateway_waiting`, which was 0.0 on every one of 163,577 ticks. The metacog "waited" cue reads the pool's lease history.
- The Hub's "Lend chat GPU" button drives a pool lend/unlend of gpu0. The old gateway lane gate and chat-lane-lend code are deleted.
- `thought` stance-react makes a single attempt; the lane fallback is removed.

## Outcome moved

- One queue sees every LLM call on circe. Before this, the pool (stages 1–2) only observed, and the gateway routed blind from env config.
- The backed-up signal can now rise when real work waits and fall back to 0 when it doesn't. The old source was structurally pinned at 0.
- Oversized prompts fail fast with the real llama.cpp overflow instead of waiting out a deadline.

## Current architecture

Before: the gateway picked an upstream from `LLM_ROUTE_*` env URLs plus a route table. It used its own concurrency and priority admission, a lane gate the Hub toggled, and a durable-runs capacity call. The pool (stage 1) ran alongside in observe mode, and nothing leased from it.

## Architecture touched

- `orion-llm-gateway`: placement via `app/pool_placement.py`, passthrough via `app/passthrough_proxy.py`, stopping in-flight calls via `app/upstream_cancel.py`. `/routes` is now a compat view built from pool state.
- `orion/gpu_pool` (client and scheduler):
  - The client reports refusal reasons and a typed `PoolRpcTimeout(full=…)`, and does a bounded background withdraw after a timeout.
  - The scheduler refuses a request with `min_ctx_exceeds_class:<max>` when the prompt is bigger than any role in its class can hold.
- `orion-gpu-pool` runtime:
  - Remembers each role's last-seen context size.
  - Refuses operator holds until the pool actuates swaps itself.
  - Uses served_by `<host>-worker-<role>`.
- `orion-field-digester`: `queue_contention` source is `gpu_pool_waiting`.
- `orion-cortex-exec`: the admission cue reads SQL on `gpu_pool_events`.
- `orion-hub`: lanes come from the pool feed, and the lend button calls the pool control. The hold hint is fixed.
- `orion-thought`: the stance-react fallback is removed.
- `orion/bus/channels.yaml`: the gateway is added as a `gpu_pool:state:request` producer and `state:reply:*` consumer.

## Files changed

102 files, +3874 / −8535. The main ones:

- `services/orion-llm-gateway/app/{main.py,pool_placement.py,passthrough_proxy.py,upstream_cancel.py,settings.py}`: pool placement, overflow re-placement, recall handling.
- `services/orion-llm-gateway/app/{capacity,upstream_admission,priority_admission,lane_gate,admission_ledger,route_catalog}.py`: deleted.
- `orion/gpu_pool/{client.py,scheduler.py,discovery.py}`, `orion/schemas/gpu_pool.py` (`DiscoveredRoleV1.model_path`).
- `services/orion-gpu-pool/app/runtime.py`.
- `config/gpu_pool.yaml`: `routes:` including `agent-burst` and `chat-burst`.
- `orion/field/queue_contention.py`, `services/orion-field-digester/app/{store.py,digestion/queue_contention.py,worker.py,settings.py}`.
- `services/orion-cortex-exec/app/admission_cue.py`.
- `services/orion-hub/{scripts/runtime_activity_routes.py,static/js/app.js,static/js/gpu_pool.js}`. Also deleted `chat_lane_lend.py` and the gate routes.
- `services/orion-thought/…` (stance-react single attempt).
- `scripts/smoke_llm_gateway_routes.py`, `scripts/analysis/record_lane_occupancy.py`, `verify_mind_llm_e2e.sh`, the poacher-gate allowlist, and AI Town's `check_llm_route_not_circe.py` (now class based).
- `docs/superpowers/specs/2026-09-20-queue-contention-metric-gate.md`: the metric quality gate for the re-pointed signal.

## Schema / bus / API changes

- Added: `DiscoveredRoleV1.model_path`. The gateway is added as a producer and consumer on the existing `orion:gpu_pool:state:*` channels. The refusal reason `min_ctx_exceeds_class:<max>` is new.
- Removed:
  - Gateway `/admission`, the lane gate routes, and the Hub hold/gate endpoints.
  - The durable-runs `/capacity` call from the gateway.
- Behavior changed:
  - Gateway errors now use `raw.error=gpu_pool_unavailable` with `details.reason`.
  - A context overflow releases the lease as `ok`/`context_overflow`, not as a GPU failure.
  - `/routes` is derived from pool state.
- Compatibility: `/routes` keeps its shape (compat view) for durable-runs and scripts.

## Env/config changes

- Added keys (gateway): `GPU_POOL_CONFIG_PATH`, `LLM_GATEWAY_POOL_WAIT_SEC`, `LLM_GATEWAY_POOL_BACKGROUND_WAIT_SEC`, `LLM_GATEWAY_POOL_PASSTHROUGH_WAIT_SEC`, `LLM_GATEWAY_EXECUTOR_WORKERS_PER_ROLE`.
- Removed keys:
  - gateway:
    - route URLs: `LLM_ROUTE_*` (URLs and served-by), `LLM_GATEWAY_ROUTE_TABLE_JSON`, `ORION_LLM_{VLLM,OLLAMA,LLAMA_COLA}_URL`, `ATLAS_METACOG_SERVICE_NAME`
    - capacity and admission: `LLM_GATEWAY_CAPACITY_{ENABLED,URL}`, `LLM_GATEWAY_BACKGROUND_*`, `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`, `LLM_ALLOW_BACKGROUND_TO_CHAT_FALLBACK`
  - `CORTEX_EXEC_ADMISSION_CUE_TIMEOUT_SEC`
  - `FIELD_DIGESTER_LLM_GATEWAY_URL`
  - `STANCE_REACT_AGENT_LANE_BUDGET_SEC`
- `.env_example` updated: yes (gateway, cortex-exec, field-digester, thought).
- Local `.env` synced: yes. The new keys are in the primary gateway `.env`. Removed keys left in local `.env` files are inert (nothing reads them).
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-llm-gateway: 267 passed
services/orion-gpu-pool: 35 passed, 2 skipped
orion/gpu_pool + orion/llm + root tests touched: 153 passed (after merging main)
services/orion-field-digester: 234 passed (after merging main)
services/orion-cortex-exec (touched tests): 33 passed
services/orion-thought: 240 passed, 1 failed. The failure is pre-existing and env-sensitive: mind_base_url comes from local .env, and the file is untouched by this PR.
services/orion-hub: 2776 passed, 38 failed. The same 38 test ids run against a clean origin/main worktree: 38 failed. None come from this branch.
node --check services/orion-hub/static/js/gpu_pool.js: ok
New regression tests (each fails without its fix):
  - a briefly-down role keeps its ctx so big prompts wait for it (runtime)
  - a refusal while queued carries the pool's reason (client)
  - a short acquire timeout does not mark the pool down (gateway)
  - acquire timeout is typed full/short and withdraws in the background (client)
  - overflows release ok (bus + anthropic/openai passthroughs)
```

## Evals run

```text
The gpu_pool scheduler eval passes (stage 1 harness, re-run on this branch).
```

## Docker/build/smoke checks

```text
Static gates:
  - git diff --check, env parity, check_gpu_pool_config, check_chat_route_poachers, check_metric_lineage, check_definition_drift, check_async_routes_not_blocking, compose mount checks, hostname refs, system-health producers, registry checks: all rc=0.
  - check_sentience_instruments --static-only (the CI lane): rc=0.
  - The live lane reports 4 DRIFTs in unrelated instruments.
Docker: UNVERIFIED on this branch until the deploy below.
```

## Review findings fixed

Round 1 (c58200851):

- Finding: oversized prompts waited out the whole deadline.
  - Fix: the pool refuses at once (`min_ctx_exceeds_class`), and the gateway re-places once on the largest role.
  - Evidence: `test_nothing_big_enough_after_overflow_returns_the_overflow`.
- Finding: recalls were ignored mid-call.
  - Fix: the upstream call is stopped (socket shutdown or task cancel) and released as `cancelled`.
  - Evidence: `tests/test_pool_lease_lifecycle.py`.
- Finding: an unreachable pool cost every call a full wait.
  - Fix: acquire timeout is bounded by the deadline, plus a 5s fail-fast breaker.
- Finding: holds drained cards before swaps exist.
  - Fix: holds are refused unless in enforce mode.

Round 2 (42fb4a0b2):

- Finding: a briefly-down 131k role shrank its class's max context, so big prompts were wrongly refused.
  - Fix: the pool remembers each role's last-seen context size.
  - Evidence: the new runtime test failed before the fix (`None == 131072`).
- Finding: a queued refusal surfaced as a generic "deadline".
  - Fix: `_wait_for_grant` returns the pool's reason.
- Finding: one caller with almost no time left could trip the 5s "pool unreachable" breaker for everyone, including chat.
  - Fix: only a full-length timeout trips it. A bounded background withdraw follows any acquire timeout.
- Finding: context overflows were counted as GPU failures.
  - Fix: they are released as `ok`/`context_overflow` on both the bus and passthrough paths.
- Finding: the Hub hid the Release button for an existing hold outside enforce mode.
  - Fix: `held` is checked first.
- Finding: a docstring pointed at a non-existent test file.
  - Fix: corrected.

Round 3 (review of round 2):

- Finding: the remembered context size was written into the pool's live role table, so it also fed swap-seat and serviceability decisions. A seen-once, now-unloaded `agent-gpu2` would start raising swap requests for big leases.
  - Fix: the remembered sizes are passed to the scheduler separately (`schedule(..., seen_ctx=)`) and are used only by the "too big for this class" check.
  - Evidence: `test_a_down_roles_last_seen_context_only_stops_a_false_too_big` asserts no refusal and the same swap loads as a lease with no minimum context.
- Not bugs (the reviewer checked each):
  - A down role can never be granted: the health and slot checks are unchanged.
  - `PoolRpcTimeout` is still caught by every `TimeoutError` handler.
  - The passthrough overflow refactor keeps the same retry semantics.
  - Counting an overflow as `ok` does not affect retry or dead-letter handling: gateway leases are non-retryable.

## Restart required

Order matters: pool first, so the gateway's first leases find it.

```bash
# on circe, in /mnt/scripts/Orion-Sapienform after merge + pull
scripts/safe_docker_build.sh orion-gpu-pool up -d --build     # then set mode=enforce, re-lend gpu0 if lent
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
scripts/safe_docker_build.sh orion-thought up -d --build
scripts/safe_docker_build.sh orion-cortex-exec up -d --build
scripts/safe_docker_build.sh orion-field-digester up -d --build
```

## Risks / concerns

- Severity: high. Concern: the gateway now depends on the pool for every call. Mitigation:
  - A pool outage fails calls fast (`gpu_pool_unavailable`, reason `pool_unreachable`) and never guesses a URL.
  - The pool is deployed and has been live since stage 1.
- Severity: medium. Concern: `upstream_cancel.py` reaches into httpx's private `transport._pool._network_backend`. Mitigation:
  - httpx is pinned at 0.27.2.
  - The lease lifecycle tests exercise the real socket shutdown.
- Severity: low. Concern: a remembered context size goes stale if a role is swapped to a smaller-context model while down. Effect: a too-big prompt waits instead of being refused at once. Once the role returns, the next tick refuses it and the gateway re-places it. Placement never uses the remembered size.
- Severity: low. Concern: the background withdraw after an acquire timeout re-sends the acquire (keyed on request_id) and then cancels it. If the pool never saw the first acquire, this can briefly grant and then cancel a lease. If the pool is too slow even for the 2s withdraw, the lease is left until its 30s heartbeat expires. Mitigation: before this PR the timeout path never withdrew at all.

## PR link

(filled on open)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
