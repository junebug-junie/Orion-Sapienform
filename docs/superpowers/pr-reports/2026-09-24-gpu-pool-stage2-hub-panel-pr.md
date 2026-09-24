## Summary

- **New Hub tab, "GPU pool".** It is also a standalone page at `/gpu-pool`. It shows:
  - every card and the roles on it, with the model the pool **discovered and confirmed** (model file, profile, slots in use, context per slot);
  - who owns each role and who may borrow it;
  - swap seats and multi-card seats;
  - the raw `config/gpu_pool.yaml`.
- **Traffic at three zoom levels:** per role; per class, holder and priority; and individual lease events.
  - **Live** comes from the bus (`orion:gpu_pool:state` and `orion:gpu_pool:event`, streamed to the browser).
  - **Historical** covers 15 minutes to 7 days and comes from `gpu_pool_events`.
  - Both include a chart of grants and p95 wait.
- **Lease walker.** Click any event and you see that lease's actual path through the lease graph, with timings, plus replay and cancel.
- **Controls:**
  - lend and unlend gpu0;
  - operator hold and release, such as the multi-card experiment seat;
  - backfill, which previews a count first and then replays exactly what was previewed.
- **Security:**
  - Control messages to the pool are **HMAC-signed**, fresh within 60 s and single-use, so the operator secret never travels on the bus.
  - The Hub control route refuses requests shaped like cross-site forgery.
- Includes #2320, so pool grammar carries exceptions only. **Close #2320 when this merges.**

## Outcome moved

Stage 2 of `docs/superpowers/specs/2026-09-24-gpu-pool-design.md`: there is now one place to see what is on every GPU, what is waiting, what happened, and why. Before this, the only view was `curl :8127/v1/pool`.

## Current architecture

orion-gpu-pool (stage 1) runs live in observe mode on athena. All four circe LLM roles show as `confirmed`. Hub had no view of it.

## Architecture touched

- **Hub:**
  - new router and page;
  - an SSE feed subscribed to two pool channels;
  - bus RPC to the pool for state and control;
  - a read-only SQL query on `gpu_pool_events`.
- **Pool:**
  - A state request can now include the parsed YAML and one lease's history (bus, not HTTP).
  - Control messages are now signed instead of carrying the token.

## Files changed

- `services/orion-hub/scripts/gpu_pool_routes.py` (new): the feed, the page, and the `/api/gpu-pool/{state,stream,history,control}` routes.
- `services/orion-hub/scripts/operator_guard.py` (new): Hub's operator cookie and header check, moved out of `api_routes.py`, which re-imports it. All existing callers are unchanged.
- `services/orion-hub/templates/gpu_pool.html`, `static/js/gpu_pool.js`, `static/js/gpu_pool_tab.js` (new): the page, its script and the tab wiring.
- `services/orion-hub/templates/index.html`: the nav tab, the panel iframe and the script tag.
- `services/orion-hub/scripts/main.py`: starts and stops the feed and registers the routers.
- `services/orion-hub/app/settings.py`, `.env_example`: `HUB_GPU_POOL_ENABLED`, `HUB_GPU_POOL_RPC_TIMEOUT_SEC`, `GPU_POOL_OPERATOR_TOKEN`.
- `orion/gpu_pool/control_auth.py` (new): signing, verification and the nonce ledger.
- `orion/schemas/gpu_pool.py`:
  - `GpuPoolControlV1` drops `operator_token` and gains `issued_at`, `nonce` and `signature`;
  - the state request and state models gain `include_config`, `history_for`, `config`, `config_yaml` and `history`.
- `orion/gpu_pool/config.py`: keeps the YAML source text.
- `services/orion-gpu-pool/app/{runtime,main}.py`: verifies the signature, returns config and history on request, and trims grammar (#2320).
- Tests:
  - `services/orion-hub/tests/test_gpu_pool_routes.py`, `test_gpu_pool_panel_browser_smoke.py`
  - `static/js/gpu_pool.test.js`
  - `services/orion-gpu-pool/tests/test_runtime.py`
- CI:
  - `.github/workflows/orion-gpu-pool-tests.yml` gains the Hub route tests on Postgres;
  - `schedule-browser-smoke.yml` gains the panel browser smoke.
- Docs: the spec and `services/orion-gpu-pool/README.md`.

## Schema / bus / API changes

- **Added:**
  - `GpuPoolStateRequestV1.include_config` and `history_for`;
  - `GpuPoolStateV1.config`, `config_yaml`, `history_lease_id` and `history`;
  - Hub routes `/gpu-pool` and `/api/gpu-pool/*`.
- **Changed (breaking for controls):** `GpuPoolControlV1` replaces `operator_token` with `issued_at`, `nonce` and `signature`. The only producer is Hub, and it is new in this PR.
- **Compatibility:**
  - State reads send only non-default fields, so an old pool still answers them.
  - **Controls need the new pool:** deploy the pool before Hub.

## Env/config changes

- **Added keys (Hub):**
  - `HUB_GPU_POOL_ENABLED=true`
  - `HUB_GPU_POOL_RPC_TIMEOUT_SEC=5.0`
  - `GPU_POOL_OPERATOR_TOKEN=` (secret, left empty in the example)
- **`.env_example` updated:** yes.
- **Local `.env` synced** with `python scripts/sync_local_env_from_example.py --all-keys orion-hub`: all 3 keys were added. `GPU_POOL_OPERATOR_TOKEN` was set to the same value as orion-gpu-pool's.
- **Operator action needed:** Hub's `SUBSTRATE_MUTATION_OPERATOR_TOKEN` is **unset**, so Hub's operator guard answers 503 to every control. This is a pre-existing Hub-wide switch, and setting it also enables Hub's other operator-only routes, so it is Juniper's call. Until it is set, the panel is read-only.

## Tests run

```text
node --test services/orion-hub/static/js/gpu_pool.test.js      11 pass (incl. walker edges == lease_graph._TABLE)
services/orion-hub tests/test_gpu_pool_routes.py               10 passed (+ history SQL on real postgres:16)
services/orion-hub tests/test_gpu_pool_panel_browser_smoke.py   1 passed (Chromium, real template+JS: discovery cards,
    mismatch explained, live SSE event, click-to-walk, lend POST with CSRF header, history range, STALE indicator,
    recovery after the pool 504s at page load)
orion/gpu_pool/tests                                           69 passed
services/orion-gpu-pool/tests                                  27 passed (+ Postgres suite in CI) -- incl. forged /
    stale / replayed control refused, secret absent from the message
gates: definition drift, metric lineage, async routes not blocking, env template parity (hub, gpu-pool),
    gpu_pool config, bus reply channels -- all PASS
```

## Evals run

```text
python services/orion-gpu-pool/evals/run_pool_day_eval.py      VERDICT: PASS (scheduling unchanged by this PR)
```
The panel has no eval harness: it is read and control UI, and the browser smoke covers its interactions.

## Docker/build/smoke checks

```text
Rendered with the LIVE pool state in headless Chromium and inspected by eye (cards, traffic, walker).
Hub templates/static are bind-mounted; Python changes need a Hub rebuild. Not deployed (see Restart required).
```

## Review findings fixed

The review subagent reported 10 findings (0 high, 4 medium, 6 low or low-medium). All are fixed.

- **Finding (medium):** control could be reached by a cross-site no-cors POST that rode the auto-issued operator cookie.
  - Fix: require `X-Requested-With: orion-hub` and a JSON content type (the custom header forces a CORS preflight, which Hub never grants).
  - Evidence: `test_control_refuses_cross_site_shaped_requests`.
- **Finding (medium):** the panel died permanently if the pool was unreachable at page load.
  - Fix: guarded rendering, `loadConfig().finally(connect)`, and a config reload on the first state frame.
  - Evidence: the browser smoke's first `/state` returns 504 and the page recovers.
- **Finding (medium):** a new Hub broke state reads against an old pool (`extra="forbid"`).
  - Fix: send only non-default fields.
  - Evidence: `test_state_route_sends_only_non_default_fields_and_caps_lease_id`.
- **Finding (medium):** a dead pool read as "live".
  - Fix: a STALE indicator based on the pool's own `generated_at`, older than 20 s.
  - Evidence: node test plus browser smoke.
- **Finding (low-medium):** backfill Run could differ from the preview, and the preview count was silently capped.
  - Fix: Run sends exactly the frozen previewed spec, any form edit resets it, `limit=1000` is explicit, and the button shows "N+".
- **Finding (low-medium):** the operator token travelled on the bus, where bus-mirror and bus-tap see everything.
  - Fix: HMAC signing, a 60 s window and a single-use nonce.
  - Evidence: `test_control_signature_freshness_and_single_use`, and the Hub test asserts the secret is absent from the payload.
- **Finding (low):** a 7-day history query could hold a connection.
  - Fix: `SET LOCAL statement_timeout = '5s'`.
- **Finding (low):** an overlong `history_for` returned 500.
  - Fix: `Query(max_length=128)` now returns 422.
- **Finding (low):** hold assumed the class name equalled the role name, and a double-click could submit twice.
  - Fix: `holdClassFor(config, role)`, and each button is disabled while its request is in flight.
- **Finding (low):** the backfill class dropdown had no "any class" option and could go stale.
  - Fix: added "any class", and the options now rebuild from a signature of the class names.

## Restart required

```bash
# athena, after merge, in this order (controls need the new pool; reads work either way)
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-gpu-pool up -d --build
ORION_ALLOW_SHARED_CHECKOUT_WRITE=1 scripts/safe_docker_build.sh orion-hub up -d --build
# open Hub -> "GPU pool" tab (or /gpu-pool). Controls stay 503 until SUBSTRATE_MUTATION_OPERATOR_TOKEN is set in Hub's .env.
```

## Risks / concerns

- **Severity:** medium
  - **Concern:** Hub's operator cookie is issued to anyone who can load the Hub index. The operator guard therefore proves "can reach Hub", not identity. This is the existing Hub-wide model.
  - **Mitigation:** the CSRF guard blocks cross-site use. Real operator identity is a Hub-wide auth change, out of scope here.
- **Severity:** low
  - **Concern:** 7-day history is capped by the 5 s timeout. At stage-3 volume it may time out before a rollup exists.
  - **Mitigation:** add a rollup when stage 3 lands.
- **Severity:** low
  - **Concern:** the lend and hold controls only change the pool, which is still in observe mode. They affect real traffic only after the stage 3 gateway cutover.

## PR link

(see PR)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
