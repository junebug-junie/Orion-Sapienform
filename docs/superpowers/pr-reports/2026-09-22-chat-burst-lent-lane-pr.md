# chat-burst: lend Juniper's chat lane to the burst queue, with an email hold and a poacher gate

Branch: `feat/chat-burst-lane`

## Summary

- New gateway route `chat-burst`: the same worker as `chat`/`harness` (circe-worker-1), offered to
  the durable-runs burst queue only while an operator gate is open. System-only, lease-only.
- The gate is one Redis key on the bus, fail-closed, owned by the gateway
  (`services/orion-llm-gateway/app/lane_gate.py`): `GET/PUT /routes/chat-burst/gate`. The catalog
  reports the lane `operator_closed` while shut; every dispatch surface refuses it.
- Hub gets a "Lend chat lane: on/off" button in the composer. While on, every message typed
  into the Hub chat box (WebSocket or HTTP) is stored in chat history and emailed to Juniper via
  orion-notify instead of being sent to Orion; the UI shows a yellow held notice.
- Durable-runs lane policy declares `chat-burst` compatible with `agent`, so new agent-lane runs
  widen onto it after the usual 20-minute wait when the gate is open.
- Six non-Hub code paths that defaulted or fell back onto `chat` are re-pointed
  (metacog / quick / quick_background), and `scripts/check_chat_route_poachers.py` is a CI gate
  that refuses any new one unless allow-listed with a reason.

## Outcome moved

Live on 2026-09-21 the agent lane had 6 durable runs queued, the oldest waiting over 12 hours,
while `agent-burst` was down (GPU2 held by diffusion, `visual_baseline_urgent`) and the chat
worker sat idle. With the gate open, that queue has a second physical worker to widen onto.
Hub chat messages typed while the lane is lent are no longer lost (previously an HTTP-path
failure dropped the user message entirely).

## Current architecture

Burst capacity was one lane, `agent-burst`, borrowed physically from GPU2 by draining
diffusion. Durable admission (`orion/durable_admission/`) widened `agent`-preferring runs onto
it after `DURABLE_RUNS_WIDENING_AFTER_SEC` when the gateway catalog reported it `up`. The chat
worker was Juniper's reserved, unthrottled, `n_parallel: 1` lane and nothing could lend it.
Several services used `chat` as a default or fallback route.

## Architecture touched

- `orion/llm/routes.py` (shared route vocabulary): `chat-burst` accepted, displayed,
  `SYSTEM_LLM_ROUTES`; new `BURST_LLM_ROUTES`, `OPERATOR_GATED_LLM_ROUTES`,
  `CHAT_BURST_LENDS_ROUTE`.
- orion-llm-gateway: gate module, two endpoints, catalog status/field, dispatch guards on the
  bus/HTTP chat path and both passthroughs, `CapacityPermit` lease rule for both burst lanes,
  code default route `quick`.
- orion-durable-runs: lane policy entry; no code change (gate arrives as catalog health).
- orion-hub: gateway client gate helpers, gate proxy routes, hold module, WS + HTTP hold,
  button + JS.
- CI: `orion-static-gates.yml` runs the poacher gate; `Makefile` target.

## Files changed

- `orion/llm/routes.py`: the new route and the three sets; tests in `orion/llm/tests/test_routes.py`.
- `services/orion-llm-gateway/app/lane_gate.py`: Redis-backed operator gate, fail-closed.
- `services/orion-llm-gateway/app/main.py`: gate check in `_dispatch_chat`; `GET/PUT /routes/{id}/gate`.
- `services/orion-llm-gateway/app/anthropic_passthrough.py`, `openai_passthrough.py`: 503 `route_operator_closed`.
- `services/orion-llm-gateway/app/route_catalog.py`: `gate_open` field, `operator_closed` status.
- `services/orion-llm-gateway/app/capacity.py`: unleased calls refused on any `BURST_LLM_ROUTES` member.
- `services/orion-llm-gateway/app/settings.py`: `LLM_ROUTE_DEFAULT` code default `quick`.
- `services/orion-llm-gateway/tests/test_lane_gate.py`: 9 tests; `test_route_catalog.py` default updated.
- `services/orion-llm-gateway/.env_example`, `README.md`: route-table entry, gate docs.
- `services/orion-durable-runs/.env_example`, `README.md`, `tests/test_admission_policy.py`: policy entry + gate-as-health test.
- `services/orion-hub/scripts/chat_lane_lend.py`: lent check (3 s cache, False on error) and email hold.
- `services/orion-hub/scripts/llm_gateway_client.py`: `fetch_route_gate`, `set_route_gate`, `gate_open` passthrough, display default `quick`.
- `services/orion-hub/scripts/api_routes.py`: gate proxies; `/api/chat` hold.
- `services/orion-hub/scripts/websocket_handler.py`: hold block before cortex dispatch.
- `services/orion-hub/templates/index.html`, `static/js/app.js`: button, render, toggle, held-notice rendering.
- `services/orion-hub/tests/test_chat_lane_lend.py` (14), `test_llm_gateway_client_routes.py`; `README.md`.
- `services/orion-cortex-orch/app/decision_router.py` (`chat` -> `metacog`), `memory_extractor.py` (`chat` -> `quick_background`).
- `services/orion-embodiment/app/settings.py`, `worker.py` (`chat` -> `quick`); `services/orion-mind/app/engine.py` (`chat` -> `metacog`); `services/orion-actions/app/settings.py` (`chat` -> `metacog`).
- `scripts/check_chat_route_poachers.py`, `tests/test_check_chat_route_poachers.py`, `.github/workflows/orion-static-gates.yml`, `Makefile`.

## Schema / bus / API changes

- Added: gateway `GET /routes/{route_id}/gate`, `PUT /routes/{route_id}/gate {open, changed_by}`
  (404 for any route not in `OPERATOR_GATED_LLM_ROUTES`); Hub `GET/PUT /api/llm-routes/{route_id}/gate`.
- Added: `GET /routes` rows carry `gate_open` (null for ungated routes); new status value
  `operator_closed`. Consumers that only test `status == "up"` (durable-runs) need no change.
- Added: error type `route_operator_closed` (HTTP 503 on passthroughs; `raw.error` on the bus path).
- Added: Redis key `orion:llm_gateway:lane_gate:chat-burst` (operator state, not a bus channel).
- Added: notify event kind `orion.hub.chat.lane_lent` (email channel).
- Added: Hub `/api/chat` may return `{"held": true, "reason": "chat_lane_lent", ...}`; WS emits
  a `turn_deferred` frame with `held: true`.
- Behavior changed: gateway code default route is `quick` (live `.env` already said so).
- Compatibility: `chat-burst` is `priority: system`, so the Hub Compute picker hides it and
  `normalize_llm_route` refuses it as a caller override, same as `harness`/`agent-burst`.

## Env/config changes

- Added keys: none.
- Value changes: `LLM_GATEWAY_ROUTE_TABLE_JSON` gains the `chat-burst` entry (gateway);
  `DURABLE_RUNS_LANE_POLICY_JSON` gains `agent` + `chat-burst` (durable-runs).
- `.env_example` updated: yes, both services, with comments.
- local `.env` synced: the sync script cannot change an existing key's value, so both live
  `.env` values were edited directly to match; `python scripts/sync_local_env_from_example.py`
  and `check_env_template_parity.py` both PASS.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=.:services/orion-llm-gateway pytest services/orion-llm-gateway/tests orion/llm/tests -q
  -> 372 passed (362 gateway + 10 routes; includes 9 new gate tests)
PYTHONPATH=.:services/orion-durable-runs pytest services/orion-durable-runs/tests/test_admission_policy.py -q
  -> 10 passed (new: chat-burst widens without elastic permission, suppressed when gate closed)
PYTHONPATH=. pytest services/orion-hub/tests/test_chat_lane_lend.py services/orion-hub/tests/test_llm_gateway_client_routes.py -q
  -> 30 passed; node --check static/js/app.js ok
PYTHONPATH=.:services/orion-cortex-orch pytest tests/test_auto_router.py tests/test_memory_extractor.py tests/test_lane_routing.py -q
  -> 53 passed
pytest services/orion-mind/tests -> 98 passed; services/orion-actions/tests -> 168 passed
PYTHONPATH=. pytest tests/test_check_chat_route_poachers.py -q -> 6 passed
python3 scripts/check_chat_route_poachers.py -> PASS (36 defaults, 20 allow entries in use);
  negative proof: re-adding route="chat" in decision_router fails the gate, reverted.
Static gates run locally: env parity, hostname refs, compose relative mounts, async routes,
  control-surface parity, definition drift, metric lineage -> all PASS.
Pre-existing failures reproduced identically on main (not touched): cortex-orch 35,
  embodiment 14, hub selector/recall/UI 22. test_elastic_policy.py needs psycopg (CI installs it).
```

## Evals run

```text
No eval harness covers lane admission or the Hub hold. The durable-runs evals
(elastic_fairness.py, admission_fairness.py) need a Postgres DSN and were not run.
Follow-up: add a chat-burst case to services/orion-durable-runs/evals/admission_fairness.py.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-llm-gateway config -> valid
scripts/safe_docker_build.sh orion-llm-gateway build  -> Image orion-llm-gateway-llm-gateway Built
scripts/safe_docker_build.sh orion-hub build          -> Image orion-hub-hub-app Built

Isolated live smoke of the NEW gateway image (scratch port 18210, scratch bus intake channel,
real bus Redis 100.92.216.81, real route table):
  GET  /routes/chat-burst/gate -> open=false (missing key = closed)
  GET  /routes -> chat up, harness up, chat-burst operator_closed gate_open=false (same served_by)
  POST /v1/messages model=llamacpp/chat-burst -> 503 route_operator_closed
  PUT  /routes/chat-burst/gate open=true -> key written; production gateway's own Redis client
       reads it back; /routes -> chat-burst up gate_open=true
  POST /v1/messages (open, unleased) -> 503 gateway_capacity_rejected
       chat_burst_requires_durable_capacity_lease
  PUT  open=false -> closed; GET /routes/chat/gate -> 404. Gate left CLOSED.
Hub email hold, live: one labelled smoke message through hold_chat_message_for_email ->
  notify logged email_send_eligible / email_send_attempted / email_send_succeeded
  (event_kind orion.hub.chat.lane_lent). Juniper's inbox should have it.
UNVERIFIED (needs deploy): a durable run actually being granted chat-burst and running its FCC
  turn on it; the Hub button in a browser; the WebSocket hold loop under real traffic.
```

## Review findings fixed

See the section appended below after review.

## Restart required

```bash
# Gateway (gate endpoints, catalog, dispatch guards, route table value):
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-llm-gateway up -d --build
# Durable-runs (lane policy value only -- restart to reload .env):
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-durable-runs up -d
# Hub (button, hold path):
cd /mnt/scripts/Orion-Sapienform && scripts/safe_docker_build.sh orion-hub up -d --build
# cortex-orch, embodiment, mind, actions: code defaults only; live .env already overrides
# actions/embodiment/mind. cortex-orch's router/extractor routes DO change on restart.
```

Deploy order: gateway first (durable-runs reads its catalog), then durable-runs, then Hub.

## Risks / concerns

- Severity: should. Concern: a run first granted `chat-burst` stays pinned to it
  (`run_assignment_locked`); closing the gate pauses that run until reopened rather than
  migrating it. Mitigation: documented in durable-runs README; same retention rule as
  `agent-burst`. A follow-up could release the lease on gate close.
- Severity: should. Concern: `quality_drop: 0` for chat-burst is a declaration (35B-A3B vs the
  27B agent model); context is smaller (65536 vs 131072) and enforced live from the catalog.
  Mitigation: curiosity runs declare no minimum context today.
- Severity: should. Concern: the long-context chat poachers are allow-listed, not moved
  (curiosity supervisor batch readings, compactor digest, context-exec, stance_react default
  and its agent->chat fallback leg). Moving a non-durable caller to `agent` would starve it
  behind the near-permanent agent lease (`capacity_wait_budget_exhausted` seen live). They
  need durable admission, which is a cognition-loop change: proposal mode, separate PR.
- Severity: note. Concern: runs queued before the policy change keep frozen alternatives and
  will not see chat-burst. Mitigation: they drain on agent; new runs derive it.
- Severity: note. Concern: while the gate is open, EVERY Hub chat message is held and emailed,
  including Juniper's own. Mitigation: that is the requested behaviour ("any traffic"); the
  button state is visible in the composer.

## PR link

(filled after `gh pr create`)
