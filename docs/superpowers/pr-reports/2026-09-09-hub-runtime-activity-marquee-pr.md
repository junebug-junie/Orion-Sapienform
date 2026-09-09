# PR #2172 — Hub "running right now" marquee + modal

Branch: `feat/hub-runtime-activity` · https://github.com/junebug-junie/Orion-Sapienform/pull/2172

## Summary

- Hub's header now has a live pill showing what Orion is running at this moment: curiosity runs (both the investigate line and the self-inquiry line), the two harness governor lanes (chat / agent) with running + queued turns, and the LLM gateway's per-worker lanes (metacog, quick, agent, ...) with in-flight and waiting counts.
- Clicking it opens a modal with the details Juniper asked for: run id, correlation id, current node, duration since dispatch, resume marker, recent transitions, the lane the run's harness turn is on, its last few motor steps, model label and served model, and per-lane queues.
- Fed over Server-Sent Events, so it moves without a reload. One frame per change, coalesced; keepalive comments in between.
- No new bus channel, schema, or producer. One in-memory reducer folds facts Hub already sees, plus a poll of two gateway HTTP endpoints that already existed.
- One additive field on the gateway's `GET /routes` (`upstream`) so its `/admission` gauges can be labelled with route names instead of ports.
- The env sync script gained the new key prefix (it was silently skipping the two new keys — found by grepping the live `.env` after the sync claimed success).

## Outcome moved

Before: the only way to know whether a curiosity run was in flight, which lane it was on, or whether a chat turn was stuck behind an agent-lane turn was hand SQL against `substrate_durable_run_state`, `docker logs`, or `curl` against the gateway. Hub Surface (PR #2137) shows history, not "now".

After: one glance at the header, one click for the detail. Runtime truth rules baked in: a turn shows as *running* only once a harness step has been observed (before that it is *queued*); a run's harness turn is `null` until its handoff is seen; a down gateway shows its error next to the last good read; a Hub restart re-adopts still-active runs from Postgres and marks them as backfilled.

## Current architecture (before)

- `orion:durable:run:state` (`DurableRunStateV1`) — every node transition of a curiosity run. Hub already subscribed (`curiosity_investigation._run_state_loop`) but only for `completed` → outreach.
- `orion:harness:run:step` (`HarnessRunStepV1`) — Hub already subscribed (`HarnessStepRelay`) but only fanned out to a WebSocket queue when one was registered; steps for turns with no listener (curiosity, world-pulse) were logged and dropped.
- `execute_unified_turn` → `HarnessGovernorClient.run()` picks the governor queue (`orion:harness:run:request` vs `:agent`) from `fcc_model_label`; nothing recorded which turns were waiting where.
- orion-llm-gateway `GET /admission` exposes per-upstream inflight/waiting gauges keyed by URL; `GET /routes` names the routes but carried no URL, so the two could not be joined.

## Architecture touched

- **New** `orion/hub/runtime_activity.py` — `RuntimeActivity` reducer + process singleton. Folds: `run_dispatched`, `run_state`, `backfill_runs`, `turn_requested`, `harness_step`, `turn_finished`, `gateway_admission`. `snapshot()` is the one contract the page reads. Bounded: finished items expire (15 min TTL, 30 max), hard caps on both maps, per-run transition list capped, prompt text never retained (motor-boot step reduced to the label "motor boot").
- **New** `services/orion-hub/scripts/runtime_activity_routes.py` — `GET /api/runtime-activity`, `GET /api/runtime-activity/stream` (SSE), `RuntimeActivityFeeds` (gateway poll loop + one-shot Postgres backfill; both fail open).
- **Hook sites** (four, each one call): `turn_orchestrator.execute_unified_turn` (requested + finished, in a `finally`), `curiosity_investigation._dispatch_durable_run` (line known only here) and `_handle_run_state` (every transition, before the outreach filter), `harness_step_relay._dispatch_step` (before the no-queue drop).
- **Gateway** `route_catalog.py` — `upstream` field on every catalog entry (`RouteTarget.url`, the exact string `upstream_admission.py` keys its gauges on; verified `plan.upstream` is `route_target.url`).
- **Page** `templates/index.html` (pill in header right cluster, modal markup, css/js links), `static/js/runtime-activity.js`, `static/css/runtime-activity.css`.

## Files changed

- `orion/hub/runtime_activity.py`: the reducer (new)
- `orion/hub/tests/test_runtime_activity.py`: fold/snapshot/eviction/backfill tests (new)
- `orion/hub/turn_orchestrator.py`: `turn_requested` before the governor RPC, `turn_finished` in its `finally`
- `services/orion-hub/scripts/runtime_activity_routes.py`: routes + feeds (new)
- `services/orion-hub/scripts/main.py`: import, start/stop feeds around the existing relays, `include_router`
- `services/orion-hub/scripts/curiosity_investigation.py`: two one-line hooks
- `services/orion-hub/scripts/harness_step_relay.py`: one hook
- `services/orion-hub/app/settings.py`, `.env_example`, `README.md`: two keys + a doc section
- `services/orion-hub/templates/index.html`, `static/js/runtime-activity.js`, `static/js/runtime-activity.test.js`, `static/css/runtime-activity.css`: the page
- `services/orion-hub/tests/test_runtime_activity_routes.py`: routes, SSE generator, gateway merge, poll, backfill (new)
- `services/orion-llm-gateway/app/route_catalog.py`, `tests/test_route_catalog_background.py`: `upstream` join key
- `scripts/sync_local_env_from_example.py`: `HUB_RUNTIME_ACTIVITY_` added to `SYNC_PREFIXES`

## Schema / bus / API changes

- Added: `upstream` (string | null) on each entry of orion-llm-gateway `GET /routes`. Additive; same URL already visible on `GET /admission`.
- Added: Hub `GET /api/runtime-activity`, `GET /api/runtime-activity/stream` (text/event-stream, `event: snapshot`, `id: <version>`).
- Removed / Renamed: none.
- Behavior changed: none on the bus. `HarnessStepRelay` still drops steps with no WebSocket listener from its fan-out; it now also feeds the reducer first.
- Compatibility notes: no channel or registry edits (`orion/bus/channels.yaml` already lists Hub as a consumer of both channels read here).

## Env/config changes

- Added keys (orion-hub): `HUB_RUNTIME_ACTIVITY_ENABLED=true`, `HUB_RUNTIME_ACTIVITY_GATEWAY_POLL_SEC=5`
- Removed / Renamed: none
- `.env_example` updated: yes
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: the first run reported success and wrote nothing (keys matched no `SYNC_PREFIXES` entry). Keys were appended to the live `services/orion-hub/.env` by hand, then the prefix was added to the script so the next sync covers them; re-run confirmed no drift.
- skipped keys requiring operator action: none

## Tests run

```text
PYTHONPATH=<wt> .venv/bin/python -m pytest orion/hub/tests/test_runtime_activity.py -q
  14 passed   (11 before review fixes, +3 for the dispatched-staleness and backfill-line fixes)

cd services/orion-hub && PYTHONPATH=<wt> pytest -q tests/test_runtime_activity_routes.py \
  tests/test_hub_harness_step_relay.py tests/test_curiosity_investigation.py \
  tests/test_curiosity_self_inquiry.py tests/test_turn_orchestrator_cockpit_hops.py \
  tests/test_turn_orchestrator_ws_frames.py tests/test_hub_agent_mode_fcc_routing.py \
  tests/test_world_pulse_read_pipeline.py tests/test_world_pulse_read_stage2.py \
  tests/test_endogenous_outreach.py tests/test_chat_route_tagging.py ../../orion/hub/tests
  485 passed in 29.99s   (481 before review fixes, +4 for the fold-isolation fix)

cd services/orion-llm-gateway && pytest -q tests/test_route_catalog_background.py tests/test_route_catalog_n_ctx.py
  22 passed

cd services/orion-hub/static/js && node --test runtime-activity.test.js
  11 passed

Static gates (same list as .github/workflows/orion-static-gates.yml), all OK:
  check_metric_lineage --gate, check_definition_drift --gate, check_inner_state_registry,
  check_scripts_dir_no_stdlib_shadow, check_service_hostname_refs,
  check_compose_no_relative_mounts, check_journal_dispatch_registry,
  check_daily_schedule_collisions, check_sentience_instruments --static-only,
  check_env_key_single_source, check_service_env_compose_parity orion-hub (env_file: N/A)
```

## Evals run

```text
None. orion-hub has no eval harness for UI surfaces; the reducer's honesty
rules (running only after a step, absent stays absent, eviction, backfill
filter) are covered by the gate tests above. Follow-up: a live eval that
replays a recorded curiosity run's state+step events through the reducer and
asserts the marquee text at each transition.
```

## Docker/build/smoke checks

```text
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml config     -> exit 0

Live reads used to shape the code (against the running system, not a deploy of this branch):
  curl http://127.0.0.1:8210/admission?window_s=300  -> 4 upstreams (8011/8012/8013/8015),
      8013 inflight=1 shed=425 longest_wait 67.9s; ledger checked=26 queued=7 deferrals=7
  curl http://127.0.0.1:8210/routes                  -> chat/quick/quick_background/metacog/
      metacog_background/agent with served_by; no URL (the gap `upstream` closes)
  psql ... substrate_durable_run_state (48h, latest per run_id) -> 12 runs, all `completed`/`finish`;
      the backfill SQL's DISTINCT ON shape and column names were run live

NOT deployed. The live path (marquee moving on a real curiosity run) is UNVERIFIED
until orion-llm-gateway and orion-hub are rebuilt from this branch.
```

## Review findings fixed

Two `/code-review` passes ran in subagents (the first hit a session rate limit before reporting; the second completed clean). Verdict on the completed pass: `DONE_WITH_CONCERNS` — two `should`-severity gaps, both fixed here; one `nit` fixed; one `nit` was a non-issue on inspection.

- **Finding (should):** a curiosity run whose dispatch was accepted but that never received a single `DurableRunStateV1` transition (runner died before its first step) stayed `status="dispatched"` — and `RunRecord.active` treats bare "dispatched" as active forever. It was excluded from the TTL sweep entirely; only the 200-run count cap would eventually evict it, which at real curiosity cadence (3–4 runs/day) could take weeks. It would sit in the marquee as a phantom "running" chip the whole time.
  - Fix: added `dispatched_stale_sec` (default 30 min) to `RuntimeActivity`; `_evict_runs` now drops a still-"dispatched" run outright once it has gone that long with no state event, instead of routing it through the finished/TTL path (which needs a `finished_at` this record never gets).
  - Evidence: `orion/hub/tests/test_runtime_activity.py::test_dispatched_run_with_no_state_event_ages_out` and `::test_a_state_event_resets_the_dispatched_staleness_clock` (a real transition arriving resets the clock — the rule only fires while status is still bare `dispatched`).

- **Finding (should):** `HarnessStepRelay._dispatch_step`'s call into the reducer ran unguarded, ahead of the real per-queue Soft HUD delivery. An exception there would propagate out of the shared `async for` loop in `_run()`, which drops the *whole* pubsub subscription and resubscribes after a 1s sleep — costing live step delivery to every in-flight turn, not just the one with the odd payload.
  - Fix: wrapped the fold call in its own `try/except Exception: logger.exception(...)`, matching the isolation `curiosity_investigation._handle_run_state` already had per-message.
  - Evidence: `services/orion-hub/tests/test_hub_harness_step_relay.py::test_a_broken_activity_fold_never_costs_real_step_delivery` — a fold that raises `RuntimeError` still lets the real queue item through.

- **Finding (nit):** `BACKFILL_SQL` never selected `detail`, so a run adopted on cold start after a Hub restart always showed the generic "curiosity" label (no investigate/self_inquiry distinction) until a fresh live event happened to arrive for that run.
  - Fix: added `s.detail` to the SELECT; `backfill_runs` now reads `line` from it (handling both a dict, from JSONB, and a JSON string, defensively).
  - Evidence: `orion/hub/tests/test_runtime_activity.py::test_backfill_recovers_line_from_the_row_detail_column`.

- **Finding (nit, no change needed):** `run_dispatched` doesn't `move_to_end` the record the way `turn_requested`/`harness_step` do. On inspection this is the *correct* eviction order as-is — a long-stuck dispatched run should be the first thing the count cap removes, not the last — so no change was made.

All four hooked-site correctness questions (unguarded folds elsewhere, event-loop safety, the SSE generator's disconnect handling, and the gateway poll's exception/cache-clock behavior) came back clean on both review passes; see the PR diff for the reviewer's full reasoning on each.

## Restart required

Both services read this at boot. Deploy the gateway first (Hub tolerates a catalog without `upstream`; it just shows unnamed upstreams until the gateway restarts).

```bash
cd /mnt/scripts/Orion-Sapienform-hub-runtime-activity   # a worktree, per safe_docker_build policy
scripts/safe_docker_build.sh orion-llm-gateway up -d --build
scripts/safe_docker_build.sh orion-hub up -d --build
# prove the code is in the containers before trusting anything on screen:
docker exec orion-athena-hub grep -c runtime_activity /app/services/orion-hub/scripts/main.py
docker exec orion-athena-llm-gateway grep -c '"upstream"' /app/app/route_catalog.py
# then:
curl -s http://127.0.0.1:8210/routes | python3 -c 'import sys,json;print([(r["id"],r["upstream"]) for r in json.load(sys.stdin)["routes"]])'
curl -s http://<hub>/api/runtime-activity | python3 -m json.tool | head -40
curl -N http://<hub>/api/runtime-activity/stream | head -c 2000
```

## Risks / concerns

- Severity: should · Concern: the marquee's "queued" for a governor lane is Hub-side knowledge (request published, no step yet). If the governor is down, a turn reads as queued until the RPC times out (up to `HUB_HARNESS_GOVERNOR_RPC_MAX_WAIT_SEC`). · Mitigation: the modal shows `queued_sec`; a long queue with nothing running on that lane is the tell. A governor-liveness read is a follow-up, not folded in here.
- Severity: should · Concern: after a Hub restart, harness turns that were in flight are not backfilled (only durable runs are, from Postgres); their steps land on the `unknown` lane until they finish. · Mitigation: `unknown` is shown as such, never guessed onto chat/agent.
- Severity: note · Concern: `graphify-out/graph.json` was not refreshed in this PR. The refresh is a ~100MB LFS object per push on a 1GB/month free bandwidth quota. · Mitigation: run `scripts/safe_graphify_update.sh` on main after merge if the quota allows.
- Severity: note · Concern: the gateway upstream URLs (Tailscale IPs) are now on `GET /routes` as well as `/admission`. Hub is the only consumer and both endpoints are already LAN-only.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/2172
