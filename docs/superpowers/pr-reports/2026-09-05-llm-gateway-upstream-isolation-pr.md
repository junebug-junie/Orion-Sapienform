# PR: llm-gateway per-upstream isolation -- one saturated lane can no longer starve the rest

Branch: `fix/llm-gateway-upstream-isolation`
Supersedes the "fix belongs to whoever owns PR #2110" recommendation in
`2026-09-05-stance-react-attention-salience-gateway-starvation-incident.md`. The
salience threshold is untouched. The gateway is fixed so it threads under load.

## Summary

- Every bus chat request ran `asyncio.to_thread(run_llm_chat, body)` on the loop's
  stock default executor: one pool, 32 threads, shared by every route, FIFO, each
  thread held for the request's whole life including the upstream HTTP read (up to
  `READ_TIMEOUT_SEC` = 700s). When `quick`'s 4-slot worker fell behind, traffic to
  that one upstream held all 32 and every other lane's requests queued behind it.
- New `app/upstream_admission.py`: one in-flight cap per distinct upstream URL
  (`LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`, default 8), waited on an asyncio semaphore
  off the pool. The executor is sized from the route table at startup
  (`distinct upstreams x cap + 4`), so a request for an idle upstream always has a
  thread.
- A request that cannot get its lane's permit inside its own read-timeout budget is
  shed immediately with `raw.error = "gateway_overloaded"` instead of being
  generated for a caller whose RPC already timed out. That wasted generation is what
  turned a busy lane into a 20-minute backlog.
- `run_llm_chat` split into `plan_llm_chat` (routing, on the loop) + execute (on the
  thread). The blocking `wait_for_slack_sync` inside the thread is deleted; the bus
  path uses the same async `background_admission` as the OpenAI passthrough, with
  `via="bus"` in the ledger. Gate order: slack wait first (holding nothing), then
  lane permit.
- The two callers behind the flood, `orion-topic-foundry` and
  `orion-memory-consolidation` (classify), now send `gateway_read_timeout_sec` equal
  to their own RPC timeout (60s-class), the same field cortex-exec and orion-mind
  already send, so the gateway stops working 700s for a 60s caller.
- Live lane depth on `GET /admission` under `"upstreams"`.

## Outcome moved

Measured on the live gateway, `docker logs -t`, receipt -> dispatch per request:

```text
BEFORE (2026-09-05 05:00-08:00Z, one shared 32-thread pool)
lane        n    p50        p90        max        target worker
quick       306  1272.1s    2003.1s    2200.5s    saturated 4/4
metacog      48  1162.0s    1792.2s    1860.4s    idle
stance_react on chat (corr c1a8f100...)  ~18 min   idle
never completed inside the window: ~300 requests
gateway process threads: 33 (pool at its 32 cap)

AFTER (08:00Z deploy onward, same offered load, quick still saturated)
lane 8013 (quick)    inflight=8 waiting=21 longest_wait=182.8s   <- backlog visible, bounded
lane 8012 (metacog)  inflight=1 waiting=0  admitted=15 longest_wait=0.0s
lane 8015 (agent)    inflight=1 waiting=0  admitted=1  longest_wait=0.0s
foreground lanes (chat/metacog/agent/harness) that ever queued: none
gateway process threads: 12
```

The `quick` worker is still genuinely oversubscribed; that is now a capacity fact
you can read off `/admission` rather than a hidden stall of every other lane. With
the two callers' 60s hints live, their queued requests shed at 60s instead of
holding a slot for 700s.

## Current architecture

- `main.handle_chat` -> `asyncio.to_thread(run_llm_chat, body)`; default executor,
  `min(32, cpu+4)` = 32 threads here (96 cpu), never sized or observed.
- `run_llm_chat` (sync) did lane routing, route resolution, and for
  `priority: background` routes called `wait_for_slack_sync` (poll `/slots` up to
  30s) *inside the thread*, then the upstream HTTP call.
- No per-upstream bound anywhere; llama.cpp queued the overflow server-side, the
  executor queued the rest, and neither was visible.

## Architecture touched

- `services/orion-llm-gateway/app/upstream_admission.py` (new): `UpstreamAdmission`
  gate, `_Admission` context manager, process-wide accessor, `executor_workers()`.
- `services/orion-llm-gateway/app/main.py`: `_dispatch_chat`, `_overloaded_result`,
  `configure_executor`; `/admission` gains `"upstreams"`.
- `services/orion-llm-gateway/app/llm_backend.py`: `ChatDispatchPlan`,
  `plan_llm_chat`, `run_llm_chat(body, plan=None)`.
- `services/orion-llm-gateway/app/priority_admission.py`: sync variant deleted,
  `background_admission(..., via=)`.
- Callers: topic-foundry `llm_client._bus_chat`, memory-consolidation
  `classify.classify_turn` options.
- `scripts/sync_local_env_from_example.py`: `LLM_GATEWAY_UPSTREAM_` prefix.

## Files changed

- `services/orion-llm-gateway/app/upstream_admission.py`: the two invariants (isolation, no work for departed callers) and the gauges.
- `services/orion-llm-gateway/app/main.py`: admission on the loop before any thread; executor sized from the route table; `/admission` exposes lanes.
- `services/orion-llm-gateway/app/llm_backend.py`: routing split out so it is decided once, on the loop; no wait inside the thread.
- `services/orion-llm-gateway/app/priority_admission.py`: one implementation; `via` parameter.
- `services/orion-llm-gateway/app/settings.py`: `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT`.
- `services/orion-llm-gateway/.env_example`, `docker-compose.yml`, `README.md`: the key, and a "Per-upstream isolation" section with the incident numbers.
- `services/orion-llm-gateway/tests/test_upstream_admission.py`: gate unit tests + incident reproduction through `_dispatch_chat` (flood on `quick`, one `chat`, chat unaffected), shedding, gate order, `/admission`, executor sizing.
- `services/orion-llm-gateway/tests/test_priority_admission.py`, `test_deferral_instrumentation.py`, `test_llm_backend.py`: sync-path tests replaced by plan/via tests.
- `services/orion-topic-foundry/app/services/llm_client.py`: `gateway_read_timeout_sec`.
- `services/orion-memory-consolidation/app/classify.py` + `tests/test_classify_turn_change.py`: same.
- `scripts/sync_local_env_from_example.py`: prefix so the key syncs.

## Schema / bus / API changes

- Added: `GET /admission` response key `upstreams` (additive; cortex-exec's
  `admission_cue` checks only its own required fields).
- Added: chat result `raw.error = "gateway_overloaded"` with
  `raw.details.{route,upstream,served_by,waited_s,budget_s,lane}`. Same shape as
  the existing `llm_route_unavailable` early return.
- Removed: none on the bus. `priority_admission.wait_for_slack_sync` (in-service
  function) deleted.
- Behavior changed: bus-path background routes now honour
  `LLM_GATEWAY_BACKGROUND_CONCURRENCY` (they skipped it before because the sync
  path could not share the asyncio semaphore).
- Compatibility: `run_llm_chat(body)` still works without a plan.

## Env/config changes

- Added keys: `LLM_GATEWAY_UPSTREAM_MAX_INFLIGHT=8` (gateway).
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: yes.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes --
  after adding the `LLM_GATEWAY_UPSTREAM_` prefix; the first run silently skipped
  the key (no matching prefix, run reported nothing). Verified
  `services/orion-llm-gateway/.env:133` in the primary checkout and copied into the
  worktree.
- skipped keys requiring operator action: none.

## Tests run

```text
.venv/bin/python -m pytest services/orion-llm-gateway/tests -q        301 passed
.venv/bin/python -m pytest services/orion-memory-consolidation/tests -q  112 passed
.venv/bin/python scripts/check_env_key_single_source.py             OK
git diff --check                                                     clean

Mutation check: keying every upstream to one lane
(`key = LEGACY_UPSTREAM` in upstream_admission.lane) ->
tests/test_upstream_admission.py::test_a_flood_on_quick_does_not_delay_chat FAILS.

Pre-existing, not this PR: 3 tests in test_llm_backend.py / test_route_catalog.py
fail on main too whenever services/orion-llm-gateway/.env is present (settings
reads env_file=".env"; CI has none). Verified: 3 failed on main with .env,
301 passed here with .env moved aside.
```

## Evals run

```text
No eval harness exists for orion-llm-gateway (tests/ only). The incident
reproduction in test_upstream_admission.py is the behavioral check; the live
before/after above is the measurement.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-llm-gateway up -d --build           Built, Started
  docker exec: /app/app/upstream_admission.py present; env has UPSTREAM_MAX_INFLIGHT=8
  startup log: executor sized workers=36 upstreams=4 max_inflight_per_upstream=8
  GET /health ok, 7 routes; GET /admission -> upstreams populated
  live: gateway_upstream_queued lines on quick only; threads 33 -> 12
scripts/safe_docker_build.sh orion-topic-foundry up -d --build          Built, Started
  docker exec grep gateway_read_timeout_sec llm_client.py -> 1
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build   Built, Started
  docker exec grep gateway_read_timeout_sec classify.py -> 1
```

## Review findings fixed

REVIEW_SECTION

## Restart required

Already restarted from this worktree (gateway 08:00Z, topic-foundry and
memory-consolidation shortly after). Note: all three containers are now pinned to
`/mnt/scripts/Orion-Sapienform-llm-gateway-upstream-isolation` as their compose
working dir (the gateway was previously pinned to the flash-next worktree). After
merge, redeploy from main to unpin:

```bash
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
scripts/safe_docker_build.sh orion-llm-gateway up -d --build           # from a worktree on main
scripts/safe_docker_build.sh orion-topic-foundry up -d --build
scripts/safe_docker_build.sh orion-memory-consolidation up -d --build
```

## Risks / concerns

- Severity: should
- Concern: `quick` (circe:8013, 4 slots) is genuinely oversubscribed by
  topic-foundry + memory-consolidation + reverie + mind + hub. This PR makes that
  visible and bounded (`waiting`, `shed` on `/admission`); it does not add capacity.
  Expect `shed > 0` on that lane under the same load.
- Mitigation: watch `/admission -> upstreams -> 8013.shed`; the fix for a rising
  shed count is fewer background callers on `quick` or a second worker, not a
  bigger cap.
- Severity: note
- Concern: bus-path background routes now share the per-route
  `LLM_GATEWAY_BACKGROUND_CONCURRENCY=1` semaphore with the HTTP passthrough. This
  is what the background-priority design always specified; the bus path only
  skipped it because the old sync code could not participate.
- Severity: note
- Concern: the graphify update wrapper refused its incremental update (the known,
  unroot-caused shrink bug) and restored the graph. No graph change in this PR.

## PR link

PR_LINK
