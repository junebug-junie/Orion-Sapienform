# Orion learns what model it's currently running on

## Summary

- Juniper asked whether Orion knows what LLM model is generating its own chat replies. Investigated end to end: it did not, and the one field meant to carry that fact was lying.
- Fixed the root cause in `orion-llm-gateway`: `ChatResultPayload.model_used` was being stamped from the *requested* route-table label (e.g. `"Active-GGUF-Model"`), not the model that actually served the call — confirmed live, this had even logged the metacog route as running `llama-3-8b-instruct-q4_k_m` when it was actually Qwen3-8B.
- Extended `orion-llm-gateway`'s existing `GET /routes` route-health catalog with a live `model` field (a `/v1/models` probe against each route's backend), reusing its existing health-check cache rather than building a new mechanism.
- Added a new `RuntimeContextV1` to `orion-cortex-exec`'s situation brief — the existing "runtime facts about myself" prompt mechanism (already carries time/weather/lab/perception) — so Orion's own system prompt now states what model is currently serving its `chat` route, sourced from the live `/v1/models` probe above (not the unreliable `model_used` label).
- Both additions fail open by design: any probe failure, timeout, or disabled flag degrades to "unavailable; do not infer" rather than guessing or crashing.

## Outcome moved

Orion's chat prompt now carries a real, live-verified fact about its own runtime ("You are currently running on model: Qwen3.6-35B-A3B-UD-Q5_K_M.gguf (route=chat)") instead of nothing. `model_used` (already a first-class, already-consumed field) now holds the true served model instead of a route alias.

## Current architecture

`services/orion-cortex-exec/app/situation.py` builds a `SituationBriefV1` once per session (cached, TTL-gated) with sub-contexts for time/presence/weather/lab/perception, compresses it into one `SituationPromptFragmentV1.compact_text` block, and injects that into `chat_general.j2` via `ctx["situation_prompt_fragment"]`. Each sub-context follows the same shape: a `_build_X_context()` wrapping a `_fetch_X()` in try/except, degrading to `available=False` on any failure, with `diagnostics.provider_status`/`provider_errors` tracking the outcome. `services/orion-llm-gateway`'s `run_llm_chat()` already tracked `route`/`served_by`/`backend` per call and exposed a route-health catalog at `GET /routes` (15s-cached `/health` probes) — but never tracked which model was actually loaded behind a route, only the label requested.

## Architecture touched

`services/orion-llm-gateway` (backend model-echo fix, route catalog probe), `services/orion-cortex-exec` (new situation sub-context + settings + compose), `orion/schemas/situation.py` (new schema).

## Files changed

- `services/orion-llm-gateway/app/llm_backend.py`: new `_served_model(result, requested_model)` helper — prefers `result["raw"]["model"]` (echoed by both the OpenAI-compat and Ollama-native response shapes) over the requested label, falling back to the label when raw has no model key (llama.cpp's native `/completion` endpoint, error paths). Wired into all 5 `result["model"] = ...` call sites in `run_llm_chat()`.
- `services/orion-llm-gateway/app/route_catalog.py`: new `_probe_health(target)`/`_probe_model(target)` — split from the original single `_probe_one`, run concurrently via `asyncio.gather()` (review finding: sequential could ~double worst-case refresh latency). `_probe_model` is a bounded, fail-open `GET {backend}/v1/models` read; its result is discarded whenever `_probe_health` doesn't report "up". `RouteHealthEntry` gained a `model: Optional[str] = None` field; `_entry_to_dict`/`build_routes_response`'s not-yet-probed branches also carry `model: None` for shape consistency.
- `services/orion-llm-gateway/tests/test_llm_backend.py`: 3 new tests for `_served_model` (prefers raw, falls back on empty raw, falls back on blank/wrong-type raw value).
- `services/orion-llm-gateway/tests/test_route_catalog.py`: extended the existing health-probe test's mock to also answer `/v1/models`; added 2 new tests (model surfaced when up, model `None` when the route is down — `_probe_model` now runs concurrently but its result is discarded on non-"up" status).
- `services/orion-llm-gateway/README.md`: new "Model identity" section under the endpoints table documenting the bug, the fix, and the `/routes` `model` field.
- `orion/schemas/situation.py`: new `RuntimeContextV1` (available/route/model_id/served_by/backend/source, `extra="forbid"`), added as `SituationBriefV1.runtime` with `default_factory` (additive, backward compatible — confirmed the one other constructor of `SituationBriefV1`, `orion-hub`'s `/api/situation/brief` debug endpoint, doesn't pass `runtime` and gets the safe default).
- `services/orion-cortex-exec/app/situation.py`: new `_fetch_runtime_context`/`_build_runtime_context` (mirrors `_fetch_weather`/`_build_environment_context`'s shape exactly — plain `urlopen` with a short timeout, try/except degrading to unavailable, own `_RUNTIME_CACHE` dict keyed by route under the existing `_LOCK`). Wired into `build_situation_for_ctx` and `_build_prompt_fragment` (new line: `"You are currently running on model: {model_id} (route={route})."` / `"Current model: unavailable; do not infer or guess a name."`).
- `services/orion-cortex-exec/app/settings.py`: 5 new fields — `orion_situation_runtime_enabled` (default `True`), `orion_situation_runtime_route` (`"chat"`), `orion_situation_runtime_ttl_seconds` (`120`), `orion_situation_runtime_probe_timeout_sec` (`2.0`), `cortex_exec_llm_gateway_url` (`"http://orion-llm-gateway:8210"`, alias `CORTEX_EXEC_LLM_GATEWAY_URL` — renamed from an initial `LLM_GATEWAY_BASE_URL` per review, see below).
- `services/orion-cortex-exec/.env_example`, `docker-compose.yml`: the same 5 keys, matching this service's existing `ORION_SITUATION_*`/`<SERVICE>_URL` conventions.
- `services/orion-cortex-exec/README.md`: new "Situation brief" section (added per review finding, see below) documenting the whole `situation.py` subsystem's fail-open pattern plus the 5 new env vars in a table.
- `services/orion-cortex-exec/tests/test_situation_provider.py`: shared `_settings()` fixture updated with `orion_situation_runtime_enabled: False` (mirrors `weather_enabled: False`'s existing rationale — avoid a real network call from every unrelated situation test). 6 new tests: live-model-reported, gateway-unreachable degrade, route-missing-from-response degrade, disabled-by-default sanity check (asserts `urlopen` is never called), and a cache-hit test (asserts `urlopen` called exactly once across two `_build_runtime_context` calls within TTL).

## Schema / bus / API changes

- Added: `RuntimeContextV1` (`orion/schemas/situation.py`), `SituationBriefV1.runtime` field.
- Added: `route_catalog.py`'s `RouteHealthEntry.model` field, surfaced in `GET /routes`'s response.
- Behavior changed: `ChatResultPayload.model_used` now holds the real served model instead of the requested route label, for any call where the backend echoes one.
- Compatibility notes: both additions are additive/backward-compatible (`model` in `/routes` degrades to `null`; `runtime` in `SituationBriefV1` has a `default_factory`). No bus channel changes — the situation brief is in-process only, never published.

## Env/config changes

- Added keys: `ORION_SITUATION_RUNTIME_ENABLED`, `ORION_SITUATION_RUNTIME_ROUTE`, `ORION_SITUATION_RUNTIME_TTL_SECONDS`, `ORION_SITUATION_RUNTIME_PROBE_TIMEOUT_SEC`, `CORTEX_EXEC_LLM_GATEWAY_URL` (all `services/orion-cortex-exec`).
- `.env_example` updated: yes, both files' relevant keys documented inline.
- local `.env` synced: manually for `orion-cortex-exec` (the 5 new keys weren't present at all, so `sync_local_env_from_example.py` had nothing to diff against — added by hand, verified with `grep`). `orion-llm-gateway`'s `.env_example`/`.env` were not touched — no new env keys there, only code behavior.
- Skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-llm-gateway/tests/test_llm_backend.py services/orion-llm-gateway/tests/test_route_catalog.py -q
40 passed
```

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest services/orion-cortex-exec/tests/test_situation_perception_context.py services/orion-cortex-exec/tests/test_situation_provider.py services/orion-cortex-exec/tests/test_situation_settings_env.py services/orion-cortex-exec/tests/test_situation_prompt_integration.py -q
29 passed, 2 failed
```
The 2 failures (`test_prompts_render_without_situation_fragment`, `test_prompts_render_with_situation_fragment_scenarios`) are pre-existing and unrelated — reproduced identically (`jinja2.exceptions.UndefinedError: 'metadata' is undefined`) on an unmodified `main` checkout before touching anything.

Also checked: running the full `services/orion-cortex-exec/tests/` directory (not just the situation files) hits 13 pre-existing collection errors (`ValueError: Verb...`) — reproduced identically on unmodified `main`, unrelated to this patch, not something this PR should fix.

`python3 -m py_compile` on every touched Python file. Manual live verification: `curl http://100.112.254.99:8011/v1/models` confirmed the real OpenAI-compat response shape (`data[0].id`) `_served_model()`/`_probe_model()` both parse.

## Evals run

No dedicated eval harness for this seam — this is a factual-grounding fix (a prompt line either matches live reality or degrades to "unavailable"), not a quality dimension an eval would score differently.

## Docker/build/smoke checks

Not run — no local Docker daemon rebuild in this session. Confirmed via compose file inspection that `orion-llm-gateway` and `orion-cortex-exec` already share the `app-net` external network (required for the new direct HTTP call), and that `orion-llm-gateway`'s `LLM_GATEWAY_HEALTH_PORT` default (`8210`) matches the new `CORTEX_EXEC_LLM_GATEWAY_URL` default port. Live-verified beyond static inspection: `docker inspect orion-llm-gateway` confirmed both `orion-llm-gateway` and `llm-gateway` are real DNS aliases on `app-net`, and `docker run --rm --network app-net curlimages/curl ... http://orion-llm-gateway:8210/routes` returned a real, current route-health payload from the live gateway.

## Review findings fixed

- Finding (SHOULD): `route_catalog.py`'s `_probe_one` ran the `/health` check and the new `/v1/models` probe sequentially, so a refresh across 4 routes could take up to ~2x as long worst-case (bounded by the 15s cache TTL, not a correctness break, but a real latency regression on the refresh path).
  - Fix: split into `_probe_health`/`_probe_model` and run both via `asyncio.gather()`; the model result is only trusted when health reports "up" (discarded otherwise), so semantics are unchanged.
  - Evidence: `test_get_routes_payload_model_none_when_route_is_down` still passes with the updated comment reflecting that `_probe_model` now runs concurrently but its result is discarded on non-"up" status.
- Finding (minor): `orion-llm-gateway/README.md`'s new "Model identity" section pointed to a "Situation brief" section in `orion-cortex-exec/README.md` that didn't exist.
  - Fix: added that section (documents `RuntimeContextV1`, the fail-open pattern shared with weather/lab/perception, all 5 new env vars in a table, and the test file).
  - Evidence: `grep -n "### Situation brief" services/orion-cortex-exec/README.md`.
- Finding (minor): the new env key `LLM_GATEWAY_BASE_URL` didn't match this repo's established naming convention for "another service's base URL for orion-llm-gateway" (`CONTEXT_EXEC_LLM_GATEWAY_URL`, `HUB_LLM_GATEWAY_URL` — both service-prefixed).
  - Fix: renamed to `CORTEX_EXEC_LLM_GATEWAY_URL` (settings field `cortex_exec_llm_gateway_url`) across `settings.py`, `.env_example`, `docker-compose.yml`, local `.env`, and the test fixture.
  - Evidence: `grep -rn "LLM_GATEWAY_BASE_URL"` returns nothing in the worktree.
  - Side-verification while investigating this: live-checked `docker inspect orion-llm-gateway` — both `orion-llm-gateway` and `llm-gateway` are registered DNS aliases on the shared `app-net` network, and a live `docker run --rm --network app-net curlimages/curl ... http://orion-llm-gateway:8210/routes` succeeded. So the hostname in the default value was already correct either way; only the env *key name* needed to change for convention consistency.
- The review agent's sandbox lacked `pytest` and could not execute the suites itself; it reviewed statically. I ran the actual suites separately (see Tests run above), before and after applying its findings, in both cases confirming only the pre-existing, unrelated `test_prompts_render_*` failures remain.

## CI failure fixed post-push

`orion-static-gates` CI's "Service hostname references" gate (`scripts/check_service_hostname_refs.py`) failed: `services/orion-cortex-exec/.env_example:147` hardcoded `CORTEX_EXEC_LLM_GATEWAY_URL=http://orion-llm-gateway:8210` — a `services/<dirname>`-shaped hostname, not the real Compose service key (`llm-gateway`). This is the exact same bug class that gate exists to catch (its own docstring: found live 2026-07-28 in 5 other places, one of which silently broke a daily digest scheduler's entire existence). I'd chosen `orion-llm-gateway` deliberately, having live-verified with `docker inspect`/`docker run --network app-net curl` that it currently resolves as a DNS alias alongside `llm-gateway` — true today, but exactly the non-portable behavior (`container_name`-based, not the declared `services:` key) this gate's docstring documents as unreliable across different `PROJECT` values. Fixed: changed the default to `http://llm-gateway:8210` everywhere it appeared (`.env_example`, `docker-compose.yml`, `settings.py`, `situation.py`'s fallback, the test fixture, local `.env`, and the README table), matching `CONTEXT_EXEC_LLM_GATEWAY_URL`/`HUB_LLM_GATEWAY_URL`'s existing convention exactly. Verified locally: `check_service_hostname_refs.py` → OK, both service test suites unchanged (40 passed llm-gateway; 29 passed / 2 pre-existing-unrelated cortex-exec).

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-llm-gateway/.env \
  -f services/orion-llm-gateway/docker-compose.yml \
  up -d --build orion-llm-gateway

docker compose \
  --env-file .env \
  --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml \
  up -d --build orion-cortex-exec
```

## Risks / concerns

- Severity: LOW
- Concern: the new HTTP call in `situation.py` (`_fetch_runtime_context`) adds a network round-trip to session/presence-fingerprint cache misses (gated by `ORION_SITUATION_RUNTIME_TTL_SECONDS=120`, on top of the gateway's own 15s route-health cache).
- Mitigation: bounded by `ORION_SITUATION_RUNTIME_PROBE_TIMEOUT_SEC=2.0` and wrapped in the same try/except-degrade pattern every other situation sub-context already uses; a slow/dead gateway degrades the prompt line, never blocks or crashes the turn.
- Severity: LOW
- Concern: `route_catalog.py`'s `_probe_model()` adds a second HTTP call per route-health refresh (every 15s per route, only when `/health` already reported "up").
- Mitigation: cheap (a single small JSON response), same timeout as the existing health probe, and skipped entirely when the route is already down.
- Disclosed, not started: the metacog trace's persisted `model` column (`services/orion-sql-writer`) is separately, structurally broken — 73,991 rows read `"unknown"` because the code path that would populate it (`router.py`'s `reasoning_trace` gate) almost never fires. This PR does not touch that path; it's a distinct bug in a different consumer, flagged during investigation, not fixed here.
- Disclosed, not started: the FCC/Claude-Code-harness surface (`orion/harness/prefix.py`) has an equivalent seam for "tell the harness what model it's running on" that this PR does not extend to — scoped to the Hub chat surface Juniper actually asked about.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1639
