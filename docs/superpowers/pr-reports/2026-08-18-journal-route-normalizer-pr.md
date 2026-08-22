# Journal route: stop silently rewriting the configured lane to `chat`

Branch: `fix/journal-route-normalizer` · 2026-08-18

## Summary

- `orion-actions` kept a **private** allow-list of accepted LLM routes — `{chat, quick, metacog}` — and returned `"chat"` for everything outside it. `orion-cortex-exec` had long since accepted `{chat, quick, metacog, quick_background, agent}`. The two copies drifted, silently.
- Consequence: `ACTIONS_JOURNAL_LLM_ROUTE=quick_background`, set by ROADMAP A3 in both `.env` and `.env_example` **with a comment explaining why**, was rewritten to `chat` before it left the service. Every `journal.compose` went to circe's single-slot 131,072-token lane. `agent` was eaten the same way.
- Review found a **third** copy on `POST /api/chat` (`orion-hub/scripts/cortex_request_builder.py`) with the same defect — fixed here too.
- Accepted routes + legacy aliases now live in `orion/llm/routes.py`, imported by all three services, each guarded by a **behavioural** test that drives its real resolver with every route in the shared set.
- Unrecognized route now means *no override* (executor applies its verb default) and logs `actions_llm_route_unrecognized`, instead of silently defaulting to `chat`.

## Outcome moved

`journal.compose` moved off circe's contended `chat` lane onto atlas's background lane. Live, 25 minutes after deploy — **58 distinct composes, 100% on `quick_background`**:

```
llm_route_selected verb=journal.compose step=draft_journal_entry
  route=quick_background override=quick_background override_attempted=quick_background
  lane_priority=low
```

and the A4 admission instrumentation confirms they land on atlas with background admission:

```
[LLM-GW background] admission waited=0.020s polls=1 reserved=2
  url=http://100.121.214.30:8013 outcome=admitted
```

Zero `journal_compose_failed` / `journal_compose_missing_final_text` in the same window.

## Current architecture (before this patch)

Three services each carried their own literal set of route names. Only `orion-cortex-exec`'s was current.

| Site | Set | Behaviour on unrecognized |
|---|---|---|
| `orion-cortex-exec/app/executor.py` | `{chat, quick, metacog, quick_background, agent}` | reject → verb default (correct) |
| `orion-actions/app/main.py` | `{chat, quick, metacog}` | **rewrite to `chat`** |
| `orion-hub/scripts/cortex_request_builder.py` | `{chat, quick, agent, metacog}` | **drop the key** |

The A3 patch changed config only. Config truth is not runtime truth — this is exactly the failure mode CLAUDE.md §0A names, and I shipped it.

## Architecture touched

- New shared module `orion/llm/routes.py` (`ACCEPTED_LLM_ROUTES`, `LLM_ROUTE_ALIASES`, `normalize_llm_route`).
- `orion-actions`, `orion-cortex-exec`, `orion-hub` all import it. No new service, no new layer.
- `_ACCEPTED_LLM_ROUTE_OVERRIDES` deleted from the executor — after the refactor it had zero production readers and was kept alive only by a test asserting it.

## Files changed

- `orion/llm/routes.py`: the shared definition, plus an explicit scope statement naming the copies this patch does *not* close.
- `services/orion-actions/app/main.py`: use the shared normalizer; warn on rejection; omit `llm_route` entirely rather than sending null.
- `services/orion-cortex-exec/app/executor.py`: use the shared normalizer; drop the dead alias.
- `services/orion-hub/scripts/cortex_request_builder.py`: third copy closed.
- `services/orion-actions/.env_example`, `README.md`: document accepted values and the incident.
- Tests in all three services.

## Schema / bus / API changes

- Added: none.
- Removed: `_ACCEPTED_LLM_ROUTE_OVERRIDES` (module-private, not a contract).
- Behavior changed: an unrecognized `llm_route` now yields *no override* instead of `chat`. With live config nothing but the journal path changes.
- Compatibility: legacy aliases `chat_quick` / `quick_chat` / `chat_kids_story` still resolve to `quick` — and now do so in Hub's builder too, where they were previously dropped.

## Env/config changes

- Added keys: none. Removed: none. Renamed: none.
- `.env_example` updated: comments only (documents the accepted set and the incident).
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: yes — no key deltas, nothing to apply.
- Skipped keys requiring operator action: none.

## Tests run

```text
services/orion-actions/tests                                   121 passed
  (incl. new tests/test_llm_route_normalization.py             14 passed)
services/orion-hub  test_cortex_request_builder.py
                  + test_workflow_request_builder.py            56 passed
services/orion-cortex-exec  test_executor_llm_route_override.py
                          + test_autonomous_background_routing.py 29 passed
```

Pre-existing, reproduced identically on `main`, unrelated to this change:
- `services/orion-cortex-exec/tests` full-suite run: 13 collection errors, `ValueError: Verb already registered: legacy.plan`.
- `test_chat_general_route_mapping.py` (1), `test_chat_quick_plumbing.py` (2), `test_chat_kids_story_plumbing.py` (2) fail on `main` too.

## Evals run

```text
No eval harness exists for orion-actions route selection. The live-traffic
evidence above (58 real composes, 100% on the intended lane, 0 failures) is the
behavioural check; a synthetic eval would be weaker than the production trace.
```

## Docker/build/smoke checks

```text
scripts/safe_docker_build.sh orion-actions up -d --build     -> built, recreated, started
curl -fsS http://localhost:7160/health
  {"ok":true,"service":"actions","version":"0.1.0","node":"athena"}
docker exec orion-athena-actions python -c '...'             -> live config resolves:
  env journal: quick_background   -> journal : quick_background   (was: chat)
  env daily  : metacog            -> daily   : metacog            (unchanged)
  typo 'qiuck'                    -> None + actions_llm_route_unrecognized WARNING
```

## Review findings fixed

- **Finding (S1, material):** the first commit claimed the set was "defined once" while a third private copy sat on `POST /api/chat` — `cortex_request_builder.py:468` — missing `quick_background` and all aliases. Shipping the claim uncorrected would be worse than the original bug, because the next reader trusts it and skips the grep.
  - Fix: closed the third copy at the source; corrected the README/docstring to state the real scope and name the remaining stale lists with file:line.
  - Evidence: `test_quick_background_reaches_the_executor_instead_of_being_dropped`, `test_legacy_aliases_resolve_rather_than_being_dropped` — both fail pre-fix (key never set → `None`).
- **Finding (S4):** `_ACCEPTED_LLM_ROUTE_OVERRIDES` became a dead alias with zero production readers, kept alive only by its own test.
  - Fix: deleted; both referencing tests retargeted to the shared module.
  - Evidence: 0 residual references in `executor.py`; 29 exec tests pass.
- **Finding (S4b):** both anti-drift guards were tautological — `X is ACCEPTED_LLM_ROUTES` only checks a re-export and passes cleanly for anyone reintroducing a private copy, which is precisely this bug.
  - Fix: replaced in all three services with behavioural guards driving the real resolver.
  - Evidence: `test_every_shared_route_survives_this_services_wrapper` / `..._resolver` / `..._the_builder`.
- **Finding (S2, documented not fixed):** `orion/schemas/context_exec.py:13`'s `ALLOWED_CONTEXT_EXEC_LLM_PROFILES` is a competing allow-list in the same shared package that *raises* rather than degrading.
  - Fix: named in `orion/llm/routes.py` as a deliberately distinct, narrower axis — widening it is a schema decision, not a routing one. Same for `brain` (a mode, not a route).
- **Cleared by review:** caller handling of `None` (only two call sites, both guarded; every downstream reader uses `.get()`); `attempted` semantics proven byte-identical across 32 inputs × 3 ctx shapes, 0 mismatches; packaging (both Dockerfiles `COPY orion`, no bind-mount, no import cycle).

## Restart required

`orion-actions` is already deployed and verified. The other two carry latent fixes and can go on their next natural deploy:

```bash
cd /mnt/scripts/Orion-Sapienform-journal-route-normalizer
scripts/safe_docker_build.sh orion-hub up -d --build          # Hub: third allow-list
scripts/safe_docker_build.sh orion-cortex-exec up -d --build  # pure refactor, 0 behaviour delta
```

## Risks / concerns

- **Severity: low.** An operator typo in `ACTIONS_*_LLM_ROUTE` now falls through to the verb default rather than `chat`. That is the intent, but it is a behaviour change for a misconfiguration; the new WARNING is the mitigation.
- **Severity: low.** `_normalized_llm_route` never consults `fallback` when `preferred` is set but invalid — it returns `None`. This matches the old structure (which returned `"chat"` and also ignored `fallback`), so it is not a regression, but "a typo in `ACTIONS_JOURNAL_LLM_ROUTE` ignores `ACTIONS_LLM_ROUTE` entirely" may surprise an operator.
- **Severity: medium, pre-existing, NOT addressed here.** `quick_background` is invisible from the gateway route catalog through Hub's gateway client to the browser — four stale 4-name lists, and `scripts/smoke_llm_gateway_routes.py` never exercises the lane this PR exists to reach. Named with file:line in `orion/llm/routes.py`. Follow-up slice.
- **Severity: medium, pre-existing, newly VISIBLE.** The route log now makes journal volume legible: **27 `metacog_digest`-triggered journal composes in 25 minutes**, roughly one per minute, ~1,500/day. That is not a daily journal. It predates this patch (it was simply invisible on circe's lane) and plausibly explains a large share of circe `chat`'s measured 8.10% all-busy. Worth its own investigation before A5 — see below.

## Follow-ups

1. **Journal compose volume.** ~1/min metacog-digest-triggered composes. Is each producing distinct reflection, or is this an empty-shell cognition loop? Measure before optimizing.
2. **Gateway catalog chain.** Teach `route_catalog.py` / `llm_gateway_client.py` / `app.js` / the routes smoke about `quick_background`.
3. `services/orion-llm-gateway/app/lane_routes.py:6` — `_BACKGROUND_ROUTE_KEYS = ("background", "metacog")`; `"background"` is not a real route key, so the gateway's background *lane* resolves to `metacog` and can never reach the background *route*. Flagged by review, unverified by me.
4. The context-overflow retry (#1705) has not fired yet — no overflow in 25 minutes of journal traffic on the 4,096 lane. Expected: the measured tail was 2/90.

## Status

DONE_WITH_CONCERNS — shipped and live-verified; concerns 3 and 4 under "Risks" are pre-existing and explicitly out of scope.
