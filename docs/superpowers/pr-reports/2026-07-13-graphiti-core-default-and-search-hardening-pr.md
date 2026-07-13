# PR report: graphiti_core default + search-stack caching + chat-visible proof

Third follow-up in this session's graphiti_core line (#993 activation, #995 RELATES_TO schema fix, this PR: hardening now that #995 is proven). Three independent-but-related items, one PR since all three are scoped to `orion-graphiti-adapter`.

## Summary

- `GRAPHITI_BACKEND` default flipped `orion_postgres` → `graphiti_core`, `FALKORDB_ENABLED` default `false` → `true` (`settings.py`, `.env_example`). This was previously an explicit live-override-only decision ("stays `orion_postgres` until search is proven") — it's proven now (PR #995), so the shipped default follows.
- `/v1/search` no longer rebuilds its `FalkorDriver`/`Graphiti`/stub-client stack on every single request — memoized in a process-local cache, mirroring the existing `_indices_ready` lazy-init pattern.
- New `scripts/smoke_graphiti_active_packet_search_e2e.sh`: proves the search-rail fix reaches a real Hub API response (`/api/memory/active-packet`, the same endpoint a live chat turn's recall path consumes), isolated from neighborhood/links by using two crystallizations with no edge between them.
- Fixed a real regression the default-flip caused: 4 pre-existing tests silently depended on the old default to avoid needing the `graphiti_core` package (Docker-image-only, not in the dev venv) — pinned explicitly rather than weakened.
- Fixed a real correctness hazard the caching change introduced before it shipped: embed-success tracking moved from embedder instance state to a per-call `ContextVar`, since a cached/reused embedder instance would otherwise leak one request's result into a concurrent request's trace.

## Outcome moved

Fresh deployments now get the proven-working backend by default, not a stale conservative default that predates proof. `/v1/search` no longer pays full driver/client construction cost on every request. And — the most important one — there is now direct evidence the search fix is reachable from the same API surface a live chat turn actually uses, not just the adapter's own internal endpoint.

## Current architecture (before this patch)

`GRAPHITI_BACKEND=orion_postgres` was the code-level default (README: "settings.py's and .env_example's code-level default remains orion_postgres... until /v1/search is proven to find real data end to end") — a hedge from PR #993, before #995 fixed the actual bug. `search()` built its entire driver/client/`Graphiti` stack fresh per call. No test exercised whether the search fix was visible outside the adapter's own `/v1/search` endpoint.

## Architecture touched

Single service, `orion-graphiti-adapter` — settings, one backend module, its tests, and one new top-level smoke script. No schema/bus/API contract changes.

## Files changed

- `services/orion-graphiti-adapter/app/settings.py`, `.env_example`: default flip
- `services/orion-graphiti-adapter/README.md`: rewrote the paragraph declaring `orion_postgres` the hedge default — that decision is made now
- `services/orion-graphiti-adapter/app/backends/graphiti_core.py`: `_get_search_stack()` (new, memoized driver/embedder/`Graphiti` build), `_embed_used_ctx` `ContextVar` replacing `_OrionEmbedderClient.used` instance state, `search()` updated to use both
- `services/orion-graphiti-adapter/tests/conftest.py`: new autouse fixture clearing `_search_stack_cache` before/after every test
- `services/orion-graphiti-adapter/tests/{test_episodes,test_links,test_rebuild}.py`: pinned `GRAPHITI_BACKEND=orion_postgres` on the specific tests that exercise the generic (non-graphiti_core-specific) ingest/link/rebuild path, since the `graphiti_core` package isn't importable in the bare dev venv
- `services/orion-graphiti-adapter/tests/test_health.py`: updated a real stale assertion (`backend == "orion_postgres"`) to match the new default
- `services/orion-graphiti-adapter/tests/test_graphiti_core_backend.py`: 2 new regression tests — driver/`Graphiti` construction happens once across repeated `search()` calls; `embed_used` doesn't leak stale `True` across calls sharing a cached embedder
- `scripts/smoke_graphiti_active_packet_search_e2e.sh` (new): the chat-visible proof
- `docs/superpowers/specs/2026-07-13-graphiti-core-backend-activation-spec.md`: "Hardening pass" addendum documenting all three items and live verification evidence

## Schema / bus / API changes

None. No new env keys, no schema/bus/API surface changes — this PR changes defaults, adds a process-local perf cache, and adds a smoke script.

## Env/config changes

- Added keys: none
- Removed keys: none
- Renamed keys: none
- Changed defaults: `GRAPHITI_BACKEND` (`orion_postgres`→`graphiti_core`), `FALKORDB_ENABLED` (`false`→`true`) in both `settings.py` and `.env_example`
- `.env_example` updated: yes
- Local `.env` synced: this host's `services/orion-graphiti-adapter/.env` already had the live-override values set (from #993/#995), unaffected by the default-flip in code — verified unchanged post-deploy (see Docker/smoke checks below)
- Skipped keys requiring operator action: none — any environment that doesn't already have an explicit `.env` override will now start with `graphiti_core` by default, which is the intended behavior of this PR

## Tests run

```text
$ source venv/bin/activate && python -m pytest services/orion-graphiti-adapter/tests -q
23 passed, 2 warnings in 0.76s
```
21 baseline + 2 new caching/ContextVar regression tests. Independently re-run by the orchestrator, not taken from the implementing agent's report alone (note: the implementing agent's background session hit an API session-limit error partway through and did not reach the test-fixing, smoke-script, or verification stages — the orchestrator completed items 2 and 3's remaining work directly, including discovering and fixing the 4-test regression above, in the same worktree the agent had already made solid partial progress in).

## Evals run

No eval harness exists for this service (same gap noted in prior PRs this session). The three live e2e smokes below are the deterministic gate.

## Docker/build/smoke checks

```text
$ curl -sS http://localhost:8640/health
{"service":"orion-graphiti-adapter","postgres":true,"falkordb_enabled":true,"backend":"graphiti_core"}

$ bash scripts/smoke_graphiti_links_e2e.sh
PASS smoke_graphiti_links_e2e seed=1a0fafe5-dd1f-4254-b3a1-b6c87330d82b linked=7b1cbb56-2782-4de8-9a3b-0a6da8a1e1ac

$ bash scripts/smoke_graphiti_search_e2e.sh
PASS smoke_graphiti_search_e2e seed=85ec3f1c-4cbf-4859-a1c9-d1540c9519df embed_used=true

$ bash scripts/smoke_graphiti_active_packet_search_e2e.sh
PASS smoke_graphiti_active_packet_search_e2e seed=86f4134c-a281-46a9-958d-4d79c7486a74 search_reached=50ab005f-6cf7-49ca-8feb-1d82e9f459f3
```

All three run after a full rebuild+restart with this PR's code (`docker exec orion-athena-graphiti-adapter grep -c "_get_search_stack\|_embed_used_ctx" /app/app/backends/graphiti_core.py` → 12, confirming the new code is what's actually running, not a stale image).

## Review findings fixed

- Finding: default-flip broke 4 existing tests that silently depended on the old default (`ModuleNotFoundError: graphiti_core` — the package is Docker-image-only).
  - Fix: pinned `GRAPHITI_BACKEND=orion_postgres` explicitly in `test_ingest_episode_returns_ids`, `test_ingest_with_supports_link_writes_cross_edge`, `test_rebuild_batch_ingests_items`, `test_rebuild_skips_intimate_items` — these test generic ingest/link/rebuild behavior, not anything graphiti_core-specific, so pinning the backend they were implicitly already exercising is correct, not a weakening.
  - Evidence: `pytest` count went from 21 failed-5/passed-16 immediately after the default flip, to 21/21 passing after the pin, to 23/23 after adding the 2 new caching tests.
- Finding: `test_health_without_postgres` asserted `backend == "orion_postgres"` — a real stale assertion, not a test-isolation artifact.
  - Fix: updated to `"graphiti_core"`.
  - Evidence: test passes; assertion now matches the intentional new default.
- Finding (design review, before shipping): caching the embedder instance across requests would let `_OrionEmbedderClient.used` (old instance-state tracking) leak one request's embed-success result into a concurrent or later request's trace.
  - Fix: replaced with `_embed_used_ctx`, an `asyncio.ContextVar` reset at the start of every `search()` call.
  - Evidence: new regression test `test_search_embed_used_does_not_leak_across_calls_sharing_cached_embedder` — first call embeds successfully (`embed_used: True`), second call (same cached embedder) has embed fail (`embed_used: False`), asserted independently.

## Restart required

Already executed live during this session:

```bash
docker compose --profile falkordb \
  --env-file .env --env-file services/orion-graphiti-adapter/.env \
  -f services/orion-graphiti-adapter/docker-compose.yml up -d --build
```

If merged and redeployed elsewhere with no existing `.env` override: no action needed — the new default (`graphiti_core`) is now correct out of the box, assuming `CRYSTALLIZER_EMBED_HOST_URL` is reachable on that host (already required, unchanged).

## Risks / concerns

- Severity: Medium — the implementing background agent hit an API session-limit error mid-task and stopped before finishing. Its partial work (items 1 fully, item 2's core logic) was solid on inspection (correctly identified the exact `ContextVar` fix needed for the caching hazard, and added the `conftest.py` test-isolation fixture proactively) — the orchestrator verified and completed the remainder directly (the 4-test regression fix, item 3's smoke script, live verification, spec addendum) rather than re-spawning a second agent into the same constraint. Documented here for traceability, not hidden.
- Severity: Low — `_search_stack_cache` is a single-process, in-memory cache with no TTL or invalidation path if `falkordb_uri`/`graph_name`/`embed_url` ever needed to change without a process restart (they don't today — sourced once from `settings` at process start). Not a bug under current usage; worth knowing if this service ever needs hot config reload.

## PR link

Not opened via `gh` (no token, SSH-only remote, consistent with every PR this session). Branch pushed; use this to open it:

`https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/graphiti-core-default-and-search-hardening?expand=1`
