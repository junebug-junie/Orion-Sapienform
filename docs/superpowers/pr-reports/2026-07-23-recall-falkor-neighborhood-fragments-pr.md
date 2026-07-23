# PR report: Falkor-native replacement for orion-recall's last live Fuseki read path

## Summary

- `orion-recall`'s generic SPARQL neighborhood fetch (`rdf_adapter.py::fetch_rdf_fragments` / `_fetch_rdf_neighborhood_fragments`, a blind `?s ?p ?o FILTER(CONTAINS(...))` scan against Fuseki) was confirmed live -- not dead code -- by reading the running `orion-athena-fuseki` container's own request log mid-session and watching real queries fire with real keywords.
- Adds `services/orion-recall/app/storage/falkor_neighborhood_adapter.py::fetch_falkor_neighborhood_fragments`: keyword-matches real, canonical `:Entity` nodes in the `orion_recall` FalkorDB graph and walks `MENTIONS_ENTITY` edges to `:ChatTurn`s, reusing the already-existing `falkor_entity_relatedness.py::fetch_turns_mentioning_entities` and `sql_chat.py::fetch_chat_turns_by_id`.
- Wired into `worker.py::_query_backends` as a flag-gated swap (not additive), behind `RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT` -- ships dark (`false`) everywhere: code default, `.env_example`, `docker-compose.yml`, and the live primary-checkout `.env`.
- A live test run (copying the module into the running `orion-athena-recall` container and calling it against real Falkor/Postgres data) surfaced a real false-positive problem: the unfiltered keyword "and" alone matched "sandra bullock", "nelson mandela", "england", "landing pad" as `CONTAINS` substrings. Fixed with a small stopword filter before shipping, re-verified live.
- A review pass (dispatched as a subagent, since `/code-review` itself is a gated command) caught three real bugs, all fixed in this same PR: the swap was nested inside `if rdf_enabled:` (gated on `RECALL_RDF_ENDPOINT_URL`), making it permanently inert the moment Fuseki's endpoint is removed from config -- exactly the end-state this migration exists to reach; `fusion.py` had no backend-weight entry for the new source (silent 0.5 fallback instead of `rdf`'s 0.3); and no belief-source-rank entry (silent last-place default in PCR ordering).

## Outcome moved

Every other Fuseki read/write path in `orion-recall` had already been migrated to Falkor or retired outright across roughly 40 prior PRs (chatturn fetch, graphtri/Claim path, Hub memory-graph, cognition/metacog traces). This was the one remaining live read dependency. It now has a Falkor-native replacement ready to verify and flip -- once flipped, `orion-recall` has zero live Fuseki dependencies, which is the actual precondition for Phase 10 (Fuseki container/infra decommission) that this repo's own phased retirement plan requires.

## Current architecture

Before this patch: `worker.py::_query_backends` called `fetch_rdf_fragments` (real Fuseki SPARQL) unconditionally whenever a profile had `rdf_top_k > 0` and `RECALL_ENABLE_RDF`/`RECALL_RDF_ENDPOINT_URL` were set (both true live) -- with no Falkor equivalent, unlike the chatturn fetch which already had one (`RECALL_FALKOR_IN_CHAT`).

## Architecture touched

- `services/orion-recall/app/storage/falkor_neighborhood_adapter.py` (new)
- `services/orion-recall/app/worker.py` (swap wiring)
- `services/orion-recall/app/fusion.py` (backend weight + belief-source rank)
- `services/orion-recall/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`
- Primary-checkout `services/orion-recall/.env` (synced, not committed)

## Files changed

- `services/orion-recall/app/storage/falkor_neighborhood_adapter.py`: new module. `fetch_falkor_neighborhood_fragments` (keyword extraction with stopword filter -> Cypher `:Entity` match -> `fetch_turns_mentioning_entities` -> Postgres text join). Discloses three real scope narrowings vs. the RDF version in its own docstring: entity-only matches (not any triple), `chat.history`-sourced turns only (not social-room/enrichment), and `CONTAINS` not being word-bounded (live example: "nico" matches only "unicode").
- `services/orion-recall/app/worker.py`: new imports; `falkor_neighborhood_enabled` computed and its fetch block run independent of `rdf_enabled`/`RECALL_RDF_ENDPOINT_URL` (matching `falkor_chat_enabled`'s placement), gated on the flag + per-profile `enable_falkor_neighborhood` override + `rdf_top_k > 0`; the old `fetch_rdf_fragments` call inside `if rdf_enabled:` now skips when the swap already ran.
- `services/orion-recall/app/fusion.py`: `DEFAULT_BACKEND_WEIGHTS["falkor_neighborhood"] = 0.3` (matches `"rdf"`); `_BELIEF_SOURCE_ORDER["falkor_neighborhood"] = 3` (matches `"rdf"`'s rank, since `_belief_source_rank` only auto-inherits `rdf`'s rank for sources literally prefixed `"rdf"`).
- `services/orion-recall/app/settings.py`: new `RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT` field, default `False`.
- `services/orion-recall/.env_example`, `docker-compose.yml`: new key added (this service has no `env_file:` directive -- every var must be listed explicitly in compose or it never reaches the container regardless of `.env`).
- `services/orion-recall/README.md`: new table row documenting the flag and its three disclosed scope narrowings.
- `services/orion-recall/tests/test_falkor_neighborhood_adapter.py` (new): unit tests for the adapter -- full fragment shape, stopword filtering (regression for the live "and" finding), no-keyword/no-client/no-entity-match short circuits, Falkor/Postgres failure degradation.
- `services/orion-recall/tests/test_falkor_neighborhood_swap.py` (new): worker.py integration tests -- swap-not-additive, independence from `rdf_enabled` (regression for the Critical review finding), `rdf_top_k=0` suppression, profile-level suppression, failure degradation, fusion.py weight/rank regressions.
- `services/orion-recall/tests/test_falkor_chat_swap.py`, `tests/test_query_backends_compression.py`: added explicit `mock_settings.RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT = False` pins everywhere `app.worker.settings` is patched as a bare `MagicMock`, so the new flag can't silently default to a truthy mock attribute and route those tests through the real (unmocked) new function -- same bug class independently fixed the same day in a different service (`docs/superpowers/pr-reports/2026-07-22-fuseki-falkor-cognition-metacog-kill-and-graph-compression-federators-pr.md`).

## Schema / bus / API changes

- None. No bus/channel/schema changes -- this is a read-path backend swap internal to `orion-recall`.

## Env/config changes

- Added keys: `RECALL_FALKOR_NEIGHBORHOOD_IN_CHAT` (`orion-recall` -- `.env_example`, `docker-compose.yml`, `settings.py`).
- `.env_example` updated: yes.
- Local `.env` synced: yes, directly edited in the primary checkout (`services/orion-recall/.env`), value `false` (matches `.env_example` default -- not flipped live in this PR).
- Skipped keys requiring operator action: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-recall venv/bin/python3 -m pytest services/orion-recall/tests -q
-> 229 passed, 3 pre-existing failures (test_process_recall_active_turn_exclusion.py,
   test_recall_policy_harness.py, test_recall_vector_amputation.py) -- confirmed
   identical on unmodified origin/main before this diff, unrelated to this patch.

git diff --check -> clean
```

## Evals run

No eval harness exists for this service's recall backend selection; focused deterministic tests cover the changed behavior, including regression tests for every review finding.

## Docker/build/smoke checks

No container rebuild/restart performed as part of this PR (flag ships dark, no live behavior change). Live verification instead: copied the new module into the running `orion-athena-recall` container via `docker cp` and called `fetch_falkor_neighborhood_fragments` directly against real FalkorDB (`orion_recall`, 845 `:Entity` nodes) and Postgres data -- confirmed real, sensible, non-degenerate results (e.g. query "tell me about Orion and reverie" -> 5 real Orion-related chat turns), confirmed the stopword-filter fix closed the "and" false-positive problem, and confirmed the disclosed "nico"/"unicode" substring-match limitation live.

## Review findings fixed

- Finding (Critical): the `falkor_neighborhood_enabled` block was nested inside `if rdf_enabled:`, which is gated on `bool(settings.RECALL_RDF_ENDPOINT_URL)` -- flipping the new flag would do nothing unless Fuseki's endpoint URL was still configured, the opposite of the independence needed for Fuseki to ever be safely removed.
  - Fix: pulled the block out to run independent of `rdf_enabled`/the endpoint URL, mirroring `falkor_chat_enabled`'s existing placement exactly; kept `rdf_top_k > 0` as the per-profile "wants graph-neighborhood candidates" gate instead.
  - Evidence: `test_falkor_neighborhood_runs_independent_of_rdf_enabled` (new), reproduces the exact scenario (RDF fully off, flag on) and asserts the Falkor fetch still runs.
- Finding (High): `fusion.py`'s `DEFAULT_BACKEND_WEIGHTS` had no entry for `"falkor_neighborhood"`, so candidates from it would silently get the generic 0.5 fallback weight instead of `"rdf"`'s 0.3 -- a 67% relative composite-score bump nobody intended.
  - Fix: added `"falkor_neighborhood": 0.3` matching `"rdf"`.
  - Evidence: `test_falkor_neighborhood_backend_weight_matches_rdf`.
- Finding (Medium): `_belief_source_rank` only auto-inherits `"rdf"`'s rank for sources literally prefixed `"rdf"` (a pre-existing quirk); `"falkor_neighborhood"` doesn't match that prefix and had no explicit `_BELIEF_SOURCE_ORDER` entry, so it would silently sort last (rank 99) in PCR belief-digest ordering.
  - Fix: added `"falkor_neighborhood": 3` matching `"rdf"`'s rank.
  - Evidence: `test_falkor_neighborhood_belief_source_rank_matches_rdf`.
- Informational (no fix needed, verified complete): grepped every `patch("app.worker.settings")` bare-`MagicMock` usage across `services/orion-recall/tests/` -- confirmed the two files this diff touched (`test_falkor_chat_swap.py`, `test_query_backends_compression.py`) are the complete list; `test_entity_relatedness_boost_map.py` uses the same pattern but exercises a function that never reads the new flag, so it was correctly left untouched.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml up -d --build
```

Not required to merge this PR (flag ships dark, no behavior change until explicitly flipped). Required only once a broader live-verification pass across more real queries is done and the flag is flipped in a follow-up.

## Risks / concerns

- Severity: Low.
- Concern: `CONTAINS` entity matching is not word-bounded -- a short keyword can still false-positive inside an unrelated entity name (live example: "nico" matched only "unicode"). The stopword filter closes the larger, live-confirmed instance of this (generic words matching dozens of entities), not all of it.
- Mitigation: disclosed explicitly in the module's docstring, `settings.py`'s comment, and `README.md`. Not solved here (would need word-boundary-aware Cypher matching); flagged as a known limitation to watch during the live-verification pass before flipping the flag.
- Severity: Low.
- Concern: this function cannot surface social-room or enrichment content (both `fetch_turns_mentioning_entities` and its Postgres join are `chat.history`-scoped only), unlike the old SPARQL scan which had no such filter.
- Mitigation: disclosed explicitly; a `SocialRoomTurn`-aware variant would be separate, not-yet-built follow-up work if that content turns out to matter for this specific fetch.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/new/feat/recall-falkor-neighborhood-fragments
