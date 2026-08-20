## Summary

Implements Track A item 3 from `docs/superpowers/specs/2026-08-18-aitown-concept-graph-split-and-atlas-readability-design.md` ("AI Town's own concept graph") — a second, fully independent topic-foundry dataset/model + FalkorDB graph + ingestion pipeline for AI Town's own organically-clustered concepts, parallel to Orion's existing one. Interpretability-only, per that spec's own Non-goals: never feeds `concept_induced`/`chat_stance` or any other Orion cognition consumer.

- Reads from `aitown_chat_history_log` directly — the AI-Town table-split cutover (PR #1734) is already fully live, so unlike Orion's dataset this needs no `where_sql` filter; the table is AI-Town-only by write-time routing, not by filter.
- New FalkorDB graph (`FALKORDB_AITOWN_SUBSTRATE_GRAPH`, default `orion_substrate_aitown`), same instance as `orion_substrate`.
- Scheduler tick gains a second trigger/enrich/ingest step-group, with its own kill switch (`SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED`) so it can be paused without touching Orion's own production pipeline.
- New manual-ingest route (`POST /api/substrate/concepts/ingest-topic-foundry-aitown`) and a `?graph=aitown` query param on the existing summary/network routes — the design doc's own "first cut" read-path suggestion, closing what code review found was a real "data written, structurally unreachable" gap in this branch's first draft.
- **Code review found and this branch fixed 6 real issues** (of 9 confirmed + 1 plausible; 2 lower-severity items cut at the report's cap) — see "Review findings fixed" below, including one finding where the obvious fix would have introduced a worse bug, and the fix was correctly not applied.

## Outcome moved

AI Town's chat corpus now has its own, independently-clustered concept graph instead of either polluting Orion's (the problem PR #1721/#1748 already fixed) or having no organic clustering at all. The graph is reachable via API today (manual ingest route + `?graph=aitown` query param); a dedicated Concept Atlas UI page is deliberately deferred per the design doc's own sequencing (item 5: "only once someone's actually looking at the new graph regularly").

## Current architecture

Before this patch: one topic-foundry dataset/model (`orion-hub-autonomous-dataset-v2`), one FalkorDB graph (`orion_substrate`), one scheduler step-group, one ingestion route — all Orion-only, all hardcoded module constants in `concept_atlas_routes.py` with no parameterization.

## Architecture touched

`services/orion-hub` (routes, scheduler, settings, README) and `orion/substrate/falkor_store.py` (a real, minimal contract addition: a second named-graph builder).

## Files changed

- `orion/substrate/falkor_store.py`: `build_falkor_substrate_store_from_env()` gained optional `graph_name_env`/`graph_name_default` kwargs (backward compatible — both existing zero-arg call sites unchanged); new `build_aitown_falkor_substrate_store_from_env()`.
- `services/orion-hub/scripts/api_routes.py`: new `SUBSTRATE_SEMANTIC_STORE_AITOWN` singleton via `_build_aitown_substrate_store_from_env()`, with an honest (not falsely-parity-claiming) docstring and a loud warning log for unsupported non-falkor backends.
- `services/orion-hub/scripts/concept_atlas_routes.py`: `_ensure_topic_foundry_dataset_and_model()`/`trigger_topic_foundry_training_run()`/`trigger_topic_foundry_enrichment()` parameterized (all defaults match prior Orion behavior exactly) instead of duplicated; new AI Town module constants; new `trigger_topic_foundry_aitown_training_run()`/`trigger_topic_foundry_aitown_enrichment()` zero-arg wrappers; ingestion logic extracted into `_ingest_topic_foundry_run()` shared by both the existing route and the new `concept_atlas_ingest_topic_foundry_aitown()` route; new `_get_named_substrate_store()` shared helper; new `_resolve_store_for_graph_param()` + `?graph=` param on `concept_atlas_summary()`/`concept_atlas_network()`; extended dataset-drift warning to cover `source_table`, not just `where_sql`.
- `services/orion-hub/scripts/main.py`: scheduler tick gains a second trigger/enrich/ingest step-group, gated by its own `SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED`.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: `FALKORDB_AITOWN_SUBSTRATE_GRAPH` (env/compose only — no dead Settings field, see review findings), `SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED`. Local `.env` synced via `python scripts/sync_local_env_from_example.py --all-keys` (the default key-subset mode didn't pick up the new scheduler-enable key; `--all-keys` also caught one unrelated pre-existing drift, `orion-sql-writer`'s `NOTIFY_API_TOKEN`, left in place as a genuine parity fix).
- `services/orion-hub/README.md`: new "AI Town's own concept graph" subsection (route, env keys, scheduler wiring, dataset/model names) — was entirely undocumented before code review flagged it.
- Tests: `orion/substrate/tests/test_falkor_store_aitown_graph.py` (new, 7), `services/orion-hub/tests/test_aitown_substrate_store_singleton.py` (new, 4), AI Town cases in `test_topic_foundry_scheduler.py` (5: 3 wrapper tests + 2 drift tests), `test_concept_atlas_ingest_topic_foundry.py` (2), `test_concept_atlas_routes.py` (5 `?graph=` param tests). 4 pre-existing lambda fakes in `test_topic_foundry_scheduler.py` updated for the new keyword-arg call shape.

## Schema / bus / API changes

- Added: `POST /api/substrate/concepts/ingest-topic-foundry-aitown`.
- Added: optional `?graph=aitown` query param on `GET /api/substrate/concepts/summary` and `.../network` (default/unrecognized values resolve to the existing Orion behavior — fully backward compatible).
- Added: `"graph"` field in the response envelope of both routes (`"orion"` or `"aitown"`).
- Removed: none.
- Renamed: none.
- Behavior changed: none for existing callers — every new parameter/field is additive with a default matching prior behavior exactly.
- Compatibility notes: none needed.

## Env/config changes

- Added keys: `FALKORDB_AITOWN_SUBSTRATE_GRAPH` (default `orion_substrate_aitown`), `SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED` (default `true`).
- Removed keys: none (a `Settings.FALKORDB_AITOWN_SUBSTRATE_GRAPH` field was added and then removed within this same PR after code review found it had no reader — see findings below; the env key itself, read directly via `os.getenv()` in the builder function, was never removed).
- Renamed keys: none.
- `.env_example` updated: yes, both keys, in both the "Attention organ operator tab" comment section and the actual consumed "Substrate semantic graph" section (mirroring `FALKORDB_SUBSTRATE_GRAPH`'s existing duplicate-listing pattern).
- local `.env` synced with `python scripts/sync_local_env_from_example.py --all-keys`: yes.
- skipped keys requiring operator action: none.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests/test_concept_atlas_routes.py services/orion-hub/tests/test_concept_atlas_ingest_topic_foundry.py services/orion-hub/tests/test_topic_foundry_scheduler.py services/orion-hub/tests/test_aitown_substrate_store_singleton.py orion/substrate/tests/test_falkor_store_aitown_graph.py -q
  -> 104 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest orion/substrate/tests -q
  -> 579 passed

/mnt/scripts/Orion-Sapienform/.venv/bin/python3 -m pytest services/orion-hub/tests -q
  -> 1371 passed, 31 failed, 5 skipped
  Confirmed via git-stash-diff against a clean pre-feature baseline: every
  failure in this run also fails identically on the unpatched baseline,
  except one (test_substrate_mutation_manual_route_routing.py::
  test_routing_manual_apply_changes_real_live_routing_surface) -- confirmed
  order-dependent pollution, not a regression: passes cleanly in isolation
  (7/7 passed running that file alone), is in a file this branch never
  touches, and a DIFFERENT test in that same file failed in the prior
  review round's full-suite run instead -- i.e. which specific test in
  that already-flaky file fails varies run-to-run, the file itself is the
  pre-existing pollution source, not this branch.
```

## Evals run

No eval harness exists for `orion-hub`'s Concept Atlas feature or `orion/substrate` beyond `tests/`. This is additive pipeline wiring reusing already-tested logic (topic-foundry client, substrate materializer, FalkorDB store), not a new cognition-affecting capability — judged not to need a new eval harness.

## Docker/build/smoke checks

```text
python3 -c "import yaml; yaml.safe_load(open('services/orion-hub/docker-compose.yml'))"  -> OK
docker compose --env-file <primary>/.env --env-file <primary>/services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml config
  -> resolves FALKORDB_AITOWN_SUBSTRATE_GRAPH: orion_substrate_aitown correctly, no errors
```
Not deployed/restarted as part of this patch — no runtime smoke beyond `docker compose config`.

## Review findings fixed

Code review ran once (`Skill("code-review")` on branch `feat/aitown-own-concept-graph`), found 10 findings (9 CONFIRMED, 1 PLAUSIBLE; 2 lower-severity items cut at the report's 10-item cap). Fixed 6, one deliberately NOT fixed after investigation showed the obvious fix was unsafe, two documented as accepted/deferred:

- Finding (CONFIRMED, most impactful): AI Town's concept graph was written by the scheduler every tick but reachable by **zero** GET route — `concept_atlas_summary()`/`concept_atlas_network()` were both hardwired to `_get_substrate_store()` (Orion only). Data written, structurally unreachable.
  - Fix: `?graph=aitown` query param on both routes (design spec's own named "first cut" option), default/unrecognized values resolve to Orion's graph, never raise.
  - Evidence: 5 new tests in `test_concept_atlas_routes.py`.
- Finding (CONFIRMED): `_build_aitown_substrate_store_from_env()`'s docstring falsely claimed "same degrade-to-in-memory-never-raise contract as the primary store for every other backend" — it does not; a `routed`/`sparql`/`graphdb` operator would silently lose every AI Town write on restart with zero signal, unlike the primary store which builds a real durable store for those backends.
  - Fix: corrected the docstring; added a WARNING log naming the unsupported backend. Not live risk today (this deployment is `SUBSTRATE_STORE_BACKEND=falkor`, confirmed via `.env`) but now honest and loud instead of silent.
  - Evidence: code inspection; `.env` grep confirming current live backend.
- Finding (CONFIRMED): `_ensure_topic_foundry_dataset_and_model`'s `where_sql`-drift warning had no analogous check for `source_table`, a field this same branch newly parameterized.
  - Fix: extended the drift check to cover both fields independently.
  - Evidence: `test_ensure_dataset_and_model_warns_on_source_table_drift` (new).
- Finding (CONFIRMED): AI Town's scheduler step-group had no independent enable flag — only Orion's `SUBSTRATE_TOPIC_FOUNDRY_SCHEDULER_ENABLED`, which would also have to be disabled to pause AI Town's experimental pipeline.
  - Fix: new `SUBSTRATE_TOPIC_FOUNDRY_AITOWN_SCHEDULER_ENABLED` (default `true`, same interval) gates just the AI Town step-group.
  - Evidence: `main.py`'s scheduler loop; `.env_example`/settings.py.
- Finding (CONFIRMED): `Settings.FALKORDB_AITOWN_SUBSTRATE_GRAPH` was dead config — grep confirmed zero code anywhere read it off the `Settings` object; the real mechanism was always a direct `os.getenv()` read inside the builder function.
  - Fix: removed the Settings field (kept the env key/`.env_example`/docker-compose entries, which ARE read). Matches CLAUDE.md's "no keyword cathedral" gate — a schema field with no consumer is dead weight.
- Finding (CONFIRMED, lower severity, cut-list item fixed anyway): `_get_substrate_store()`/`_get_aitown_substrate_store()` were a hand-duplicated copy of each other.
  - Fix: factored through a shared `_get_named_substrate_store()` helper (same pattern already used for the other Orion/AI-Town-parameterized functions in this file).
- Finding (CONFIRMED, real, but **not fixed after investigation showed the obvious fix was unsafe**): two full eager FalkorDB hydrations at Hub import time (Orion's + AI Town's) instead of one, real added startup latency.
  - Investigated fix: pass `hydrate=False` to defer the AI Town store's hydration to first real access.
  - Why rejected: `concept_atlas_network()` calls `store.query_concept_region()` directly, which reads straight from the store's in-process cache and does **not** go through `snapshot()` — `snapshot()` is the only thing that lazily triggers a deferred hydration in this codebase. `hydrate=False` would leave the AI Town cache **permanently empty** for that route, a correctness bug worse than the startup-latency cost being traded away. Left `hydrate=True` unchanged; documented here rather than either silently skipping the finding or shipping a subtly broken "fix."
- Finding (CONFIRMED, real, judged disproportionate to fix given the 24h default scheduler interval — documented, not fixed): the Orion and AI Town step-groups run sequentially inside one tick instead of concurrently (`asyncio.to_thread` calls not parallelized via `asyncio.gather`), and `list_datasets()`/`list_models()` are fetched twice per tick (once per tenant) instead of once and shared. Both add real but small per-tick cost; at a daily default interval, judged not worth the added complexity of parallelizing/caching two multi-step scheduler groups in this patch.
- Finding (CONFIRMED, real, judged disproportionate given exactly 2 tenants exist today and both are already correctly paired — documented, not fixed): `log_prefix` is hand-paired with `dataset_name`/`model_name` at each of 4 call sites rather than derived or bundled into one config object, a real drift risk for a hypothetical future third tenant.

## Restart required

```bash
docker compose \
  --env-file .env \
  --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml \
  up -d --build
```

## Risks / concerns

- Severity: low
- Concern: AI Town's dataset/model/graph have no tuned HDBSCAN parameters yet — reuses Orion's `SUBSTRATE_TOPIC_FOUNDRY_HDBSCAN_MIN_CLUSTER_SIZE`/`_METRIC`, which may cluster very differently given AI Town's ~14x higher chat volume (per the design doc's own "Missing questions").
- Mitigation: deliberately deferred per that same doc — no real AI-Town cluster-quality data exists yet to tune against; revisit once the pipeline has run for real.
- Severity: low
- Concern: the two scheduler step-groups run sequentially and the catalog fetch is doubled per tick (documented findings above, not fixed).
- Mitigation: negligible at the 24h default interval; a real fix if/when the interval is ever shortened significantly.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1760
