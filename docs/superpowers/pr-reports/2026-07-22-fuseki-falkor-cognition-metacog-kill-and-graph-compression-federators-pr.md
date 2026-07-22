# PR report: kill cognition/metacog RDF writes, add dark Cypher-native graph-compression federators

## Summary

- Re-implements PR #1155 (reviewed, correct, closed without merging): removes `orion-rdf-writer`'s `cognition.trace`/`metacog.trace` Fuseki writes -- pure redundancy (~750 writes/6h) with Postgres (`cognition_traces`, `orion_metacognitive_trace` via orion-sql-writer).
- Adds two new Cypher-native FalkorDB federators to `orion-graph-compression` (`FalkorSubstrateFederator`, `FalkorEpisodicFederator`) as direct alternatives to its SPARQL-based Leiden-clustering federators, using the shared `orion/graph/falkor_client.py` client -- no translation/adapter layer.
- Wired in additively (unioned with existing SPARQL results, never a swap) behind two new off-by-default flags, so enabling them can only add clustering signal during verification, never regress what's already there.
- Fixed a real HIGH-severity finding from code review: free-text Falkor entity/tag names (real production data has spaces/apostrophes) would have broken the pre-existing SPARQL serializer in `writer.py`, silently failing to write exactly the regions this feature adds. Both federators now return well-formed IRIs.
- `orion/bus/channels.yaml` fixed to match reality: `orion-rdf-writer` removed as a consumer of `orion:metacog:trace`; `orion-sql-writer` added as a consumer of `orion:cognition:trace` (was already consuming it in code, never registered).

## Outcome moved

`orion-rdf-writer` no longer writes two Fuseki graphs that were pure Postgres redundancy. `orion-graph-compression` gained a path off Fuseki for its two most cognition-relevant clustering scopes -- most notably `SubstrateFederator`, which has likely been reading stale/empty data since substrate-runtime cut to Falkor-primary in PR #1153, unnoticed until this investigation.

## Current architecture

Before this patch: `orion-rdf-writer` still subscribed to and wrote `cognition.trace`/`metacog.trace` despite an already-reviewed prior attempt to kill it. `orion-graph-compression`'s `EpisodicFederator`/`SubstrateFederator` were SPARQL-only, with `SubstrateFederator` reading a graph (`orion:substrate`) nothing has written to since PR #1153's Falkor cutover.

## Architecture touched

- `services/orion-rdf-writer/app/settings.py`, `rdf_builder.py`: dead channel/handler removal.
- `services/orion-rdf-writer/.env_example`, `docker-compose.yml`: dangling key cleanup.
- `orion/bus/channels.yaml`: consumer-list correction for both channels.
- `services/orion-graph-compression/app/falkor_store.py` (new): lazy singleton FalkorDB client getters, mirrors `orion-recall/app/recall_falkor_store.py`.
- `services/orion-graph-compression/app/federators/substrate_falkor.py`, `episodic_falkor.py` (new): Cypher-native federators.
- `services/orion-graph-compression/app/worker.py`: additive wiring behind two new flags.
- `services/orion-graph-compression/app/settings.py`, `.env_example`, `docker-compose.yml`, `README.md`: flag/env documentation.

## Files changed

- `services/orion-rdf-writer/app/settings.py`: removed `cognition.trace`/`metacog.trace` from `get_all_subscribe_channels()`, removed `CHANNEL_COGNITION_TRACE_PUB` field.
- `services/orion-rdf-writer/app/rdf_builder.py`: removed dispatch branches + `_handle_cognition_trace`/`_handle_metacognitive_trace` handlers + now-unused schema imports.
- `services/orion-rdf-writer/.env_example`, `docker-compose.yml`: removed dangling `CHANNEL_COGNITION_TRACE_PUB`.
- `services/orion-rdf-writer/tests/test_autonomy_materialization.py`: new channel-not-subscribed + quiet-no-op dispatch regression tests.
- `services/orion-rdf-writer/tests/test_service_rdf_store_integration.py`: removed the two kinds from the parametrized "writes for kind" list (fixture mocks the builder entirely, cosmetic).
- `orion/bus/channels.yaml`: `orion:metacog:trace` consumer list drops `orion-rdf-writer`; `orion:cognition:trace` consumer list gains `orion-sql-writer`.
- `services/orion-graph-compression/app/falkor_store.py` (new): `get_substrate_falkor_client()`, `get_recall_falkor_client()`.
- `services/orion-graph-compression/app/federators/substrate_falkor.py` (new): `FalkorSubstrateFederator`, queries `orion_substrate` generically via `type(r)`.
- `services/orion-graph-compression/app/federators/episodic_falkor.py` (new): `FalkorEpisodicFederator`, queries `orion_recall` (ChatSession/ChatTurn/Tag/Entity) via `coalesce()` across heterogeneous identity properties.
- `services/orion-graph-compression/app/worker.py`: imports + additive union wiring in `_process_scope`.
- `services/orion-graph-compression/app/settings.py`: two new bool flags.
- `services/orion-graph-compression/.env_example`, `docker-compose.yml`, `README.md`: new env vars + Falkor federator documentation + updated scope table (cognition/metacog rows noted as no-longer-written).
- `services/orion-graph-compression/tests/test_federator_substrate_falkor.py`, `test_federator_episodic_falkor.py` (new): federator unit tests, including IRI-wrapping regression coverage.
- `services/orion-graph-compression/tests/test_worker_degraded.py`: fixed `_make_worker`'s settings mock (unset `MagicMock` attrs are truthy -- would have silently enabled the new flags in every existing test), added flag-off/flag-on wiring tests with real union-vs-replace verification.

## Schema / bus / API changes

- Removed: `orion-rdf-writer` as a subscriber to `orion:cognition:trace`/`orion:metacog:trace`.
- Added: `orion-sql-writer` registered as a consumer of `orion:cognition:trace` in `channels.yaml` (already true in code, previously unregistered).
- Behavior changed: `orion-graph-compression` can now optionally (flag-gated, default off) pull additional Leiden-clustering input from FalkorDB for the `episodic`/`substrate` scopes.
- Compatibility notes: `orion:chat:social` and `orion-rdf-writer`'s `orion:enrichment` write (Entity/Mention/hasTag/hasEntity -- separate from the already-removed `Claim` reification) are untouched by this PR.

## Env/config changes

- Removed keys: `CHANNEL_COGNITION_TRACE_PUB` (`orion-rdf-writer` `.env_example`/docker-compose; not present in the live `.env`, so no sync needed there).
- Added keys: `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH`, `FALKORDB_RECALL_GRAPH`, `GRAPH_COMPRESSION_SUBSTRATE_FALKOR_ENABLED`, `GRAPH_COMPRESSION_EPISODIC_FALKOR_ENABLED` (`orion-graph-compression`).
- `.env_example` updated: yes, both services.
- local `.env` synced: yes, directly edited in the primary checkout (`orion-graph-compression/.env`); `orion-rdf-writer/.env` already lacked the removed key.
- skipped keys requiring operator action: none.

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-rdf-writer venv/bin/python3 -m pytest services/orion-rdf-writer/tests -q
→ 49 passed

ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-graph-compression venv/bin/python3 -m pytest services/orion-graph-compression/tests -q
→ 42 passed

git diff --check → clean (all three commits)
scripts/check_service_env_compose_parity.py orion-rdf-writer → same pre-existing 17-key gap (Fuseki JVM tuning, PROJECT/NET), no new gap
scripts/check_service_env_compose_parity.py orion-graph-compression → N/A (env_file: covers all keys regardless of environment: list)
```

## Evals run

No eval harness exists for either service's channel dispatch or federator logic; focused deterministic tests cover the changed behavior (including a live-data-grounded regression test for the IRI-wrapping fix, using real messy entity-name examples pulled from the production FalkorDB instance during review).

## Docker/build/smoke checks

No Docker rebuild/restart performed. Both new Falkor flags default `false`, so no live behavior changes until explicitly flipped after a live verification pass (community/cluster sanity check against real Falkor data, matching the recall backfill's verification approach) -- not done in this PR, flagged as the next step.

## Review findings fixed

- Finding (HIGH): `FalkorEpisodicFederator` returned raw Entity/Tag `.name` values straight from FalkorDB into node-identity strings that `CompressionWriter._build_sparql_update` interpolates unescaped into a SPARQL IRIREF. Live-verified real `orion_recall` entity names include spaces and apostrophes (`"solar system"`, `"the 'sentience striving program'"`) -- both illegal inside a SPARQL IRIREF. Any community containing one of these nodes would produce an invalid SPARQL UPDATE, silently dropping exactly the regions this federator was added to surface.
  - Fix: both federators now wrap every returned node in a synthetic namespace + `urllib.parse.quote(..., safe="")` before returning it, matching every SPARQL federator's implicit "always a well-formed IRI" contract.
  - Evidence: commit `21b0f7ca`; new regression tests `test_fetch_wraps_free_text_entity_names_as_well_formed_iris` / `test_fetch_wraps_node_ids_as_well_formed_iris` using the real messy values found live.
- Finding (MODERATE): the original union-vs-replace worker test only asserted the Falkor federator got called and *something* got written -- it would pass identically if a bug fully replaced rather than unioned the SPARQL triples.
  - Fix: test now captures the actual triple set passed into `build_graph_from_triples` and asserts both sources' edges survived.
  - Evidence: `test_falkor_federators_unioned_with_sparql_when_flags_on` in `test_worker_degraded.py`.
- Informational (no fix needed): `orion:cognition:trace`'s channels.yaml edit only *adds* `orion-sql-writer` (was never a registered consumer to begin with, unlike `orion:metacog:trace` which had `orion-rdf-writer` cleanly removed) -- net effect correct, not a symmetric edit, noted for anyone diffing the two channel entries later.
- Informational (no fix needed): no local `.env` files existed in the review worktree to check sync against -- confirmed via the primary checkout instead (see Env/config changes above).

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-rdf-writer/.env \
  -f services/orion-rdf-writer/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-graph-compression/.env \
  -f services/orion-graph-compression/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low.
- Concern: `episodic_falkor.py` only covers the `orion_recall` (chat/tag/entity) slice of what `EpisodicFederator` reads -- `orion:collapse`/`orion:chat:social` have no Falkor writer yet and still depend on SPARQL. Not a regression (additive union), but worth tracking as the next phase.
- Mitigation: documented explicitly in `services/orion-graph-compression/README.md`; both new flags default off until a live verification pass is run.
- Severity: Low.
- Concern: `orion-rdf-writer`'s `orion:enrichment` write (Entity/Mention/hasTag/hasEntity into Fuseki) is still live and untouched -- retiring it was the original ask that led to this investigation, but it's not safe to cut until `episodic_falkor.py` is verified live as a full replacement for the clustering signal it currently feeds.
- Mitigation: explicitly out of scope for this PR, next step in the phased plan.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1254
