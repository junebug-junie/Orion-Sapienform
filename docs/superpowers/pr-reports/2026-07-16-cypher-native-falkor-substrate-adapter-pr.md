# PR report: Cypher-native Falkor substrate adapter

## Summary

- Replaced Falkor substrate durable writes from `payload_json` blobs to Cypher-native node/edge properties.
- Added pure codec helpers for closed native property allowlists and typed hydration.
- Hydration prefers native scalars; legacy Hub `payload_json` rows fall back and rewrite to native (REMOVE blob).
- Persists `evidence_refs` and `taxonomy_path` as JSON string properties on concepts/edges.
- Fail-closed: durable Falkor writes accept concept nodes only (non-concept raises).
- Added Concept Atlas route regression against a hydrated Falkor store test double.
- Left drive measurement, substrate-runtime env, rdf-writer, and Graphiti untouched.

## Outcome moved

Falkor is now a property graph for the Concept Atlas/substrate graph seam instead of RDF-style payload blobs behind Cypher syntax. Live Hub blob data survives restart via one-shot legacy migrate-on-hydrate.

## Current architecture

Before this patch, `FalkorSubstrateStore` persisted `n.payload_json` and `e.payload_json`, then hydrated cache state by parsing JSON blobs. Hub Concept Atlas could persist across restarts, but Falkor queries could not use native properties beyond a few helper scalars.

## Architecture touched

- `orion/substrate/falkor_store.py`: durable Falkor adapter.
- `orion/substrate/falkor_codec.py`: Cypher-native encode/decode helpers.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: route contract regression.

## Files changed

- `orion/substrate/falkor_codec.py`: native property encode/decode allowlist; concept-only encode; evidence/taxonomy JSON props.
- `orion/substrate/falkor_store.py`: native Cypher writes, native+legacy hydration, concept-only durable writes, split hydrate test double.
- `orion/substrate/tests/test_falkor_codec.py`: codec unit tests.
- `orion/substrate/tests/test_falkor_store.py`: adapter write/hydration/migration/cache tests.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: Concept Atlas compatibility test.
- `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`: acceptance evidence.
- `docs/superpowers/plans/2026-07-16-cypher-native-falkor-substrate-adapter.md`: implementation plan (committed earlier).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: Falkor substrate durable representation is native Cypher properties instead of `payload_json` SoR. Legacy blobs migrate on hydrate. Non-concept durable upserts raise.
- Compatibility notes: `SubstrateGraphStore` caller API remains unchanged for concept paths. Callers writing non-concept nodes to Falkor must catch `ValueError` or use another backend. SPARQL store remains legacy and unchanged. Runtime env is not flipped.

## Env/config changes

- Added keys: none.
- Removed keys: none.
- Renamed keys: none.
- `.env_example` updated: no.
- local `.env` synced with `python scripts/sync_local_env_from_example.py`: not needed.
- skipped keys requiring operator action: none.

## Tests run

```text
PYTHONPATH=. venv/bin/pytest orion/substrate/tests/test_falkor_codec.py \
  orion/substrate/tests/test_falkor_store.py \
  services/orion-hub/tests/test_concept_atlas_routes.py::test_summary_reads_hydrated_falkor_store -q
→ 25 passed
```

## Evals run

```text
No eval harness exists for the Falkor adapter seam; focused deterministic tests cover the adapter and Hub route contract.
```

## Docker/build/smoke checks

```text
No Docker smoke required for this adapter-only patch. Live Falkor restart smoke remains required before runtime cutover.
scripts/safe_graphify_update.sh REFUSED (~92% node-loss guard); graphify-out left unchanged.
```

## Review findings fixed

- Finding: Live Hub Falkor data still uses blob `payload_json`; native-only hydrate would empty Concept Atlas after restart.
 - Fix: Hydrate legacy `payload_json` rows, rewrite via native upsert, `REMOVE n.payload_json` / `REMOVE e.payload_json`.
 - Evidence: `test_falkor_hydrates_legacy_payload_json_and_rewrites_native` passes.

- Finding: Non-concept kinds could be written but only concepts hydrate → silent loss.
 - Fix: `upsert_node` and `encode_node_properties` fail closed for non-concept.
 - Evidence: `test_falkor_rejects_non_concept_durable_write`, `test_encode_node_properties_rejects_non_concept`.

- Finding: `evidence_refs` and `taxonomy_path` not persisted.
 - Fix: Durable `evidence_refs_json` / `taxonomy_path_json` encode/decode.
 - Evidence: `test_encode_preserves_evidence_refs_and_taxonomy_path_round_trip`.

- Finding: `RecordingFalkorClient` reused one `hydrate_rows` list for node and edge queries.
 - Fix: Split `hydrate_node_rows` / `hydrate_edge_rows` / legacy lists; keep `hydrate_rows=` alias for nodes.
 - Evidence: `test_recording_client_splits_node_and_edge_hydrate_rows`.

- Finding (Critical): redis-py `result_set` list rows discarded headers → empty hydrate on live Hub restart.
 - Fix: `RedisGraphQueryClient` zips `QueryResult.header` to named dicts; `_normalize_rows(..., fields=)` zips positional rows to RETURN allowlists.
 - Evidence: `test_falkor_hydrates_from_redis_py_result_set_lists`, `test_redis_graph_client_returns_named_dicts_from_header`.

- Finding: Legacy migrate only cached after successful write.
 - Fix: Seed in-process cache before best-effort native rewrite.
 - Evidence: `test_falkor_legacy_migrate_keeps_cache_when_rewrite_fails`.

- Finding: Corrupt `*_json` list props decoded to `[]` silently.
 - Fix: `_parse_json_list` raises; hydrate skips the row.
 - Evidence: `test_decode_rejects_corrupt_evidence_refs_json`.

- Finding: Plan sample used `result.nodes` on `SubstrateQueryResultV1`.
 - Fix: Assert against `result.slice.nodes`.
 - Evidence: `test_falkor_hydrated_concepts_support_concept_region_query` passes.

## Restart required

```text
No restart required for merged code until the affected services are redeployed. For live validation after deploy, restart orion-hub only; do not flip substrate-runtime yet.
```

## Risks / concerns

- Severity: Medium
- Concern: Hydration reconstructs the native Concept Atlas subset first; non-concept substrate node kinds remain outside the first cutover (now fail-closed on write). Live Falkor restart smoke still required.
- Mitigation: Runtime cutover is deferred until graph-shaped runtime writers have their own tests and codec coverage.

## PR link

See GitHub PR created from this branch.
