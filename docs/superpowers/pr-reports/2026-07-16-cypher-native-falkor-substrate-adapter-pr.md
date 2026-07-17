# PR report: Cypher-native Falkor substrate adapter

## Summary

- Replaced Falkor substrate durable writes from `payload_json` blobs to Cypher-native node/edge properties.
- Added pure codec helpers for closed native property allowlists and typed hydration.
- Updated Falkor store hydration to rebuild concept nodes and edges from native rows.
- Added Concept Atlas route regression against a hydrated Falkor store test double.
- Left drive measurement, substrate-runtime env, rdf-writer, and Graphiti untouched.

## Outcome moved

Falkor is now a property graph for the Concept Atlas/substrate graph seam instead of RDF-style payload blobs behind Cypher syntax.

## Current architecture

Before this patch, `FalkorSubstrateStore` persisted `n.payload_json` and `e.payload_json`, then hydrated cache state by parsing JSON blobs. Hub Concept Atlas could persist across restarts, but Falkor queries could not use native properties beyond a few helper scalars.

## Architecture touched

- `orion/substrate/falkor_store.py`: durable Falkor adapter.
- `orion/substrate/falkor_codec.py`: Cypher-native encode/decode helpers.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: route contract regression.

## Files changed

- `orion/substrate/falkor_codec.py`: native property encode/decode allowlist.
- `orion/substrate/falkor_store.py`: native Cypher writes and hydration.
- `orion/substrate/tests/test_falkor_codec.py`: codec unit tests.
- `orion/substrate/tests/test_falkor_store.py`: adapter write/hydration/cache tests.
- `services/orion-hub/tests/test_concept_atlas_routes.py`: Concept Atlas compatibility test.
- `docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md`: acceptance evidence.
- `docs/superpowers/plans/2026-07-16-cypher-native-falkor-substrate-adapter.md`: implementation plan (committed earlier).

## Schema / bus / API changes

- Added: none.
- Removed: none.
- Renamed: none.
- Behavior changed: Falkor substrate durable representation is native Cypher properties instead of `payload_json` SoR.
- Compatibility notes: `SubstrateGraphStore` caller API remains unchanged. SPARQL store remains legacy and unchanged. Runtime env is not flipped. Hydration reconstructs concept nodes first; non-concept kinds remain out of this cutover.

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
  services/orion-hub/tests/test_concept_atlas_routes.py -q
→ 31 passed (focused adapter + Concept Atlas suite)

TDD RED evidence (Task 1):
  test_falkor_upsert_concept_uses_native_cypher_properties FAILED (payload_json still present)
  test_falkor_hydrates_concept_from_native_properties FAILED (node None under payload_json hydrate)
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

- Finding: Plan sample used `result.nodes` on `SubstrateQueryResultV1`.
 - Fix: Assert against `result.slice.nodes`.
 - Evidence: `test_falkor_hydrated_concepts_support_concept_region_query` passes.

- Finding: Subagent Task tool unavailable (usage limit); orchestrator executed plan inline.
 - Fix: Same TDD order and commits; noted here.
 - Evidence: commits `865af66a`, `b077eab9`.

## Restart required

```text
No restart required for merged code until the affected services are redeployed. For live validation after deploy, restart orion-hub only; do not flip substrate-runtime yet.
```

## Risks / concerns

- Severity: Medium
- Concern: Hydration reconstructs the native Concept Atlas subset first; non-concept substrate node kinds remain outside the first cutover.
- Mitigation: Runtime cutover is deferred until graph-shaped runtime writers have their own tests and codec coverage.

## PR link

UNVERIFIED — branch not pushed in this session.
