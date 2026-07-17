# PR: Concept induction → Cypher-native Falkor materialization

## Summary

- Cut Spark concept-induction post-save graph materialization off RDF (`rdf.write.request` → Fuseki) onto Cypher-native `FalkorSubstrateStore` (shared Hub Concept Atlas graph `orion_substrate`).
- Added env contract: `CONCEPT_PROFILE_GRAPH_BACKEND=falkor|rdf|disabled`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH` (synced into local `.env`).
- Concept-only durable writes (Option A): filter mapper output to `ConceptNodeV1` + concept↔concept edges; skip evidence/hypothesis/contradiction until codec is extended.
- `LocalProfileStore` remains profile SoR; graph path stays additive and failure-isolated.

## Outcome moved

Induction profiles land as native Falkor concept nodes/edges Hub Atlas can hydrate — no blob/`payload_json` SoR, no SPARQL smoosh for this path.

## Current architecture

`_materialize_profile_graph` always built RDF via `build_concept_profile_rdf_request` and published `rdf.write.request`. Substrate mapper existed but was unused by the worker. Concept-induction `.env_example` had no Falkor keys.

## Architecture touched

- Service: `orion-spark-concept-induction` (+ shared `orion/spark/concept_induction`, `orion/substrate` reuse only)
- Contracts: none new (reuses #1120 Falkor codec/store)
- Config: service `.env_example`, compose, settings, sync script exact key

## Files changed

- `orion/spark/concept_induction/falkor_materialization.py`: map → filter → upsert helper
- `orion/spark/concept_induction/bus_worker.py`: backend-switched materialization; injectable store
- `orion/spark/concept_induction/settings.py`: new env fields + fail-closed validator
- `orion/spark/concept_induction/tests/test_falkor_materialization.py`: focused Falkor tests
- `orion/spark/concept_induction/tests/test_concept_induction.py`: default path asserts no RDF
- `services/orion-spark-concept-induction/.env_example` + `docker-compose.yml`
- `scripts/sync_local_env_from_example.py`: `CONCEPT_PROFILE_GRAPH_BACKEND` in `SYNC_EXACT`
- Specs/plan/PR report docs

## Schema / bus / API changes

- Added: none
- Removed: none (RDF path retained behind `CONCEPT_PROFILE_GRAPH_BACKEND=rdf`)
- Renamed: none
- Behavior changed: default post-save graph path is Falkor; no `rdf.write.request` when backend=`falkor`
- Compatibility notes: set `CONCEPT_PROFILE_GRAPH_BACKEND=rdf` to restore legacy RDF publish

## Env/config changes

- Added keys: `CONCEPT_PROFILE_GRAPH_BACKEND`, `FALKORDB_URI`, `FALKORDB_SUBSTRATE_GRAPH`
- Removed keys: none
- Renamed keys: none
- `.env_example` updated: yes
- local `.env` synced with `python3 scripts/sync_local_env_from_example.py`: yes (worktree + main checkout operator `.env`)
- skipped keys requiring operator action: none for this change (`PUBLISH_CORTEX_EXEC_GRAMMAR` N/A)

Default `FALKORDB_URI=redis://orion-athena-falkordb:6379` (bridge DNS). Hub host-mode still uses `127.0.0.1:6380`.

## Tests run

```text
/mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest \
  orion/spark/concept_induction/tests/test_falkor_materialization.py \
  orion/spark/concept_induction/tests/test_concept_induction.py \
  orion/spark/concept_induction/tests/test_rdf_materialization.py -q
→ 39 passed
```

## Evals run

```text
No concept-induction eval harness for this seam; unit/regression coverage above.
```

## Docker/build/smoke checks

```text
Not run in this session (code + unit tests only). Restart commands below for operator live verify.
```

## Review findings fixed

- Finding: Subagent Task dispatch blocked by usage limits; orchestrator implemented + self-reviewed.
- Finding: Bridge-network URI must not copy Hub's `127.0.0.1:6380` — fixed to `orion-athena-falkordb:6379`.
- Evidence: compose default + `.env_example` comments + synced `.env`.

## Restart required

```bash
cd /mnt/scripts/Orion-Sapienform-concept-induction-falkor-materialize
./scripts/safe_docker_build.sh orion-spark-concept-induction up -d --build
# Hub restart not required unless you want to re-check Atlas against newly written induction concepts:
# ./scripts/safe_docker_build.sh orion-hub up -d --build
```

## Risks / concerns

- Severity: medium
- Concern: Live mesh smoke (concept lands in Falkor, Hub Atlas sees it) is UNVERIFIED until restart + induction tick.
- Mitigation: unit tests cover Cypher-native props + no RDF; operator restart + Atlas check.

## PR link

(filled after `gh pr create`)
