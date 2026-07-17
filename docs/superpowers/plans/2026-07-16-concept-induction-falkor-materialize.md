# Concept induction → Cypher-native Falkor materialization

**Date:** 2026-07-16
**Branch:** `feat/concept-induction-falkor-materialize`
**Worktree:** `/mnt/scripts/Orion-Sapienform-concept-induction-falkor-materialize`
**Depends on:** PR #1120 (merged) — Cypher-native `FalkorSubstrateStore`
**Ambiguity resolution:** **Option A** — concept-only Falkor writes (skip evidence/hypothesis/contradiction nodes and non-concept↔concept edges). Do not extend codec in this PR.

## Goal

Stop Spark concept induction from materializing profiles via RDF (`rdf.write.request` → Fuseki). Persist Concept Atlas–compatible concept nodes through `FalkorSubstrateStore` into the shared `orion_substrate` graph Hub already uses (concept↔concept edges when the mapper emits them; live outcome today is nodes only).

## Global Constraints

1. **No SPARQL smoosh / no `payload_json` SoR** into Falkor. Writes must use existing `FalkorSubstrateStore` + `falkor_codec` only — do not invent a second Falkor client in Spark.
2. **Concept-only durable Falkor writes** (PR #1120 contract). Filter `map_concept_profile_to_substrate` output to `ConceptNodeV1` + edges where both endpoints are `node_kind="concept"`. Non-concept nodes/edges are **skipped** (documented), not upserted.
3. **Do not** flip `orion-substrate-runtime` to Falkor.
4. **Do not** HTTP into sql-writer. Drives/measurement stay Postgres/bus.
5. Keep `LocalProfileStore` as profile SoR (graph path remains additive post-save).
6. Failures on the Falkor path must be isolated like today's RDF path (log + return False; do not crash the worker).
7. Env parity: if `.env_example` changes, run `python scripts/sync_local_env_from_example.py` from worktree root in the same patch.
8. Worktree commits only. No `--no-verify`.

## Current architecture (gap)

- `_materialize_profile_graph` in `bus_worker.py` always builds RDF and publishes `rdf.write.request`.
- `map_concept_profile_to_substrate` exists but is unused by the worker; it emits Concept + Evidence + Hypothesis + Contradiction.
- `FalkorSubstrateStore.upsert_node` raises on non-concept kinds.
- concept-induction `.env_example` has no `FALKORDB_*` keys.

## Task 1: Env contract

**Files:**
- `services/orion-spark-concept-induction/.env_example`
- `services/orion-spark-concept-induction/docker-compose.yml`
- `orion/spark/concept_induction/settings.py`
- local `.env` via sync script (do not commit `.env`)

**Add keys (match Hub host-mode Falkor block):**
- `FALKORDB_URI=redis://orion-athena-falkordb:6379` (bridge DNS; Hub host-mode uses `127.0.0.1:6380`)
- `FALKORDB_SUBSTRATE_GRAPH=orion_substrate`
- `CONCEPT_PROFILE_GRAPH_BACKEND=falkor` with allowed values `falkor|rdf|disabled`

**Settings:**
- `falkordb_uri`, `falkordb_substrate_graph`, `concept_profile_graph_backend`
- Validator: normalize backend to one of `falkor|rdf|disabled`; invalid → `disabled` (fail-closed)
- Wire new keys into `docker-compose.yml` environment list

**Acceptance:**
- [ ] `.env_example` contains the three keys with comments pointing at Hub shared graph
- [ ] `python scripts/sync_local_env_from_example.py` run; report skipped keys if any
- [ ] Settings load and validate the new fields
- [ ] Commit: `feat(concept-induction): add Falkor graph backend env contract`

## Task 2: Falkor materialization + worker wire (TDD)

**Files:**
- New helper preferred: `orion/spark/concept_induction/falkor_materialization.py` (thin: map → filter concepts/edges → upsert via injectable store)
- `orion/spark/concept_induction/bus_worker.py` — `_materialize_profile_graph`
- Tests under `orion/spark/concept_induction/tests/` (new focused file + update RDF assertions in `test_concept_induction.py`)

**Behavior for `CONCEPT_PROFILE_GRAPH_BACKEND`:**
- `falkor`: map profile → substrate; filter to concepts + concept↔concept edges; `FalkorSubstrateStore.upsert_node` / `upsert_edge`; **do not** publish `rdf.write.request`
- `rdf`: keep existing RDF publish path
- `disabled`: no-op success (or skip with log); no RDF, no Falkor

**Store construction:**
- Reuse `FalkorSubstrateStore` + `FalkorSubstrateStoreConfig` + `RedisGraphQueryClient` (or injectable client)
- Allow injecting a store/client on `ConceptWorker` for tests (mirror RecordingFalkorClient patterns from `orion/substrate/tests/test_falkor_store.py`)

**TDD acceptance:**
- [ ] RED then GREEN: with backend `falkor`, post-save does **not** emit `rdf.write.request`
- [ ] Recorded Cypher has native props (`evidence_refs_json` / labels), no `$payload_json` SoR as write source of truth
- [ ] Non-concept mapper nodes are skipped (assert no ValueError; assert only concept upserts)
- [ ] A second store (or hydrate path) can read back concept-shaped native rows (reuse RecordingFalkorClient / codec patterns)
- [ ] Existing tests that assumed RDF update for default/`falkor` path; RDF path still covered when backend=`rdf`
- [ ] Commit: `feat(concept-induction): materialize profiles to Cypher-native Falkor`

## Task 3: Docs + design ground-truth

**Files:**
- Short update in `docs/superpowers/specs/2026-07-16-falkordb-property-graph-routing-design.md` ground-truth / acceptance: induction no longer RDF-materializes when Falkor path is live
- Optional pointer in cypher-native substrate design if it claims induction still RDF
- `docs/superpowers/pr-reports/2026-07-16-concept-induction-falkor-materialize-pr.md` (full AGENTS.md PR template)

**Acceptance:**
- [ ] Spec ground-truth updated
- [ ] PR report written
- [ ] Commit: `docs: record concept-induction Falkor materialization cutover`

## Non-goals

- Deleting DriveEngine / concept-induction package teardown
- `CONCEPT_PROFILE_REPOSITORY_BACKEND=graph` Fuseki read cutover
- substrate-runtime SPARQL → Falkor
- Extending codec for evidence/hypothesis
- Drive measurement into Falkor

## Restart (print for operator; do not sudo)

```bash
cd /mnt/scripts/Orion-Sapienform-concept-induction-falkor-materialize
./scripts/safe_docker_build.sh orion-spark-concept-induction up -d --build
# Hub only if shared graph contract changes (should not in this PR):
# ./scripts/safe_docker_build.sh orion-hub up -d --build
```
