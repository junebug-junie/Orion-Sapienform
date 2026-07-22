# PR report: Fuseki Phase 8/9 -- kill orionmem memory-graph RDF path (test-fixture pollution)

## Summary

- Investigated Fuseki retirement Phases 8 (Hub's memory-graph-approval RDF write) and 9 (chat_stance.py's orionmem read paths) from `docs/superpowers/specs/2026-07-17-recall-rdf-writer-falkor-cutover-phase2-spec.md`.
- Live audit found the entire Fuseki content behind this feature (1075 triples, 19 approval batches, `AffectiveDisposition`/`TypedEntity`/`Situation`/`UtteranceSlice`) traced back to fixture utterance ids (`turn-id-1..5`, matching `services/orion-hub/tests/test_memory_graph_consolidation_draft_routes.py`'s fixture convention) with zero matching Postgres `memory_cards` rows despite the write function being atomic-with-compensation. **Test-fixture pollution in production Fuseki, not real approved memory** -- purged with explicit go-ahead after reviewing the evidence together.
- Separately found the *read* side of this same feature was **already dead in production** regardless: `RECALL_MEMORY_GRAPH_SPARQL_ENABLED=false` live in orion-recall, `CHAT_STANCE_MEMORY_GRAPH_GRAPHS=` empty live in orion-cortex-exec.
- Net result: killed the whole vertical (Fuseki write in Hub's approve flow, both dead read paths, the substrate adapter, and their producer registrations) rather than migrating it -- there was no real signal to preserve.

## Outcome moved

Hub's memory-graph-approval endpoint is now Postgres-only (`memory_cards` + edges), matching every other successful cut in this Fuseki-retirement series. Two additional dead-code paths (orion-recall's memory_graph SPARQL augment, chat_stance's orionmem hints) removed as a side effect of tracing this one feature end to end.

## Current architecture

Before this patch: `orion/memory_graph/approve.py::approve_memory_graph_draft` wrote to Fuseki (graph-store HTTP + SPARQL update, with legacy GraphDB support) and Postgres atomically, with SPARQL-delete compensation on Postgres failure. `orion-recall` and `orion-cortex-exec` each had their own (both already-inert) SPARQL read paths for the same `orionmem:AffectiveDisposition` shape.

## Architecture touched

- `orion/memory_graph/approve.py`, `graphdb.py` (deleted), `rdf_target.py` (deleted).
- `services/orion-hub/scripts/memory_graph_routes.py`, `app/settings.py`, `.env_example`, `docker-compose.yml`.
- `services/orion-recall/app/{worker.py,fusion.py,settings.py}`, `memory_graph_sparql.py` (deleted), `.env_example`, `docker-compose.yml`.
- `services/orion-cortex-exec/app/chat_stance.py`, `.env_example`.
- `orion/cognition/projection_builder.py` (a second, independent producer registry shared by cortex-exec and cortex-orch).
- `orion/substrate/relational/adapters/orionmem.py` (deleted) and its three re-export layers.

## Files changed

- `orion/memory_graph/approve.py`: dropped the Fuseki/GraphDB write + compensation logic and the `graphdb_url`/`graphdb_repo`/`graphdb_user`/`graphdb_pass` params -- now Postgres-only. Exception on Postgres failure now propagates directly (no compensation needed, nothing external to roll back).
- `orion/memory_graph/graphdb.py`, `rdf_target.py`: deleted, zero remaining callers.
- `services/orion-hub/scripts/memory_graph_routes.py`: removed the `resolve_memory_graph_rdf_target()` 503-gate and the `requests.RequestException`/Fuseki-lock-exhaustion handler; `named_graph_iri_required` is now the only precondition before Postgres.
- `services/orion-hub/app/settings.py`, `.env_example`, `docker-compose.yml`: removed `MEMORY_GRAPH_APPROVAL_BACKEND`, `RDF_STORE_GRAPH_STORE_URL`/`UPDATE_URL`, `GRAPHDB_URL`/`REPO`/`USER`/`PASS`, `FUSEKI_USER`/`PASS`. Kept `RDF_STORE_QUERY_URL`/`USER`/`PASS` (still used elsewhere, read-side).
- `services/orion-recall/app/memory_graph_sparql.py`: deleted. `worker.py`: removed the `RECALL_MEMORY_GRAPH_SPARQL_ENABLED` augment block. `fusion.py`: removed the `"memory_graph_sparql"` weight/rank entries. `settings.py`, `.env_example`, `docker-compose.yml`: removed the three related keys.
- `services/orion-cortex-exec/app/chat_stance.py`: removed `fetch_chat_stance_memory_graph_hints()` (inline SPARQL, dead) and `_project_memory_graph_hints_from_beliefs()` (its intended replacement, fed only by the orionmem substrate adapter -- equally dead) and their call site. Removed the `"orionmem"` producer from `_build_unification_registry()` (chat_stance's own registry). `.env_example`: removed `CHAT_STANCE_MEMORY_GRAPH_GRAPHS`/`TIMEOUT_SEC`.
- `orion/cognition/projection_builder.py`: removed the `"orionmem"` producer from the separate shared registry used by both cortex-exec and cortex-orch.
- `orion/substrate/relational/adapters/orionmem.py`: deleted. Re-exports removed from `adapters/__init__.py`, `relational/__init__.py`, `orion/substrate/__init__.py`.
- `tests/test_memory_graph_approve.py`: rewritten (old test exercised the now-deleted GraphDB compensation path).
- `tests/test_memory_graph_graphdb_mocked.py`: deleted (tested deleted `insert_batch`).
- `services/orion-hub/tests/test_memory_graph_routes.py`: renamed/rewrote the RDF-backend-gate test to test `named_graph_iri_required` instead; kept the same `(400, 503)` tolerance the original already used (bare `TestClient` doesn't mock the DB pool, so `_pool()`'s own 503 can fire first depending on lifespan pool attachment -- pre-existing nondeterminism, not new).
- `services/orion-hub/tests/test_memory_graph_consolidation_draft_approve.py`: removed a monkeypatch targeting the now-deleted `rdf_target` module and a stale `GRAPHDB_URL` env default.
- `orion/substrate/relational/tests/test_adapters.py`: removed `TestOrionmemAdapter`.
- `orion/substrate/relational/tests/test_reducer_lane_adapters.py`: updated the hardcoded producer-count assertion (15 -> 14 from an unrelated same-day main commit's `self_state_ctx` removal, picked up by rebase -> 13 from this patch).

## Schema / bus / API changes

- Removed: Fuseki as a write target for `/api/memory/graph/approve`. No bus/channel changes -- this feature was HTTP request/response and direct SPARQL polling, not bus-driven.
- Behavior changed: `/api/memory/graph/approve` no longer returns `503 graph_backend_unconfigured`; `named_graph_iri_required` (400) is now the only remaining precondition before Postgres.
- Compatibility notes: Hub's memory-graph-approval Postgres write (`memory_cards` + edges) is unchanged -- it was already the real, working durability path.

## Env/config changes

- Removed keys: `MEMORY_GRAPH_APPROVAL_BACKEND`, `RDF_STORE_GRAPH_STORE_URL`, `RDF_STORE_UPDATE_URL`, `FUSEKI_USER`, `FUSEKI_PASS`, `GRAPHDB_URL`, `GRAPHDB_REPO`, `GRAPHDB_USER`, `GRAPHDB_PASS` (orion-hub); `RECALL_MEMORY_GRAPH_SPARQL_ENABLED`, `RECALL_MEMORY_GRAPH_NAMED_GRAPHS`, `RECALL_MEMORY_GRAPH_SPARQL_TIMEOUT_SEC` (orion-recall); `CHAT_STANCE_MEMORY_GRAPH_GRAPHS`, `CHAT_STANCE_MEMORY_GRAPH_TIMEOUT_SEC` (orion-cortex-exec).
- `.env_example` updated: yes, all three services.
- local `.env` synced: yes, directly edited in the primary checkout for all three services (confirmed independently by the review agent against live `.env` state).
- skipped keys requiring operator action: none.

## Live data operation (not part of the git diff)

Purged 1075 triples (the full content of Fuseki named graph `https://orion.example/ns/memory/ng/session/local`) after confirming test-fixture pollution -- see "Outcome moved" above for the evidence trace. Snapshot, verification queries, and report: `/tmp/purge-orionmem-test-pollution/report.md` + `snapshot.ttl` (not committed, per AGENTS.md's backfill-protocol snapshot requirement -- this was well under the 100k-row/100MB threshold).

## Tests run

```text
ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=. venv/bin/python3 -m pytest tests/test_memory_graph_approve.py orion/substrate/relational/tests -q
→ 123 passed

ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-cortex-exec venv/bin/python3 -m pytest services/orion-cortex-exec/tests/test_chat_stance_*.py -q
→ 90 passed (required temporarily copying the real .env into the worktree -- worktrees don't get gitignored files; confirmed a failure without it was a pre-existing SUBSTRATE_STORE_BACKEND-driven environment artifact, reproduced identically on unmodified origin/main, not a regression)

ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-recall venv/bin/python3 -m pytest services/orion-recall/tests -q
→ 211 passed, 3 pre-existing failures confirmed identical on origin/main

ORION_BUS_URL=redis://127.0.0.1:6379/0 PYTHONPATH=.:services/orion-hub venv/bin/python3 -m pytest services/orion-hub/tests/test_memory_graph_routes.py services/orion-hub/tests/test_memory_graph_consolidation_draft_approve.py -q
→ 1 passed, 4 pre-existing failures confirmed identical on origin/main (bare TestClient without a live Postgres pool)

git diff --check → clean
scripts/check_service_env_compose_parity.py orion-hub → N/A (env_file covers everything)
scripts/check_service_env_compose_parity.py orion-recall → 2 pre-existing missing keys, unrelated to this patch
scripts/check_service_env_compose_parity.py orion-cortex-exec → script crashes on a pre-existing !override YAML tag, confirmed same crash on origin/main
```

## Evals run

No eval harness exists for this feature; focused deterministic tests cover the changed behavior.

## Docker/build/smoke checks

No Docker rebuild/restart performed. Live data purge (Fuseki DELETE) was run directly and verified via follow-up SPARQL query (0 triples remaining).

## Review findings fixed

- Finding (informational, must-fix-before-merge but not a defect in the patch): branch was 8 commits behind `origin/main` at review time, including an unrelated same-day commit that also touches `chat_stance.py`-named files (a different module, `orion/signals/adapters/chat_stance.py`, not this branch's `services/orion-cortex-exec/app/chat_stance.py` -- confirmed no real overlap).
  - Fix: rebased onto `origin/main`, clean (no conflicts). Updated the producer-count test assertion to account for main's own independent `self_state_ctx` producer removal landing the same day (15->14 from main, ->13 from this patch).
  - Evidence: commit `b8bdce45`; re-ran full test suite post-rebase, 123 + 90 passed.
- Informational (no fix needed, real caller confirmed safe): removing `approve_memory_graph_draft`'s try/except/compensate around the Postgres call is safe -- the only production caller already has its own `except asyncpg.PostgresError` handler.
- Informational (no fix needed): the two `orionmem` producer-registry removals (chat_stance.py's own + `projection_builder.py`'s shared one) are genuinely separate registries (the latter also consumed by cortex-orch), confirmed non-redundant.
- Informational, not fixed (optional follow-up): three doc-comments elsewhere in the repo (`self_study.py`, `registry.py`, `backend_config.py`) still mention "the orionmem adapter" by name -- harmless (comments only), left for a future cleanup pass rather than scope-creeping this PR.

## Restart required

```bash
docker compose --env-file .env --env-file services/orion-hub/.env \
  -f services/orion-hub/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-recall/.env \
  -f services/orion-recall/docker-compose.yml up -d --build

docker compose --env-file .env --env-file services/orion-cortex-exec/.env \
  -f services/orion-cortex-exec/docker-compose.yml up -d --build
```

## Risks / concerns

- Severity: Low.
- Concern: the live data purge (Fuseki DELETE) is not reversible via git -- relies on the snapshot at `/tmp/purge-orionmem-test-pollution/snapshot.ttl` for recovery if the test-pollution conclusion turns out to be wrong.
- Mitigation: snapshot taken before delete, verified complete (1075 triples), decision made jointly after reviewing the wasDerivedFrom/Postgres-mismatch evidence together, not unilaterally.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1259
