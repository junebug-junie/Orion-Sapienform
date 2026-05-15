# PR: RDF Store V1 cutover safety

## Summary

- **Canonical general RDF writes:** `services/orion-rdf-writer` only (backend-neutral `RDF_STORE_*`; bus producers unchanged).
- **Legacy writer quarantined:** `services/orion-gdb-client` defaults `GDB_CLIENT_ENABLED=false` — no GraphDB wait, repo bootstrap, listener, or heartbeat unless explicitly enabled.
- **Substrate blast-radius:** `build_substrate_store_from_env()` uses in-memory when `SUBSTRATE_STORE_BACKEND` is unset; GraphDB substrate requires explicit `SUBSTRATE_STORE_BACKEND=graphdb` (global `GRAPHDB_URL` alone no longer auto-selects).
- **Memory graph approval:** remains GraphDB-only; Hub returns `503` with `memory_graph_approval_requires_graphdb` when GraphDB is not configured (no silent migration to Fuseki/generic).
- **Operator doc:** `docs/architecture/rdf_store_v1_cutover.md` (checklist, env, rollback).

## V1 operator env (reference)

```env
RDF_STORE_BACKEND=fuseki
RDF_STORE_BASE_URL=http://orion-athena-fuseki:3030
RDF_STORE_DATASET=orion
RDF_STORE_GRAPH_STORE_URL=http://orion-athena-fuseki:3030/orion/data
RDF_STORE_QUERY_URL=http://orion-athena-fuseki:3030/orion/query
RDF_STORE_UPDATE_URL=http://orion-athena-fuseki:3030/orion/update
GDB_CLIENT_ENABLED=false
SUBSTRATE_STORE_BACKEND=in_memory
```

Hub deployments that need durable substrate may keep `SUBSTRATE_STORE_BACKEND=graphdb` in Hub `.env` / compose (explicit, not auto from `GRAPHDB_URL` alone).

## Test plan

- [x] `PYTHONPYCACHEPREFIX=/tmp/orion_pycache_$$ ./venv/bin/python -m compileall services/orion-rdf-writer services/orion-gdb-client orion/substrate orion/memory_graph -q`
- [x] `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-rdf-writer/tests -q` — **37 passed**
- [x] `PYTHONPATH=. ./venv/bin/python -m pytest orion/substrate/tests -q` — **87 passed**
- [x] `PYTHONPATH=. ./venv/bin/python -m pytest services/orion-gdb-client/tests -q` — **3 passed**
- [x] `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_memory_graph_graphdb_mocked.py tests/test_memory_graph_approve.py -q` — **2 passed**
- [ ] Runtime: `orion-rdf-writer` `/health` shows `rdf_store_backend` + queue fields (no credentials in URLs)
- [ ] Runtime: `orion-gdb-client` `/health` shows `"enabled": false` with default env
- [ ] Runtime: publish `rdf.write.request` on `orion:rdf:enqueue` → `rdf_write_enqueued` / `rdf_write_committed` logs

## Out of scope (V2)

- Backend-neutral memory graph approval store
- Backend-neutral substrate graph store
- Concept profile / self-study read cutover
- GraphDB retirement

## Files touched

| Area | Change |
|------|--------|
| `services/orion-rdf-writer/` | Startup `rdf_store_backend_selected` log; `.env_example` V1 block; compose comment; factory tests |
| `services/orion-gdb-client/` | `GDB_CLIENT_ENABLED` gate; compose default false; tests; README |
| `orion/substrate/graphdb_store.py` | Explicit GraphDB backend only; V1 default in-memory |
| `orion/memory_graph/`, Hub routes | GraphDB-only doc + `memory_graph_approval_requires_graphdb` |
| `docs/architecture/rdf_store_v1_cutover.md` | Operator cutover / rollback |

## Risk notes

- **gdb-client compose:** `depends_on: graphdb` removed; with `GDB_CLIENT_ENABLED=true`, operators must ensure GraphDB is reachable separately.
- **Hub compose** still defaults `SUBSTRATE_STORE_BACKEND=graphdb` at compose level — intentional for durable Hub stacks; code default for unset env is in-memory.
