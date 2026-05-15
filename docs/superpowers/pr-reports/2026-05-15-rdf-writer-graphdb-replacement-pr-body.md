# RDF writer GraphDB replacement — PR description report

**Branch:** `chore/rdf-writer-env-compose-parity` vs `origin/main`  
**Saved:** `docs/superpowers/pr-reports/2026-05-15-rdf-writer-graphdb-replacement-pr-body.md`

Paste **from `## Summary` through end of file** into the GitHub PR description.

---

## Summary

This change set implements **GraphDB replacement spike v0** for `orion-rdf-writer`: a backend-neutral RDF persistence layer with **GraphDB as default**, **Apache Jena Fuseki** as the first alternate, **async write decoupling** from the bus hot path (bounded queue, workers, retries, dead-letter), and **operator/deploy parity** between `.env_example` and `docker-compose.yml`.

High-level goals (from plan `docs/superpowers/plans/2026-05-14-graphdb-replacement-rdf-store-spike-v0.md`):

- `RdfStoreClient` protocol + `build_rdf_store_client(settings)` for `graphdb` | `fuseki` | `generic` | `rdf4j` (alias).
- Bus-driven RDF still flows `build_triples_from_envelope` → queue or direct write → `write_graph` (no `rdf_builder.py` semantic edits in this PR).
- **Out of scope** (unchanged here): recall/hub substrate reads, concept induction, ontology edits, removing GraphDB, migrating non-writer consumers.

## Commits (9, oldest → newest)

| Commit | Subject |
|--------|---------|
| `bb7d7f88` | feat(rdf-writer): add RDF store and async pipeline settings |
| `1f264cc4` | feat(rdf-writer): add RDF store clients and graph name normalization |
| `86a138ec` | feat(rdf-writer): async RDF write queue with retries and dead-letter |
| `6d8b26d5` | fix(rdf-writer): repair /rdf/ingest triple build and RDF store write path |
| `528b2c5f` | chore(deploy): wire RDF store envs and add Fuseki compose fragment |
| `29640588` | test(smoke): add chat.history RDF store readback script |
| `9ba3c119` | test(rdf-writer): cover RDF store, write queue, and envelope integration |
| `6176c847` | docs(rdf-store): document multi-backend RDF persistence and Fuseki spike |
| `f45498a1` | chore(orion-rdf-writer): align env example and compose with RDF store spike |

## Files changed (`origin/main`…HEAD)

```
17 files changed, 1336 insertions(+), 86 deletions(-)
```

| Path | Role |
|------|------|
| `services/orion-rdf-writer/app/settings.py` | Optional `GRAPHDB_URL`; `RDF_STORE_*`, `RDF_WRITE_*`; channel contracts preserved. |
| `services/orion-rdf-writer/app/rdf_store.py` | **New.** `RdfWriteResult`, `normalize_graph_name`, clients, `build_rdf_store_client`, httpx limits helper. |
| `services/orion-rdf-writer/app/service.py` | Async queue, workers, semaphore, retries, dead-letter NDJSON, bus error hook registration, `_push_to_rdf_store`. |
| `services/orion-rdf-writer/app/main.py` | Lifespan: init/shutdown pipeline, `register_rdf_write_publisher`, extended `/health`. |
| `services/orion-rdf-writer/app/router.py` | Fix ingest: `build_triples_from_envelope`, `await _push_to_rdf_store`, `503` on queue full. |
| `services/orion-rdf-writer/docker-compose.yml` | `CHANNEL_WORLD_PULSE_GRAPH`, all `RDF_STORE_*` / `RDF_WRITE_*`, `WORLD_PULSE_GRAPH_*` passthrough. |
| `services/orion-rdf-writer/.env_example` | Documents new vars + Fuseki operator knobs. |
| `services/orion-rdf-writer/README.md` | Backends, async queue, smoke command, chat-as-canary note. |
| `services/orion-rdf-store/**` | Operator Fuseki stack (compose, Makefile, env); replaces `services/rdf-store/`. |
| `scripts/smoke_chat_to_rdf_store.py` | **New.** Synthetic `chat.history` publish + SPARQL readback with polling. |
| `scripts/smoke_chat_to_rdf.py` | One-line pointer to store-aware smoke script. |
| `services/orion-rdf-writer/tests/test_rdf_store.py` | **New.** Normalization, factory, endpoint defaults. |
| `services/orion-rdf-writer/tests/test_rdf_write_queue.py` | **New.** Retries, dead-letter, queue full. |
| `services/orion-rdf-writer/tests/test_service_rdf_store_integration.py` | **New.** Parametrized kinds → store path (sync mode). |

## Test plan

- [ ] `python3 -m compileall services/orion-rdf-writer/app` — exit 0.
- [ ] `PYTHONPATH=.:services/orion-rdf-writer ./venv/bin/python -m pytest services/orion-rdf-writer/tests -q --tb=short`  
  or `./scripts/test_service.sh orion-rdf-writer services/orion-rdf-writer/tests -q --tb=short`
- [ ] `docker compose -f services/orion-rdf-writer/docker-compose.yml config` with a valid `.env` (`PROJECT`, `NET`, `ORION_BUS_URL`, `GRAPHDB_*`, etc.).
- [ ] Optional stack smoke: with bus + writer + store running,  
  `PYTHONPATH=<repo>:<repo>/services/orion-rdf-writer ./venv/bin/python scripts/smoke_chat_to_rdf_store.py`  
  (GraphDB or Fuseki per env).

## Risk / follow-ups

- **Graph IRI parity:** GraphDB uses raw `context=<{graph}>`; Fuseki normalizes compact names to `http://conjourney.net/graph/...`. Confirm query patterns per backend.
- **Bus catalog:** RDF errors may publish a **dict** on `CHANNEL_RDF_ERROR`; validate if `ORION_BUS_ENFORCE_CATALOG=true` requires envelope-shaped payloads.
- **Fuseki image:** Verify `JVM_ARGS` vs image docs (`stain/jena-fuseki`) in operator runbooks.

## Plan reference

Implementation plan: `docs/superpowers/plans/2026-05-14-graphdb-replacement-rdf-store-spike-v0.md`
