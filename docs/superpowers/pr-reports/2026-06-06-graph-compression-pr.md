# PR: Graph compression service + orion-recall adapter

**Branch:** `feat/graph-compression`  
**Base:** `main`  
**Compare:** https://github.com/junebug-junie/Orion-Sapienform/compare/main...feat/graph-compression

## Summary

Adds a new offline **`orion-graph-compression`** service that federates Orion's RDF named graphs, detects dense regions via Leiden community detection, summarizes them (LLM Gateway with a structural fallback), and writes compression artifacts to Fuseki + a Postgres index. Also wires those artifacts into **`orion-recall`** as a new `graph_compression` backend with three recall profiles.

Two plans executed via subagent-driven development (TDD per task, two-stage review per task, plus a final full-branch review):

- `docs/superpowers/plans/2026-06-06-graph-compression-service.md`
- `docs/superpowers/plans/2026-06-06-graph-compression-recall-adapter.md`

## What's included

### New service — `services/orion-graph-compression/`

| Component | Role |
|-----------|------|
| `app/federators/` | Episodic (9 named graphs), Substrate, Self-Study SPARQL federators. Graph clauses **UNION**-ed; object terms filtered to `uri`/`bnode`. |
| `app/clustering/` | Leiden community detection + `CompressionRegionV1` builder with stable region IDs. |
| `app/summarizer.py` | `RegionSummarizer` over LLM Gateway bus RPC, structural fallback. |
| `app/writer.py` | Fuseki SPARQL UPDATE into `orion:compressions`; post-write bus events. |
| `app/worker.py` | Poll-loop worker; per-tick LLM budget; substrate clusters default `kind="hotspot"`. |
| `app/store.py` | Postgres artifact index + coalesced stale queue. |
| `app/main.py` | FastAPI lifespan, heartbeat, `/health` `/regions` `/artifacts`. |
| Infra | `settings.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt` (incl. `redis[hiredis]`), policy YAML, `.env_example`. |

### `orion-recall` integration

- `app/storage/graph_compression_adapter.py` — Postgres index + Fuseki summary fetch; cached engine per DSN; parametrized scope IN-list.
- `app/worker.py` — backend via `asyncio.to_thread`.
- `app/settings.py` — `RECALL_COMPRESSION_*` settings.
- Profiles: `graph.compressions.{global,local,v1}.yaml`.

### Shared / platform

- `orion/schemas/graph_compression.py` — region/staleness/materialized schemas (registered).
- `orion/bus/channels.yaml` — `orion:substrate:mutation:pressure`; consumer on `orion:rdf:enqueue`.
- `orion-rdf-writer` — `normalize_graph_name` for `orion:self*` and `orion:compressions`.
- `README.md` — service listed under memory services.

## Code review

Full-branch review flagged 2 Critical + 6 Important issues; all fixed in `68a54af6` and re-verified:

| ID | Issue | Fix |
|----|-------|-----|
| C1 | Missing `redis` dep → bus-less boot | Added `redis[hiredis]==5.0.7` |
| C2 | Federator conjunction → empty results | UNION between graph clauses |
| I1 | Literals serialized as IRIs | Filter objects to uri/bnode |
| I2 | `RegionSummarizer` dead code | Wired with budget + fallback |
| I3 | Recall adapter blocked loop + leaked engines | `to_thread`, cached engine, parametrized SQL |
| I4 | All substrate clusters `contradiction` | Default `hotspot` |
| I5 | Unbounded stale queue | Coalesced `WHERE NOT EXISTS` |
| I6 | Wrong autonomy graph IRIs | Corrected to `graph/autonomy/*` |

**Re-review verdict:** Ready to merge. 39 tests green.

## Test plan

- [x] `PYTHONPATH=.:services/orion-graph-compression venv/bin/python -m pytest services/orion-graph-compression/tests/ -q` → **28 passed**
- [x] `cd services/orion-recall && PYTHONPATH=. ../../venv/bin/python -m pytest tests/test_graph_compression_adapter.py tests/test_recall_profiles_compression.py tests/test_query_backends_compression.py -q` → **11 passed**

## Local `.env` sync (not committed)

Synced on operator machine from `.env_example`:

- `services/orion-graph-compression/.env` — `CHANNEL_SUBSTRATE_MUTATION_PRESSURE`, `ENABLE_LLM_SUMMARIES`
- `services/orion-recall/.env` — `RECALL_COMPRESSION_TIMEOUT_SEC`

## Notes / follow-ups (non-blocking)

- Substrate **contradiction** pressure path is intentionally inert until contradiction detection lands; channel + wiring are in place.
- `docker-compose.yml` doesn't enumerate the two newest env vars in `environment:` (defaults + `.env_example` cover them).
