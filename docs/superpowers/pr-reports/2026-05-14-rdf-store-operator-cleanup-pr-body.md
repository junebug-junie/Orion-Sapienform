# PR: RDF store operator cleanup + RDF writer review fixes

## Summary

- **Remove** the non-conventional `services/rdf-store/` tree (nested Fuseki + GraphDB pointer docs).
- **Add** `services/orion-rdf-store/` as a **Fuseki-only operator stack** (no Python `app/`): `docker-compose.yml`, `.env_example`, `Makefile`, `README.md` — external `app-net`, service `orion-athena-fuseki`, durable bind mount to `/fuseki`, `JVM_ARGS` from `FUSEKI_JVM_ARGS`.
- **Keep** `services/orion-rdf-writer/` as the canonical bus → triples → store path; **GraphDB** remains default; **Fuseki** remains optional via `RDF_STORE_*`.
- **Address merge-review gaps**: tests for `RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT`, smoke Basic-auth only when credentials are non-empty, writer `.env_example` compose placeholders, stale `rdf_store` comment, README `docker compose` + correct service name, `orion-rdf-store` `make config` using `--env-file .env_example`.

## Test plan

- [ ] `rg 'services/rdf-store|services/rdf-store/fuseki|services/rdf-store/graphdb' .` — only historical prose (if any) remains.
- [ ] `python3 -m compileall services/orion-rdf-writer/app`
- [ ] `PYTHONPATH=.:services/orion-rdf-writer ./venv/bin/python -m pytest services/orion-rdf-writer/tests -q --tb=short` (expect **34** tests passing).
- [ ] `docker compose -f services/orion-rdf-store/docker-compose.yml --env-file services/orion-rdf-store/.env_example config`
- [ ] `docker compose -f services/orion-rdf-writer/docker-compose.yml --env-file services/orion-rdf-writer/.env_example config` — no missing `PROJECT`/`NET` warnings; `GRAPHDB_URL` and `container_name` render with `orion-athena` prefix; dead-letter path `/app/logs/orion-rdf-writer-deadletter.ndjson`; logs volume `/mnt/storage-lukewarm/rdf_logs:/app/logs`.
- [ ] From `services/orion-rdf-store/`: `make config` (uses `.env_example`).
- [ ] Optional: run Fuseki stack `make preflight && make up`, then writer against Fuseki per README.

## Out of scope (unchanged)

- No `rdf_builder.py` semantic changes.
- No autonomy / substrate / recall / concept-induction migrations.
- No fake Python scaffolding under `services/orion-rdf-store/`.
- GraphDB images and `orion-gdb-client` remain the GraphDB path; **`.env` files are gitignored** — operators copy from `.env_example`.

## Files touched (high level)

| Area | Change |
|------|--------|
| `services/rdf-store/**` | Deleted |
| `services/orion-rdf-store/**` | New operator stack |
| `services/orion-rdf-writer/` | `.env_example`, `docker-compose.yml`, `settings.py`, `rdf_store.py`, `README.md`, `tests/test_rdf_store.py` |
| `scripts/smoke_chat_to_rdf_store.py` | Auth resolution uses non-empty credential checks |
| `services/orion-rdf-store/Makefile` | `config` target uses `--env-file .env_example` |

## Risk notes

- **Fuseki healthcheck** uses `curl` + `$/ping`; image `stain/jena-fuseki` typically includes `curl`.
- **GraphDB context**: default `RDF_STORE_NORMALIZE_GRAPHDB_CONTEXT=false` preserves legacy `context=<orion:chat>`; `true` normalizes to the conjourney graph IRI (tests assert both).
