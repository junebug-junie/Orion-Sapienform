# Fuseki operator runbook (spike)

## Image

Default: `stain/jena-fuseki:latest` (see `docker-compose.yml` in this folder).

Verify JVM-related environment variable names against the image documentation on Docker Hub (`stain/jena-fuseki`). This repo uses `JVM_ARGS` in compose; adjust one line there and in `services/orion-rdf-writer/.env_example` if the image expects `JAVA_OPTIONS` or another name.

## Network

The compose file attaches to **external** `app-net` (or `${NET:-app-net}`) so `orion-rdf-writer` can reach the service at **`http://orion-athena-fuseki:3030`** by container DNS.

## Persistence

- Host directory: default `${FUSEKI_DATA_DIR:-/mnt/storage-lukewarm/rdf-store/fuseki}` bind-mounted to `/fuseki` in the container.
- Pre-create on the host: `mkdir -p "${FUSEKI_DATA_DIR}"`.
- **Backups:** snapshot the host `fuseki` directory while the container is stopped, or use your platform backup tool on that path (details intentionally out of scope for this spike).

## Endpoints

- UI / admin: port `3030` (override with `FUSEKI_PORT`).
- SPARQL query (dataset `orion` default): `{base}/{dataset}/query`
- Graph store POST: `{base}/{dataset}/data`

## Smoke

With the bus and writer running, from the repo root (see `services/orion-rdf-writer/README.md`):

```bash
PYTHONPATH=/path/to/Orion-Sapienform:/path/to/Orion-Sapienform/services/orion-rdf-writer \
  ./venv/bin/python scripts/smoke_chat_to_rdf_store.py
```

Set `RDF_STORE_BACKEND=fuseki`, `RDF_STORE_BASE_URL=http://orion-athena-fuseki:3030`, and `RDF_STORE_DATASET=orion` on the writer (and matching query URL overrides if non-default).
