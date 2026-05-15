# Orion RDF Store (operator stack)

This directory is the **operator / deployment stack** for Orion’s primary **RDF datastore containers** (currently **Apache Jena Fuseki**). It is **not** a Python Orion service: there is no `app/`, `settings.py`, or `requirements.txt` here—only compose, env templates, and Makefile targets around the upstream Fuseki image.

## What this is not

- **Not the RDF writer.** The canonical bus → triples → store write path remains [`services/orion-rdf-writer`](../orion-rdf-writer/). Configure that service with `RDF_STORE_*` to point at this stack when using Fuseki.
- **Not GraphDB.** GraphDB images and the GDB client relay live under [`services/graphdb`](../graphdb/) and [`services/orion-gdb-client`](../orion-gdb-client/). This stack does not duplicate them.

This layout **replaces** the earlier `services/rdf-store/` experiment (nested `fuseki/` + redundant `graphdb/` pointer), which did not match Orion repo conventions.

## Backend and networking

- **Compose service name:** `orion-athena-fuseki` (stable container DNS: **`http://orion-athena-fuseki:3030`** on the shared external `app-net`).
- **Image:** `stain/jena-fuseki:latest`. **JVM:** the image honors **`JVM_ARGS`** (mapped from `FUSEKI_JVM_ARGS` in `.env_example`); see [Docker Hub — stain/jena-fuseki](https://hub.docker.com/r/stain/jena-fuseki).
- **Persistence:** bind-mount **`FUSEKI_DATA_DIR`** → `/fuseki` in the container (no anonymous Docker volumes). Use a **durable host path** (e.g. Athena U2-backed storage under `/mnt/storage-lukewarm/rdf-store/`).

## Quick start

```bash
cd services/orion-rdf-store
cp .env_example .env
# edit .env — set NET, passwords, and RDF_STORE_DATA_ROOT if needed
make preflight
make up
```

Writer env (typical Fuseki):

- `RDF_STORE_BACKEND=fuseki`
- `RDF_STORE_BASE_URL=http://orion-athena-fuseki:3030`
- `RDF_STORE_DATASET=orion`

## Smoke

With the bus and writer running, from the repo root (see [`services/orion-rdf-writer/README.md`](../orion-rdf-writer/README.md)):

```bash
PYTHONPATH=/path/to/Orion-Sapienform:/path/to/Orion-Sapienform/services/orion-rdf-writer \
  ./venv/bin/python scripts/smoke_chat_to_rdf_store.py
```

## Backups

Snapshot `FUSEKI_DATA_DIR` (and optionally `FUSEKI_BACKUP_DIR`) while the container is stopped, or use your platform backup tool on those host paths.

## Performance note

The writer uses an **async queue** when enabled so bus handlers are not blocked on store HTTP. Tune `RDF_WRITE_*` in `services/orion-rdf-writer/.env_example` rather than only raising bus-side timeouts.
