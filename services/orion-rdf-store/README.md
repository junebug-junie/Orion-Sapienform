# Orion RDF Store (operator stack)

This directory is the **operator / deployment stack** for Orion’s primary **RDF datastore containers** (currently **Apache Jena Fuseki**). It is **not** a Python Orion service: there is no `app/`, `settings.py`, or `requirements.txt` here—only compose, env templates, Makefile targets, and shell scripts around the upstream Fuseki image.

## What this is not

- **Not the RDF writer.** The canonical bus → triples → store write path remains [`services/orion-rdf-writer`](../orion-rdf-writer/). Configure that service with `RDF_STORE_*` to point at this stack when using Fuseki.
- **Not GraphDB.** GraphDB images and the GDB client relay live under [`services/graphdb`](../graphdb/) and [`services/orion-gdb-client`](../orion-gdb-client/). This stack does not duplicate them.

This layout **replaces** the earlier `services/rdf-store/` experiment (nested `fuseki/` + redundant `graphdb/` pointer), which did not match Orion repo conventions.

## Backend and networking

- **Compose service name:** `orion-athena-fuseki` (stable container DNS: **`http://orion-athena-fuseki:3030`** on the shared external `app-net`).
- **Image:** pin `stain/jena-fuseki:5.1.0` (or newer 5.x). **JVM:** honors **`JVM_ARGS`** from `.env`; see [Docker Hub — stain/jena-fuseki](https://hub.docker.com/r/stain/jena-fuseki).
- **Persistence:** bind-mount **`FUSEKI_DATA_DIR`** → `/fuseki` in the container (no anonymous Docker volumes). Prefer **nvme** (`/mnt/graphdb/rdf-store/fuseki`); lukewarm HDD is a legacy tier.

## Quick start

```bash
cd services/orion-rdf-store
cp .env_example .env
make preflight
make up
make health-probe
```

Writer env (typical Fuseki):

- `RDF_STORE_BACKEND=fuseki`
- `RDF_STORE_BASE_URL=http://orion-athena-fuseki:3030`
- `RDF_STORE_DATASET=orion`

---

## Disk: why Fuseki hurt us (read this before migrating again)

Orion’s RDF store is **append-heavy** and **index-amplified**. This is expected TDB2 behaviour, not a misconfigured mount path.

### What grows on disk

| Layer | What it is |
|-------|------------|
| **TDB2 indexes** | Each triple is stored in multiple permuted B-trees (SPO, POS, OSP, graph variants). On-disk size is **several×** raw N-Quads. |
| **Named graphs** | Autonomy (`autonomy/drives`, `identity`, `goals`), chat, cognition, collapse, enrichment — each adds graph-aware index pairs. |
| **Bus ingest** | `orion-rdf-writer` POSTs N-Triples continuously (drives audits, identity snapshots, chat, traces, …). |
| **No automatic shrink** | Deletes and compaction do **not** happen unless you run retention + offline compact. |

We observed **~36M triples → ~1.3 TB** on disk (~37 KB effective per triple). A GraphDB export of ~18 GB expanded into a much larger TDB footprint after migration + ongoing writes.

### Migration trap (2× disk)

`rsync` copies the **entire** TDB tree. Until the stale copy is removed you hold **two full datasets**:

1. Active mount (Fuseki container)
2. Stale copy (lukewarm or nvme, depending on direction)

If the **destination filesystem hits 100%**, Fuseki fails with `No space left on device` — TDB cannot even write lock files. Always monitor **both** sides during migration.

### Operator commands (storage)

| Step | Command |
|------|---------|
| Active vs stale sizes | `make storage-status` |
| Remove duplicate after cutover | `CONFIRM=1 make delete-stale-copy` |
| Block start when disk low | `make disk-guard` (`FUSEKI_MIN_FREE_GB`, default 50) |
| Shrink index bloat (offline) | `SOURCE=…/databases/orion DEST=…/fuseki-compact/databases/orion make compact` |
| Remount if active tree truncated | `make restore-mount` |

**Typical cutover cleanup:** verify health on nvme → `CONFIRM=1 make delete-stale-copy` (reclaim ~1.3T on lukewarm) → run **`make compact`** to shrink TDB index bloat.

---

## TDB bloat vs Orion memory (read this)

These are **different problems**:

| | **TDB index bloat** (Fuseki/TDB2 issue) | **Orion memory** (semantic data) |
|--|--|--|
| What it is | Multiple on-disk indexes per triple, journal holes, write amplification under load ([jena#2584](https://github.com/apache/jena/issues/2584)) | Identity snapshots, drive audits, chat, collapse — ~36M triples you **want** |
| GraphDB comparison | Same logical triples were ~18 GB export; TDB landed at ~1.3 TB on disk (~70×) — mostly **storage engine overhead + bloat**, not 70× more knowledge | GraphDB held the same *kind* of data; TDB2 layout is inherently wider on disk |
| Fix | **`make compact`** (offline rebuild) — **keeps your triples**, rewrites indexes tighter | Do **not** SPARQL DELETE memory graphs |
| Wrong fix | Hoping deletes aren't needed | Retention/prune on autonomy/chat (deletes memory) |

**`make compact` is the primary tool for your goal.** It stops Fuseki briefly, runs `tdb2.tdbcompact`, swaps in a tighter database. Logical triple count stays the same; disk size should drop dramatically (exact ratio depends on how bloated indexes are).

```bash
# Needs free space on DEST filesystem (see script output). Example:
SOURCE=/mnt/storage-lukewarm/rdf-store/fuseki/databases/orion \
DEST=/mnt/graphdb/rdf-store/fuseki-compact/databases/orion \
make compact
```

Schedule compact **quarterly** or when `make storage-status` shows disk creeping up while triple count is stable.

Also keep write pressure low (`RDF_WRITE_WORKERS=2`, `RDF_WRITE_MAX_IN_FLIGHT=8`) to slow future bloat buildup.

---

## Optional retention (off by default)

`RDF_RETENTION_ENABLED=false` by default. **No memory graphs are pruned unless you explicitly set `RDF_RETENTION_POLICIES`.**

Retention runs SPARQL DELETE — it removes **old triples**, not TDB garbage. Only enable if you deliberately want to cap a **non-memory** graph (e.g. cognition trace telemetry). Never point it at autonomy/chat/collapse unless you accept losing history.

```bash
make prune-dry-run   # only when RDF_RETENTION_ENABLED=true + policies set
make prune
```

---

## Performance and lock exhaustion

The writer uses an **async queue** so bus handlers are not blocked on store HTTP. Tune `RDF_WRITE_*` in [`orion-rdf-writer/.env_example`](../orion-rdf-writer/.env_example).

Under heavy write load, Fuseki/TDB2 can return **`Maximum lock count exceeded`** ([apache/jena#2584](https://github.com/apache/jena/issues/2584)):

1. Pin `FUSEKI_IMAGE=stain/jena-fuseki:5.1.0+`
2. `make health-probe` — ping + query + graph-store POST
3. `make recover` — restart when write probe fails
4. Cron every 15–30 min: `make recover`
5. Lower writer concurrency: `RDF_WRITE_WORKERS=2`, `RDF_WRITE_MAX_IN_FLIGHT=8`
6. Client retries via `FUSEKI_HTTP_RETRY_*` on Hub/writer paths

---

## Recommended Athena cron (copy/paste)

```cron
# Fuseki lock recovery
*/20 * * * * cd /path/to/Orion-Sapienform/services/orion-rdf-store && make recover >/dev/null 2>&1

# Weekly TDB compact (reclaim index bloat — keeps all memory triples)
0 3 * * 0 cd /path/to/Orion-Sapienform/services/orion-rdf-store && \
  SOURCE=/mnt/storage-lukewarm/rdf-store/fuseki/databases/orion \
  DEST=/mnt/graphdb/rdf-store/fuseki-compact/databases/orion \
  make compact >> /mnt/graphdb/rdf_logs/fuseki-compact.log 2>&1
```

Adjust paths and ensure `SOURCE`/`DEST` are set for `compact` in that cron entry or in a wrapper script.

---

## Backups

Snapshot `FUSEKI_DATA_DIR` (and optionally `FUSEKI_BACKUP_DIR`) while the container is **stopped**, or use your platform backup tool on those host paths.

## Smoke

```bash
PYTHONPATH=/path/to/Orion-Sapienform:/path/to/Orion-Sapienform/services/orion-rdf-writer \
  ./venv/bin/python scripts/smoke_chat_to_rdf_store.py
```
