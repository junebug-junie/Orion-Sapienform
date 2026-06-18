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

### Athena production layout (Jun 2026)

| Path | Role |
|------|------|
| `/mnt/graphdb/rdf-store/fuseki` | **Active** Fuseki data dir (nvme) — bind-mount → `/fuseki` |
| `/mnt/storage-lukewarm/rdf-store/fuseki` | Legacy tier; should be empty after cutover |
| `/mnt/graphdb/rdf_logs/fuseki-compact-run.log` | Compact job log |
| `/mnt/scripts/Orion-Sapienform/logs/orion-fuseki-recover.log` | Recover cron log |

**JVM (`.env`):** `FUSEKI_JVM_XMS=8g`, `FUSEKI_JVM_XMX=96g`. Fuseki runs as uid **100** (`fuseki`) inside `stain/jena-fuseki:5.1.0`.

**Writer throttling** ([`orion-rdf-writer/.env`](../orion-rdf-writer/.env)): `RDF_WRITE_WORKERS=2`, `RDF_WRITE_MAX_IN_FLIGHT=8` — slows TDB bloat; does not replace compact.

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
| Shrink index bloat (offline) | `SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion make compact` |
| Remount if active tree truncated | `make restore-mount` |

**Typical cutover cleanup:** verify health on nvme → `CONFIRM=1 make delete-stale-copy` (reclaim stale lukewarm copy) → `make compact` to shrink TDB index bloat.

---

## TDB bloat vs Orion memory (read this)

These are **different problems**:

| | **TDB index bloat** (Fuseki/TDB2 issue) | **Orion memory** (semantic data) |
|--|--|--|
| What it is | Multiple on-disk indexes per triple, journal holes, write amplification under load ([jena#2584](https://github.com/apache/jena/issues/2584)) | Identity snapshots, drive audits, chat, collapse — ~36M triples you **want** |
| GraphDB comparison | Same logical triples were ~18 GB export; TDB landed at ~1.3 TB on disk (~70×) — mostly **storage engine overhead + bloat**, not 70× more knowledge | GraphDB held the same *kind* of data; TDB2 layout is inherently wider on disk |
| Fix | **`make compact`** (offline rebuild) — **keeps your triples**, rewrites indexes tighter | Do **not** SPARQL DELETE memory graphs |
| Wrong fix | Hoping deletes aren't needed | Retention/prune on autonomy/chat (deletes memory) |

**`make compact` is the primary tool for index bloat.** It stops Fuseki and rdf-writer, runs Jena 5 **`tdb2.tdbcompact --loc --deleteOld`** in-place, fixes dataset ownership, restarts services, and runs `make health-probe`. Logical triple count stays the same; on-disk size should drop dramatically.

```bash
cd services/orion-rdf-store

# Dry-run (checks disk space, prints plan):
SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion DRY_RUN=1 make compact

# Manual compact (expect Fuseki downtime; duration scales with dataset size):
SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion make compact
```

**Jun 2026 reference:** ~395 GB bloated → ~33 GB after compact (~30 min on nvme).

### How `scripts/fuseki_tdb_compact.sh` works

Jena 5 compacts **in-place** (creates a new `Data-NNNN` tree, then `--deleteOld` removes the previous one). There is **no `--loc2`** destination flag — older docs/scripts that used `--loc2` will fail.

The script:

1. Creates **`FUSEKI_DATA_DIR/.compact-in-progress`** so `make recover` skips restarts during compact
2. Stops **`orion-athena-fuseki`** and **`orion-athena-rdf-writer`**
3. Runs **`tdb2.tdbcompact`** via host Java, or **`eclipse-temurin:21-jre-jammy`** Docker if no host JDK
4. **`chown -R 100:101`** on the dataset (compact in Docker runs as root; without this step Fuseki returns **503** / `AccessDeniedException` on `tdb.lock`)
5. Restarts Fuseki + rdf-writer, runs **`make health-probe`**

Requires free space on the dataset filesystem **≥ current dataset size** (peak usage during compact).

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
4. Cron every 20 min: `make recover` (see [Scheduled maintenance](#scheduled-maintenance-athena-cron))
5. Lower writer concurrency: `RDF_WRITE_WORKERS=2`, `RDF_WRITE_MAX_IN_FLIGHT=8`
6. Client retries via `FUSEKI_HTTP_RETRY_*` on Hub/writer paths
7. Weekly **`make compact`** — primary defense against index bloat causing lock/GC pressure

---

## Scheduled maintenance (Athena cron)

Install on the host that runs Fuseki (`crontab -e` as `athena`):

```cron
# Fuseki lock recovery — restarts when health-probe fails; skips if compact lock present
*/20 * * * * cd /mnt/scripts/Orion-Sapienform/services/orion-rdf-store && make recover >> /mnt/scripts/Orion-Sapienform/logs/orion-fuseki-recover.log 2>&1

# Weekly TDB compact — reclaims index bloat; ~downtime scales with dataset size
0 3 * * 0 cd /mnt/scripts/Orion-Sapienform/services/orion-rdf-store && SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion make compact >> /mnt/graphdb/rdf_logs/fuseki-compact-run.log 2>&1
```

| Job | Prevents bloat? | What it does |
|-----|-----------------|--------------|
| **`make recover`** (every 20 min) | No | Ping + query + write probe; restart on failure |
| **`make compact`** (weekly) | **Yes** | Offline in-place TDB rebuild; keeps all triples |
| Writer `RDF_WRITE_*` limits | Slows bloat | Continuous; configured in orion-rdf-writer |
| **`make disk-guard`** | No (ENOSPC guard) | Only runs on manual `make up` |

After weekly compact, confirm success:

```bash
tail -30 /mnt/graphdb/rdf_logs/fuseki-compact-run.log   # look for "Compact complete"
cd services/orion-rdf-store && make health-probe
du -sh /mnt/graphdb/rdf-store/fuseki/databases/orion
```

Schedule compact more often if `make storage-status` shows dataset size creeping up while triple count is stable.

---

## Troubleshooting

### Fuseki slow / high CPU / OOM

- Check dataset size: `du -sh $FUSEKI_DATA_DIR/databases/orion` — bloat shows up here before RAM helps
- Confirm JVM heap in `.env` (`FUSEKI_JVM_XMX`); container must be recreated after changes
- Ensure nothing else is hammering the store (stuck integration tests, runaway pytest)
- Run **`make compact`** if disk >> expected for triple count

### Fuseki 503 after compact

Usually **`tdb.lock` owned by root** after Docker compact. Fix and restart:

```bash
docker stop orion-athena-fuseki
docker run --rm -v /mnt/graphdb/rdf-store/fuseki/databases/orion:/db alpine \
  sh -c 'chown -R 100:101 /db && chmod -R u+rwX,g+rwX /db'
rm -f /mnt/graphdb/rdf-store/fuseki/.compact-in-progress
cd services/orion-rdf-store && docker compose restart orion-athena-fuseki && make health-probe
```

The compact script performs this chown automatically; manual fix only needed if compact was interrupted or run outside the script.

### Recover cron restarted Fuseki during compact

`make recover` checks **`$FUSEKI_DATA_DIR/.compact-in-progress`** and exits without restarting. If compact was killed mid-run, remove a stale lock:

```bash
rm -f /mnt/graphdb/rdf-store/fuseki/.compact-in-progress
```

### `Maximum lock count exceeded`

See [Performance and lock exhaustion](#performance-and-lock-exhaustion) above.

---

## Recreate Athena Fuseki ops (checklist)

Use this after a fresh host, lost crontab, or disaster recovery.

1. **Env & stack**
   ```bash
   cd services/orion-rdf-store
   cp .env_example .env
   # Confirm: FUSEKI_DATA_DIR=/mnt/graphdb/rdf-store/fuseki
   # Confirm: FUSEKI_JVM_XMS=8g, FUSEKI_JVM_XMX=96g
   make preflight && make disk-guard && make up && make health-probe
   ```

2. **Writer** — point [`orion-rdf-writer`](../orion-rdf-writer/) at `http://orion-athena-fuseki:3030`; keep `RDF_WRITE_WORKERS=2`, `RDF_WRITE_MAX_IN_FLIGHT=8`.

3. **Cron** — paste the two lines from [Scheduled maintenance](#scheduled-maintenance-athena-cron); ensure log dirs exist:
   ```bash
   mkdir -p /mnt/graphdb/rdf_logs /mnt/scripts/Orion-Sapienform/logs
   ```

4. **Post-migration cleanup** (if moving or rsyncing data):
   ```bash
   make storage-status
   CONFIRM=1 make delete-stale-copy   # after nvme is healthy active mount
   SOURCE=/mnt/graphdb/rdf-store/fuseki/databases/orion make compact
   ```

5. **Verify**
   ```bash
   make health-probe
   du -sh /mnt/graphdb/rdf-store/fuseki/databases/orion
   df -h /mnt/graphdb /mnt/storage-lukewarm
   ```

**Do not use:** `--loc2` on `tdb2.tdbcompact` (Jena 5), or `DEST=` swap-based compact flows from pre-2026 scripts — they fail on current Jena.

---

## Backups

Snapshot `FUSEKI_DATA_DIR` (and optionally `FUSEKI_BACKUP_DIR`) while the container is **stopped**, or use your platform backup tool on those host paths.

## Smoke

```bash
PYTHONPATH=/path/to/Orion-Sapienform:/path/to/Orion-Sapienform/services/orion-rdf-writer \
  ./venv/bin/python scripts/smoke_chat_to_rdf_store.py
```
