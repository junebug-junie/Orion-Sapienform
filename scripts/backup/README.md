# Local `/mnt/scripts` backup and nightly database backups

Nightly hard-linked snapshots of `/mnt/scripts` into `/mnt/storage-warm`, with
local evidence and optional Orion failure alerts.

This folder is the operator-facing home for the runner. Full design:
[`docs/superpowers/specs/2026-05-09-local-mnt-scripts-backup-design.md`](../../docs/superpowers/specs/2026-05-09-local-mnt-scripts-backup-design.md).
Ops runbook (install/restore):
[`docs/operations/local-mnt-scripts-backup.md`](../../docs/operations/local-mnt-scripts-backup.md).

## Why this exists

`/mnt/scripts` and `/mnt/storage-warm` are host mounts (UUID/`fstab` per node).
There is no portable “copy the disk layout” backup. What we need is a
**local-first file-restore** copy of the scripts tree onto warm storage, with:

- restorable plain directories (not a opaque backup repo)
- incremental storage via hard links
- failure evidence that survives even when Orion is down
- no hardcoded disk UUIDs in the runner

## Design (short)

| Choice | Decision |
|--------|----------|
| Mechanism | `rsync -aHAX --numeric-ids --delete` with `--link-dest` after the first success |
| Destination | `/mnt/storage-warm/backups/<node-name>/mnt-scripts/` |
| Consistency | Best-effort live copy (v1 does **not** stop Orion services) |
| Retention | Keep the latest **14** successful snapshots |
| Staging | Write to `snapshots/.incomplete-<run-id>/`, rename to `snapshots/<run-id>/` only on success |
| Identity | `<run-id>` = UTC timestamp to the second + PID (avoids same-second collisions) |
| Locking | Exclusive `backup.lock` under the backup root (dry-run skips the lock) |
| Alerts | On failure, optional POST to Orion Notify `/attention/request` |

Rejected for v1: `restic`, filesystem-native snapshots (Btrfs/ZFS/LVM), active
cross-node replication, bootable volume images, automatic restore.

## Layout on disk

```text
/mnt/storage-warm/backups/<node-name>/mnt-scripts/
  snapshots/<run-id>/          # completed snapshot (hard-linked tree)
  snapshots/.incomplete-.../   # in-flight or leftover (cleaned on next run start)
  latest -> snapshots/<run-id> # symlink updated only after success
  status/latest.json
  status/runs/<run-id>.json
  logs/<run-id>.log
  manifests/<run-id>.json
  backup.lock
```

## Files in this folder

| File | Role |
|------|------|
| `orion_backup_mnt_scripts.py` | CLI runner (stdlib + `rsync`) |
| `__init__.py` | Makes `scripts.backup` importable for tests |
| `README.md` | This document |

Systemd templates live under `deploy/systemd/`:

- `orion-backup-mnt-scripts.service`
- `orion-backup-mnt-scripts.timer` (03:15 **host local** time, `Persistent=true`)

Tests: `tests/test_orion_backup_mnt_scripts.py` (from repo root).

## Prerequisites

```bash
findmnt /mnt/scripts
findmnt /mnt/storage-warm
rsync --version
```

The backup root parent must be writable by the user that runs the job
(`/mnt/storage-warm` is often root-owned — use `sudo` for real mounts, matching
the systemd service).

## Usage

All commands from the **repo root**.

### Dry run (no copy, no lock)

```bash
sudo PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_mnt_scripts.py --dry-run
```

Prints JSON with `base_root`, `previous_snapshot`, and the planned `rsync_command`.

### Manual run

```bash
sudo PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_mnt_scripts.py
```

Inspect:

```bash
readlink -f /mnt/storage-warm/backups/$(hostname)/mnt-scripts/latest
jq . /mnt/storage-warm/backups/$(hostname)/mnt-scripts/status/latest.json
```

### Fixture / non-mount smoke

```bash
tmpdir="$(mktemp -d)"
mkdir -p "$tmpdir/source" "$tmpdir/target"
echo hello > "$tmpdir/source/hello.txt"
PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_mnt_scripts.py \
  --source "$tmpdir/source" \
  --storage-warm "$tmpdir/target" \
  --node-name smoke-node \
  --no-require-mounts
```

### CLI flags

| Flag / env | Meaning |
|------------|---------|
| `--source` | Default `/mnt/scripts` |
| `--storage-warm` | Default `/mnt/storage-warm` |
| `--node-name` | Default hostname |
| `--keep-successful` | Default `14` |
| `--no-require-mounts` | Skip mount-point checks (fixtures / smoke) |
| `--dry-run` | Plan only; no lock, no rsync |
| `--notify-url` / `ORION_BACKUP_NOTIFY_URL` | Orion Notify attention endpoint |
| `--notify-token` / `ORION_BACKUP_NOTIFY_TOKEN` | Optional token header |

### Install the nightly timer

```bash
sudo cp deploy/systemd/orion-backup-mnt-scripts.service /etc/systemd/system/
sudo cp deploy/systemd/orion-backup-mnt-scripts.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-backup-mnt-scripts.timer
systemctl list-timers orion-backup-mnt-scripts.timer
```

If you previously installed units that pointed at
`scripts/orion_backup_mnt_scripts.py`, re-copy the service file and
`daemon-reload` so `ExecStart` uses `scripts/backup/...`.

Optional `/etc/orion-backup-mnt-scripts.env`:

```bash
ORION_BACKUP_NOTIFY_URL=http://localhost:7140/attention/request
ORION_BACKUP_NOTIFY_TOKEN=<token-if-required>
```

### Restore a file or directory

```bash
ls -1 /mnt/storage-warm/backups/$(hostname)/mnt-scripts/snapshots
sudo rsync -aHAX --numeric-ids \
  /mnt/storage-warm/backups/$(hostname)/mnt-scripts/snapshots/<run-id>/<relative-path> \
  /mnt/scripts/<restore-target>
```

Do not restore over live service trees without deciding whether services should
stop first. Do **not** copy another node’s `/etc/fstab` UUIDs — remount with the
destination node’s own disk IDs, then restore files.

## Failure behavior

1. Evidence JSON is written under `status/` and `manifests/` **before** notify.
2. `latest` is left pointing at the last successful snapshot.
3. Leftover `.incomplete-*` dirs are removed at the **start** of the next run.
4. If notify is configured and the run fails, the runner POSTs a critical
   attention request (`source_service=orion-backup`, `reason=backup_failed`).

## Tests

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q
```

---

## Nightly database backups (Postgres, FalkorDB, Chroma, Convex)

`orion_backup_databases.py` -- a separate tool from the `/mnt/scripts` backup
above, because a live database data directory is not safe to `rsync`: files
mutate mid-copy (confirmed live 2026-07-24 -- a plain tar of Convex's
`db.sqlite3` failed with "file changed as we read it" while the backend was
running). Each target uses its own consistency-safe capture instead of a raw
file copy:

| Target | Capture method | Why |
|--------|-----------------|-----|
| Postgres | `docker exec ... pg_dumpall`, streamed straight to the destination file | Logical dump, always consistent regardless of concurrent writes; streamed rather than buffered in memory since a full cluster dump has been observed at 1.2GB+ live and only grows |
| FalkorDB | `redis-cli BGSAVE`, poll `INFO persistence` for `rdb_bgsave_in_progress`/`rdb_last_bgsave_status`, then copy the resulting `dump.rdb` | Atomic point-in-time RDB snapshot; polling the explicit status fields (not just watching `LASTSAVE` change) avoids both false-negatives from `LASTSAVE`'s 1-second resolution and false-failures on a save that legitimately takes longer than a short fixed timeout |
| Chroma | `docker stop`, plain copy of the host bind-mount path, `docker start` | Chroma's per-collection index segments mutate on write with no CLI-accessible checkpoint API available here; an earlier version of this tool tried a live `sqlite3 .backup` for the metadata file only and left the segment files raw-copied, which doesn't actually solve the consistency problem for those files (and can leave a stale `-wal`/`-shm` sidecar next to a `.backup` output, corrupting restore) -- stopping for the copy window is simple and actually consistent |
| Convex | Same stop/copy/start treatment, against the resolved Docker volume mountpoint (`docker volume inspect ... --format '{{.Mountpoint}}'`) | Same reason -- confirmed live that a raw copy of Convex's `db.sqlite3` tears mid-write ("file changed as we read it"), and its data directory also mixes in live RocksDB-style segment files that have no safe online-backup API exposed to this tool either |

Chroma and Convex are non-critical services (an internal vector store and an
AI-town simulation backend, not user-facing production data paths), so a few
seconds of downtime during the 03:45 backup window is an acceptable trade for
actual consistency -- unlike Postgres/FalkorDB above, which stay fully up via
their own online-backup mechanisms.

Layout mirrors the `/mnt/scripts` backup, one level down per target:
`/mnt/storage-warm/backups/<node-name>/db/<target>/{snapshots,latest,status,logs,manifests}`,
plus a shared `db/backup.lock` and `db/status/latest.json` for the whole run.
Each target has its own retention count (`Target.keep_successful`, default 14)
and is pruned independently, since a text SQL dump and a ~20GB Convex copy
have very different reasonable retention windows.

### Manual run

```bash
sudo PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_databases.py
sudo PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_databases.py --only postgres falkordb
```

Root is required for the real run: the backup root under `/mnt/storage-warm`
is root-owned (same as the `/mnt/scripts` backup above), and Convex's volume
mountpoint under `/mnt/docker/volumes/.../_data` is only host-readable as
root. The systemd service below runs as root by default (no `User=` override),
matching this.

### Install the nightly timer

```bash
sudo cp deploy/systemd/orion-backup-databases.service /etc/systemd/system/
sudo cp deploy/systemd/orion-backup-databases.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-backup-databases.timer
systemctl list-timers orion-backup-databases.timer
```

Runs at 03:45 local time, offset 30 minutes after the `/mnt/scripts` backup's
03:15 slot to avoid the two jobs contending for disk I/O at the same moment.

### Notifications

Unlike the `/mnt/scripts` backup above (failure-only), this tool notifies on
**every run** via `--notify-url`/`ORION_BACKUP_NOTIFY_URL` (same env var
names, so it can share `/etc/orion-backup-databases.env`) -- a silent
success and a silent failure otherwise look identical from the outside.
Failures are `severity: critical` with `require_ack: true`; successes are
`severity: info` with `require_ack: false`, so a healthy run doesn't demand
attention, but a real record still lands in Orion Notify's `/attention`
history either way. Point it at Orion Notify's `/attention/request` endpoint
(confirmed live on port 7140):

```bash
sudo tee /etc/orion-backup-databases.env <<'EOF'
ORION_BACKUP_NOTIFY_URL=http://localhost:7140/attention/request
EOF
```

### Tests

```bash
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_databases.py -q
```

### Known gaps (not yet covered)

- **Fuseki**: intentionally not included -- it's mid-decommission (see the
  Fuseki removal campaign), so a recurring nightly backup of a store being
  actively killed didn't seem worth building. A one-off manual archival
  snapshot before final teardown is a `docker exec ... tar` away if wanted,
  but isn't wired into this tool.
- **bus-mirror**: not a backup target -- it's a disabled debug tap, not a
  database (see `services/orion-bus-mirror/README.md`).
