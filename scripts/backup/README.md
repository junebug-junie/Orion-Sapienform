# Local `/mnt/scripts` backup

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
