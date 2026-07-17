# Local `/mnt/scripts` Backup Operations

This runbook installs and operates the local-first backup for `/mnt/scripts`.

Package home (design + usage summary): [`scripts/backup/README.md`](../../scripts/backup/README.md).
Runner: `scripts/backup/orion_backup_mnt_scripts.py`.

## What It Does

- Copies `/mnt/scripts/` to `/mnt/storage-warm/backups/<node-name>/mnt-scripts/snapshots/<run-id>/`, where `<run-id>` is UTC time to second resolution plus the process ID (for example `2026-05-09T22-39-35Z-966029`) so consecutive runs in the same second do not collide.
- Uses `rsync --link-dest` after the first successful run.
- Updates `latest` only after a successful snapshot.
- Keeps the latest 14 successful snapshots.
- Writes local status, logs, and manifests for every run.
- Sends an Orion critical attention request on failure when `ORION_BACKUP_NOTIFY_URL` is configured.

## Before Enabling

Confirm mounts:

```bash
findmnt /mnt/scripts
findmnt /mnt/storage-warm
```

Confirm rsync:

```bash
rsync --version
```

## Dry Run

Dry-run does **not** acquire the backup lock. You can run it while a real backup is in progress to inspect configuration and observe state without blocking or conflicting with the active run.

Dry-run validates that the backup root is writable by the user running the command, so operators should run it with the same privileges as the systemd service for the real `/mnt/storage-warm` target.

```bash
cd /mnt/scripts/Orion-Sapienform
sudo PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_mnt_scripts.py --dry-run
```

## First Manual Run

```bash
cd /mnt/scripts/Orion-Sapienform
sudo PYTHONPATH=. ./venv/bin/python scripts/backup/orion_backup_mnt_scripts.py
```

Inspect:

```bash
readlink -f /mnt/storage-warm/backups/$(hostname)/mnt-scripts/latest
jq . /mnt/storage-warm/backups/$(hostname)/mnt-scripts/status/latest.json
```

## Optional Orion Notification

Create `/etc/orion-backup-mnt-scripts.env`:

```bash
ORION_BACKUP_NOTIFY_URL=http://localhost:7140/attention/request
ORION_BACKUP_NOTIFY_TOKEN=<token-if-required>
```

Keep the file readable only by the service user if it contains a token.

## Install Timer

Copy the shipped units (paths are relative to the repo root):

```bash
sudo cp deploy/systemd/orion-backup-mnt-scripts.service /etc/systemd/system/
sudo cp deploy/systemd/orion-backup-mnt-scripts.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-backup-mnt-scripts.timer
systemctl list-timers orion-backup-mnt-scripts.timer
```

### Timer schedule (host local time)

The timer uses `OnCalendar=*-*-* 03:15:00`, which fires at **03:15** in the **host’s local timezone** (systemd calendar semantics). Adjust the unit file if you need a different wall-clock time or timezone policy.

### Service runtime assumptions

- **Interpreter:** `ExecStart` is `/mnt/scripts/Orion-Sapienform/venv/bin/python` (see `deploy/systemd/orion-backup-mnt-scripts.service`). Ensure that venv exists and has the dependencies the backup script needs.
- **User:** A system-installed unit runs as **root** by default unless you override `User=` / `Group=` (or use a drop-in). Change deliberately if you want a dedicated service account.
- **Long jobs:** `TimeoutStartSec=infinity` is set on purpose so large `rsync` snapshots are not killed by systemd’s default start timeout.

## Restore A File Or Directory

Choose a snapshot:

```bash
ls -1 /mnt/storage-warm/backups/$(hostname)/mnt-scripts/snapshots
```

Restore with metadata preservation:

```bash
sudo rsync -aHAX --numeric-ids /mnt/storage-warm/backups/$(hostname)/mnt-scripts/snapshots/<run-id>/<relative-path> /mnt/scripts/<restore-target>
```

The `-A` and `-X` flags preserve POSIX ACLs and extended attributes. These only work on Linux filesystems that support them (ext4, xfs, btrfs with `user_xattr`, zfs with appropriate properties). If you are restoring to a filesystem that does **not** support ACLs/xattrs, the command will fail or silently drop that metadata. Test the restore against a scratch path first, and either accept the metadata gap or drop the `-A`/`-X` flags from the invocation before restoring over real data.

Do not restore over active service directories without deciding whether services need to stop first.

## Cross-Node Note

Do not copy `/etc/fstab` UUIDs from another node. Mount `/mnt/scripts` and `/mnt/storage-warm` using the destination node's own disk identifiers, then restore files from the chosen snapshot.
