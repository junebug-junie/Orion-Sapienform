# Local `/mnt/scripts` Backup Strategy - Design

**Date:** 2026-05-09  
**Status:** Approved design; awaiting written spec review  
**Primary Goal:** Nightly local file-restore snapshots of `/mnt/scripts` to `/mnt/storage-warm`

---

## Problem

The current node mounts `/mnt/scripts` and `/mnt/storage-warm` through host-specific filesystem UUID entries in `/etc/fstab`. That is correct for the current machine, but it is not a portable backup contract across nodes. Copying those UUID assumptions into a reusable backup strategy would break as soon as another node has different disks or volume IDs.

Orion also needs a backup failure surface that is visible to operators. A silent nightly failure is worse than no automation because it creates false confidence.

---

## Goal

Deliver a local-first backup strategy that:

1. Creates nightly snapshots of the full `/mnt/scripts` filesystem tree.
2. Stores those snapshots under `/mnt/storage-warm`.
3. Keeps the last 14 successful daily snapshots once at least 14 exist.
4. Supports simple file and directory restore from timestamped snapshot directories.
5. Writes local machine-readable evidence for every run.
6. Attempts an Orion critical notification on failure when notification settings are configured.
7. Avoids hardcoded disk UUIDs or copied `fstab` assumptions in the backup logic.

---

## Non-Goals

- No bootable volume image in v1.
- No active cross-node replication in v1.
- No service quiescing or stopping Orion services in v1.
- No database-aware dumps unless those dumps already exist inside `/mnt/scripts`.
- No automatic restore flow.
- No replacement for host-local `fstab`; mount configuration remains per-node.

---

## Selected Approach

Adopt **Approach B: local snapshot runner with status and alert sidecar**.

The runner uses `rsync --link-dest` to create complete-looking timestamped snapshot directories while hard-linking unchanged files to the previous successful snapshot. It writes local status, logs, and manifests for every run. On failure, it also attempts to notify Orion through the existing notification/attention surface.

Why this approach:

- Plain directories make restore easy to inspect and perform manually.
- Hard-linked snapshots keep daily retention storage-efficient without requiring a new backup repository format.
- Local status files remain useful even when Orion is unavailable.
- Structured manifests give future Orion ingestion and later replication a stable contract.

Rejected alternatives:

- `restic`: stronger deduplication and verification, but introduces repository/password management before it is needed.
- Filesystem-native snapshots: better atomicity when the source supports it, but more tied to disk layout and less portable across nodes.
- Minimal rsync only: simpler, but loses the structured evidence needed for operator alerts and future ingestion.

---

## Backup Scope

V1 backs up the full `/mnt/scripts` filesystem tree using a best-effort live copy. The source is treated as a mounted path, not as a specific UUID or block device.

The backup is file-restore scoped:

- Operators can restore individual files or directories from a snapshot.
- The snapshot is not guaranteed to be crash-consistent for actively written files.
- The snapshot is not intended to recreate a bootable node by itself.
- Restore onto live service paths requires operator judgment.

No broad exclude policy is part of v1. Excludes should be added only for paths that are proven unsafe or impossible to copy.

---

## Snapshot Layout

The backup root is:

```text
/mnt/storage-warm/backups/<node-name>/mnt-scripts/
```

Directory layout:

```text
snapshots/<timestamp>/
snapshots/.incomplete-<run-id>/
latest -> snapshots/<timestamp>
status/latest.json
status/runs/<run-id>.json
logs/<run-id>.log
manifests/<run-id>.json
```

Snapshot timestamps use a sortable UTC format, for example:

```text
2026-05-09T22-00-00Z
```

`latest` points only to the newest successful snapshot. Failed or interrupted runs never update `latest`.

---

## Runner Behavior

Each run:

1. Creates a unique run ID and opens a lock so only one backup can run at a time.
2. Verifies `/mnt/scripts` exists and is a mount point.
3. Verifies `/mnt/storage-warm` exists and is a mount point.
4. Verifies source and target are not the same path.
5. Verifies the backup root is writable.
6. Verifies `rsync` is available.
7. Finds the previous successful snapshot through `latest` or the newest valid snapshot directory.
8. Creates `snapshots/.incomplete-<run-id>/`.
9. Runs `rsync` into the incomplete directory.
10. If `rsync` succeeds, renames the incomplete directory to `snapshots/<timestamp>/`.
11. Updates `latest` to point at the new snapshot.
12. Prunes successful snapshots when more than 14 exist.
13. Writes status, manifest, and log evidence.
14. On failure, attempts Orion notification after local evidence is written.

The copy command should preserve normal restore metadata. When a previous successful snapshot exists, use it as the hard-link base:

```bash
rsync -aHAX --numeric-ids --delete --link-dest "$previous_snapshot" /mnt/scripts/ "$incomplete_snapshot"/
```

On the first successful run, omit `--link-dest` because there is no previous snapshot.

If a target filesystem or environment does not support `-A` or `-X`, implementation should detect and document the fallback rather than silently claiming full metadata preservation.

---

## Retention Policy

Retention keeps up to the last 14 successful snapshot directories, sorted by timestamp. After the fifteenth successful run, every successful run prunes older snapshots so only the latest 14 remain.

Rules:

- Only finalized `snapshots/<timestamp>/` directories count.
- `.incomplete-*` directories do not count as successful snapshots.
- Retention runs only after a successful new snapshot.
- Pruned snapshots are recorded in the run manifest.
- Failed run evidence is kept separately under `status/runs`, `logs`, and `manifests`.

---

## Scheduling

V1 uses a `systemd` service and timer.

The timer:

- Runs nightly.
- Uses `Persistent=true` so missed runs execute after the machine comes back.
- Delegates all backup logic to the repo-tracked runner script.

The service:

- Runs the backup runner as an operator-approved user, likely root if preserving ownership and full metadata is required.
- Emits logs to journald and the runner log file.
- Does not embed node-specific UUIDs.

Unit installation can be handled by a small install script or documented commands during implementation.

---

## Status And Manifest Contract

Every run writes machine-readable local evidence.

`status/latest.json` and `status/runs/<run-id>.json` include:

- `run_id`
- `status`: `success` or `failure`
- `node_name`
- `started_at_utc`
- `finished_at_utc`
- `source_path`
- `target_root`
- `snapshot_path`, when successful
- `latest_symlink`
- `rsync_exit_code`
- `error_summary`, when failed
- `log_path`
- `manifest_path`
- `notification_attempt`, when applicable

`manifests/<run-id>.json` includes the status fields plus operational details:

- resolved previous snapshot
- final snapshot ID
- retention actions
- filesystem usage before and after, if available
- rsync summary metrics, if available
- mount validation results

This schema is intentionally stable enough for later Orion ingestion or cross-node replication planning.

---

## Alerting

The runner always writes local evidence before attempting notification.

On failure, it attempts to send an Orion attention request through the existing notification surface with:

- `source_service=orion-backup`
- `reason=backup_failed`
- `severity=critical`
- message containing node name, source, target, run ID, and short failure summary
- context containing run ID, snapshot root, log path, manifest path, and rsync exit code

Notification failure is recorded in the status and manifest files. It does not replace or obscure the original backup failure.

Successful runs do not notify Orion by default. They update local status only to avoid nightly alert fatigue.

---

## Restore Model

Restore is manual in v1.

Basic flow:

1. Choose a snapshot under `snapshots/<timestamp>/`.
2. Inspect the file or directory to restore.
3. Restore with `rsync`, preserving metadata when appropriate:

```bash
rsync -aHAX --numeric-ids /mnt/storage-warm/backups/<node-name>/mnt-scripts/snapshots/<timestamp>/<relative-path> /mnt/scripts/<restore-target>
```

Operators should not restore over actively used service directories without deciding whether services need to be stopped first.

Cross-node restore notes:

- Do not copy the source node's `/etc/fstab` UUID assumptions.
- Mount `/mnt/scripts` and `/mnt/storage-warm` according to the destination node's own disks.
- Use paths and snapshot manifests as restore guides, not as block-device identity.

---

## Testing And Verification

Implementation should include focused verification:

- Dry-run mode confirms mount checks, lock behavior, target layout, and rsync command construction.
- A small fixture tree test verifies `--link-dest` behavior creates complete snapshots.
- Failure-path test verifies status/log/manifest are written before notification attempt.
- Retention test verifies only 14 successful snapshots remain.
- Manual smoke command runs the real runner against a temporary source and target before enabling the timer.

The first production run should be operator-triggered manually before enabling the nightly timer.

---

## Open Implementation Decisions

These are implementation details, not unresolved product requirements:

- Whether the runner is shell or Python.
- Exact nightly timer time.
- Exact notification endpoint and token configuration source.
- Whether unit installation is script-driven or documented command-driven.

The design requirement is that these choices stay node-portable and keep local evidence as the source of truth.
