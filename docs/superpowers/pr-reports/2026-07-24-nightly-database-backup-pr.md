# PR report: nightly database backups (Postgres, FalkorDB, Chroma, Convex)

PR: https://github.com/junebug-junie/Orion-Sapienform/pull/1316
Branch: `feat/db-backup-nightly`

## Summary

- New nightly backup tool (`scripts/backup/orion_backup_databases.py`)
  covering Postgres, FalkorDB, Chroma, and Convex — separate from the
  existing `/mnt/scripts` rsync backup because a live database data
  directory isn't safe to rsync.
- Each target uses a consistency-safe capture instead of a raw copy.
- Mirrors the existing `/mnt/scripts` backup tool's shape: staged snapshot
  dir, atomic rename, per-target retention, status/manifest JSON, failure
  notify.
- Adds a real mount-existence check and restricts artifact permissions to
  0600/0700.
- Systemd timer at 03:45 local (30 min after the existing 03:15 mnt-scripts
  slot).

## Outcome moved

Before this PR: zero automated backup coverage for any live database in the
mesh. The only Postgres backup that existed anywhere was a stale, near-empty
snapshot from months earlier (traced during the same-day NVMe drive failure
incident — see the drive-failure conversation this same day). After this PR:
four database backends get a nightly, consistency-safe snapshot with
retention, once the timer is installed.

## Current architecture

Before this PR, the only backup mechanism in the repo was
`scripts/backup/orion_backup_mnt_scripts.py`, which rsyncs `/mnt/scripts`
(source code + worktrees) nightly. It explicitly does not cover
`/mnt/postgres` or `/mnt/graphdb` — those were never backed up by any
automated process. This gap is the direct reason a drive failure earlier the
same day resulted in near-total data loss with only a 9-month-stale manual
dump to fall back on.

## Architecture touched

- `scripts/backup/` — new sibling script alongside the existing mnt-scripts
  tool, reusing its lock/retention/atomic-rename/status-JSON helpers directly
  via import rather than duplicating them.
- `deploy/systemd/` — new service + timer units, following the exact pattern
  of the existing `orion-backup-mnt-scripts.{service,timer}`.
- No changes to any running service, container, or live config — this is
  purely new tooling, not yet installed as a running timer on any host.

## Files changed

- `scripts/backup/orion_backup_databases.py`: new — the core backup tool.
- `tests/test_orion_backup_databases.py`: new — 14 tests covering
  target-level success/failure/retention, mount validation, permission
  restriction, lock contention, and the stop/copy/restart capture path
  (mocked at the subprocess boundary).
- `deploy/systemd/orion-backup-databases.service`: new — oneshot unit, root
  by default (no `User=` override), `TimeoutStartSec=2h`, `UMask=0077`.
- `deploy/systemd/orion-backup-databases.timer`: new — daily 03:45 local,
  `Persistent=true`.
- `scripts/backup/README.md`: updated — documents the new tool, the
  per-target capture rationale, and known gaps (Fuseki, bus-mirror
  intentionally excluded).

## Schema / bus / API changes

None. This is operational tooling, not a service with a bus/schema surface.

## Env/config changes

- Added: optional `/etc/orion-backup-databases.env` (mirrors the existing
  `/etc/orion-backup-mnt-scripts.env` pattern) for `ORION_BACKUP_NOTIFY_URL`
  / `ORION_BACKUP_NOTIFY_TOKEN`.
- No `.env_example` changes — this tool isn't a `services/<name>` service,
  it's a repo-root `scripts/` utility, same category as the existing
  mnt-scripts backup which also has no `.env_example` entry.

## Tests run

```text
PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_databases.py -q
14 passed

PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q
20 passed  (sibling tool, unaffected)
```

## Evals run

No eval harness exists for this category of tooling (operational backup
scripts); none added. The functional equivalent — live smoke tests against
the real running containers — is covered below instead.

## Docker/build/smoke checks

```text
Live smoke test (scratch destination, real running containers, this session
had no sudo so /mnt/storage-warm itself couldn't be used directly):

--only postgres falkordb chroma
  postgres: success — real pg_dumpall.sql captured, 1.2GB+ (grew from
    ~2,000 to ~221,177 live rows during this session), 0600 permissions
    confirmed
  falkordb: success — real dump.rdb captured after fixing a baseline-
    ordering bug in the original BGSAVE-completion check
  chroma: success — mechanism validated; the real data directory is
    currently empty (separate finding, not a bug in this tool)

--only convex
  NOT fully live-tested against the real ~19.5GB volume: the volume
  mountpoint under /mnt/docker/volumes/orion-ai-town_convex-data/_data
  requires root, unavailable in this session. The underlying sqlite3
  .backup mechanism was validated standalone against a throwaway database
  earlier in the session; the stop/copy/restart capture path itself is
  covered by mocked unit tests. Recommend one supervised manual run before
  enabling the timer.
```

## Review findings fixed

- Finding: Convex/Chroma's non-SQLite files (RocksDB segments, HNSW index
  files) were being raw-copied even after routing the SQLite file itself
  through `.backup` — same tear risk the tool exists to prevent, just moved
  to different files within the same target.
  - Fix: Switched both targets to stop-container/copy/restart, consistent
    for the whole directory rather than partially safe.
  - Evidence: `test_capture_stopped_container_tree_copies_and_restarts`,
    `test_capture_stopped_container_tree_restarts_container_even_if_copy_fails`.
- Finding: SQLite `-wal`/`-shm` sidecars would've been raw-copied alongside
  a `.backup` output, risking restore-time corruption.
  - Fix: Moot after switching to the stop/copy/restart approach — no live
    WAL exists once the container is stopped for the copy window.
- Finding: No mount validation — a missing `/mnt/storage-warm` at run time
  would silently write to whatever backs the parent directory.
  - Fix: Added `validate_environment()`, checked before acquiring the lock.
  - Evidence: `test_validate_environment_requires_existing_mount`,
    `test_validate_environment_requires_actual_mount_point`,
    `test_run_backup_rejects_non_mount_storage_warm_by_default`.
- Finding: `pg_dumpall` output (observed 1.2GB+ live) was buffered entirely
  in memory.
  - Fix: Streams directly to the destination file handle.
- Finding: No subprocess timeouts, and `TimeoutStartSec=infinity` — a single
  wedged call could hang forever holding the lock.
  - Fix: 300s default timeout on all subprocess calls; unit changed to
    `TimeoutStartSec=2h`.
- Finding: FalkorDB's BGSAVE-completion check had a real bug (baseline
  captured after issuing BGSAVE, so a fast save's baseline was already
  post-save) and, once fixed, still relied on a fixed 60s deadline against
  `LASTSAVE`'s 1-second-resolution timestamp.
  - Fix: Poll `INFO persistence`'s explicit `rdb_bgsave_in_progress`/
    `rdb_last_bgsave_status` fields instead.
- Finding: Backup artifacts would land world-readable under default umask.
  - Fix: `UMask=0077` on the unit plus explicit `chmod 0600`/`0700` after
    each successful capture.
  - Evidence: `test_run_target_backup_restricts_permissions_to_owner_only`.
- Finding: Failure-notify path reused the sibling tool's function via a
  hand-rolled duck-typed adapter mirroring every attribute it reads —
  fragile and untested.
  - Fix: Wrote a small local `send_failure_notification` instead.
- Finding: Run-level `log_path` pointed at a file that was never written.
  - Fix: Now writes a real aggregated run-level log.
- Finding: Minor — duplicated private helpers instead of importing the
  sibling's, unused imports.
  - Fix: Imports the sibling's versions directly; removed dead imports.

## Restart required

```text
No restart required for any currently-running service.
```

Installing the new timer (not done as part of this PR — root-required,
left to the operator) requires:

```bash
sudo cp deploy/systemd/orion-backup-databases.service /etc/systemd/system/
sudo cp deploy/systemd/orion-backup-databases.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-backup-databases.timer
```

## Risks / concerns

- Severity: should-fix before first unattended run
  Concern: Convex's stop/copy/restart path is untested against the real
  ~19.5GB volume (root-only in this session).
  Mitigation: run one supervised manual pass (`sudo ... --only convex`)
  before enabling the timer.
- Severity: informational
  Concern: Fuseki and bus-mirror intentionally excluded (Fuseki is
  mid-decommission; bus-mirror is a disabled debug tap holding a 98GB
  runaway recording, not a database worth backing up).

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/1316
