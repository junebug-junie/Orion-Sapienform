# Local `/mnt/scripts` Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first nightly backup runner that snapshots `/mnt/scripts` into `/mnt/storage-warm` with hard-linked rsync snapshots, 14-snapshot retention, local evidence, and failure notification.

**Architecture:** Implement a focused Python runner in `scripts/orion_backup_mnt_scripts.py` with pure helper functions for paths, validation, command construction, evidence writing, retention, and notification. Keep runtime installation outside service code through `deploy/systemd/` unit templates and an operator guide.

**Tech Stack:** Python 3 standard library, `rsync`, pytest, systemd service/timer units, Orion Notify `/attention/request` HTTP endpoint.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/orion_backup_mnt_scripts.py` | CLI runner, mount validation, locking, rsync snapshot execution, evidence writing, retention, optional Orion notification |
| `tests/test_orion_backup_mnt_scripts.py` | Unit tests for layout, validation, command construction, success/failure evidence, retention, and notification payloads |
| `deploy/systemd/orion-backup-mnt-scripts.service` | Service unit template that runs the Python runner |
| `deploy/systemd/orion-backup-mnt-scripts.timer` | Nightly persistent timer unit |
| `docs/operations/local-mnt-scripts-backup.md` | Operator install, dry-run, manual first run, status inspection, restore, and timer enablement guide |

Keep the runner self-contained and importable. Avoid adding a service package unless implementation discovers the script has become too large to test comfortably.

---

### Task 1: Add Failing Layout And Validation Tests

**Files:**
- Create: `tests/test_orion_backup_mnt_scripts.py`
- Later Modify: `scripts/orion_backup_mnt_scripts.py`

- [ ] **Step 1: Create tests for config defaults, layout, timestamps, and validation**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.orion_backup_mnt_scripts import (
    BackupConfig,
    build_paths,
    snapshot_timestamp,
    validate_environment,
)


def test_snapshot_timestamp_is_sortable_utc() -> None:
    assert snapshot_timestamp("2026-05-09T22:00:00+00:00") == "2026-05-09T22-00-00Z"


def test_build_paths_uses_node_scoped_backup_root(tmp_path: Path) -> None:
    cfg = BackupConfig(
        source_path=tmp_path / "scripts",
        storage_warm_path=tmp_path / "storage-warm",
        node_name="node-a",
    )

    paths = build_paths(cfg, run_id="run-1", timestamp="2026-05-09T22-00-00Z")

    assert paths.base_root == tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts"
    assert paths.snapshots_dir == paths.base_root / "snapshots"
    assert paths.incomplete_snapshot == paths.snapshots_dir / ".incomplete-run-1"
    assert paths.final_snapshot == paths.snapshots_dir / "2026-05-09T22-00-00Z"
    assert paths.latest_symlink == paths.base_root / "latest"
    assert paths.status_latest == paths.base_root / "status" / "latest.json"
    assert paths.status_run == paths.base_root / "status" / "runs" / "run-1.json"
    assert paths.log_path == paths.base_root / "logs" / "run-1.log"
    assert paths.manifest_path == paths.base_root / "manifests" / "run-1.json"


def test_validate_environment_requires_mounted_source_and_target(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a")

    with pytest.raises(RuntimeError, match="/mnt/scripts.*mount point|source.*mount point"):
        validate_environment(cfg, is_mount=lambda path: False, rsync_path="/usr/bin/rsync")


def test_validate_environment_accepts_mounted_paths(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a")

    validate_environment(cfg, is_mount=lambda path: True, rsync_path="/usr/bin/rsync")
```

- [ ] **Step 2: Run tests and verify they fail because runner module does not exist**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.orion_backup_mnt_scripts'`.

- [ ] **Step 3: Commit tests**

```bash
git add tests/test_orion_backup_mnt_scripts.py
git commit -m "test(backup): define local mnt scripts backup layout"
```

---

### Task 2: Implement Config, Path Layout, Timestamp, And Validation Helpers

**Files:**
- Create: `scripts/orion_backup_mnt_scripts.py`
- Modify: `tests/test_orion_backup_mnt_scripts.py` only if assertions need exact error text alignment

- [ ] **Step 1: Add runner module with pure helpers**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_SOURCE = Path("/mnt/scripts")
DEFAULT_STORAGE_WARM = Path("/mnt/storage-warm")
DEFAULT_KEEP_SUCCESSFUL = 14


@dataclass(frozen=True)
class BackupConfig:
    source_path: Path = DEFAULT_SOURCE
    storage_warm_path: Path = DEFAULT_STORAGE_WARM
    node_name: str = socket.gethostname()
    keep_successful: int = DEFAULT_KEEP_SUCCESSFUL
    notify_url: str | None = None
    notify_token: str | None = None
    require_mounts: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class BackupPaths:
    base_root: Path
    snapshots_dir: Path
    incomplete_snapshot: Path
    final_snapshot: Path
    latest_symlink: Path
    status_latest: Path
    status_run: Path
    log_path: Path
    manifest_path: Path


def _coerce_datetime(value: str | dt.datetime | None) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.timezone.utc)
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def snapshot_timestamp(value: str | dt.datetime | None = None) -> str:
    return _coerce_datetime(value).strftime("%Y-%m-%dT%H-%M-%SZ")


def build_paths(cfg: BackupConfig, *, run_id: str, timestamp: str) -> BackupPaths:
    base_root = cfg.storage_warm_path / "backups" / cfg.node_name / "mnt-scripts"
    snapshots_dir = base_root / "snapshots"
    return BackupPaths(
        base_root=base_root,
        snapshots_dir=snapshots_dir,
        incomplete_snapshot=snapshots_dir / f".incomplete-{run_id}",
        final_snapshot=snapshots_dir / timestamp,
        latest_symlink=base_root / "latest",
        status_latest=base_root / "status" / "latest.json",
        status_run=base_root / "status" / "runs" / f"{run_id}.json",
        log_path=base_root / "logs" / f"{run_id}.log",
        manifest_path=base_root / "manifests" / f"{run_id}.json",
    )


def validate_environment(
    cfg: BackupConfig,
    *,
    is_mount: Callable[[Path], bool] | None = None,
    rsync_path: str | None = None,
) -> None:
    mount_check = is_mount or (lambda path: path.is_mount())
    if not cfg.source_path.exists():
        raise RuntimeError(f"source path does not exist: {cfg.source_path}")
    if not cfg.storage_warm_path.exists():
        raise RuntimeError(f"target path does not exist: {cfg.storage_warm_path}")
    if cfg.require_mounts and not mount_check(cfg.source_path):
        raise RuntimeError(f"source path must be a mount point: {cfg.source_path}")
    if cfg.require_mounts and not mount_check(cfg.storage_warm_path):
        raise RuntimeError(f"target path must be a mount point: {cfg.storage_warm_path}")
    if cfg.source_path.resolve() == cfg.storage_warm_path.resolve():
        raise RuntimeError("source and target paths must be different")
    if not (rsync_path or shutil.which("rsync")):
        raise RuntimeError("rsync binary not found on PATH")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create local hard-linked snapshots of /mnt/scripts.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--storage-warm", type=Path, default=DEFAULT_STORAGE_WARM)
    parser.add_argument("--node-name", default=socket.gethostname())
    parser.add_argument("--keep-successful", type=int, default=DEFAULT_KEEP_SUCCESSFUL)
    parser.add_argument("--notify-url", default=os.environ.get("ORION_BACKUP_NOTIFY_URL"))
    parser.add_argument("--notify-token", default=os.environ.get("ORION_BACKUP_NOTIFY_TOKEN"))
    parser.add_argument("--no-require-mounts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = BackupConfig(
        source_path=args.source,
        storage_warm_path=args.storage_warm,
        node_name=args.node_name,
        keep_successful=args.keep_successful,
        notify_url=args.notify_url,
        notify_token=args.notify_token,
        require_mounts=not args.no_require_mounts,
        dry_run=args.dry_run,
    )
    validate_environment(cfg)
    print(json.dumps({"ok": True, "dry_run": cfg.dry_run, "node_name": cfg.node_name}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run layout tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: PASS for the initial helper tests.

- [ ] **Step 3: Commit helpers**

```bash
git add scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py
git commit -m "feat(backup): add local snapshot runner helpers"
```

---

### Task 3: Add Snapshot Discovery, Rsync Command Construction, And Dry-Run Plan

**Files:**
- Modify: `scripts/orion_backup_mnt_scripts.py`
- Modify: `tests/test_orion_backup_mnt_scripts.py`

- [ ] **Step 1: Add failing tests for previous snapshot and rsync command**

```python
from scripts.orion_backup_mnt_scripts import build_rsync_command, find_previous_snapshot


def test_find_previous_snapshot_uses_latest_symlink(tmp_path: Path) -> None:
    root = tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts"
    previous = root / "snapshots" / "2026-05-08T22-00-00Z"
    previous.mkdir(parents=True)
    (root / "latest").symlink_to(previous)

    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-2", timestamp="2026-05-09T22-00-00Z")

    assert find_previous_snapshot(paths) == previous


def test_build_rsync_command_omits_link_dest_without_previous(tmp_path: Path) -> None:
    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-1", timestamp="2026-05-09T22-00-00Z")

    cmd = build_rsync_command(cfg, paths, previous_snapshot=None)

    assert "--link-dest" not in cmd
    assert str(cfg.source_path) + "/" in cmd
    assert str(paths.incomplete_snapshot) + "/" in cmd


def test_build_rsync_command_uses_previous_snapshot_for_link_dest(tmp_path: Path) -> None:
    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-2", timestamp="2026-05-09T22-00-00Z")
    previous = tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts" / "snapshots" / "2026-05-08T22-00-00Z"

    cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)

    assert "--link-dest" in cmd
    assert str(previous) in cmd
```

- [ ] **Step 2: Run tests and verify failures**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: FAIL with missing `find_previous_snapshot` and `build_rsync_command`.

- [ ] **Step 3: Implement snapshot discovery and command construction**

```python
def find_previous_snapshot(paths: BackupPaths) -> Path | None:
    latest = paths.latest_symlink
    if latest.is_symlink():
        target = latest.resolve()
        if target.exists() and target.is_dir():
            return target
    if not paths.snapshots_dir.exists():
        return None
    candidates = sorted(
        path for path in paths.snapshots_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".incomplete-")
    )
    return candidates[-1] if candidates else None


def build_rsync_command(cfg: BackupConfig, paths: BackupPaths, *, previous_snapshot: Path | None) -> list[str]:
    cmd = ["rsync", "-aHAX", "--numeric-ids", "--delete"]
    if previous_snapshot is not None:
        cmd.extend(["--link-dest", str(previous_snapshot)])
    cmd.extend([str(cfg.source_path) + "/", str(paths.incomplete_snapshot) + "/"])
    return cmd
```

- [ ] **Step 4: Extend `main()` dry-run output**

```python
    now = _coerce_datetime(None)
    timestamp = snapshot_timestamp(now)
    run_id = f"{timestamp}-{os.getpid()}"
    paths = build_paths(cfg, run_id=run_id, timestamp=timestamp)
    previous = find_previous_snapshot(paths)
    cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)
    if cfg.dry_run:
        print(json.dumps({
            "ok": True,
            "dry_run": True,
            "node_name": cfg.node_name,
            "base_root": str(paths.base_root),
            "previous_snapshot": str(previous) if previous else None,
            "rsync_command": cmd,
        }, sort_keys=True))
        return 0
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit command planning**

```bash
git add scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py
git commit -m "feat(backup): plan rsync link-dest snapshots"
```

---

### Task 4: Implement Evidence Writing, Successful Snapshot Finalization, And Retention

**Files:**
- Modify: `scripts/orion_backup_mnt_scripts.py`
- Modify: `tests/test_orion_backup_mnt_scripts.py`

- [ ] **Step 1: Add failing tests for success evidence and retention**

```python
import json

from scripts.orion_backup_mnt_scripts import RunOutcome, prune_successful_snapshots, write_run_evidence


def test_write_run_evidence_updates_latest_status_and_manifest(tmp_path: Path) -> None:
    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-1", timestamp="2026-05-09T22-00-00Z")
    outcome = RunOutcome(
        run_id="run-1",
        status="success",
        node_name="node-a",
        started_at_utc="2026-05-09T22:00:00Z",
        finished_at_utc="2026-05-09T22:01:00Z",
        source_path=str(cfg.source_path),
        target_root=str(paths.base_root),
        snapshot_path=str(paths.final_snapshot),
        latest_symlink=str(paths.latest_symlink),
        rsync_exit_code=0,
        error_summary=None,
        log_path=str(paths.log_path),
        manifest_path=str(paths.manifest_path),
        notification_attempt=None,
        retention_actions=[],
        previous_snapshot=None,
        mount_validation={"source": True, "target": True},
    )

    write_run_evidence(paths, outcome)

    assert json.loads(paths.status_latest.read_text())["status"] == "success"
    assert json.loads(paths.status_run.read_text())["run_id"] == "run-1"
    assert json.loads(paths.manifest_path.read_text())["snapshot_path"] == str(paths.final_snapshot)


def test_prune_successful_snapshots_keeps_latest_14(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    for day in range(1, 17):
        (snapshots / f"2026-05-{day:02d}T22-00-00Z").mkdir()
    (snapshots / ".incomplete-run").mkdir()

    removed = prune_successful_snapshots(snapshots, keep=14)

    assert removed == [
        str(snapshots / "2026-05-01T22-00-00Z"),
        str(snapshots / "2026-05-02T22-00-00Z"),
    ]
    assert not (snapshots / "2026-05-01T22-00-00Z").exists()
    assert (snapshots / ".incomplete-run").exists()
```

- [ ] **Step 2: Run tests and verify failures**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: FAIL with missing `RunOutcome`, `write_run_evidence`, and `prune_successful_snapshots`.

- [ ] **Step 3: Implement evidence and retention helpers**

```python
@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: str
    node_name: str
    started_at_utc: str
    finished_at_utc: str
    source_path: str
    target_root: str
    snapshot_path: str | None
    latest_symlink: str
    rsync_exit_code: int | None
    error_summary: str | None
    log_path: str
    manifest_path: str
    notification_attempt: dict[str, Any] | None
    retention_actions: list[str]
    previous_snapshot: str | None
    mount_validation: dict[str, bool]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_name = handle.name
    Path(tmp_name).replace(path)


def write_run_evidence(paths: BackupPaths, outcome: RunOutcome) -> None:
    payload = asdict(outcome)
    _write_json_atomic(paths.status_run, payload)
    _write_json_atomic(paths.status_latest, payload)
    _write_json_atomic(paths.manifest_path, payload)


def prune_successful_snapshots(snapshots_dir: Path, *, keep: int) -> list[str]:
    if keep < 1:
        raise ValueError("keep must be at least 1")
    if not snapshots_dir.exists():
        return []
    candidates = sorted(
        path for path in snapshots_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".incomplete-")
    )
    to_remove = candidates[:-keep]
    removed: list[str] = []
    for path in to_remove:
        shutil.rmtree(path)
        removed.append(str(path))
    return removed
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit evidence and retention helpers**

```bash
git add scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py
git commit -m "feat(backup): write snapshot evidence and retention state"
```

---

### Task 5: Implement Backup Execution And Failure Evidence Before Notification

**Files:**
- Modify: `scripts/orion_backup_mnt_scripts.py`
- Modify: `tests/test_orion_backup_mnt_scripts.py`

- [ ] **Step 1: Add failing tests for successful and failed runs**

```python
from scripts.orion_backup_mnt_scripts import run_backup


def test_run_backup_finalizes_snapshot_and_latest_symlink(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a", require_mounts=False)

    def fake_run(cmd: list[str], log_path: Path) -> int:
        destination = Path(cmd[-1].rstrip("/"))
        destination.mkdir(parents=True)
        (destination / "hello.txt").write_text("world\n")
        log_path.write_text("fake rsync ok\n")
        return 0

    outcome = run_backup(cfg, now="2026-05-09T22:00:00+00:00", process_runner=fake_run)

    assert outcome.status == "success"
    assert Path(outcome.snapshot_path or "").is_dir()
    latest = target / "backups" / "node-a" / "mnt-scripts" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == Path(outcome.snapshot_path or "").resolve()


def test_run_backup_writes_failure_evidence_before_notification(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a", require_mounts=False, notify_url="http://notify.invalid/attention/request")
    seen_status = {}

    def fake_run(cmd: list[str], log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("rsync failed\n")
        return 23

    def fake_notify(outcome: RunOutcome, cfg: BackupConfig) -> dict[str, Any]:
        seen_status["exists"] = Path(outcome.manifest_path).exists()
        return {"attempted": True, "ok": False, "error": "blocked"}

    outcome = run_backup(cfg, now="2026-05-09T22:00:00+00:00", process_runner=fake_run, notifier=fake_notify)

    assert outcome.status == "failure"
    assert seen_status == {"exists": True}
    assert Path(outcome.manifest_path).exists()
```

- [ ] **Step 2: Run tests and verify failures**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: FAIL with missing `run_backup`.

- [ ] **Step 3: Implement process execution and run orchestration**

```python
def _utc_iso(value: str | dt.datetime | None = None) -> str:
    return _coerce_datetime(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_process_runner(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def _update_latest_symlink(latest: Path, target: Path) -> None:
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_name(f".{latest.name}.tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target)
    tmp.replace(latest)


def send_failure_notification(outcome: RunOutcome, cfg: BackupConfig) -> dict[str, Any]:
    if not cfg.notify_url:
        return {"attempted": False, "ok": False, "reason": "notify_url_not_configured"}
    payload = {
        "source_service": "orion-backup",
        "reason": "backup_failed",
        "severity": "critical",
        "message": f"Backup failed on {outcome.node_name}: {outcome.error_summary}",
        "require_ack": True,
        "context": {
            "run_id": outcome.run_id,
            "source_path": outcome.source_path,
            "target_root": outcome.target_root,
            "log_path": outcome.log_path,
            "manifest_path": outcome.manifest_path,
            "rsync_exit_code": outcome.rsync_exit_code,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(cfg.notify_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    if cfg.notify_token:
        request.add_header("X-Orion-Notify-Token", cfg.notify_token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"attempted": True, "ok": 200 <= response.status < 300, "status": response.status}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def run_backup(
    cfg: BackupConfig,
    *,
    now: str | dt.datetime | None = None,
    process_runner: Callable[[list[str], Path], int] = _default_process_runner,
    notifier: Callable[[RunOutcome, BackupConfig], dict[str, Any]] = send_failure_notification,
) -> RunOutcome:
    started = _utc_iso(now)
    timestamp = snapshot_timestamp(now)
    run_id = f"{timestamp}-{os.getpid()}"
    paths = build_paths(cfg, run_id=run_id, timestamp=timestamp)
    paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    previous = find_previous_snapshot(paths)
    cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)
    rsync_exit = None
    error_summary = None
    snapshot_path: str | None = None
    retention_actions: list[str] = []
    try:
        if paths.incomplete_snapshot.exists():
            shutil.rmtree(paths.incomplete_snapshot)
        paths.incomplete_snapshot.mkdir(parents=True)
        rsync_exit = process_runner(cmd, paths.log_path)
        if rsync_exit != 0:
            raise RuntimeError(f"rsync exited with {rsync_exit}")
        paths.incomplete_snapshot.replace(paths.final_snapshot)
        _update_latest_symlink(paths.latest_symlink, paths.final_snapshot)
        retention_actions = prune_successful_snapshots(paths.snapshots_dir, keep=cfg.keep_successful)
        snapshot_path = str(paths.final_snapshot)
        status = "success"
    except Exception as exc:
        status = "failure"
        error_summary = str(exc)
    outcome = RunOutcome(
        run_id=run_id,
        status=status,
        node_name=cfg.node_name,
        started_at_utc=started,
        finished_at_utc=_utc_iso(None),
        source_path=str(cfg.source_path),
        target_root=str(paths.base_root),
        snapshot_path=snapshot_path,
        latest_symlink=str(paths.latest_symlink),
        rsync_exit_code=rsync_exit,
        error_summary=error_summary,
        log_path=str(paths.log_path),
        manifest_path=str(paths.manifest_path),
        notification_attempt=None,
        retention_actions=retention_actions,
        previous_snapshot=str(previous) if previous else None,
        mount_validation={"source": True, "target": True},
    )
    write_run_evidence(paths, outcome)
    if outcome.status == "failure":
        notify_result = notifier(outcome, cfg)
        outcome = RunOutcome(**{**asdict(outcome), "notification_attempt": notify_result})
        write_run_evidence(paths, outcome)
    return outcome
```

- [ ] **Step 4: Wire `main()` to call `run_backup()`**

```python
    if cfg.dry_run:
        now = _coerce_datetime(None)
        timestamp = snapshot_timestamp(now)
        run_id = f"{timestamp}-{os.getpid()}"
        paths = build_paths(cfg, run_id=run_id, timestamp=timestamp)
        previous = find_previous_snapshot(paths)
        cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)
        print(json.dumps({
            "ok": True,
            "dry_run": True,
            "node_name": cfg.node_name,
            "base_root": str(paths.base_root),
            "previous_snapshot": str(previous) if previous else None,
            "rsync_command": cmd,
        }, sort_keys=True))
        return 0
    outcome = run_backup(cfg)
    print(json.dumps(asdict(outcome), sort_keys=True))
    return 0 if outcome.status == "success" else 1
```

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit execution path**

```bash
git add scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py
git commit -m "feat(backup): execute local rsync snapshots with evidence"
```

---

### Task 6: Add Locking, CLI Mount Overrides, And Real Temporary Smoke Test

**Files:**
- Modify: `scripts/orion_backup_mnt_scripts.py`
- Modify: `tests/test_orion_backup_mnt_scripts.py`

- [ ] **Step 1: Add failing tests for lock contention and dry-run CLI**

```python
from scripts.orion_backup_mnt_scripts import acquire_lock


def test_acquire_lock_rejects_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "backup.lock"
    first = acquire_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            acquire_lock(lock_path)
    finally:
        first.close()


def test_cli_dry_run_allows_non_mount_fixture_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.orion_backup_mnt_scripts import main

    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()

    rc = main([
        "--source", str(source),
        "--storage-warm", str(target),
        "--node-name", "node-a",
        "--no-require-mounts",
        "--dry-run",
    ])

    assert rc == 0
    assert '"dry_run": true' in capsys.readouterr().out
```

- [ ] **Step 2: Run tests and verify lock failure**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: FAIL with missing `acquire_lock`.

- [ ] **Step 3: Implement lock helper and use it in `run_backup()`**

```python
import fcntl


def acquire_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"backup already running: {lock_path}") from exc
    handle.write(str(os.getpid()))
    handle.flush()
    return handle
```

In `run_backup()`, after `paths = build_paths(cfg, run_id=run_id, timestamp=timestamp)`, acquire a lock and release it on every return path:

```python
    lock_handle = acquire_lock(paths.base_root / "backup.lock")
    try:
        paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        previous = find_previous_snapshot(paths)
        cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)
        rsync_exit = None
        error_summary = None
        snapshot_path: str | None = None
        retention_actions: list[str] = []
        try:
            if paths.incomplete_snapshot.exists():
                shutil.rmtree(paths.incomplete_snapshot)
            paths.incomplete_snapshot.mkdir(parents=True)
            rsync_exit = process_runner(cmd, paths.log_path)
            if rsync_exit != 0:
                raise RuntimeError(f"rsync exited with {rsync_exit}")
            paths.incomplete_snapshot.replace(paths.final_snapshot)
            _update_latest_symlink(paths.latest_symlink, paths.final_snapshot)
            retention_actions = prune_successful_snapshots(paths.snapshots_dir, keep=cfg.keep_successful)
            snapshot_path = str(paths.final_snapshot)
            status = "success"
        except Exception as exc:
            status = "failure"
            error_summary = str(exc)
        outcome = RunOutcome(
            run_id=run_id,
            status=status,
            node_name=cfg.node_name,
            started_at_utc=started,
            finished_at_utc=_utc_iso(None),
            source_path=str(cfg.source_path),
            target_root=str(paths.base_root),
            snapshot_path=snapshot_path,
            latest_symlink=str(paths.latest_symlink),
            rsync_exit_code=rsync_exit,
            error_summary=error_summary,
            log_path=str(paths.log_path),
            manifest_path=str(paths.manifest_path),
            notification_attempt=None,
            retention_actions=retention_actions,
            previous_snapshot=str(previous) if previous else None,
            mount_validation={"source": True, "target": True},
        )
        write_run_evidence(paths, outcome)
        if outcome.status == "failure":
            notify_result = notifier(outcome, cfg)
            outcome = RunOutcome(**{**asdict(outcome), "notification_attempt": notify_result})
            write_run_evidence(paths, outcome)
        return outcome
    finally:
        lock_handle.close()
```

- [ ] **Step 4: Run unit tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: PASS.

- [ ] **Step 5: Run real temporary smoke command**

Run:

```bash
tmpdir="$(mktemp -d)" && mkdir -p "$tmpdir/source" "$tmpdir/target" && echo "hello" > "$tmpdir/source/hello.txt" && PYTHONPATH=. ./venv/bin/python scripts/orion_backup_mnt_scripts.py --source "$tmpdir/source" --storage-warm "$tmpdir/target" --node-name smoke-node --no-require-mounts && test -L "$tmpdir/target/backups/smoke-node/mnt-scripts/latest"
```

Expected: exit code `0`; output JSON includes `"status": "success"`; `test -L` passes.

- [ ] **Step 6: Commit locking and smoke-ready CLI**

```bash
git add scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py
git commit -m "feat(backup): guard snapshot runs with a lock"
```

---

### Task 7: Add Systemd Units

**Files:**
- Create: `deploy/systemd/orion-backup-mnt-scripts.service`
- Create: `deploy/systemd/orion-backup-mnt-scripts.timer`

- [ ] **Step 1: Create service unit**

```ini
[Unit]
Description=Orion local /mnt/scripts backup
Documentation=file:/mnt/scripts/Orion-Sapienform/docs/operations/local-mnt-scripts-backup.md
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/mnt/scripts/Orion-Sapienform
EnvironmentFile=-/etc/orion-backup-mnt-scripts.env
ExecStart=/mnt/scripts/Orion-Sapienform/venv/bin/python /mnt/scripts/Orion-Sapienform/scripts/orion_backup_mnt_scripts.py
Nice=10
IOSchedulingClass=best-effort
IOSchedulingPriority=7
```

- [ ] **Step 2: Create timer unit**

```ini
[Unit]
Description=Nightly Orion local /mnt/scripts backup

[Timer]
OnCalendar=*-*-* 03:15:00
Persistent=true
Unit=orion-backup-mnt-scripts.service

[Install]
WantedBy=timers.target
```

- [ ] **Step 3: Verify unit syntax if systemd tools are available**

Run: `systemd-analyze verify deploy/systemd/orion-backup-mnt-scripts.service deploy/systemd/orion-backup-mnt-scripts.timer`

Expected: exit code `0`. If `systemd-analyze` is unavailable in the environment, record that verification blocker and still run the text checks in Step 4.

- [ ] **Step 4: Run text checks for unit contents**

Run:

```bash
grep -R "Persistent=true" deploy/systemd/orion-backup-mnt-scripts.timer && grep -R "orion_backup_mnt_scripts.py" deploy/systemd/orion-backup-mnt-scripts.service
```

Expected: exit code `0`; both expected strings print.

- [ ] **Step 5: Commit systemd units**

```bash
git add deploy/systemd/orion-backup-mnt-scripts.service deploy/systemd/orion-backup-mnt-scripts.timer
git commit -m "chore(backup): add nightly systemd timer units"
```

---

### Task 8: Add Operator Guide

**Files:**
- Create: `docs/operations/local-mnt-scripts-backup.md`

- [ ] **Step 1: Write operator guide**

```markdown
# Local `/mnt/scripts` Backup Operations

This runbook installs and operates the local-first backup for `/mnt/scripts`.

## What It Does

- Copies `/mnt/scripts/` to `/mnt/storage-warm/backups/<node-name>/mnt-scripts/snapshots/<timestamp>/`.
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

```bash
cd /mnt/scripts/Orion-Sapienform
PYTHONPATH=. ./venv/bin/python scripts/orion_backup_mnt_scripts.py --dry-run
```

## First Manual Run

```bash
cd /mnt/scripts/Orion-Sapienform
sudo PYTHONPATH=. ./venv/bin/python scripts/orion_backup_mnt_scripts.py
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

```bash
sudo cp deploy/systemd/orion-backup-mnt-scripts.service /etc/systemd/system/
sudo cp deploy/systemd/orion-backup-mnt-scripts.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now orion-backup-mnt-scripts.timer
systemctl list-timers orion-backup-mnt-scripts.timer
```

## Restore A File Or Directory

Choose a snapshot:

```bash
ls -1 /mnt/storage-warm/backups/$(hostname)/mnt-scripts/snapshots
```

Restore with metadata preservation:

```bash
sudo rsync -aHAX --numeric-ids /mnt/storage-warm/backups/$(hostname)/mnt-scripts/snapshots/<timestamp>/<relative-path> /mnt/scripts/<restore-target>
```

Do not restore over active service directories without deciding whether services need to stop first.

## Cross-Node Note

Do not copy `/etc/fstab` UUIDs from another node. Mount `/mnt/scripts` and `/mnt/storage-warm` using the destination node's own disk identifiers, then restore files from the chosen snapshot.
```

- [ ] **Step 2: Verify guide references existing files**

Run:

```bash
test -f docs/operations/local-mnt-scripts-backup.md && test -f deploy/systemd/orion-backup-mnt-scripts.service && test -f deploy/systemd/orion-backup-mnt-scripts.timer
```

Expected: exit code `0`.

- [ ] **Step 3: Commit guide**

```bash
git add docs/operations/local-mnt-scripts-backup.md
git commit -m "docs(backup): add local mnt scripts backup runbook"
```

---

### Task 9: Final Verification And Evidence

**Files:**
- Verify: `scripts/orion_backup_mnt_scripts.py`
- Verify: `tests/test_orion_backup_mnt_scripts.py`
- Verify: `deploy/systemd/orion-backup-mnt-scripts.service`
- Verify: `deploy/systemd/orion-backup-mnt-scripts.timer`
- Verify: `docs/operations/local-mnt-scripts-backup.md`

- [ ] **Step 1: Run targeted unit tests**

Run: `PYTHONPATH=. ./venv/bin/python -m pytest tests/test_orion_backup_mnt_scripts.py -q`

Expected: exit code `0`; all backup runner tests pass.

- [ ] **Step 2: Run temporary real snapshot smoke**

Run:

```bash
tmpdir="$(mktemp -d)" && mkdir -p "$tmpdir/source" "$tmpdir/target" && printf "v1\n" > "$tmpdir/source/file.txt" && PYTHONPATH=. ./venv/bin/python scripts/orion_backup_mnt_scripts.py --source "$tmpdir/source" --storage-warm "$tmpdir/target" --node-name smoke-node --no-require-mounts && printf "v2\n" > "$tmpdir/source/file.txt" && PYTHONPATH=. ./venv/bin/python scripts/orion_backup_mnt_scripts.py --source "$tmpdir/source" --storage-warm "$tmpdir/target" --node-name smoke-node --no-require-mounts && test "$(find "$tmpdir/target/backups/smoke-node/mnt-scripts/snapshots" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 2
```

Expected: exit code `0`; two successful snapshot directories exist.

- [ ] **Step 3: Run dry-run against production paths without copying**

Run: `PYTHONPATH=. ./venv/bin/python scripts/orion_backup_mnt_scripts.py --dry-run`

Expected: exit code `0` if `/mnt/scripts`, `/mnt/storage-warm`, and `rsync` are present; output includes `base_root`, `rsync_command`, and `dry_run`.

- [ ] **Step 4: Verify systemd units**

Run: `systemd-analyze verify deploy/systemd/orion-backup-mnt-scripts.service deploy/systemd/orion-backup-mnt-scripts.timer`

Expected: exit code `0`, or record `UNVERIFIED` for systemd syntax if the command is unavailable.

- [ ] **Step 5: Check git diff and commit final fixes if needed**

```bash
git status --short
git diff -- scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py deploy/systemd/orion-backup-mnt-scripts.service deploy/systemd/orion-backup-mnt-scripts.timer docs/operations/local-mnt-scripts-backup.md
```

Expected: no uncommitted changes for the backup implementation files after all task commits.

If verification required small fixes, commit them:

```bash
git add scripts/orion_backup_mnt_scripts.py tests/test_orion_backup_mnt_scripts.py deploy/systemd/orion-backup-mnt-scripts.service deploy/systemd/orion-backup-mnt-scripts.timer docs/operations/local-mnt-scripts-backup.md
git commit -m "fix(backup): finalize local snapshot verification"
```
