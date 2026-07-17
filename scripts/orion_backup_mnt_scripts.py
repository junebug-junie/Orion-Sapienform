#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import shutil
import socket
import subprocess
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
        path
        for path in snapshots_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".incomplete-")
    )
    to_remove = candidates[:-keep]
    removed: list[str] = []
    for path in to_remove:
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


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


def find_previous_snapshot(paths: BackupPaths) -> Path | None:
    latest = paths.latest_symlink
    if latest.is_symlink():
        target = latest.resolve()
        if target.exists() and target.is_dir() and not target.name.startswith(".incomplete-"):
            return target
    if not paths.snapshots_dir.exists():
        return None
    candidates = sorted(
        path
        for path in paths.snapshots_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".incomplete-")
    )
    return candidates[-1] if candidates else None


def build_rsync_command(cfg: BackupConfig, paths: BackupPaths, *, previous_snapshot: Path | None) -> list[str]:
    cmd = ["rsync", "-aHAX", "--numeric-ids", "--delete"]
    if previous_snapshot is not None:
        cmd.extend(["--link-dest", str(previous_snapshot)])
    cmd.extend([str(cfg.source_path) + "/", str(paths.incomplete_snapshot) + "/"])
    return cmd


def build_paths(cfg: BackupConfig, *, run_id: str) -> BackupPaths:
    base_root = cfg.storage_warm_path / "backups" / cfg.node_name / "mnt-scripts"
    snapshots_dir = base_root / "snapshots"
    return BackupPaths(
        base_root=base_root,
        snapshots_dir=snapshots_dir,
        incomplete_snapshot=snapshots_dir / f".incomplete-{run_id}",
        final_snapshot=snapshots_dir / run_id,
        latest_symlink=base_root / "latest",
        status_latest=base_root / "status" / "latest.json",
        status_run=base_root / "status" / "runs" / f"{run_id}.json",
        log_path=base_root / "logs" / f"{run_id}.log",
        manifest_path=base_root / "manifests" / f"{run_id}.json",
    )


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def validate_environment(
    cfg: BackupConfig,
    *,
    is_mount: Callable[[Path], bool] | None = None,
    is_writable: Callable[[Path], bool] | None = None,
    rsync_path: str | None = None,
) -> None:
    mount_check = is_mount or (lambda path: path.is_mount())
    writable_check = is_writable or (lambda path: os.access(path, os.W_OK | os.X_OK))
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
    backup_root = cfg.storage_warm_path / "backups" / cfg.node_name / "mnt-scripts"
    writable_parent = _nearest_existing_parent(backup_root)
    if not writable_check(writable_parent):
        raise RuntimeError(
            f"backup root parent is not writable: {writable_parent}"
        )


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


def send_failure_notification(outcome: RunOutcome, cfg: BackupConfig) -> dict[str, Any]:
    if not cfg.notify_url:
        return {"attempted": False, "ok": False, "reason": "notify_url_not_configured"}
    message = (
        f"Backup failed on {outcome.node_name} "
        f"(run_id={outcome.run_id}): "
        f"source={outcome.source_path} -> target={outcome.target_root}: "
        f"{outcome.error_summary}"
    )
    payload = {
        "source_service": "orion-backup",
        "reason": "backup_failed",
        "severity": "critical",
        "message": message,
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
    validate_environment(cfg)
    started = _utc_iso(now)
    timestamp = snapshot_timestamp(now)
    run_id = f"{timestamp}-{os.getpid()}"
    paths = build_paths(cfg, run_id=run_id)
    lock_handle = acquire_lock(paths.base_root / "backup.lock")
    try:
        paths.snapshots_dir.mkdir(parents=True, exist_ok=True)
        paths.log_path.parent.mkdir(parents=True, exist_ok=True)
        previous = find_previous_snapshot(paths)
        cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)
        rsync_exit: int | None = None
        error_summary: str | None = None
        snapshot_path: str | None = None
        retention_actions: list[str] = []
        try:
            if paths.final_snapshot.exists():
                raise RuntimeError(f"snapshot already exists: {paths.final_snapshot}")
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
            mount_validation={
                "source": (not cfg.require_mounts) or cfg.source_path.is_mount(),
                "target": (not cfg.require_mounts) or cfg.storage_warm_path.is_mount(),
            },
        )
        write_run_evidence(paths, outcome)
        if outcome.status == "failure":
            notify_result = notifier(outcome, cfg)
            outcome = RunOutcome(**{**asdict(outcome), "notification_attempt": notify_result})
            write_run_evidence(paths, outcome)
        return outcome
    finally:
        lock_handle.close()


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
    if cfg.dry_run:
        validate_environment(cfg)
        now = _coerce_datetime(None)
        timestamp = snapshot_timestamp(now)
        run_id = f"{timestamp}-{os.getpid()}"
        paths = build_paths(cfg, run_id=run_id)
        previous = find_previous_snapshot(paths)
        cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "node_name": cfg.node_name,
                    "base_root": str(paths.base_root),
                    "previous_snapshot": str(previous) if previous else None,
                    "rsync_command": cmd,
                },
                sort_keys=True,
            )
        )
        return 0
    outcome = run_backup(cfg)
    print(json.dumps(asdict(outcome), sort_keys=True))
    return 0 if outcome.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
