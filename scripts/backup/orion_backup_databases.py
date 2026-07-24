#!/usr/bin/env python3
"""Nightly backups for the live database backends (Postgres, FalkorDB, Chroma,
Convex).

Reuses the same shape as orion_backup_mnt_scripts.py (staged snapshot dir,
atomic rename on success, retention, status/manifest JSON, failure notify)
but each target has its own consistency-safe capture method instead of a
single rsync, because a live database data directory is not safe to rsync
while it is being written -- confirmed live: a plain tar of Convex's
db.sqlite3 failed with "file changed as we read it" while the backend was
running. See scripts/backup/README.md for the full per-target rationale.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from scripts.backup.orion_backup_mnt_scripts import (
    _update_latest_symlink,
    _utc_iso,
    _write_json_atomic,
    acquire_lock,
    cleanup_incomplete_snapshots,
    prune_successful_snapshots,
    snapshot_timestamp,
)

DEFAULT_STORAGE_WARM = Path("/mnt/storage-warm")
DEFAULT_KEEP_SUCCESSFUL = 14
DEFAULT_SUBPROCESS_TIMEOUT_SEC = 300


@dataclass(frozen=True)
class TargetOutcome:
    name: str
    status: str
    snapshot_path: str | None
    error_summary: str | None
    retention_actions: list[str]


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: str
    node_name: str
    started_at_utc: str
    finished_at_utc: str
    storage_warm_path: str
    targets: list[dict[str, Any]]
    log_path: str
    manifest_path: str
    notification_attempt: dict[str, Any] | None


def _target_root(storage_warm: Path, node_name: str, target_name: str) -> Path:
    return storage_warm / "backups" / node_name / "db" / target_name


def _run(
    cmd: list[str], *, log: list[str], timeout: int = DEFAULT_SUBPROCESS_TIMEOUT_SEC
) -> subprocess.CompletedProcess:
    log.append("$ " + " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def validate_environment(storage_warm: Path, *, require_mount: bool = True) -> None:
    """Fail loudly before writing anything if storage-warm isn't actually
    mounted -- otherwise a missing mount silently falls through to whatever
    filesystem backs the parent directory (root, in production), which is
    exactly the phantom-mount failure mode this whole backup system exists
    to protect against."""
    if not storage_warm.exists():
        raise RuntimeError(f"storage-warm path does not exist: {storage_warm}")
    if require_mount and not storage_warm.is_mount():
        raise RuntimeError(f"storage-warm path must be a mount point: {storage_warm}")
    if not os.access(storage_warm, os.W_OK | os.X_OK):
        raise RuntimeError(f"storage-warm path is not writable: {storage_warm}")


def _chmod_tree_owner_only(root: Path) -> None:
    """Backup artifacts (a full Postgres cluster dump, in plaintext) must not
    be world- or group-readable by default -- restrict after writing rather
    than relying on umask alone, since this runs as root and the resulting
    files would otherwise inherit whatever the process umask happens to be."""
    for path in root.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
    root.chmod(0o700)


# --- per-target capture -----------------------------------------------------

def capture_postgres(dest_dir: Path, *, container: str, pg_user: str, log: list[str]) -> None:
    """pg_dumpall inside the running container -- logical dump, always
    consistent regardless of concurrent writes, unlike a raw data-dir copy.

    Streams stdout straight to the destination file instead of buffering it
    in memory: a full cluster dump has been observed at 1.2GB+ live and only
    grows, so capturing it as a Python bytes object first would double peak
    memory for no reason.
    """
    dest_dir.mkdir(parents=True)
    dest_file = dest_dir / "pg_dumpall.sql"
    cmd = ["docker", "exec", container, "pg_dumpall", "-U", pg_user]
    log.append("$ " + " ".join(cmd) + f" > {dest_file}")
    with dest_file.open("wb") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, timeout=DEFAULT_SUBPROCESS_TIMEOUT_SEC)
    if proc.returncode != 0:
        dest_file.unlink(missing_ok=True)
        raise RuntimeError(f"pg_dumpall failed ({proc.returncode}): {proc.stderr.decode(errors='replace')[:500]}")


def capture_falkordb(
    dest_dir: Path,
    *,
    container: str,
    data_dir_in_container: str,
    log: list[str],
    bgsave_deadline_sec: int = 300,
) -> None:
    """BGSAVE first for an atomic point-in-time RDB snapshot, then copy the
    resulting dump.rdb out -- do not rsync/cp a live RDB file mid-write.

    Polls `INFO persistence` for rdb_bgsave_in_progress/rdb_last_bgsave_status
    rather than just watching LASTSAVE change: LASTSAVE has 1-second
    resolution and BGSAVE always updates it on completion (even with zero
    dirty keys), but INFO's explicit in-progress/status fields remove any
    ambiguity and let a large save run past a short fixed timeout.
    """
    dest_dir.mkdir(parents=True)
    proc = _run(["docker", "exec", container, "redis-cli", "BGSAVE"], log=log)
    if proc.returncode != 0:
        raise RuntimeError(f"BGSAVE failed: {proc.stderr.decode(errors='replace')[:500]}")
    deadline = time.monotonic() + bgsave_deadline_sec
    while True:
        info = _run(["docker", "exec", container, "redis-cli", "INFO", "persistence"], log=log).stdout.decode(
            errors="replace"
        )
        fields = dict(
            line.split(":", 1) for line in info.splitlines() if ":" in line and not line.startswith("#")
        )
        if fields.get("rdb_bgsave_in_progress") == "0":
            if fields.get("rdb_last_bgsave_status") != "ok":
                raise RuntimeError(f"BGSAVE finished with status={fields.get('rdb_last_bgsave_status')!r}")
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"BGSAVE did not complete within {bgsave_deadline_sec}s")
        time.sleep(1)
    proc = _run(
        ["docker", "cp", f"{container}:{data_dir_in_container}/dump.rdb", str(dest_dir / "dump.rdb")],
        log=log,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker cp of dump.rdb failed: {proc.stderr.decode(errors='replace')[:500]}")


def capture_stopped_container_tree(
    dest_dir: Path, *, container: str, host_path: Path, log: list[str], stop_timeout_sec: int = 30
) -> None:
    """Stop the container, plain-copy its data directory, restart it.

    Used for stores with no safe online-backup mechanism available to this
    tool (Convex's data dir mixes a SQLite file with live RocksDB-style
    segment files that have no CLI-accessible checkpoint API here; Chroma
    similarly has per-collection index segments that mutate on write).
    Copying any of that raw while the process is live risks tearing exactly
    like the sqlite-only case already confirmed live for Convex. Stopping
    for the copy window is the simple, actually-consistent alternative --
    both of these are non-critical services where a few seconds of downtime
    during the 03:45 backup window is an acceptable trade, unlike Postgres/
    FalkorDB above which stay up via their own online-backup mechanisms.
    """
    dest_dir.mkdir(parents=True)
    if not host_path.exists():
        raise RuntimeError(f"host path does not exist: {host_path}")
    stop_proc = _run(["docker", "stop", "-t", str(stop_timeout_sec), container], log=log)
    if stop_proc.returncode != 0:
        raise RuntimeError(f"docker stop failed: {stop_proc.stderr.decode(errors='replace')[:500]}")
    try:
        shutil.copytree(host_path, dest_dir, dirs_exist_ok=True)
    finally:
        start_proc = _run(["docker", "start", container], log=log)
        if start_proc.returncode != 0:
            # Surface this loudly rather than swallowing it under whatever
            # copytree exception (if any) is already propagating -- a
            # container that didn't come back up is worse than a failed
            # backup.
            log.append(f"WARNING: docker start failed to restart {container} after backup")


@dataclass(frozen=True)
class Target:
    name: str
    capture: Callable[[Path, list[str]], None]
    keep_successful: int = DEFAULT_KEEP_SUCCESSFUL


def default_targets() -> list[Target]:
    return [
        Target(
            "postgres",
            lambda dest, log: capture_postgres(
                dest, container="orion-athena-sql-db", pg_user="postgres", log=log
            ),
        ),
        Target(
            "falkordb",
            lambda dest, log: capture_falkordb(
                dest,
                container="orion-athena-falkordb",
                data_dir_in_container="/var/lib/falkordb/data",
                log=log,
            ),
        ),
        Target(
            "chroma",
            lambda dest, log: capture_stopped_container_tree(
                dest,
                container="orion-athena-vector-db",
                host_path=Path("/mnt/postgres/collapse-mirrors/chroma"),
                log=log,
            ),
        ),
        Target(
            "convex",
            lambda dest, log: capture_stopped_container_tree(
                dest,
                container="orion-ai-town-backend-1",
                host_path=Path("/mnt/docker/volumes/orion-ai-town_convex-data/_data"),
                log=log,
            ),
        ),
    ]


def run_target_backup(
    target: Target,
    *,
    storage_warm: Path,
    node_name: str,
    run_id: str,
) -> tuple[TargetOutcome, list[str]]:
    root = _target_root(storage_warm, node_name, target.name)
    snapshots_dir = root / "snapshots"
    incomplete = snapshots_dir / f".incomplete-{run_id}"
    final = snapshots_dir / run_id
    latest = root / "latest"
    log: list[str] = []
    try:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        cleanup_incomplete_snapshots(snapshots_dir)
        if final.exists():
            raise RuntimeError(f"snapshot already exists: {final}")
        target.capture(incomplete, log)
        _chmod_tree_owner_only(incomplete)
        incomplete.replace(final)
        _update_latest_symlink(latest, final)
        retention_actions = prune_successful_snapshots(snapshots_dir, keep=target.keep_successful)
        status = "success"
        snapshot_path: str | None = str(final)
        error_summary = None
    except Exception as exc:  # noqa: BLE001 - captured into evidence, not swallowed
        status = "failure"
        snapshot_path = None
        error_summary = str(exc)
        retention_actions = []
        if incomplete.exists():
            shutil.rmtree(incomplete, ignore_errors=True)
    log_path = root / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log) + "\n")
    return (
        TargetOutcome(
            name=target.name,
            status=status,
            snapshot_path=snapshot_path,
            error_summary=error_summary,
            retention_actions=retention_actions,
        ),
        log,
    )


def send_failure_notification(
    outcome: RunOutcome, *, notify_url: str | None, notify_token: str | None
) -> dict[str, Any]:
    if not notify_url:
        return {"attempted": False, "ok": False, "reason": "notify_url_not_configured"}
    failed_summary = "; ".join(
        f"{t['name']}: {t['error_summary']}" for t in outcome.targets if t["status"] == "failure"
    )
    message = f"Database backup failed on {outcome.node_name} (run_id={outcome.run_id}): {failed_summary}"
    payload = {
        "source_service": "orion-backup",
        "reason": "db_backup_failed",
        "severity": "critical",
        "message": message,
        "require_ack": True,
        "context": {"run_id": outcome.run_id, "targets": outcome.targets, "manifest_path": outcome.manifest_path},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(notify_url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    if notify_token:
        request.add_header("X-Orion-Notify-Token", notify_token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return {"attempted": True, "ok": 200 <= response.status < 300, "status": response.status}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def run_backup(
    *,
    storage_warm: Path = DEFAULT_STORAGE_WARM,
    node_name: str | None = None,
    targets: list[Target] | None = None,
    notify_url: str | None = None,
    notify_token: str | None = None,
    require_mount: bool = True,
) -> RunOutcome:
    node_name = node_name or socket.gethostname()
    targets = targets if targets is not None else default_targets()
    validate_environment(storage_warm, require_mount=require_mount)
    started = _utc_iso(None)
    run_id = f"{snapshot_timestamp(None)}-{os.getpid()}"
    base_root = storage_warm / "backups" / node_name / "db"
    lock_handle = acquire_lock(base_root / "backup.lock")
    try:
        run_log: list[str] = []
        outcomes: list[TargetOutcome] = []
        for t in targets:
            outcome, target_log = run_target_backup(t, storage_warm=storage_warm, node_name=node_name, run_id=run_id)
            outcomes.append(outcome)
            run_log.append(f"=== {t.name} ===")
            run_log.extend(target_log)
        log_path = base_root / "logs" / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(run_log) + "\n")
        overall_status = "success" if all(o.status == "success" for o in outcomes) else "failure"
        outcome = RunOutcome(
            run_id=run_id,
            status=overall_status,
            node_name=node_name,
            started_at_utc=started,
            finished_at_utc=_utc_iso(None),
            storage_warm_path=str(storage_warm),
            targets=[asdict(o) for o in outcomes],
            log_path=str(log_path),
            manifest_path=str(base_root / "manifests" / f"{run_id}.json"),
            notification_attempt=None,
        )
        _write_json_atomic(base_root / "status" / "latest.json", asdict(outcome))
        _write_json_atomic(base_root / "status" / "runs" / f"{run_id}.json", asdict(outcome))
        _write_json_atomic(Path(outcome.manifest_path), asdict(outcome))
        if outcome.status == "failure":
            notify_result = send_failure_notification(outcome, notify_url=notify_url, notify_token=notify_token)
            outcome = RunOutcome(**{**asdict(outcome), "notification_attempt": notify_result})
            _write_json_atomic(base_root / "status" / "latest.json", asdict(outcome))
        return outcome
    finally:
        lock_handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nightly logical backups of Orion's live database backends.")
    parser.add_argument("--storage-warm", type=Path, default=DEFAULT_STORAGE_WARM)
    parser.add_argument("--node-name", default=socket.gethostname())
    parser.add_argument("--notify-url", default=os.environ.get("ORION_BACKUP_NOTIFY_URL"))
    parser.add_argument("--notify-token", default=os.environ.get("ORION_BACKUP_NOTIFY_TOKEN"))
    parser.add_argument("--no-require-mount", action="store_true")
    parser.add_argument(
        "--only", nargs="*", default=None, help="Restrict to these target names (default: all)"
    )
    args = parser.parse_args(argv)
    targets = default_targets()
    if args.only:
        targets = [t for t in targets if t.name in args.only]
    outcome = run_backup(
        storage_warm=args.storage_warm,
        node_name=args.node_name,
        targets=targets,
        notify_url=args.notify_url,
        notify_token=args.notify_token,
        require_mount=not args.no_require_mount,
    )
    print(json.dumps(asdict(outcome), sort_keys=True))
    return 0 if outcome.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
