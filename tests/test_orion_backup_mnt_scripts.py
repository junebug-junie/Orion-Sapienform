from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.orion_backup_mnt_scripts import (
    BackupConfig,
    RunOutcome,
    acquire_lock,
    build_paths,
    build_rsync_command,
    find_previous_snapshot,
    prune_successful_snapshots,
    run_backup,
    send_failure_notification,
    snapshot_timestamp,
    validate_environment,
    write_run_evidence,
)


def test_snapshot_timestamp_is_sortable_utc() -> None:
    assert snapshot_timestamp("2026-05-09T22:00:00+00:00") == "2026-05-09T22-00-00Z"


def test_build_paths_uses_node_scoped_backup_root(tmp_path: Path) -> None:
    cfg = BackupConfig(
        source_path=tmp_path / "scripts",
        storage_warm_path=tmp_path / "storage-warm",
        node_name="node-a",
    )

    paths = build_paths(cfg, run_id="run-1")

    assert paths.base_root == tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts"
    assert paths.snapshots_dir == paths.base_root / "snapshots"
    assert paths.incomplete_snapshot == paths.snapshots_dir / ".incomplete-run-1"
    assert paths.final_snapshot == paths.snapshots_dir / "run-1"
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


def test_validate_environment_requires_writable_backup_root_parent(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a")

    with pytest.raises(RuntimeError, match="backup root.*writable"):
        validate_environment(
            cfg,
            is_mount=lambda path: True,
            is_writable=lambda path: False,
            rsync_path="/usr/bin/rsync",
        )


def test_validate_environment_writable_check_uses_nearest_existing_parent(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a")
    seen: list[Path] = []

    def fake_is_writable(path: Path) -> bool:
        seen.append(path)
        return True

    validate_environment(
        cfg,
        is_mount=lambda path: True,
        is_writable=fake_is_writable,
        rsync_path="/usr/bin/rsync",
    )

    assert seen == [target]


def test_find_previous_snapshot_uses_latest_symlink(tmp_path: Path) -> None:
    root = tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts"
    previous = root / "snapshots" / "2026-05-08T22-00-00Z"
    previous.mkdir(parents=True)
    (root / "latest").symlink_to(previous)

    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-2")

    assert find_previous_snapshot(paths) == previous


def test_find_previous_snapshot_ignores_latest_if_it_points_to_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts"
    snapshots = root / "snapshots"
    complete = snapshots / "2026-05-08T22-00-00Z"
    incomplete = snapshots / ".incomplete-run-9"
    complete.mkdir(parents=True)
    incomplete.mkdir()
    (root / "latest").symlink_to(incomplete)

    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-2")

    assert find_previous_snapshot(paths) == complete


def test_find_previous_snapshot_falls_back_to_newest_complete_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts"
    snapshots = root / "snapshots"
    older = snapshots / "2026-05-07T22-00-00Z"
    newest = snapshots / "2026-05-08T22-00-00Z"
    incomplete = snapshots / ".incomplete-run-9"
    older.mkdir(parents=True)
    newest.mkdir()
    incomplete.mkdir()

    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-2")

    assert find_previous_snapshot(paths) == newest


def test_build_rsync_command_omits_link_dest_without_previous(tmp_path: Path) -> None:
    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-1")

    cmd = build_rsync_command(cfg, paths, previous_snapshot=None)

    assert cmd[:4] == ["rsync", "-aHAX", "--numeric-ids", "--delete"]
    assert "--link-dest" not in cmd
    assert str(cfg.source_path) + "/" in cmd
    assert str(paths.incomplete_snapshot) + "/" in cmd


def test_build_rsync_command_uses_previous_snapshot_for_link_dest(tmp_path: Path) -> None:
    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-2")
    previous = tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts" / "snapshots" / "2026-05-08T22-00-00Z"

    cmd = build_rsync_command(cfg, paths, previous_snapshot=previous)

    assert cmd[:4] == ["rsync", "-aHAX", "--numeric-ids", "--delete"]
    assert "--link-dest" in cmd
    assert str(previous) in cmd


def test_write_run_evidence_updates_latest_status_and_manifest(tmp_path: Path) -> None:
    cfg = BackupConfig(source_path=tmp_path / "scripts", storage_warm_path=tmp_path / "storage-warm", node_name="node-a")
    paths = build_paths(cfg, run_id="run-1")
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


def test_run_backup_finalizes_snapshot_and_latest_symlink(tmp_path: Path) -> None:
    source = tmp_path / "scripts"
    target = tmp_path / "storage-warm"
    source.mkdir()
    target.mkdir()
    cfg = BackupConfig(source_path=source, storage_warm_path=target, node_name="node-a", require_mounts=False)

    def fake_run(cmd: list[str], log_path: Path) -> int:
        destination = Path(cmd[-1].rstrip("/"))
        destination.mkdir(parents=True, exist_ok=True)
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
    cfg = BackupConfig(
        source_path=source,
        storage_warm_path=target,
        node_name="node-a",
        require_mounts=False,
        notify_url="http://notify.invalid/attention/request",
    )
    seen_status: dict[str, Any] = {}

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


def test_acquire_lock_rejects_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "backup.lock"
    first = acquire_lock(lock_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            acquire_lock(lock_path)
    finally:
        first.close()


def test_send_failure_notification_message_includes_run_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = BackupConfig(
        source_path=tmp_path / "scripts",
        storage_warm_path=tmp_path / "storage-warm",
        node_name="node-a",
        notify_url="http://notify.invalid/attention/request",
    )
    target_root = str(tmp_path / "storage-warm" / "backups" / "node-a" / "mnt-scripts")
    outcome = RunOutcome(
        run_id="2026-05-09T22-00-00Z-12345",
        status="failure",
        node_name="node-a",
        started_at_utc="2026-05-09T22:00:00Z",
        finished_at_utc="2026-05-09T22:01:00Z",
        source_path="/mnt/scripts",
        target_root=target_root,
        snapshot_path=None,
        latest_symlink=str(tmp_path / "latest"),
        rsync_exit_code=23,
        error_summary="rsync exited with 23",
        log_path=str(tmp_path / "log.log"),
        manifest_path=str(tmp_path / "manifest.json"),
        notification_attempt=None,
        retention_actions=[],
        previous_snapshot=None,
        mount_validation={"source": True, "target": True},
    )
    captured: dict[str, Any] = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

    def fake_urlopen(request: Any, timeout: int = 10) -> _FakeResponse:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["url"] = request.full_url
        return _FakeResponse()

    monkeypatch.setattr(
        "scripts.orion_backup_mnt_scripts.urllib.request.urlopen", fake_urlopen
    )

    result = send_failure_notification(outcome, cfg)

    assert result == {"attempted": True, "ok": True, "status": 200}
    payload = captured["payload"]
    message = payload["message"]
    assert "node-a" in message
    assert "/mnt/scripts" in message
    assert target_root in message
    assert "2026-05-09T22-00-00Z-12345" in message
    assert "rsync exited with 23" in message


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
