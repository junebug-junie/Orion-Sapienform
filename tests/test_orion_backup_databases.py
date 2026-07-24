from __future__ import annotations

import fcntl
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.backup.orion_backup_databases import (
    Target,
    capture_stopped_container_tree,
    run_backup,
    run_target_backup,
    validate_environment,
)


def _ok_capture(dest: Path, log: list[str]) -> None:
    dest.mkdir(parents=True)
    (dest / "payload.txt").write_text("ok")


def _failing_capture(dest: Path, log: list[str]) -> None:
    raise RuntimeError("boom")


def test_run_target_backup_success_updates_latest_and_prunes(tmp_path: Path) -> None:
    outcome, _log = run_target_backup(
        Target("widget", _ok_capture, keep_successful=2),
        storage_warm=tmp_path,
        node_name="node-a",
        run_id="run-1",
    )

    assert outcome.status == "success"
    assert outcome.snapshot_path is not None
    snapshot = Path(outcome.snapshot_path)
    assert snapshot.is_dir()
    assert (snapshot / "payload.txt").read_text() == "ok"
    latest = tmp_path / "backups" / "node-a" / "db" / "widget" / "latest"
    assert latest.resolve() == snapshot.resolve()


def test_run_target_backup_restricts_permissions_to_owner_only(tmp_path: Path) -> None:
    outcome, _log = run_target_backup(
        Target("widget", _ok_capture), storage_warm=tmp_path, node_name="node-a", run_id="run-1"
    )
    snapshot = Path(outcome.snapshot_path)
    assert (snapshot.stat().st_mode & 0o777) == 0o700
    assert ((snapshot / "payload.txt").stat().st_mode & 0o777) == 0o600


def test_run_target_backup_failure_cleans_incomplete_dir(tmp_path: Path) -> None:
    outcome, _log = run_target_backup(
        Target("widget", _failing_capture), storage_warm=tmp_path, node_name="node-a", run_id="run-1"
    )

    assert outcome.status == "failure"
    assert outcome.snapshot_path is None
    assert "boom" in outcome.error_summary
    snapshots_dir = tmp_path / "backups" / "node-a" / "db" / "widget" / "snapshots"
    assert list(snapshots_dir.glob(".incomplete-*")) == []
    assert list(snapshots_dir.glob("run-1")) == []


def test_run_target_backup_prunes_old_snapshots_beyond_keep(tmp_path: Path) -> None:
    for i in range(3):
        run_target_backup(
            Target("widget", _ok_capture, keep_successful=2),
            storage_warm=tmp_path,
            node_name="node-a",
            run_id=f"run-{i}",
        )
    snapshots_dir = tmp_path / "backups" / "node-a" / "db" / "widget" / "snapshots"
    remaining = sorted(p.name for p in snapshots_dir.iterdir())
    assert remaining == ["run-1", "run-2"]


def test_validate_environment_requires_existing_mount(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RuntimeError, match="does not exist"):
        validate_environment(missing)


def test_validate_environment_requires_actual_mount_point(tmp_path: Path) -> None:
    # tmp_path is a plain directory, not a mount point -- this is exactly the
    # case that must be rejected so a missing storage-warm mount can't
    # silently fall through to writing on root.
    with pytest.raises(RuntimeError, match="mount point"):
        validate_environment(tmp_path, require_mount=True)


def test_validate_environment_allows_non_mount_when_not_required(tmp_path: Path) -> None:
    validate_environment(tmp_path, require_mount=False)


def test_run_backup_rejects_non_mount_storage_warm_by_default(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="mount point"):
        run_backup(storage_warm=tmp_path, node_name="node-a", targets=[Target("good", _ok_capture)])


def test_run_backup_overall_status_is_failure_if_any_target_fails(tmp_path: Path) -> None:
    outcome = run_backup(
        storage_warm=tmp_path,
        node_name="node-a",
        targets=[Target("good", _ok_capture), Target("bad", _failing_capture)],
        require_mount=False,
    )

    assert outcome.status == "failure"
    names_by_status = {t["name"]: t["status"] for t in outcome.targets}
    assert names_by_status == {"good": "success", "bad": "failure"}


def test_run_backup_writes_status_and_manifest_json(tmp_path: Path) -> None:
    outcome = run_backup(
        storage_warm=tmp_path, node_name="node-a", targets=[Target("good", _ok_capture)], require_mount=False
    )

    status_latest = tmp_path / "backups" / "node-a" / "db" / "status" / "latest.json"
    assert status_latest.exists()
    assert Path(outcome.manifest_path).exists()
    assert Path(outcome.log_path).exists()


def test_run_backup_lock_prevents_concurrent_runs(tmp_path: Path) -> None:
    lock_path = tmp_path / "backups" / "node-a" / "db" / "backup.lock"
    lock_path.parent.mkdir(parents=True)
    handle = lock_path.open("w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            run_backup(
                storage_warm=tmp_path, node_name="node-a", targets=[Target("good", _ok_capture)], require_mount=False
            )
    finally:
        handle.close()


def test_capture_stopped_container_tree_copies_and_restarts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.bin").write_bytes(b"segment-contents")

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], capture_output: bool, timeout: int) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    dest = tmp_path / "dest"
    capture_stopped_container_tree(dest, container="my-container", host_path=source, log=[])

    assert (dest / "data.bin").read_bytes() == b"segment-contents"
    assert calls[0][:2] == ["docker", "stop"]
    assert calls[-1] == ["docker", "start", "my-container"]


def test_capture_stopped_container_tree_restarts_container_even_if_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], capture_output: bool, timeout: int) -> subprocess.CompletedProcess:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "copytree", MagicMock(side_effect=OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        capture_stopped_container_tree(
            tmp_path / "dest", container="my-container", host_path=tmp_path, log=[]
        )

    assert calls[-1] == ["docker", "start", "my-container"]


def test_capture_stopped_container_tree_raises_if_host_path_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(cmd: list[str], capture_output: bool, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="does not exist"):
        capture_stopped_container_tree(
            tmp_path / "dest", container="my-container", host_path=tmp_path / "missing", log=[]
        )
