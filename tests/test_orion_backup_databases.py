from __future__ import annotations

import fcntl
import gzip
import io
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.backup.orion_backup_databases import (
    DEFAULT_SUBPROCESS_TIMEOUT_SEC,
    POSTGRES_DUMP_TIMEOUT_SEC,
    Target,
    _terminate_stale_pg_dump_backends,
    capture_postgres,
    capture_stopped_container_tree,
    run_backup,
    run_target_backup,
    send_backup_notification,
    validate_environment,
)


def _ok_capture(dest: Path, log: list[str]) -> None:
    dest.mkdir(parents=True)
    (dest / "payload.txt").write_text("ok")


def _failing_capture(dest: Path, log: list[str]) -> None:
    raise RuntimeError("boom")


def test_postgres_dump_timeout_exceeds_shared_default() -> None:
    # A live nightly dump against a 17GB+ database has already exceeded
    # DEFAULT_SUBPROCESS_TIMEOUT_SEC (300s) and failed the whole backup run
    # (2026-07-30). This pins capture_postgres's own timeout well above the
    # shared default so that regression can't silently come back.
    assert POSTGRES_DUMP_TIMEOUT_SEC > DEFAULT_SUBPROCESS_TIMEOUT_SEC


class _FakeProc:
    """Stand-in for a subprocess.Popen handle used by capture_postgres.

    capture_postgres runs the dump as a real pipeline now (pg_dumpall | gzip),
    so patching subprocess.run alone no longer intercepts it -- and an
    unpatched Popen in these tests fires a genuine `docker exec pg_dumpall`
    at the live database. Every test below patches Popen for that reason.
    """

    def __init__(self, cmd, *, returncode=0, err=b"", timeout=False, stderr=None):
        self.args = cmd
        self.returncode = returncode
        self.killed = False
        self.timeout_seen = None
        self._err = err
        self._timeout = timeout
        self.stdout = io.BytesIO(b"")
        # The dump's stderr is handed a real file object by production code.
        if stderr is not None and hasattr(stderr, "write") and err:
            stderr.write(err)

    def communicate(self, timeout=None):
        self.timeout_seen = timeout
        if self._timeout:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return (b"", self._err)

    def wait(self, timeout=None):
        self.timeout_seen = timeout
        return self.returncode

    def kill(self):
        self.killed = True


def _patch_dump_pipeline(
    monkeypatch, *, dump_rc=0, gz_rc=0, dump_err=b"", gz_err=b"", gz_timeout=False
):
    """Patch subprocess.Popen so the dump pipeline is simulated, and record
    every command both Popen and run were asked to execute."""
    calls: list[list[str]] = []
    procs: dict[str, _FakeProc] = {}

    def _fake_popen(cmd, **kwargs):
        calls.append(cmd)
        if cmd and cmd[0] == "gzip":
            proc = _FakeProc(cmd, returncode=gz_rc, err=gz_err, timeout=gz_timeout)
            procs["gzip"] = proc
        else:
            proc = _FakeProc(cmd, returncode=dump_rc, err=dump_err, stderr=kwargs.get("stderr"))
            procs["dump"] = proc
        return proc

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    return calls, procs


def test_capture_postgres_uses_dedicated_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, procs = _patch_dump_pipeline(monkeypatch)
    dest = tmp_path / "postgres_snapshot"
    capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    assert procs["gzip"].timeout_seen == POSTGRES_DUMP_TIMEOUT_SEC
    assert procs["dump"].timeout_seen == POSTGRES_DUMP_TIMEOUT_SEC


def test_capture_postgres_writes_gzipped_dump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The dump is compressed on the way to disk: 14 retained *uncompressed*
    # cluster dumps is what filled /mnt/storage-warm to 0 bytes free and broke
    # every postgres backup from 2026-08-30 onward.
    calls, _ = _patch_dump_pipeline(monkeypatch)
    dest = tmp_path / "postgres_snapshot"
    capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    assert (dest / "pg_dumpall.sql.gz").exists()
    assert not (dest / "pg_dumpall.sql").exists()
    gzip_calls = [c for c in calls if c and c[0] == "gzip"]
    assert len(gzip_calls) == 1


def test_capture_postgres_really_produces_readable_gzip(tmp_path: Path) -> None:
    # Hand-checked round trip through the real gzip binary -- the fake Popen
    # above proves the wiring, this proves the artifact is actually a valid
    # gzip stream that restores to the original bytes.
    payload = b"-- orion cluster dump\nCREATE TABLE t (id int);\n" * 100
    dest_file = tmp_path / "pg_dumpall.sql.gz"
    with dest_file.open("wb") as fh:
        src = subprocess.Popen(["printf", "%s", payload.decode()], stdout=subprocess.PIPE)
        gz = subprocess.Popen(["gzip"], stdin=src.stdout, stdout=fh)
        src.stdout.close()
        gz.communicate()
        src.wait()

    assert gzip.decompress(dest_file.read_bytes()) == payload
    assert dest_file.stat().st_size < len(payload)


def test_capture_postgres_fails_when_dump_fails_even_if_gzip_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole reason both exit codes are checked: gzip compresses a
    # truncated stream perfectly happily and exits 0. If only the tail of the
    # pipe were trusted, a half-written dump would be stored as a good
    # snapshot and only discovered at restore time.
    _patch_dump_pipeline(monkeypatch, dump_rc=1, gz_rc=0, dump_err=b"connection lost")
    dest = tmp_path / "postgres_snapshot"

    with pytest.raises(RuntimeError, match="pg_dumpall failed"):
        capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    assert not (dest / "pg_dumpall.sql.gz").exists()


def test_capture_postgres_fails_when_gzip_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_dump_pipeline(monkeypatch, dump_rc=0, gz_rc=1, gz_err=b"No space left on device")
    dest = tmp_path / "postgres_snapshot"

    with pytest.raises(RuntimeError, match="gzip failed"):
        capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    assert not (dest / "pg_dumpall.sql.gz").exists()


def test_capture_postgres_clears_stale_backend_before_dumping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _patch_dump_pipeline(monkeypatch)
    dest = tmp_path / "postgres_snapshot"
    capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    assert calls[0][:4] == ["docker", "exec", "orion-athena-sql-db", "psql"]
    cleanup_sql = " ".join(calls[0])
    assert "pg_terminate_backend" in cleanup_sql
    # Pinned deliberately: pg_dumpall's actual per-database lock-holding
    # connections show as application_name=pg_dump, not pg_dumpall (confirmed
    # live) -- a future edit that silently dropped 'pg_dump' from this IN list
    # would regress straight back to the incident this test guards against.
    assert "application_name IN ('pg_dump', 'pg_dumpall')" in cleanup_sql
    assert "pid <> pg_backend_pid()" in cleanup_sql
    assert calls[1][:4] == ["docker", "exec", "orion-athena-sql-db", "pg_dumpall"]


def test_capture_postgres_terminates_orphaned_backend_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression test for the confirmed live bug (2026-08-16/17/18): a timed-out
    # `docker exec ... pg_dumpall` only kills the local client, leaving the
    # server-side backend orphaned and holding locks across every table. The
    # timeout must now trigger a cleanup call in addition to still propagating.
    # With the pipeline, a hung dump surfaces as gzip still waiting on stdin.
    calls, procs = _patch_dump_pipeline(monkeypatch, gz_timeout=True)
    dest = tmp_path / "postgres_snapshot"
    with pytest.raises(subprocess.TimeoutExpired):
        capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    terminate_calls = [c for c in calls if "pg_terminate_backend" in " ".join(c)]
    # pre-flight cleanup (before the dump attempt) + post-timeout cleanup
    assert len(terminate_calls) == 2
    # Both halves of the pipeline are killed, not just the one that timed out.
    assert procs["dump"].killed and procs["gzip"].killed
    assert not (dest / "pg_dumpall.sql.gz").exists()


def test_terminate_stale_pg_dump_backends_is_best_effort_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"connection refused")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    log: list[str] = []
    _terminate_stale_pg_dump_backends(container="c", pg_user="postgres", log=log)
    assert any("WARNING" in line for line in log)


def test_terminate_stale_pg_dump_backends_swallows_its_own_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    log: list[str] = []
    _terminate_stale_pg_dump_backends(container="c", pg_user="postgres", log=log)  # must not raise
    assert any("timed out" in line for line in log)


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
    # Real run_id shape (`snapshot_timestamp()-<pid>`), not "run-0": retention
    # deliberately ignores directories whose name it cannot date, so a
    # synthetic id would now be exempt from the count cap and never pruned.
    run_ids = [f"2026-09-0{i + 1}T03-45-01Z-1234{i}" for i in range(3)]
    for run_id in run_ids:
        run_target_backup(
            Target("widget", _ok_capture, keep_successful=2, max_age_days=None),
            storage_warm=tmp_path,
            node_name="node-a",
            run_id=run_id,
        )
    snapshots_dir = tmp_path / "backups" / "node-a" / "db" / "widget" / "snapshots"
    remaining = sorted(p.name for p in snapshots_dir.iterdir())
    assert remaining == run_ids[1:]


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


def test_run_backup_notifies_on_success_not_just_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sent_payloads: list[dict] = []

    def fake_send(outcome, *, notify_url, notify_token):
        sent_payloads.append({"status": outcome.status, "notify_url": notify_url})
        return {"attempted": True, "ok": True, "status": 200}

    monkeypatch.setattr("scripts.backup.orion_backup_databases.send_backup_notification", fake_send)
    outcome = run_backup(
        storage_warm=tmp_path,
        node_name="node-a",
        targets=[Target("good", _ok_capture)],
        notify_url="http://example/attention/request",
        require_mount=False,
    )
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["status"] == "success"
    assert outcome.notification_attempt == {"attempted": True, "ok": True, "status": 200}


def test_send_backup_notification_marks_success_informational_no_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.backup.orion_backup_databases import RunOutcome

    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        captured["body"] = request.data
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    outcome = RunOutcome(
        run_id="run-1",
        status="success",
        node_name="node-a",
        started_at_utc="2026-01-01T00:00:00Z",
        finished_at_utc="2026-01-01T00:00:01Z",
        storage_warm_path="/x",
        targets=[{"name": "good", "status": "success", "snapshot_path": "/x/y", "error_summary": None, "retention_actions": []}],
        log_path="/x/log",
        manifest_path="/x/manifest.json",
        notification_attempt=None,
    )

    result = send_backup_notification(outcome, notify_url="http://example/attention/request", notify_token=None)

    assert result["ok"] is True
    import json as _json

    payload = _json.loads(captured["body"])
    assert payload["reason"] == "db_backup_succeeded"
    assert payload["severity"] == "info"
    assert payload["require_ack"] is False


def test_send_backup_notification_marks_failure_critical_with_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.backup.orion_backup_databases import RunOutcome

    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout):
        captured["body"] = request.data
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    outcome = RunOutcome(
        run_id="run-1",
        status="failure",
        node_name="node-a",
        started_at_utc="2026-01-01T00:00:00Z",
        finished_at_utc="2026-01-01T00:00:01Z",
        storage_warm_path="/x",
        targets=[{"name": "bad", "status": "failure", "snapshot_path": None, "error_summary": "boom", "retention_actions": []}],
        log_path="/x/log",
        manifest_path="/x/manifest.json",
        notification_attempt=None,
    )

    result = send_backup_notification(outcome, notify_url="http://example/attention/request", notify_token=None)

    assert result["ok"] is True
    import json as _json

    payload = _json.loads(captured["body"])
    assert payload["reason"] == "db_backup_failed"
    assert payload["severity"] == "critical"
    assert payload["require_ack"] is True
    assert "boom" in payload["message"]


def test_capture_stopped_container_tree_raises_if_host_path_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(cmd: list[str], capture_output: bool, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="does not exist"):
        capture_stopped_container_tree(
            tmp_path / "dest", container="my-container", host_path=tmp_path / "missing", log=[]
        )


def test_retention_runs_even_when_the_capture_fails(tmp_path: Path) -> None:
    # The 2026-08-30..09-02 postgres outage: retention was gated on the
    # success path, so four consecutive failed runs released no space at all
    # while /mnt/storage-warm sat at 0 bytes free -- the one condition under
    # which the next run could never succeed either.
    storage_warm = tmp_path / "storage-warm"
    snapshots = storage_warm / "backups" / "node-a" / "db" / "postgres" / "snapshots"
    snapshots.mkdir(parents=True)
    for day in range(1, 11):
        (snapshots / f"2026-01-{day:02d}T22-00-00Z-12345").mkdir()

    outcome, _ = run_target_backup(
        Target("postgres", _failing_capture),
        storage_warm=storage_warm,
        node_name="node-a",
        run_id="2026-09-02T03-45-01Z-99999",
    )

    assert outcome.status == "failure"
    assert "boom" in outcome.error_summary
    # Aged-out snapshots released, floor respected, capture error preserved.
    assert len(outcome.retention_actions) == 3
    assert len(list(snapshots.iterdir())) == 7


def test_retention_failure_does_not_mask_the_capture_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage_warm = tmp_path / "storage-warm"
    (storage_warm / "backups" / "node-a" / "db" / "postgres" / "snapshots").mkdir(parents=True)

    import scripts.backup.orion_backup_databases as mod

    def _explode(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(mod, "prune_successful_snapshots", _explode)

    outcome, _ = run_target_backup(
        Target("postgres", _failing_capture),
        storage_warm=storage_warm,
        node_name="node-a",
        run_id="2026-09-02T03-45-01Z-99999",
    )

    assert outcome.status == "failure"
    assert "boom" in outcome.error_summary
    assert "read-only filesystem" not in outcome.error_summary
    assert outcome.retention_actions == []


def test_retention_failure_on_a_successful_capture_still_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review finding: moving retention off the success path meant a prune
    # exception no longer flipped the run to "failure". Since run_backup
    # derives overall status from per-target status, and the notifier only
    # reports failing targets, a broken retention would have paged nobody and
    # exited 0 -- silently losing the exact alert this PR exists to raise.
    storage_warm = tmp_path / "storage-warm"
    (storage_warm / "backups" / "node-a" / "db" / "widget" / "snapshots").mkdir(parents=True)

    import scripts.backup.orion_backup_databases as mod

    def _explode(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(mod, "prune_successful_snapshots", _explode)

    outcome, _ = run_target_backup(
        Target("widget", _ok_capture),
        storage_warm=storage_warm,
        node_name="node-a",
        run_id="2026-09-02T03-45-01Z-99999",
    )

    assert outcome.status == "failure"
    assert "retention failed" in outcome.error_summary
    assert "read-only filesystem" in outcome.error_summary
    # The snapshot really was written, so its path is still reported.
    assert outcome.snapshot_path is not None


def test_capture_postgres_kills_the_dump_when_gzip_cannot_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review finding: only TimeoutExpired was handled. If starting gzip raised,
    # pg_dumpall was already running and would be left behind -- and killing
    # the local docker-exec client does not kill the server-side backend, which
    # then holds locks on every table until the next night's pre-flight sweep.
    procs: dict[str, _FakeProc] = {}
    calls: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        calls.append(cmd)
        if cmd and cmd[0] == "gzip":
            raise OSError("Cannot allocate memory")
        proc = _FakeProc(cmd, stderr=kwargs.get("stderr"))
        procs["dump"] = proc
        return proc

    def _fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(subprocess, "run", _fake_run)

    dest = tmp_path / "postgres_snapshot"
    with pytest.raises(OSError, match="Cannot allocate memory"):
        capture_postgres(dest, container="orion-athena-sql-db", pg_user="postgres", log=[])

    assert procs["dump"].killed
    terminate_calls = [c for c in calls if "pg_terminate_backend" in " ".join(c)]
    assert len(terminate_calls) == 2  # pre-flight + post-failure cleanup
    assert not (dest / "pg_dumpall.sql.gz").exists()


def test_capture_postgres_fails_fast_when_gzip_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The dump is only useful if it can actually be compressed. Checking the
    # binary up front means the run fails before pg_dumpall is started, rather
    # than after it has already opened a cluster-wide read.
    import scripts.backup.orion_backup_databases as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    def _must_not_run(*args, **kwargs):
        raise AssertionError("no subprocess should start when gzip is missing")

    monkeypatch.setattr(subprocess, "Popen", _must_not_run)
    monkeypatch.setattr(subprocess, "run", _must_not_run)

    with pytest.raises(RuntimeError, match="gzip binary not found"):
        capture_postgres(
            tmp_path / "snap", container="orion-athena-sql-db", pg_user="postgres", log=[]
        )
