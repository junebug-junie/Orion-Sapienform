"""Regression coverage for `collect_disk_capacity()` (app/metrics.py).

Added alongside real host disk-usage telemetry piggybacked onto orion-biometrics'
existing bus-native SystemHealthV1 heartbeat (see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md,
"Cross-node disk telemetry: bus-piggyback vs. SSH?"). This collector is deliberately
separate from `_collect_disk()` (I/O throughput, feeds BiometricsPipeline's `disk`
pressure) -- it reports filesystem capacity via `shutil.disk_usage()` against
read-only bind mounts (see docker-compose.yml), independent of the biometrics
sample/pipeline/grammar path.

`collect_disk_capacity()` requires `os.path.ismount(path)`, not just `os.path.isdir(path)`
(2026-07-24 review fix). A `tmp_path` subdirectory is a real directory but never a real
mount point, so most tests here monkeypatch `os.path.ismount` to simulate "this path is a
real bind-mount root" without needing an actual host mount inside the test sandbox --
`shutil.disk_usage()` itself is still called for real against `tmp_path`, so the
percent-used computation is exercised end to end, not mocked.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.metrics import collect_disk_capacity


def _mount_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate every path handed to os.path.ismount() as a real mount point."""
    monkeypatch.setattr(os.path, "ismount", lambda path: True)


def test_collect_disk_capacity_reports_percent_used_for_real_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real, existing, mounted directory should produce a plausible 0-100
    percent-used value and no error entry. shutil.disk_usage() is called for real
    against tmp_path (not mocked), so this exercises the real stdlib call end to end."""
    _mount_everything(monkeypatch)
    mount_a = tmp_path / "docker"
    mount_a.mkdir()
    mount_b = tmp_path / "scripts"
    mount_b.mkdir()

    result = collect_disk_capacity({"docker": str(mount_a), "scripts": str(mount_b)})

    assert "disk_usage_errors" not in result
    pct = result["disk_usage_pct"]
    assert set(pct.keys()) == {"docker", "scripts"}
    for value in pct.values():
        assert isinstance(value, float)
        assert 0.0 <= value <= 100.0


def test_collect_disk_capacity_skips_missing_mount_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One missing/not-yet-bind-mounted path (e.g. a host path that doesn't exist on
    atlas/circe today) must be skipped, not raise -- the heartbeat must still publish
    with whatever mounts *are* present."""
    _mount_everything(monkeypatch)
    real_mount = tmp_path / "graphdb"
    real_mount.mkdir()
    missing_mount = tmp_path / "does_not_exist_at_all"

    result = collect_disk_capacity(
        {"graphdb": str(real_mount), "telemetry": str(missing_mount)}
    )

    assert "graphdb" in result["disk_usage_pct"]
    assert "telemetry" not in result["disk_usage_pct"]
    assert result["disk_usage_errors"] == {"telemetry": "not_mounted"}


def test_collect_disk_capacity_treats_unmounted_dir_as_not_mounted(tmp_path: Path) -> None:
    """Regression for the Docker auto-create-empty-dir footgun: when a bind-mount
    source doesn't exist on the host, Docker silently creates an empty directory at
    the container target instead of failing the mount. A plain os.path.isdir() check
    can't tell that apart from a real mount, and would then report the *container's
    own* overlay filesystem usage as if it were the host mount's -- a plausible-looking
    but meaningless number. No monkeypatching here: tmp_path is a real directory that
    is genuinely not a mount point, exactly this scenario."""
    unmounted_dir = tmp_path / "telemetry"
    unmounted_dir.mkdir()
    assert not os.path.ismount(unmounted_dir)  # sanity: this really isn't a mount

    result = collect_disk_capacity({"telemetry": str(unmounted_dir)})

    assert result["disk_usage_pct"] == {}
    assert result["disk_usage_errors"] == {"telemetry": "not_mounted"}


def test_collect_disk_capacity_empty_mounts_returns_empty_dict() -> None:
    result = collect_disk_capacity({})
    assert result == {"disk_usage_pct": {}}


def test_collect_disk_capacity_rejects_empty_path_value() -> None:
    """A mount configured with an empty string path (e.g. a malformed
    DISK_CAPACITY_MOUNTS env override) is treated as not-mounted, not as cwd."""
    result = collect_disk_capacity({"bad": ""})
    assert result["disk_usage_errors"] == {"bad": "not_mounted"}
    assert result["disk_usage_pct"] == {}
