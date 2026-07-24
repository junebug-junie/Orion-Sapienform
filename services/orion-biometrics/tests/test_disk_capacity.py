"""Regression coverage for `collect_disk_capacity()` (app/metrics.py).

Added alongside real host disk-usage telemetry piggybacked onto orion-biometrics'
existing bus-native SystemHealthV1 heartbeat (see
docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md,
"Cross-node disk telemetry: bus-piggyback vs. SSH?"). This collector is deliberately
separate from `_collect_disk()` (I/O throughput, feeds BiometricsPipeline's `disk`
pressure) -- it reports filesystem capacity via `shutil.disk_usage()` against
read-only bind mounts (see docker-compose.yml), independent of the biometrics
sample/pipeline/grammar path.
"""

from __future__ import annotations

from pathlib import Path

from app.metrics import collect_disk_capacity


def test_collect_disk_capacity_reports_percent_used_for_real_paths(tmp_path: Path) -> None:
    """A real, existing directory should produce a plausible 0-100 percent-used value
    and no error entry. Uses tmp_path (a real filesystem path) rather than mocking
    shutil.disk_usage, so this exercises the real stdlib call end to end."""
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


def test_collect_disk_capacity_skips_missing_mount_without_failing(tmp_path: Path) -> None:
    """One missing/not-yet-bind-mounted path (e.g. a host path that doesn't exist on
    atlas/circe today) must be skipped, not raise -- the heartbeat must still publish
    with whatever mounts *are* present."""
    real_mount = tmp_path / "graphdb"
    real_mount.mkdir()
    missing_mount = tmp_path / "does_not_exist_at_all"

    result = collect_disk_capacity(
        {"graphdb": str(real_mount), "telemetry": str(missing_mount)}
    )

    assert "graphdb" in result["disk_usage_pct"]
    assert "telemetry" not in result["disk_usage_pct"]
    assert result["disk_usage_errors"] == {"telemetry": "not_mounted"}


def test_collect_disk_capacity_empty_mounts_returns_empty_dict() -> None:
    result = collect_disk_capacity({})
    assert result == {"disk_usage_pct": {}}


def test_collect_disk_capacity_rejects_empty_path_value(tmp_path: Path) -> None:
    """A mount configured with an empty string path (e.g. a malformed
    DISK_CAPACITY_MOUNTS env override) is treated as not-mounted, not as cwd."""
    result = collect_disk_capacity({"bad": ""})
    assert result["disk_usage_errors"] == {"bad": "not_mounted"}
    assert result["disk_usage_pct"] == {}
