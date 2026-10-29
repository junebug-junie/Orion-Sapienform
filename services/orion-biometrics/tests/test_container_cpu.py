"""Per-container busy cores from cgroup v2 cpu.stat (act energy cost, 2026-10-29).

Layout is athena's live one: systemd cgroup driver, /sys/fs/cgroup/system.slice/docker-<id>.scope,
names in <docker root>/containers/<id>/config.v2.json. vector-host's live counter was
2,911,569,556,451 usec on 2026-10-29.
"""
from __future__ import annotations

import json

import pytest

from app.metrics import BiometricsCollector

VEC = "3673251d6726472607500132e4a019e5104e59613b8d802ab97b840443ef60bf"
DOJO = "0108b11a8e8cd4b16ab60019ecd0565dcf360070de3bf8618c7ea1eb1cea6005"


def _cgroup(sys_root, cid, usec, *, systemd=True):
    d = (sys_root / "fs/cgroup/system.slice" / f"docker-{cid}.scope") if systemd else (sys_root / "fs/cgroup/docker" / cid)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cpu.stat").write_text(f"usage_usec {usec}\nuser_usec 1\nsystem_usec 1\n")


def _name(docker_root, cid, name):
    d = docker_root / "containers" / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.v2.json").write_text(json.dumps({"Name": f"/{name}"}))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    sys_root, docker_root = tmp_path / "sys", tmp_path / "docker"
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(sys_root))
    monkeypatch.setattr("app.metrics.settings.DOCKER_ROOT_PATH", str(docker_root))
    monkeypatch.setattr("app.metrics.settings.NODE_NAME", "athena")
    monkeypatch.setattr("app.metrics.settings.CPU_IDLE_WATTS_BY_NODE", '{"athena":99.0}')
    return sys_root, docker_root


def test_busy_cores_per_named_container(env):
    sys_root, docker_root = env
    _cgroup(sys_root, VEC, 2_911_569_556_451)
    _cgroup(sys_root, DOJO, 1_000_000_000)
    _name(docker_root, VEC, "orion-athena-vector-host")
    c = BiometricsCollector()
    first = c._collect_containers(dt=30.0)
    assert "cpu_cores" not in first  # first tick: no delta, absent not zero
    assert first["cpu_idle_watts"] == 99.0
    _cgroup(sys_root, VEC, 2_911_569_556_451 + 60_000_000)   # 60 core-seconds in 30 s
    _cgroup(sys_root, DOJO, 1_000_000_000 + 30_000_000)      # 30 core-seconds
    out = c._collect_containers(dt=30.0)
    assert out["cpu_cores"]["orion-athena-vector-host"] == pytest.approx(2.0)
    assert out["cpu_cores"][DOJO[:12]] == pytest.approx(1.0)  # no config.v2.json: the short id


def test_cgroupfs_layout_is_read_too(env):
    sys_root, _ = env
    _cgroup(sys_root, VEC, 100, systemd=False)
    c = BiometricsCollector()
    c._collect_containers(dt=10.0)
    _cgroup(sys_root, VEC, 100 + 5_000_000, systemd=False)
    assert c._collect_containers(dt=10.0)["cpu_cores"][VEC[:12]] == pytest.approx(0.5)


def test_a_restarted_counter_and_a_long_gap_report_nothing(env):
    sys_root, _ = env
    _cgroup(sys_root, VEC, 5_000_000)
    c = BiometricsCollector()
    c._collect_containers(dt=30.0)
    _cgroup(sys_root, VEC, 1_000)  # counter went backwards: a new cgroup under the same id
    assert c._collect_containers(dt=30.0)["cpu_cores"] == {}
    _cgroup(sys_root, VEC, 9_000_000)
    assert "cpu_cores" not in c._collect_containers(dt=10_000.0)


def test_non_container_entries_are_ignored(env):
    sys_root, _ = env
    (sys_root / "fs/cgroup/system.slice/docker.service").mkdir(parents=True)
    (sys_root / "fs/cgroup/system.slice/docker.service/cpu.stat").write_text("usage_usec 5\n")
    assert BiometricsCollector()._read_container_usec() == {}


def test_missing_cgroup_tree_is_absent(env, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "")
    c = BiometricsCollector()
    c._collect_containers(dt=30.0)
    assert "cpu_cores" not in c._collect_containers(dt=30.0)


@pytest.mark.parametrize("raw,want", [
    ('{"athena":99.0}', 99.0), ("", 99.0), ("not json", None), ('{"circe":100}', None), ('{"athena":-1}', None),
])
def test_idle_floor_setting(monkeypatch, raw, want):
    from app.settings import settings

    monkeypatch.setattr(settings, "NODE_NAME", "athena")
    monkeypatch.setattr(settings, "CPU_IDLE_WATTS_BY_NODE", raw)
    assert settings.cpu_idle_watts() == want
