"""CPU package power from RAPL (ROADMAP B4).

athena draws 407 W at the PSU with the GPU at 44 W. The other 363 W was entirely
unattributed on a node whose whole job is CPU orchestration. RAPL says 210 W of it is the two
CPU packages -- more than half the machine.

`energy_uj` is a CUMULATIVE microjoule counter, so power is a delta over time. Two properties
make that harder than it looks and both are tested here:

  - it WRAPS at max_energy_range_uj (262,143,328,850 uJ on athena), which at the measured
    ~105 W per package is roughly every 41 minutes -- about once every 83 ticks at
    TELEMETRY_INTERVAL=30. A negative delta is normal, not an error.
  - a gap longer than one wrap period is AMBIGUOUS. One wrap and three wraps are
    indistinguishable, so a stale baseline must be discarded, never guessed at.

Values below are athena's real ones, measured 2026-08-14.
"""
from __future__ import annotations

import pytest

from app.metrics import _RAPL_MAX_GAP_SEC, BiometricsCollector

# athena: two sockets, identical range.
MAX_RANGE = 262_143_328_850


@pytest.fixture()
def collector() -> BiometricsCollector:
    return BiometricsCollector()


def _rapl_tree(root, domains):
    """domains: {dirname: (name, energy_uj)}. Writes a powercap sysfs tree."""
    base = root / "class" / "powercap"
    for dirname, (name, energy) in domains.items():
        d = base / dirname
        d.mkdir(parents=True, exist_ok=True)
        (d / "name").write_text(name + "\n")
        (d / "energy_uj").write_text(str(energy) + "\n")
        (d / "max_energy_range_uj").write_text(str(MAX_RANGE) + "\n")
    return root


def _two_sockets(p0: int, p1: int):
    return {"intel-rapl:0": ("package-0", p0), "intel-rapl:1": ("package-1", p1)}


# ------------------------------------------------------------------ the measurement

def test_the_live_athena_reading(collector, tmp_path, monkeypatch):
    """Hand-computed from the real 20 s probe:
        package-0  2,072,483,098 uJ / 20 s = 103.62 W
        package-1  2,134,517,729 uJ / 20 s = 106.73 W
        total                               210.35 W
    """
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, _two_sockets(91_900_177_637, 21_748_398_913))
    assert collector._rapl_package_watts(dt=20.0) == (None, {})  # first tick: no baseline

    _rapl_tree(tmp_path, _two_sockets(91_900_177_637 + 2_072_483_098,
                                      21_748_398_913 + 2_134_517_729))
    total, packages = collector._rapl_package_watts(dt=20.0)
    assert packages["package-0"] == pytest.approx(103.62, abs=0.01)
    assert packages["package-1"] == pytest.approx(106.73, abs=0.01)
    assert total == pytest.approx(210.35, abs=0.01)


def test_both_sockets_are_summed_not_averaged(collector, tmp_path, monkeypatch):
    """A dual-socket box averaged would read like a single-socket box -- the same error
    `_power_pressure` makes with GPUs and the reason gpu_watts_total exists separately."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, _two_sockets(0, 0))
    collector._rapl_package_watts(dt=1.0)
    _rapl_tree(tmp_path, _two_sockets(100_000_000, 100_000_000))  # 100 J each in 1 s
    total, _ = collector._rapl_package_watts(dt=1.0)
    assert total == pytest.approx(200.0)


# ------------------------------------------------------------------ wrap-around

def test_a_wrap_is_corrected_not_reported_as_a_power_drop(collector, tmp_path, monkeypatch):
    """Hand-computed. Baseline 100 uJ below the range, then 900 uJ past the wrap:
        delta = 900 - (MAX_RANGE - 100) = -262,143,328,750
        corrected = -262,143,328,750 + MAX_RANGE + 1 = 1001 uJ
    Over 1 s on one socket that is 0.001001 W. Uncorrected it would be -262,143 W.
    """
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    one = {"intel-rapl:0": ("package-0", MAX_RANGE - 100)}
    _rapl_tree(tmp_path, one)
    collector._rapl_package_watts(dt=1.0)

    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", 900)})
    total, packages = collector._rapl_package_watts(dt=1.0)
    assert packages["package-0"] == pytest.approx(1001 / 1e6)
    assert total > 0


def test_a_realistic_wrap_at_load_gives_a_realistic_wattage(collector, tmp_path, monkeypatch):
    """The case that actually happens every ~41 minutes. 30 s at ~105 W is 3,150,000,000 uJ;
    straddle the wrap and the answer must still be 105 W, not a negative spike."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    before = MAX_RANGE - 1_000_000_000
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", before)})
    collector._rapl_package_watts(dt=30.0)

    after = 3_150_000_000 - 1_000_000_000 - 1  # wrapped past the top by this much
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", after)})
    total, _ = collector._rapl_package_watts(dt=30.0)
    assert total == pytest.approx(105.0, abs=0.01)


def test_a_gap_longer_than_the_guard_is_discarded_not_guessed(collector, tmp_path, monkeypatch):
    """After a long stall the counter may have wrapped more than once, and one wrap is
    indistinguishable from three. Correcting by a single range would under-report by whole
    multiples. Absent beats confidently wrong."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, _two_sockets(1_000, 1_000))
    collector._rapl_package_watts(dt=30.0)
    _rapl_tree(tmp_path, _two_sockets(2_000, 2_000))
    assert collector._rapl_package_watts(dt=_RAPL_MAX_GAP_SEC + 1) == (None, {})


def test_the_guard_allows_a_merely_slow_tick(collector, tmp_path, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", 0)})
    collector._rapl_package_watts(dt=30.0)
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", 100_000_000)})
    total, _ = collector._rapl_package_watts(dt=_RAPL_MAX_GAP_SEC - 1)
    assert total == pytest.approx(100.0 / (_RAPL_MAX_GAP_SEC - 1))


# ------------------------------------------------------------------ domain selection

def test_subdomains_are_not_double_counted(collector, tmp_path, monkeypatch):
    """`intel-rapl:0:0` (core) and `intel-rapl:0:1` (dram) are already inside package-0.
    Counting them would roughly double the figure -- the same bug class as summing disk
    partitions alongside their whole disk."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    tree = {
        "intel-rapl:0": ("package-0", 0),
        "intel-rapl:0:0": ("core", 0),
        "intel-rapl:0:1": ("dram", 0),
    }
    _rapl_tree(tmp_path, tree)
    collector._rapl_package_watts(dt=1.0)
    _rapl_tree(tmp_path, {
        "intel-rapl:0": ("package-0", 100_000_000),
        "intel-rapl:0:0": ("core", 80_000_000),
        "intel-rapl:0:1": ("dram", 20_000_000),
    })
    total, packages = collector._rapl_package_watts(dt=1.0)
    assert total == pytest.approx(100.0)  # not 200.0
    assert list(packages) == ["package-0"]


def test_a_socket_appearing_mid_run_suppresses_the_tick(collector, tmp_path, monkeypatch):
    """Reporting the sum without the new socket would silently understate the node by a whole
    package -- a number that looks plausible and is wrong by ~50% on a dual-socket box."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", 0)})
    collector._rapl_package_watts(dt=1.0)
    _rapl_tree(tmp_path, _two_sockets(100_000_000, 100_000_000))
    assert collector._rapl_package_watts(dt=1.0) == (None, {})
    # ...and it recovers on the tick after, once both have a baseline.
    _rapl_tree(tmp_path, _two_sockets(200_000_000, 200_000_000))
    total, _ = collector._rapl_package_watts(dt=1.0)
    assert total == pytest.approx(200.0)


# ------------------------------------------------------------------ absence

def test_no_powercap_tree_is_absent_not_zero(collector, tmp_path, monkeypatch):
    """A node with no RAPL (or an AMD box on an older kernel) must be ABSENT from the fleet
    CPU total, not counted as drawing 0 W."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    (tmp_path / "class" / "powercap").mkdir(parents=True)
    assert collector._rapl_package_watts(dt=1.0) == (None, {})


def test_missing_host_sys_mount_is_absent(collector, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "/nonexistent-host-sys")
    assert collector._rapl_package_watts(dt=1.0) == (None, {})


def test_unreadable_energy_file_is_absent_not_a_crash(collector, tmp_path, monkeypatch):
    """energy_uj is mode 400. This service runs as uid 0 so it reads fine, but a deployment
    that did not would get None rather than an exception escaping the collector."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    d = tmp_path / "class" / "powercap" / "intel-rapl:0"
    d.mkdir(parents=True)
    (d / "name").write_text("package-0\n")
    (d / "max_energy_range_uj").write_text(str(MAX_RANGE) + "\n")
    # no energy_uj at all
    assert collector._rapl_package_watts(dt=1.0) == (None, {})


@pytest.mark.parametrize("dt", [None, 0.0, -5.0])
def test_a_nonsense_dt_reports_nothing(collector, tmp_path, monkeypatch, dt):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", 0)})
    collector._rapl_package_watts(dt=1.0)
    _rapl_tree(tmp_path, {"intel-rapl:0": ("package-0", 100_000_000)})
    assert collector._rapl_package_watts(dt=dt) == (None, {})


# ------------------------------------------------------------------ payload wiring

def test_collect_power_carries_the_reading_and_the_breakdown(collector, tmp_path, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    _rapl_tree(tmp_path, _two_sockets(0, 0))
    collector._collect_power({"gpus": []}, {}, 1.0)
    _rapl_tree(tmp_path, _two_sockets(103_620_000, 106_730_000))
    out = collector._collect_power({"gpus": []}, {}, 1.0)
    assert out["cpu_package_watts"] == pytest.approx(210.35, abs=0.01)
    assert out["cpu_package_watts_by_domain"]["package-0"] == pytest.approx(103.62, abs=0.01)


def test_collect_power_omits_the_key_entirely_when_unavailable(collector, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "/nonexistent-host-sys")
    out = collector._collect_power({"gpus": []}, {}, 1.0)
    assert "cpu_package_watts" not in out
    assert "gpu_power_watts" in out  # the pre-existing payload is untouched
