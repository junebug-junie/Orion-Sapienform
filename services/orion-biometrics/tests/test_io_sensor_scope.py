"""The disk and network sensors measure the NODE, not the container (ROADMAP B3).

Two independent bugs made `disk_pressure` and `net_pressure` report quantities that were not
the node's:

  - `/proc/diskstats` lists whole disks AND their partitions, and the partition counters are
    already included in the whole disk's. Summing both double-counts. athena: 1.956x.
  - `/proc/net/dev` is network-namespaced. From inside a bridged container it shows only that
    container's own veth. athena: 1,357 B/s against the host's 159,304 B/s.

The second one is the reason `net_pressure` read 3.2e-05 on athena, 2.2e-05 on atlas and
2.1e-05 on circe -- degenerate on every node in the fleet, incapable of ever signalling
anything, and silently contributing a zero to `strain`.

Device names below are athena's real ones (`lsblk` / `/proc/diskstats`, 2026-08-14).
"""
from __future__ import annotations

import pytest

from app.metrics import BiometricsCollector


@pytest.fixture()
def collector() -> BiometricsCollector:
    return BiometricsCollector()


# ------------------------------------------------------------ disk: whole devices only

# athena, verbatim. Six of the ten devices are partitioned.
ATHENA_DISKSTATS_NAMES = [
    "nvme0n1", "nvme0n1p1",
    "sda", "sdb", "sdc",
    "sdd", "sdd1", "sdd2", "sdd3",
    "sde", "sde1",
    "sdf", "sdf1",
    "sdg", "sdg1", "sdg2",
    "sdh", "sdi",
]


def test_athena_partitions_are_all_rejected(collector):
    accepted = [n for n in ATHENA_DISKSTATS_NAMES if collector._is_disk_device(n)]
    assert accepted == ["nvme0n1", "sda", "sdb", "sdc", "sdd", "sde", "sdf", "sdg", "sdh", "sdi"]
    # Ten whole devices, matching `lsblk -d`. Eight partition rows dropped.
    assert len(accepted) == 10
    assert len(ATHENA_DISKSTATS_NAMES) - len(accepted) == 8


@pytest.mark.parametrize("name", ["sda1", "sdd3", "sdg2", "nvme0n1p1", "vda1", "mmcblk0p1"])
def test_partition_names_are_rejected(collector, name):
    assert collector._is_disk_device(name) is False


@pytest.mark.parametrize("name", ["sda", "sdaa", "nvme0n1", "nvme1n2", "vda", "mmcblk0"])
def test_whole_device_names_are_accepted(collector, name):
    assert collector._is_disk_device(name) is True


@pytest.mark.parametrize("name", ["loop0", "ram0", "dm-0", "md0", "sr0", "zram0"])
def test_non_disk_devices_stay_rejected(collector, name):
    """Unchanged behaviour -- the fix must not widen what counts as a disk."""
    assert collector._is_disk_device(name) is False


def test_nvme_namespace_is_a_device_but_its_partition_is_not(collector):
    """The one naming rule that differs: NVMe separates the partition with a literal 'p',
    so a digit-suffix rule alone would reject `nvme0n1` itself."""
    assert collector._is_disk_device("nvme0n1") is True
    assert collector._is_disk_device("nvme0n1p1") is False


def test_diskstats_totals_drop_by_the_measured_factor(collector, tmp_path, monkeypatch):
    """Hand-computed. Give every device 1000 read sectors and 1000 write sectors.

    Old rule accepted all 18 rows; new rule accepts 10. Bytes = sectors * 512.
      old: 18 * 2000 * 512 = 18,432,000
      new: 10 * 2000 * 512 = 10,240,000
      ratio 1.8 -- the same shape as the 1.956 measured live, where the partitioned devices
      happen to carry most of the traffic.
    """
    rows = []
    for i, name in enumerate(ATHENA_DISKSTATS_NAMES):
        # /proc/diskstats: major minor name then 11+ fields; [5] reads sectors, [9] writes.
        rows.append(f"   8  {i} {name} 0 0 1000 0 0 0 1000 0 0 0 0")
    path = tmp_path / "diskstats"
    path.write_text("\n".join(rows) + "\n")

    real_open = open

    def fake_open(p, *a, **k):
        return real_open(path if p == "/proc/diskstats" else p, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    stats = collector._read_diskstats()
    assert stats.read_bytes == 10 * 1000 * 512
    assert stats.write_bytes == 10 * 1000 * 512
    assert stats.read_bytes + stats.write_bytes == 10_240_000


# ------------------------------------------------------------ net: host namespace

ATHENA_NETDEV = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop
    lo: 999999  100 0 0 0 0 0 0 999999  100 0 0 0 0 0 0
  eno1: 1000 10 1 2 0 0 0 0 2000 20 3 4 0 0 0 0
docker0: 500000 50 0 0 0 0 0 0 500000 50 0 0 0 0 0 0
 veth1: 7 1 0 0 0 0 0 0 9 1 0 0 0 0 0 0
"""


def test_only_the_named_physical_nic_is_counted(collector, tmp_path):
    """docker0 and veth1 carry the same packets eno1 does, one hop earlier. Counting all
    three multiply-counts container traffic -- and docker0 alone here is 1,000,000 bytes
    against eno1's 3,000."""
    path = tmp_path / "dev"
    path.write_text(ATHENA_NETDEV)
    stats = collector._read_netdev(path=str(path), only=["eno1"])
    assert stats.rx_bytes == 1000
    assert stats.tx_bytes == 2000
    assert stats.rx_errors == 1 and stats.rx_drops == 2
    assert stats.tx_errors == 3 and stats.tx_drops == 4


def test_without_a_filter_every_non_loopback_interface_is_counted(collector, tmp_path):
    """The fallback path, unchanged: 1000 + 500000 + 7 rx. Loopback still excluded."""
    path = tmp_path / "dev"
    path.write_text(ATHENA_NETDEV)
    stats = collector._read_netdev(path=str(path))
    assert stats.rx_bytes == 1000 + 500_000 + 7
    assert stats.tx_bytes == 2000 + 500_000 + 9


def test_a_filter_naming_nothing_present_yields_zero_not_the_unfiltered_total(collector, tmp_path):
    path = tmp_path / "dev"
    path.write_text(ATHENA_NETDEV)
    stats = collector._read_netdev(path=str(path), only=["eno9"])
    assert stats.rx_bytes == 0 and stats.tx_bytes == 0


# ------------------------------------------------------------ physical NIC discovery

def _fake_sysfs(root, ifaces):
    """ifaces: {name: (has_device, operstate, speed_or_None)}"""
    base = root / "class" / "net"
    for name, (has_device, operstate, speed) in ifaces.items():
        d = base / name
        d.mkdir(parents=True)
        if has_device:
            (d / "device").mkdir()
        (d / "operstate").write_text(operstate + "\n")
        if speed is not None:
            (d / "speed").write_text(str(speed) + "\n")
    return root


# athena verbatim: six onboard NICs, one plugged in; three docker bridges and docker0 all
# claiming a fabricated 10 Gb/s; tailscale0 reporting -1.
ATHENA_SYSFS = {
    "eno1": (True, "up", 1000),
    "eno2": (True, "down", None),
    "eno3": (True, "down", None),
    "eno4": (True, "down", None),
    "eno5": (True, "down", None),
    "eno6": (True, "down", None),
    "docker0": (False, "up", 10000),
    "br-6d7d917f145a": (False, "up", 10000),
    "br-8a71dda15b30": (False, "up", 10000),
    "tailscale0": (False, "unknown", -1),
    "lo": (False, "unknown", None),
}


def test_athena_resolves_to_exactly_one_physical_nic(collector, tmp_path, monkeypatch):
    _fake_sysfs(tmp_path, ATHENA_SYSFS)
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    assert collector._physical_interfaces() == ["eno1"]


def test_link_speed_is_one_gigabit_not_thirty_one(collector, tmp_path, monkeypatch):
    """The number this replaces: summing every interface reporting a `speed` would give
    1000 + 10000*3 = 31,000 Mb/s of capacity on a box with one 1 Gb link, understating
    net_pressure 31-fold."""
    _fake_sysfs(tmp_path, ATHENA_SYSFS)
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    ifaces = collector._physical_interfaces()
    assert collector._link_speed_mbps(ifaces) == 1000.0


def test_dark_ports_contribute_no_capacity(collector, tmp_path, monkeypatch):
    """eno2..eno6 are real NICs with no cable. Counting them would claim 6 Gb/s."""
    _fake_sysfs(tmp_path, ATHENA_SYSFS)
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    assert "eno2" not in collector._physical_interfaces()


def test_two_live_nics_sum(collector, tmp_path, monkeypatch):
    _fake_sysfs(tmp_path, {"eno1": (True, "up", 1000), "eno2": (True, "up", 10000)})
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    assert collector._link_speed_mbps(collector._physical_interfaces()) == 11000.0


def test_speed_of_minus_one_is_unknown_not_a_capacity(collector, tmp_path, monkeypatch):
    """A driver that does not know the link speed writes -1. Treating it as a number would
    make the denominator negative and flip the pressure's sign."""
    _fake_sysfs(tmp_path, {"eth0": (True, "up", -1)})
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    assert collector._link_speed_mbps(collector._physical_interfaces()) is None


def test_no_readable_speed_returns_none_not_zero(collector, tmp_path, monkeypatch):
    """None -> caller falls back to NET_BW_MBPS. 0.0 would be a division by zero, or worse,
    'this node has no network capacity'."""
    _fake_sysfs(tmp_path, {"eth0": (True, "up", None)})
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path))
    assert collector._link_speed_mbps(collector._physical_interfaces()) is None


def test_missing_host_sysfs_mount_returns_none_rather_than_an_empty_list(collector, monkeypatch):
    """None and [] mean different things: 'could not look' vs 'looked, found no NIC'. The
    caller uses None to fall back to the container namespace."""
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "/nonexistent-host-sys")
    assert collector._physical_interfaces() is None


def test_empty_host_sys_setting_disables_the_host_read(collector, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "")
    assert collector._physical_interfaces() is None


# ------------------------------------------------------------ scope reporting

def test_collect_network_reports_container_scope_when_unmounted(collector, monkeypatch):
    monkeypatch.setattr("app.metrics.settings.HOST_PROC_PATH", "/nonexistent-host-proc")
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "/nonexistent-host-sys")
    errors: list = []
    out = collector._collect_network(errors, dt=1.0)
    assert out["scope"] == "container"
    assert "link_mbps" not in out
    assert not errors


def test_a_scope_change_drops_the_baseline_instead_of_spiking(collector, tmp_path, monkeypatch):
    """Host counters are far larger than container ones. Subtracting a container baseline
    from a host reading would report a one-tick burst of the entire host's lifetime traffic."""
    monkeypatch.setattr("app.metrics.settings.HOST_PROC_PATH", "/nonexistent-host-proc")
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", "/nonexistent-host-sys")
    collector._collect_network([], dt=1.0)
    assert collector._prev_net_scope == "container"

    # Now the host mounts appear (a redeploy), and the counters jump by orders of magnitude.
    proc = tmp_path / "proc" / "1" / "net"
    proc.mkdir(parents=True)
    (proc / "dev").write_text(ATHENA_NETDEV.replace("  eno1: 1000 10", "  eno1: 99999999999 10"))
    _fake_sysfs(tmp_path / "sys", {"eno1": (True, "up", 1000)})
    monkeypatch.setattr("app.metrics.settings.HOST_PROC_PATH", str(tmp_path / "proc"))
    monkeypatch.setattr("app.metrics.settings.HOST_SYS_PATH", str(tmp_path / "sys"))

    out = collector._collect_network([], dt=1.0)
    assert out["scope"] == "host"
    assert out["rx_bytes_per_sec"] == 0.0  # baseline dropped, not a 99 GB/s spike
    assert out["link_mbps"] == 1000.0
    assert out["interfaces"] == ["eno1"]

    # The tick after, the delta is real again.
    (proc / "dev").write_text(ATHENA_NETDEV.replace("  eno1: 1000 10", "  eno1: 100000000499 10"))
    out2 = collector._collect_network([], dt=1.0)
    assert out2["rx_bytes_per_sec"] == 500.0
