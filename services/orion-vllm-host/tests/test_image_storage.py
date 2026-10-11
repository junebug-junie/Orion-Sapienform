import runpy
from pathlib import Path

import pytest

CHECK = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_image_storage.py"))
INFO = {"docker_root": "/mnt/docker", "driver_status": [["driver-type", "io.containerd.snapshotter.v1"]]}


def test_docker_data_root_does_not_hide_containerd_default_on_root():
    root = CHECK["image_root"](INFO, '#root = "/var/lib/containerd"\n')
    assert root == "/var/lib/containerd"
    with pytest.raises(ValueError, match="not the Docker SSD"):
        CHECK["validate_mount"](root, "/")


def test_containerd_store_on_docker_ssd_is_accepted():
    root = CHECK["image_root"](INFO, 'root = "/mnt/docker/containerd-store"\n')
    assert root == "/mnt/docker/containerd-store"
    CHECK["validate_mount"](root, "/mnt/docker")


def test_unmounted_docker_ssd_is_rejected():
    root = CHECK["image_root"](INFO, 'root = "/mnt/docker/containerd-store"\n')
    with pytest.raises(ValueError):
        CHECK["validate_mount"](root, "/")


def test_classic_storage_uses_docker_data_root():
    assert CHECK["image_root"]({"docker_root": "/mnt/docker", "driver_status": []}, "") == "/mnt/docker"


def test_relative_containerd_root_is_rejected():
    with pytest.raises(ValueError, match="absolute path"):
        CHECK["image_root"](INFO, 'root = "relative"\n')


def mountinfo(root):
    return f"541 390 0:55 / / rw - overlay overlay rw,lowerdir={root}/io.containerd.snapshotter.v1.overlayfs/snapshots/1/fs,upperdir={root}/io.containerd.snapshotter.v1.overlayfs/snapshots/2/fs\n"


def test_edited_config_does_not_hide_daemon_still_using_root():
    with pytest.raises(ValueError, match="differs from live"):
        CHECK["validate_live_root"]("/mnt/docker/containerd-store", mountinfo("/var/lib/containerd"))


def test_matching_live_storage_is_accepted():
    CHECK["validate_live_root"]("/mnt/docker/containerd-store", mountinfo("/mnt/docker/containerd-store"))


@pytest.mark.parametrize("data", ["", "1 2 0:1 / / rw - ext4 /dev/sda rw\n"])
def test_missing_live_snapshot_proof_fails_closed(data):
    with pytest.raises(ValueError, match="Cannot establish"):
        CHECK["live_snapshot_root"](data)
