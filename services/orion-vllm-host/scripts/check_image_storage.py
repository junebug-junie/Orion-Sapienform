#!/usr/bin/env python3
"""Refuse Hecate builds unless configured AND live image storage is on its SSD."""
import json
from pathlib import Path
import subprocess
import tomllib


def image_root(info, containerd_config):
    if ["driver-type", "io.containerd.snapshotter.v1"] in (info.get("driver_status") or []):
        root = tomllib.loads(containerd_config).get("root", "/var/lib/containerd")
    else:
        root = info["docker_root"]
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("Image storage root must be an absolute path")
    return root


def validate_mount(root, mount):
    if mount != "/mnt/docker":
        raise ValueError(f"Image store {root} is on {mount}, not the Docker SSD /mnt/docker")


def mount_for_root(root, mounts):
    # Read the mount table without stat-ing containerd's root-only directories.
    path = Path(root).resolve()
    parents = [Path(item["target"]) for item in mounts if path.is_relative_to(item["target"])]
    if not parents:
        raise ValueError(f"No filesystem found for image store {root}")
    return str(max(parents, key=lambda parent: len(parent.parts)))


def live_snapshot_root(mountinfo):
    """Read the host snapshot path exposed by a running container's root mount."""
    marker = "/io.containerd.snapshotter.v1.overlayfs/"
    roots = set()
    for line in mountinfo.splitlines():
        fields = line.split()
        if len(fields) < 7 or fields[4] != "/" or " - overlay " not in line:
            continue
        options = line.split(" - overlay ", 1)[1].split()[-1]
        for option in options.split(","):
            if option.startswith(("lowerdir=", "upperdir=", "workdir=")):
                for path in option.split("=", 1)[1].split(":"):
                    if marker in path:
                        roots.add(path.split(marker, 1)[0])
    if len(roots) != 1:
        raise ValueError("Cannot establish one live containerd snapshot root")
    return roots.pop()


def validate_live_root(root, mountinfo):
    live = live_snapshot_root(mountinfo)
    if live != root:
        raise ValueError(f"Configured image store {root} differs from live store {live}")


def main():
    info = json.loads(subprocess.check_output([
        "docker", "info", "--format",
        '{"docker_root":{{json .DockerRootDir}},"driver_status":{{json .DriverStatus}}}',
    ], text=True))
    config = Path("/etc/containerd/config.toml")
    root = image_root(info, config.read_text() if config.exists() else "")
    mounts = json.loads(subprocess.check_output([
        "findmnt", "--json", "--list", "--output", "TARGET",
    ], text=True))["filesystems"]
    mount = mount_for_root(root, mounts)
    validate_mount(root, mount)
    if ["driver-type", "io.containerd.snapshotter.v1"] in (info.get("driver_status") or []):
        containers = subprocess.check_output(["docker", "ps", "-q"], text=True).split()
        checked = 0
        for container in containers:
            result = subprocess.run(
                ["docker", "exec", container, "cat", "/proc/self/mountinfo"],
                text=True, capture_output=True, timeout=10,
            )
            if result.returncode:
                continue  # A distroless or concurrently stopped container cannot supply proof.
            validate_live_root(root, result.stdout)
            checked += 1
        if not checked:
            raise ValueError("No running container supplied live storage proof; start the existing biometrics container and retry")
    print(f"Verified image store: {root} on {mount}")


if __name__ == "__main__":
    main()
