#!/usr/bin/env python3
"""Read-only hardware evidence, bounded commands; missing tools are recorded, not guessed."""
import json
import subprocess
from datetime import datetime, timezone

COMMANDS = [
    ["hostname"], ["nvidia-smi"], ["nvidia-smi", "topo", "-m"],
    ["nvidia-smi", "topo", "-p2p", "r"], ["nvidia-smi", "topo", "-p2p", "w"],
    ["nvidia-smi", "nvlink", "-s"], ["nvidia-smi", "-q", "-d", "PCI,ECC"],
    ["lscpu"], ["numactl", "--hardware"], ["free", "-b"],
    ["df", "-B1", "/", "/mnt/models", "/mnt/docker"],
    ["lspci", "-tv"], ["gcc", "--version"], ["nvcc", "--version"],
    ["docker", "ps", "--format", "{{.Names}} {{.Status}} {{.Ports}}"],
]


def capture(cmd):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
        return {"command": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": cmd, "error": str(exc), "status": "UNVERIFIED"}


if __name__ == "__main__":
    print(json.dumps({"captured_at": datetime.now(timezone.utc).isoformat(),
                      "commands": [capture(cmd) for cmd in COMMANDS]}, indent=2))
