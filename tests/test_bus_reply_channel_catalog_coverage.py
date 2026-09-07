"""Every resolvable per-correlation RPC reply channel must be registered in
`orion/bus/channels.yaml`.

Found live 2026-09-07 (`orion-durable-runs`, PR #2128): the reply channel
`orion:curiosity:turn:reply:*` was never added, so Hub's own
`ORION_BUS_ENFORCE_CATALOG=true` made every reply publish raise `ValueError`
before it ever reached the caller -- the request channel worked, the (real,
expensive) work happened, and only the reply silently vanished as a timeout.
`scripts/check_bus_reply_channels.py` is the deterministic gate; this wires
it into the regular test run so a future gap fails a `pytest` run instead of
a live turn.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_bus_reply_channels.py"


def test_check_bus_reply_channels_script_exists() -> None:
    assert SCRIPT.is_file()


def test_every_resolvable_reply_channel_prefix_is_registered() -> None:
    result = subprocess.run(["python3", str(SCRIPT)], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        "one or more per-correlation reply channels have no matching catalog "
        "entry -- add the missing \"<prefix>:*\" wildcard to "
        "orion/bus/channels.yaml (see the script's own output):\n"
        + result.stdout + result.stderr
    )
