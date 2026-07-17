from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "scripts" / "hooks" / "graphify_hook_guard_gate.sh"


def _fake_graphify_bin(tmp_path: Path) -> Path:
    """A stand-in for the real (out-of-repo, uv-tool-installed) graphify
    binary that just echoes its argv so tests don't need it installed."""
    fake = tmp_path / "fake_graphify.sh"
    fake.write_text("#!/usr/bin/env bash\necho \"called: $*\"\n", encoding="utf-8")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return fake


def test_gate_skips_graphify_for_fcc_subprocess(tmp_path: Path) -> None:
    """orion/harness/fcc_motor.py tags its subprocess env with
    ORION_FCC_SUBPROCESS=1 -- the gate must no-op (no stdout, exit 0)
    without even invoking graphify, since a chat/forensics turn gets no
    value from the "run graphify before grepping" nudge."""
    fake = _fake_graphify_bin(tmp_path)
    env = {**os.environ, "ORION_FCC_SUBPROCESS": "1", "GRAPHIFY_HOOK_GUARD_BIN": str(fake)}

    proc = subprocess.run(
        [str(GATE), "search"], env=env, input="{}", capture_output=True, text=True, timeout=10
    )

    assert proc.returncode == 0
    assert proc.stdout == ""


def test_gate_forwards_to_graphify_outside_fcc_subprocess(tmp_path: Path) -> None:
    """Without the FCC marker (a real interactive Claude Code session), the
    gate must still forward to the real hook-guard -- this is where future
    light development work through this repo would want the nudge."""
    fake = _fake_graphify_bin(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "ORION_FCC_SUBPROCESS"}
    env["GRAPHIFY_HOOK_GUARD_BIN"] = str(fake)

    proc = subprocess.run(
        [str(GATE), "read"], env=env, input="{}", capture_output=True, text=True, timeout=10
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == "called: hook-guard read"


def test_gate_script_is_executable() -> None:
    mode = GATE.stat().st_mode
    assert mode & stat.S_IXUSR, f"{GATE} must be executable (chmod +x)"
