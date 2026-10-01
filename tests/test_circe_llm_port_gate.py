"""Tests for scripts/ops/circe_llm_port_gate.sh (GPU pool stage 6.6, port gate runtime layer).

The script needs root to run for real (Juniper runs it on circe). These tests run it with
DRY_RUN=1, which only prints the iptables commands, and check that the firewall and the CI gate
cover the same ports.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "ops" / "circe_llm_port_gate.sh"
sys.path.insert(0, str(_REPO / "scripts"))
import check_circe_worker_refs as gate  # noqa: E402


def _dry(mode: str) -> str:
    env = {**os.environ, "DRY_RUN": "1"}
    out = subprocess.run(["bash", str(_SCRIPT), mode], env=env, capture_output=True, text=True, check=True)
    return out.stdout


def _covered(text: str) -> set[int]:
    ports: set[int] = set()
    for spec in re.findall(r"--ctorigdstport (\S+)", text):
        lo, _, hi = spec.partition(":")
        ports.update(range(int(lo), int(hi or lo) + 1))
    return ports


def test_apply_allows_athena_and_local_then_drops() -> None:
    out = _dry("apply").splitlines()
    chain = [line for line in out if line.startswith("iptables -A ORION-LLM-GATE")]
    assert chain == [
        "iptables -A ORION-LLM-GATE -s 100.92.216.81/32 -j RETURN",
        "iptables -A ORION-LLM-GATE -s 172.16.0.0/12 -j RETURN",
        "iptables -A ORION-LLM-GATE -s 127.0.0.0/8 -j RETURN",
        "iptables -A ORION-LLM-GATE -j DROP",
    ]
    # idempotent: apply starts by removing whatever an earlier apply left
    assert out.index("iptables -F ORION-LLM-GATE || true") < out.index("iptables -N ORION-LLM-GATE")
    assert any(line.startswith("ip6tables -I INPUT 1") and "! -i lo" in line for line in out)


def test_firewall_covers_every_worker_port_the_ci_gate_knows() -> None:
    ident = gate.load_identity(_REPO)
    missing = set(ident.ports) - _covered(_dry("apply"))
    assert not missing, f"ports the CI gate guards but the firewall does not: {sorted(missing)}"


def test_remove_deletes_everything_apply_adds() -> None:
    out = _dry("remove")
    assert "iptables -X ORION-LLM-GATE" in out
    assert _covered(out) == _covered(_dry("apply"))
    assert "ip6tables -D INPUT" in out


def test_bad_mode_exits_2() -> None:
    res = subprocess.run(["bash", str(_SCRIPT), "nope"], capture_output=True, text=True)
    assert res.returncode == 2
