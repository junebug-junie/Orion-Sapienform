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


def _dry(mode: str) -> list[str]:
    env = {**os.environ, "DRY_RUN": "1"}
    out = subprocess.run(["bash", str(_SCRIPT), mode], env=env, capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def _ports(lines: list[str], ipt: str) -> set[int]:
    ports: set[int] = set()
    for line in lines:
        if not line.startswith(f"{ipt} ") or "-I PREROUTING" not in line:
            continue
        for spec in re.search(r"--dports (\S+)", line).group(1).split(","):
            lo, _, hi = spec.partition(":")
            ports.update(range(int(lo), int(hi or lo) + 1))
    return ports


def test_apply_allows_local_and_athena_then_drops_for_both_families() -> None:
    out = _dry("apply")
    for ipt, athena in (("iptables", "100.92.216.81/32"), ("ip6tables", "fd7a:115c:a1e0::733:d851/128")):
        chain = [line for line in out if line.startswith(f"{ipt} -w -t mangle -A ORION-LLM-GATE")]
        assert chain == [
            f"{ipt} -w -t mangle -A ORION-LLM-GATE -i lo -j RETURN",
            f"{ipt} -w -t mangle -A ORION-LLM-GATE -i docker0 -j RETURN",
            f"{ipt} -w -t mangle -A ORION-LLM-GATE -i br-+ -j RETURN",
            f"{ipt} -w -t mangle -A ORION-LLM-GATE -s {athena} -j RETURN",
            f"{ipt} -w -t mangle -A ORION-LLM-GATE -j DROP",
        ]
        jump = [line for line in out if line.startswith(f"{ipt} -w -t mangle -I PREROUTING 1")]
        assert len(jump) == 1 and "--dst-type LOCAL" in jump[0] and jump[0].endswith("-j ORION-LLM-GATE")
        # idempotent: every apply starts from a remove of whatever an earlier version left
        assert out.index(f"{ipt} -w -t mangle -X ORION-LLM-GATE || true") < out.index(
            f"{ipt} -w -t mangle -N ORION-LLM-GATE")
    # never the FORWARD/DOCKER-USER path tailscale's ts-forward can jump ahead of
    assert not any("DOCKER-USER" in line or "FORWARD" in line for line in out)


def test_firewall_covers_every_worker_port_the_ci_gate_knows_in_both_families() -> None:
    ident = gate.load_identity(_REPO)
    out = _dry("apply")
    for ipt in ("iptables", "ip6tables"):
        missing = set(ident.ports) - _ports(out, ipt)
        assert not missing, f"{ipt}: ports the CI gate guards but the firewall does not: {sorted(missing)}"
    assert _ports(out, "iptables") == _ports(out, "ip6tables")


def test_remove_deletes_jumps_by_chain_name_not_by_port_list() -> None:
    out = "\n".join(_dry("remove"))
    for ipt in ("iptables", "ip6tables"):
        assert f'{ipt} -w -t mangle -S PREROUTING | grep -- "-j ORION-LLM-GATE"' in out
        assert f"{ipt} -w -t mangle -X ORION-LLM-GATE" in out


def test_bad_mode_exits_2() -> None:
    res = subprocess.run(["bash", str(_SCRIPT), "nope"], capture_output=True, text=True)
    assert res.returncode == 2


def test_script_parses() -> None:
    subprocess.run(["bash", "-n", str(_SCRIPT)], check=True)


def test_install_is_one_command_that_places_enables_reapplies_and_shows_status() -> None:
    out = _dry("install")
    assert out[0].startswith("install -m 0755 ") and out[0].endswith("/usr/local/sbin/orion-llm-port-gate")
    assert out[1].endswith("/etc/systemd/system/orion-llm-port-gate.service")
    assert out[2:] == [
        "systemctl daemon-reload",
        "systemctl enable orion-llm-port-gate.service",
        "systemctl restart orion-llm-port-gate.service",  # a re-install re-applies changed rules
        "/usr/local/sbin/orion-llm-port-gate status",
    ]


def test_uninstall_stops_the_unit_which_removes_the_rules() -> None:
    assert _dry("uninstall")[0] == "systemctl disable --now orion-llm-port-gate.service"
