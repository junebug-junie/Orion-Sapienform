"""The llama.cpp operator scripts call a model only under a GPU pool operator hold, through the gateway
(leased-evals design; lane-door PR 1). The probes' Python is covered by test_probe_*; this pins the
shell scripts."""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
QUICK = SCRIPTS / "verify_atlas_quick_llamacpp_thinking_off.sh"
QWEN3 = SCRIPTS / "verify_qwen3_thinking_off_live.sh"


def test_quick_check_uses_the_leased_helper_and_no_worker_url():
    src = QUICK.read_text()
    assert "-m orion.evals.leased_llm chat" in src and "--holder operator:" in src
    assert "ATLAS_LLAMACPP_QUICK_URL" not in src
    assert not re.search(r"curl[^\n]*/v1/chat/completions", src)


def test_qwen3_container_check_runs_under_an_operator_hold():
    src = QWEN3.read_text()
    hold = src.index("-m orion.evals.leased_llm hold")
    assert hold < src.index("docker run --rm --init")
    assert "trap 'docker rm -f" in src and "CUDA_DEVICE_ORDER=PCI_BUS_ID" in src
    assert "--holder operator:verify-qwen3-thinking-off" in src


def _stub(tmp_path: Path, body: str, code: int = 0) -> Path:
    stub = tmp_path / "py"
    stub.write_text(f"#!/usr/bin/env bash\ncat >/dev/null\necho '{body}'\nexit {code}\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
@pytest.mark.parametrize("body,code,want", [
    ('{"choices":[{"message":{"content":"LIVE-ATLAS-QUICK-NO-THINK-OK"}}]}', 0, 0),
    ('{"choices":[{"message":{"content":"<think>x</think>LIVE-ATLAS-QUICK-NO-THINK-OK"}}]}', 0, 1),
    ('{"choices":[{"message":{"content":"nope"}}]}', 0, 1),
    ("", 3, 3),   # recalled: the helper's exit code passes through
])
def test_quick_check_verdicts(tmp_path, body, code, want):
    env = {**os.environ, "ORION_PYTHON": str(_stub(tmp_path, body, code)), "ORION_BUS_URL": "redis://x:1/0"}
    got = subprocess.run(["bash", str(QUICK)], env=env, capture_output=True, text=True).returncode
    assert got == want


def test_every_model_call_in_the_hosts_scripts_and_evals_goes_through_the_lease_helper():
    """The address scan cannot see a loopback or tunnel URL (review finding). This one can: a file in the
    llama.cpp host's scripts/ or evals/ that names a model-call path must use orion.evals.leased_llm."""
    root = SCRIPTS.parent
    offenders = []
    for folder in ("scripts", "evals"):
        for path in sorted((root / folder).rglob("*")):
            if path.suffix not in (".py", ".sh") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            if re.search(r"/v1/chat/completions|/v1/completions|/completion\b|/v1/messages", text) \
                    and "leased_llm" not in text:
                offenders.append(path.relative_to(root).as_posix())
    assert offenders == [], offenders
