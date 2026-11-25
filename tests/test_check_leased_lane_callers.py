"""Tests for scripts/check_leased_lane_callers.py (#3921): a committed eval/script that calls a model goes
through orion/evals/leased_llm.py, or is a reviewed gateway caller."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))
import check_leased_lane_callers as gate  # noqa: E402

POOL_YAML = "host:\n  address: 100.112.254.99\n  name: circe\nroles:\n  agent: {kind: llm, port: 8015}\n"


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def tree(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(gate, "GATEWAY_CALLERS", {})
    _write(tmp_path, "config/gpu_pool.yaml", POOL_YAML)
    return tmp_path


# The shape of the bug: a bare --url, a docstring that names the agent seat, no lease header.
UNLEASED_EVAL = '''"""Run under an operator lease:
    python eval.py --url http://<strata-agent-host>:8015
"""
import urllib.request
def call(url, b):
    return urllib.request.Request(url + "/v1/chat/completions", data=b)
'''
LEASED_EVAL = '''from orion.evals.leased_llm import operator_lane_sync
def call(b):
    with operator_lane_sync("agent", holder="operator:x") as lane:
        return lane.post("/v1/chat/completions", b)
'''
GATEWAY_EVAL = '''import httpx
def call(gw, b):
    return httpx.post(gw + "/v1/chat/completions", json={"model": "agent", **b})
'''


def test_real_tree_passes() -> None:
    out = gate.run(_REPO)
    assert out["ok"], out


@pytest.mark.parametrize("rel", ["services/orion-cortex-exec/evals/eval_json_task_turn_end_after_brace.py",
                                 "scripts/analysis/eval_curiosity_register_judge.py"])
def test_the_two_callers_fixed_by_3921_use_the_helper(rel: str) -> None:
    text = (_REPO / rel).read_text()
    assert gate.is_leased(rel, text)
    port_re = gate.lane_port_re(gate.port_gate.load_identity(_REPO).ports)
    assert gate.classify_text(rel, text, port_re)[0] in ("none", "leased")


def test_helper_and_gate_are_not_scanned() -> None:
    scanned = {rel for _, rel in gate.iter_scope(_REPO)}
    assert not scanned & gate.OWNERS


def test_unleased_eval_naming_a_lane_port_fails(tree: Path) -> None:
    _write(tree, "services/orion-x/evals/eval_thing.py", UNLEASED_EVAL)
    out = gate.run(tree)
    assert not out["ok"]
    assert [(f["path"], f["problem"]) for f in out["findings"]] == [
        ("services/orion-x/evals/eval_thing.py", "names_lane_port")]


def test_allow_list_cannot_cover_a_file_that_names_a_lane_port(tree: Path, monkeypatch) -> None:
    _write(tree, "scripts/judge.py", UNLEASED_EVAL)
    monkeypatch.setattr(gate, "GATEWAY_CALLERS", {"scripts/judge.py": "says it is the gateway"})
    out = gate.run(tree)
    assert [f["problem"] for f in out["findings"]] == ["names_lane_port"]
    assert out["stale_gateway_callers"] == {}   # reported once, as the finding


def test_unclassified_model_caller_fails_even_without_a_port(tree: Path) -> None:
    _write(tree, "scripts/analysis/judge.py", GATEWAY_EVAL.replace('"model": "agent", ', ""))
    out = gate.run(tree)
    assert [f["problem"] for f in out["findings"]] == ["unclassified"]


@pytest.mark.parametrize("line", [
    'URL = base + "/v1/completions"', 'URL = base + "/v1/messages"', "URL = base + '/completion'",
    'URL = f"{base}/completion"', 'URL = f"{b}/chat/completions"', 'URL = f"{base}/infill"',
    'c = OpenAI(base_url=u)', 'r = c.chat.completions.create(model="x", messages=[])',
])
def test_every_model_call_shape_counts(tree: Path, line: str) -> None:
    _write(tree, "bench/x.py", line + "\n")
    assert not gate.run(tree)["ok"]


@pytest.mark.parametrize("line", ['p = "/completions_report"', 'p = "/completion/stats"', 'name = "OpenAIish"'])
def test_near_misses_are_not_model_calls(tree: Path, line: str) -> None:
    _write(tree, "bench/x.py", line + "\n")
    assert gate.run(tree)["ok"]


def test_importing_only_an_exception_is_not_a_lease(tree: Path) -> None:
    _write(tree, "scripts/x.py", "from orion.evals.leased_llm import LaneCallError\n" + GATEWAY_EVAL)
    assert [f["problem"] for f in gate.run(tree)["findings"]] == ["unclassified"]


def test_a_commented_shell_cli_call_is_not_a_lease(tree: Path) -> None:
    _write(tree, "scripts/x.sh", "# python -m orion.evals.leased_llm chat --lane agent\n"
                                 'curl "$U/v1/chat/completions"\n')
    assert not gate.run(tree)["ok"]


def test_reviewed_direct_caller_passes_even_naming_a_port(tree: Path, monkeypatch) -> None:
    _write(tree, "scripts/bench_x.py", UNLEASED_EVAL)
    monkeypatch.setattr(gate.port_gate, "DIRECT_CALLERS", {"scripts/bench_x.py": "loopback on circe"})
    out = gate.run(tree)
    assert out["ok"] and out["direct"] == ["scripts/bench_x.py"]


def test_leased_caller_passes(tree: Path) -> None:
    _write(tree, "services/orion-x/evals/eval_thing.py", LEASED_EVAL + "# agent seat :8015\n")
    out = gate.run(tree)
    assert out["ok"] and out["leased"] == ["services/orion-x/evals/eval_thing.py"]


def test_a_mention_in_a_comment_is_not_a_lease(tree: Path) -> None:
    _write(tree, "scripts/x.py", "# see orion/evals/leased_llm.py\n" + GATEWAY_EVAL)
    assert not gate.run(tree)["ok"]


def test_shell_caller_through_the_cli_passes(tree: Path) -> None:
    _write(tree, "scripts/x.sh", 'echo "$BODY" | python -m orion.evals.leased_llm chat --lane agent '
                                 '--holder operator:x --path /v1/chat/completions\n')
    assert gate.run(tree)["ok"]


def test_gateway_caller_in_the_allow_list_passes(tree: Path, monkeypatch) -> None:
    _write(tree, "scripts/x.py", GATEWAY_EVAL)
    monkeypatch.setattr(gate, "GATEWAY_CALLERS", {"scripts/x.py": "gateway by route"})
    out = gate.run(tree)
    assert out["ok"] and out["gateway"] == ["scripts/x.py"]


@pytest.mark.parametrize("content", [None, "print('no model call')\n", LEASED_EVAL])
def test_stale_allow_entry_fails(tree: Path, monkeypatch, content) -> None:
    if content is not None:
        _write(tree, "scripts/x.py", content)
    monkeypatch.setattr(gate, "GATEWAY_CALLERS", {"scripts/x.py": "gateway by route"})
    out = gate.run(tree)
    assert not out["ok"] and list(out["stale_gateway_callers"]) == ["scripts/x.py"]


@pytest.mark.parametrize("rel", [
    "services/orion-x/tests/test_a.py",           # tests
    "services/orion-x/evals/test_a.py",           # test file inside evals
    "services/orion-x/app/main.py",               # service runtime: not operator tooling
    "services/orion-lane-door/evals/latency.py",  # the door's own zone
    "services/orion-llm-gateway/scripts/x.py",    # the gateway's own zone
])
def test_out_of_scope_paths_are_ignored(tree: Path, rel: str) -> None:
    _write(tree, rel, UNLEASED_EVAL)
    _write(tree, "services/orion-lane-door/app/door.py", "")
    _write(tree, "services/orion-llm-gateway/app/main.py", "")
    assert gate.run(tree)["ok"]


def test_main_exit_codes(tree: Path, capsys) -> None:
    assert gate.main(["--root", str(tree)]) == 0
    _write(tree, "scripts/x.py", UNLEASED_EVAL)
    assert gate.main(["--root", str(tree)]) == 1
    assert "scripts/x.py [names_lane_port]" in capsys.readouterr().out


def test_cli_passes_on_the_real_tree() -> None:
    script = str(_REPO / "scripts" / "check_leased_lane_callers.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
