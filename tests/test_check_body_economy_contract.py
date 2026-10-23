"""Body-economy "no unregistered veto" gate (scripts/check_body_economy_contract.py).

Seeded fixture repos (tmp_path, not git) with hand-written modules; the expected verdicts are
written from the rule, not read back off the checker. Plus one test that the real repo passes.
"""
from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_body_economy_contract",
                                               REPO / "scripts" / "check_body_economy_contract.py")
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate  # dataclasses and the process pool resolve it by name
_spec.loader.exec_module(gate)

CONTRACT = "config/physiology/contract.yaml"


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text), encoding="utf-8")


def _contract(root: Path, readers=(), sources=(), extra: str = "") -> Path:
    lines = ["version: 1", "protected_paths:", "  - {glob: 'orion/physiology/**', organ: all}", "readers:"]
    lines += [f"  - {{module: {m}, role: {role}, reason: test}}" for m, role in readers] or []
    if not readers:
        lines[-1] = "readers: []"
    lines.append("sources:" if sources else "sources: []")
    lines += [f"  - {{module: {m}, reason: test}}" for m in sources]
    _write(root, CONTRACT, "\n".join(lines) + "\n" + extra)
    return root / CONTRACT


STRAINED_VETO = """
    def maybe_run(regulation):
        if regulation.arousal == "strained":
            return None
        return "ran"
"""


def test_new_strained_veto_module_fails_and_names_module_and_fix(tmp_path):
    """G1: a new `if regulation.arousal == "strained": return` module is refused."""
    _write(tmp_path, "orion/new_feature/loop.py", STRAINED_VETO)
    errors, _ = gate.check(tmp_path, _contract(tmp_path))
    assert len(errors) == 1
    msg = errors[0]
    assert "UNREGISTERED READER orion/new_feature/loop.py" in msg
    assert "compares against 'strained'" in msg
    assert "add it to `readers`" in msg and CONTRACT in msg


def test_listed_module_passes(tmp_path):
    _write(tmp_path, "orion/new_feature/loop.py", STRAINED_VETO)
    errors, hits = gate.check(tmp_path, _contract(tmp_path, readers=[("orion/new_feature/loop.py", "bidder")]))
    assert errors == []
    assert list(hits) == ["orion/new_feature/loop.py"]


def test_stale_entry_fails_when_module_no_longer_reads(tmp_path):
    _write(tmp_path, "orion/new_feature/loop.py", "def maybe_run():\n    return 'ran'\n")
    errors, _ = gate.check(tmp_path, _contract(tmp_path, readers=[("orion/new_feature/loop.py", "bidder")]))
    assert len(errors) == 1 and errors[0].startswith("STALE ENTRY orion/new_feature/loop.py")
    assert "no longer reads" in errors[0]


def test_stale_entry_fails_when_file_is_gone(tmp_path):
    errors, _ = gate.check(tmp_path, _contract(tmp_path, sources=["orion/gone.py"]))
    assert len(errors) == 1 and "does not exist" in errors[0]


@pytest.mark.parametrize("src, evidence", [
    ("from orion.regulation.arousal_read import read_arousal_sync\n", "from orion.regulation.arousal_read"),
    ("from orion.regulation import arousal_view\n", "import arousal_view"),
    ("import orion.autonomy.thermal_gate as tg\n", "import orion.autonomy.thermal_gate"),
    ("from orion.physiology.prices import posted\n", "from orion.physiology.prices"),
    ("def f(s):\n    return s.RegulationStateV1\n", "attribute .RegulationStateV1"),
    ("KEY = 'orion:regulation:latest'\n", "literal 'orion:regulation:latest'"),
    ("REASON = 'arousal_strained'\n", "literal 'arousal_strained'"),
    ("def f(p):\n    return p.get('arousal_level')\n", "literal 'arousal_level'"),
    ("def f(level):\n    return level in ('engaged', 'strained')\n", "compares against 'strained'"),
    ("def f(level):\n    return 'strained' != level\n", "compares against 'strained'"),
])
def test_each_structural_rule_detects(src, evidence):
    ev = gate.scan_source(src)
    assert any(evidence in e for e in ev), ev


@pytest.mark.parametrize("src", [
    "# reads orion:regulation:latest and checks strained\nx = 1\n",           # comments only
    '"""Docstring: when arousal is strained we hold."""\n',                    # prose
    "def f(posture):\n    return 'strained' in posture\n",                     # substring test
    "from orion.regulation.juniper_turns import JUNIPER_TURN_PREDICATE\n",     # unwatched module
    "def f(x):\n    return x == 'engaged'\n",                                  # not the veto level
])
def test_non_reads_are_not_detected(src):
    assert gate.scan_source(src) == []


def test_tests_and_evals_are_not_scanned(tmp_path):
    _write(tmp_path, "services/svc/tests/test_loop.py", STRAINED_VETO)
    _write(tmp_path, "services/svc/evals/run_eval.py", STRAINED_VETO)
    _write(tmp_path, "tests/test_top.py", STRAINED_VETO)
    errors, hits = gate.check(tmp_path, _contract(tmp_path))
    assert errors == [] and hits == {}


@pytest.mark.parametrize("body, needle", [
    ("readers: [\n", "malformed YAML"),
    ("protected_paths: []\nsources: []\n", "`readers` must be a list"),
    ("protected_paths: []\nreaders:\n  - {module: a.py, role: veto, reason: x}\nsources: []\n", "role 'veto'"),
    ("protected_paths: []\nreaders:\n  - {module: a.py, role: display, reason: ''}\nsources: []\n", "non-empty `reason`"),
    ("protected_paths: []\nreaders:\n  - {module: a.py, role: display, reason: x}\n"
     "  - {module: a.py, role: display, reason: x}\nsources: []\n", "lists a.py twice"),
    ("protected_paths:\n  - {glob: x}\nreaders: []\nsources: []\n", "protected_paths[0]"),
])
def test_malformed_contract_fails(tmp_path, body, needle):
    _write(tmp_path, CONTRACT, body)
    errors, _ = gate.check(tmp_path, tmp_path / CONTRACT)
    assert errors and any(needle in e for e in errors), errors


def test_main_exit_codes(tmp_path, capsys):
    _write(tmp_path, "orion/new_feature/loop.py", STRAINED_VETO)
    _contract(tmp_path)
    assert gate.main(["--root", str(tmp_path)]) == 1
    assert "UNREGISTERED READER" in capsys.readouterr().err
    _contract(tmp_path, readers=[("orion/new_feature/loop.py", "bidder")])
    assert gate.main(["--root", str(tmp_path)]) == 0


def test_real_repo_contract_is_green():
    errors, hits = gate.check(REPO, REPO / CONTRACT)
    assert errors == [], "\n".join(errors)
    assert len(hits) >= 40  # the scan is not silently empty
