"""ATTENTION_WORLD_FIRST_ENABLED=false restores the previous ranking EXACTLY.

Golden outputs were produced from origin/main (dc8eab06c, before world-first)
by tests/fixtures/world_first_parity/scenarios.py. With world-first off, both
contests must reproduce them byte for byte -- including the "always a winner
at 1.0" frame (`all_calm`) that world-first exists to end.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "fixtures" / "world_first_parity"


def _scenarios():
    spec = importlib.util.spec_from_file_location("wf_parity_scenarios", HERE / "scenarios.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _golden() -> dict:
    return json.loads((HERE / "golden.json").read_text())


def test_field_contest_flag_off_matches_pre_world_first_golden() -> None:
    got = json.loads(json.dumps(_scenarios().run_field(), sort_keys=True))
    assert got == _golden()["field"]
    # The old behaviour really is "always a winner at 1.0", even when calm.
    assert got["all_calm"]["dominant_targets"][0]["salience_score"] == 1.0


def test_broadcast_flag_off_matches_pre_world_first_golden(monkeypatch) -> None:
    monkeypatch.setenv("ORION_ATTENTION_TOPDOWN_ENABLED", "false")
    got = json.loads(json.dumps(_scenarios().run_broadcast(), sort_keys=True))
    golden = _golden()["broadcast"]
    for frame in (got["mixed"], golden["mixed"]):
        frame.pop("generated_at", None)
    assert got == golden
    assert "world_first" not in got["mixed"]["debug"]
