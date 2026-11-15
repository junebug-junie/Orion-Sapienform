"""#3394: idle curiosity ticks now carry a gate, and it is not a System One decision."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "eval_system_one_appraisal",
    Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "eval_system_one_appraisal.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_idle_ticks_are_counted_apart_from_gate_decisions() -> None:
    rows = [
        {"gate_json": {"gate_result": "idle_no_seeds"}},
        {"gate_json": '{"gate_result": "idle_no_seeds"}'},
        {"gate_json": {"gate_result": "system_one_curiosity_admit", "selected_level": 1}},
    ]
    report = _MOD.evaluate_curiosity_gates(rows)
    assert report["seeded_ticks_with_gate"] == 1
    assert report["idle_ticks_no_seeds"] == 2
    assert "idle_no_seeds" not in report["gate_results"]
