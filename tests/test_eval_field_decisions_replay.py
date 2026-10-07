"""Gate the replay eval's compare step: it must fail on a natural-data move or
unpaired rows, and pass the intended rename/retirement."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "eval_field_decisions_replay.py"


def _row(rn: int, *, merged: dict, dims: dict, caps: dict) -> dict:
    return {
        "rn": rn, "scenario": "natural", "dims": dims, "winners": {"resource_pressure": ["pressure", "node:x"]},
        "merged": merged, "cap_vectors": {"capability:transport": caps}, "targets": {}, "top_capability": None,
        "dominant": [], "transport_reliability_measured": True, "backed": {"resource_pressure": True},
        "guard": {"resource_pressure": False},
    }


def _run(tmp_path: Path, before: list[dict], after: list[dict]) -> subprocess.CompletedProcess:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in before))
    b.write_text("".join(json.dumps(r) + "\n" for r in after))
    return subprocess.run([sys.executable, str(SCRIPT), "compare", str(a), str(b)], capture_output=True, text=True)


def test_rename_and_retirement_alone_pass(tmp_path: Path) -> None:
    main = _row(60, merged={"catalog_drift_pressure": 0.02, "contract_pressure": 0.017, "observer_failure_pressure": 0.0},
                dims={"resource_pressure": 0.3}, caps={"contract_pressure": 0.017})
    branch = _row(60, merged={"catalog_drift_pressure": 0.02}, dims={"resource_pressure": 0.3},
                  caps={"catalog_drift_pressure": 0.017})
    r = _run(tmp_path, [main], [branch])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "GATE PASS" in r.stdout


def test_a_moved_dimension_or_unpaired_row_fails(tmp_path: Path) -> None:
    main = _row(60, merged={}, dims={"resource_pressure": 0.3}, caps={})
    moved = _row(60, merged={}, dims={"resource_pressure": 0.4}, caps={})
    assert _run(tmp_path, [main], [moved]).returncode == 1
    assert _run(tmp_path, [main], [main, _row(120, merged={}, dims={}, caps={})]).returncode == 1
