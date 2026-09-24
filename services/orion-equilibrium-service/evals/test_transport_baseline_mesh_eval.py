"""pytest entry for run_transport_baseline_mesh_eval.py (synthetic 24h mesh replay)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_transport_baseline_mesh_eval import run_eval  # noqa: E402


def test_transport_baseline_mesh_eval_passes():
    report = run_eval()
    failed = [k for k, ok in report["checks"].items() if not ok]
    assert not failed, report
    # volume: episodes, not windows
    assert report["triggers_total"] <= 20
    assert report["legacy_pooled_p95_fires"] > 100 * report["triggers_total"]
