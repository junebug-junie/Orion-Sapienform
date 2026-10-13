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


def test_sparse_hop_replay_states_one_regime_shift_instead_of_absorbing():
    """2026-10-13: real stored hourly rows for the two hops whose floor was
    absorbed live (see run_transport_baseline_sparse_hop_replay.py)."""
    from run_transport_baseline_sparse_hop_replay import run_eval as run_sparse

    reports = run_sparse()
    failed = [(r["hop"], r["spacing"], k) for r in reports for k, ok in r["checks"].items() if not ok]
    assert not failed, reports


def test_gpu_shed_replay_degraded_refusals_never_drop_the_floor():
    """2026-10-13: the 2026-10-08 GPU shed replayed from stored counts (see
    run_transport_baseline_degraded_shed_replay.py). The control must reproduce the
    live collapse, or the fixed run's pass would prove nothing."""
    from run_transport_baseline_degraded_shed_replay import REAL_AFTER_MS, replay

    control = replay(refused_as_success=True)
    fixed = replay(refused_as_success=False)
    assert control["floor_at_2026-10-08T19:00_ms"] < 2500, control
    assert fixed["floor_min_ms"] >= 0.8 * REAL_AFTER_MS, fixed
    assert not fixed["events"], fixed
    assert fixed["degraded_counted"] == fixed["refusals_in_fixture"] > 0, fixed
    assert 0.8 <= fixed["ratio_median_after_shed"] <= 1.3, fixed
