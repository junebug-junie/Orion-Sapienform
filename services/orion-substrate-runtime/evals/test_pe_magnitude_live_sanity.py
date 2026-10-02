"""The live sanity eval's pure evaluator flags degenerate domains on synthetic data."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_pe_magnitude_live_sanity import evaluate  # noqa: E402

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def _rows(node_id, values, step=timedelta(minutes=1)):
    n = len(values)
    return [(node_id, NOW - step * (n - 1 - i), v) for i, v in enumerate(values)]


def test_healthy_varied_node_passes():
    # Alternating calm/active stretches so trend labels are mixed.
    vals = []
    for block in range(48):
        vals += [0.0 if block % 2 else 0.3 + 0.001 * block] * 60
    result = evaluate(_rows("node:substrate.chat", vals), now=NOW,
                      expected_nodes=("node:substrate.chat",))
    entry = result["nodes"]["node:substrate.chat"]
    assert entry["checks"]["coverage"] and entry["checks"]["not_flat"]
    assert entry["checks"]["trend_not_degenerate"], entry["trend"]
    assert result["ok"]


def test_flat_node_and_missing_domain_fail():
    rows = _rows("node:substrate.route", [0.0] * 3000)
    result = evaluate(rows, now=NOW,
                      expected_nodes=("node:substrate.route", "node:substrate.perception"))
    assert not result["ok"]
    route = result["nodes"]["node:substrate.route"]
    assert route["checks"]["not_flat"] is False
    assert route["checks"]["trend_not_degenerate"] is False  # 100% "flat"
    assert route["share_at_min"] == 1.0
    assert result["nodes"]["node:substrate.perception"]["checks"]["present"] is False


def test_sparse_node_fails_coverage():
    result = evaluate(_rows("node:substrate.cabinet", [0.1, 0.2] * 50), now=NOW,
                      expected_nodes=("node:substrate.cabinet",))
    assert result["nodes"]["node:substrate.cabinet"]["checks"]["coverage"] is False
