"""Metric-gate evidence for rpc_timeout_pressure, pinned on live 2026-09-25 data
(see run_rpc_delivery_eval.py for how the fixtures were captured).

Asserts the claims the PR makes, not just that the replay runs:
- it rests at a real 0.0 most of the time on a healthy mesh;
- it moves off 0.0 on real timeouts, and names the hop;
- without the probe label it could not rest (the structural floor this patch
  removes);
- the denominator floor is what keeps single timeouts small;
- the raw wire shape folds.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_rpc_delivery_eval as ev  # noqa: E402

RESULT = ev.run()


def test_shipped_config_rests_at_a_measured_zero():
    s = RESULT["shipped"]
    assert s["unmeasured_ticks"] == 0
    assert s["zero_fraction"] >= 0.6
    assert s["p50"] == 0.0


def test_shipped_config_moves_on_real_timeouts_and_names_the_hop():
    s = RESULT["shipped"]
    assert 0.0 < s["max"] <= 0.1
    assert set(s["nonzero_worst_hops"]) <= {
        "orion:exec:request:LLMGatewayService",
        "orion:cortex:request",
    }


def test_unlabelled_probe_would_pin_a_floor_under_calm():
    """Before the cortex-exec label, the probe's by-design 3 s deadline kept the
    reading off zero almost all the time: not a metric that can read calm."""
    p = RESULT["probe_unlabelled"]
    assert p["zero_fraction"] < 0.1
    assert p["p50"] > 0.05
    assert p["nonzero_worst_hops"].get("orion:exec:request:LLMGatewayService", 0) > 200


def test_denominator_floor_caps_isolated_timeouts():
    f = RESULT["floor_sensitivity"]
    assert f[10] <= 0.1
    assert f[1] > f[10] > f[20]


def test_raw_wire_payloads_fold_and_read():
    w = RESULT["wire"]
    assert w["unmeasured_ticks"] == 0
    assert w["producers_max"] == 14
    assert 0.0 < w["max"] <= 1.0
