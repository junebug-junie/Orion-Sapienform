"""publish_cluster() carries the per-node measurements breakdown (chassis_watts, etc.) through
to BiometricsClusterV1.measurements_by_node instead of losing it once summed into a fleet
total -- see orion/telemetry/biometrics_pipeline.py's aggregate_fleet_measurements for the
sum, and tests/test_pdu_proxy_polling.py for the proxy-fill merge this dict comes out of.

This covers app.main._measurements_by_node_for_cluster in isolation: the one-line
transformation actually applied at the BiometricsClusterV1(...) call site inside
publish_cluster(), which itself needs a live bus/hub/clock to exercise directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SVC = Path(__file__).resolve().parents[1]
_REPO = _SVC.parents[1]
sys.path.insert(0, str(_SVC))
os.environ.setdefault(
    "NODE_CATALOG_PATH", str(_REPO / "config" / "biometrics" / "node_catalog.yaml")
)

import app.main as main  # noqa: E402


def test_a_full_per_node_dict_passes_through_unchanged():
    per_node = {
        "athena": {"chassis_watts": 390.0},
        "circe": {"chassis_watts": 512.0, "pdu_watts": 512.0},
    }
    assert main._measurements_by_node_for_cluster(per_node) == per_node


def test_a_node_with_none_measurements_is_dropped_not_a_validation_crash():
    """A producer predating the `measurements` field maps to None here -- the schema field is
    Dict[str, float] per node (not Optional), so passing None through would raise instead of
    just omitting that one node, same treatment aggregate_fleet_measurements already gives it."""
    per_node = {"athena": {"chassis_watts": 390.0}, "legacy_node": None}
    result = main._measurements_by_node_for_cluster(per_node)
    assert result == {"athena": {"chassis_watts": 390.0}}
    assert "legacy_node" not in result


def test_all_none_collapses_to_none_not_an_empty_dict():
    assert main._measurements_by_node_for_cluster({"a": None, "b": None}) is None


def test_empty_input_is_none():
    assert main._measurements_by_node_for_cluster({}) is None
