"""Deterministic unit tests for measure_society_of_mind_magnitude_probe.py.

No DB, no network. Same sibling-module-by-file-path loading pattern as
test_measure_precision_weighted_salience_probe.py / test_measure_emergent_clustering_probe.py.

Lives in top-level `tests/`, not `scripts/analysis/tests/` -- `scripts` is in
`pyproject.toml`'s `norecursedirs`, so a bare `pytest` run from repo root never
discovers anything placed there (same defect found and fixed for Candidate A's replay
test in this same review cycle).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "measure_society_of_mind_magnitude_probe.py"
)
_spec = importlib.util.spec_from_file_location(
    "measure_society_of_mind_magnitude_probe", _MODULE_PATH
)
mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
sys.modules["measure_society_of_mind_magnitude_probe"] = mod
_spec.loader.exec_module(mod)


# ===========================================================================
# parse_prediction_error -- pure scalar parsing, must never raise.
# ===========================================================================


def test_parse_prediction_error_valid_string() -> None:
    assert mod.parse_prediction_error("0.0429") == 0.0429


def test_parse_prediction_error_none_returns_none() -> None:
    assert mod.parse_prediction_error(None) is None


def test_parse_prediction_error_malformed_returns_none() -> None:
    assert mod.parse_prediction_error("not-a-number") is None
    assert mod.parse_prediction_error("") is None


def test_parse_prediction_error_non_finite_returns_none() -> None:
    assert mod.parse_prediction_error("nan") is None
    assert mod.parse_prediction_error("inf") is None


# ===========================================================================
# REDUCER_TO_TARGET_ID -- must match the real node:substrate.* convention
# _write_prediction_error_node() uses, one entry per real orion/substrate/
# prediction_error.py instrument.
# ===========================================================================


def test_reducer_to_target_id_covers_all_five_real_reducers() -> None:
    assert set(mod.REDUCER_TO_TARGET_ID) == {
        "substrate.node_biometrics",
        "substrate.execution_trajectory",
        "substrate.transport_bus",
        "substrate.chat_session",
        "substrate.route_arbitration",
    }


def test_reducer_to_target_id_values_match_node_substrate_convention() -> None:
    assert mod.REDUCER_TO_TARGET_ID["substrate.node_biometrics"] == "node:substrate.biometrics"
    assert mod.REDUCER_TO_TARGET_ID["substrate.execution_trajectory"] == "node:substrate.execution"
    assert mod.REDUCER_TO_TARGET_ID["substrate.route_arbitration"] == "node:substrate.route"


# ===========================================================================
# magnitude_scorer integration -- real import, exercised via the probe's own
# per-tick call shape (single-entry dict per real historical value).
# ===========================================================================


def test_magnitude_scorer_real_import_clamps_and_drops_like_the_module_promises() -> None:
    from orion.attention.field_attention.candidate_society_of_mind import magnitude_scorer

    assert magnitude_scorer({"node:substrate.biometrics": 1.5}) == {
        "node:substrate.biometrics": 1.0
    }
    assert magnitude_scorer({"node:substrate.biometrics": float("nan")}) == {}
