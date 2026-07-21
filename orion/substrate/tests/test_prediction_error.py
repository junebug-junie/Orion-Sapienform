"""Unit tests for the field-native prediction-error instruments in
``orion/substrate/prediction_error.py`` -- 0-1 surprise scores diffing successive
reducer-projection snapshots (not ``SelfStateV1``, not ``tensions.py``'s bucket
vocabulary; see the Sentience Striving Program charter §9b item 3)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from orion.schemas.biometrics_projection import (
    NodeBiometricsProjectionV1,
    NodeBiometricsStateV1,
)
from orion.substrate.prediction_error import biometrics_prediction_error

_NOW = datetime(2026, 7, 21, 0, 0, 0, tzinfo=timezone.utc)


def _node(node_id: str, pressure_hints: dict) -> NodeBiometricsStateV1:
    return NodeBiometricsStateV1(node_id=node_id, pressure_hints=pressure_hints)


def _projection(nodes: dict[str, NodeBiometricsStateV1]) -> NodeBiometricsProjectionV1:
    return NodeBiometricsProjectionV1(
        projection_id="node_biometrics_projection",
        generated_at=_NOW,
        nodes=nodes,
    )


def test_biometrics_prediction_error_zero_when_no_change() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.8, "strain": 0.1})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 0.8, "strain": 0.1})})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_biometrics_prediction_error_scales_with_delta_magnitude() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.5})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 0.8})})
    # |0.8 - 0.5| = 0.3 == _THRESHOLD -> saturates at 1.0
    assert biometrics_prediction_error(prev, curr) == pytest.approx(1.0)


def test_biometrics_prediction_error_partial_delta_below_threshold() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.5})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 0.65})})
    # |0.65 - 0.5| = 0.15 -> 0.15 / 0.30 = 0.5
    assert biometrics_prediction_error(prev, curr) == pytest.approx(0.5)


def test_biometrics_prediction_error_zero_when_node_not_in_prev() -> None:
    """A brand-new node in curr with no prev counterpart contributes no delta --
    mirrors execution_prediction_error's ``prev_run is None: continue`` skip."""
    prev = _projection({})
    curr = _projection({"circe": _node("circe", {"gpu": 0.9})})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_biometrics_prediction_error_zero_when_projections_empty() -> None:
    prev = _projection({})
    curr = _projection({})
    assert biometrics_prediction_error(prev, curr) == 0.0


def test_biometrics_prediction_error_handles_disjoint_pressure_hint_keys() -> None:
    """Real biometrics nodes carry different pressure_hints key sets depending on
    node role (GPU nodes: gpu/strain; orchestration nodes: disk/memory/thermal
    pressure) -- confirmed live 2026-07-21 against substrate_node_biometrics_
    projection. A key appearing only on one side of the diff (e.g. a pressure
    signal that newly starts or stops firing) must still be diffed against an
    implicit 0.0, not silently dropped."""
    prev = _projection({"athena": _node("athena", {"disk_pressure": 0.1})})
    curr = _projection(
        {"athena": _node("athena", {"disk_pressure": 0.1, "thermal_pressure": 0.3})}
    )
    # thermal_pressure delta: |0.3 - 0.0| = 0.3; disk_pressure delta: 0.0
    # mean(0.3, 0.0) = 0.15 -> 0.15 / 0.30 = 0.5
    assert biometrics_prediction_error(prev, curr) == pytest.approx(0.5)


def test_biometrics_prediction_error_averages_across_multiple_nodes() -> None:
    prev = _projection(
        {
            "atlas": _node("atlas", {"gpu": 0.5}),
            "athena": _node("athena", {"memory_pressure": 0.2}),
        }
    )
    curr = _projection(
        {
            "atlas": _node("atlas", {"gpu": 0.8}),  # delta 0.3
            "athena": _node("athena", {"memory_pressure": 0.2}),  # delta 0.0
        }
    )
    # mean(0.3, 0.0) = 0.15 -> 0.15 / 0.30 = 0.5
    assert biometrics_prediction_error(prev, curr) == pytest.approx(0.5)


def test_biometrics_prediction_error_clamps_to_one() -> None:
    prev = _projection({"atlas": _node("atlas", {"gpu": 0.0})})
    curr = _projection({"atlas": _node("atlas", {"gpu": 1.0})})
    assert biometrics_prediction_error(prev, curr) == pytest.approx(1.0)


def test_biometrics_prediction_error_fail_open_on_non_numeric_value() -> None:
    """pressure_hints is dict[str, Any] (unlike execution's dict[str, float]) --
    a malformed/non-numeric value for one key must not raise, just be skipped,
    while a real numeric key on the same node still contributes its delta."""
    prev = _projection(
        {"athena": _node("athena", {"thermal_pressure": "not-a-number", "gpu": 0.5})}
    )
    curr = _projection(
        {"athena": _node("athena", {"thermal_pressure": "still-not-a-number", "gpu": 0.8})}
    )
    # thermal_pressure delta skipped (non-numeric); gpu delta: |0.8-0.5| = 0.3 -> 1.0
    assert biometrics_prediction_error(prev, curr) == pytest.approx(1.0)
