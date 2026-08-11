from datetime import datetime, timezone

from orion.feedback.extractors import (
    classify_pressure_deltas,
    extract_field_pressure_snapshot,
    pressure_delta,
)
from orion.schemas.field_state import FieldStateV1

NOW = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
CHANNELS = ["execution_pressure", "reasoning_pressure", "resource_pressure", "reliability_pressure"]


def _field(node_vectors: dict[str, dict[str, float]]) -> FieldStateV1:
    return FieldStateV1(
        generated_at=NOW,
        tick_id="field.tick:test",
        node_vectors=node_vectors,
    )


def test_extract_snapshot() -> None:
    snap = extract_field_pressure_snapshot(
        _field({"node:test": {"execution_pressure": 1.0, "pressure": 0.2}}),
        CHANNELS,
    )
    assert snap["execution_pressure"] == 1.0
    assert snap.get("resource_pressure", 0.0) == 0.2
    assert snap.get("reliability_pressure", 0.0) == 0.0


def test_thermal_pressure_does_not_route_to_resource_pressure() -> None:
    """Regression guard for Patch A (2026-08-11): `thermal_pressure` was removed
    from `orion/field/pressure.py::CHANNEL_DIMENSION_MAP`.

    This assertion is the inverse of what `test_extract_snapshot` asserted
    before that patch. It is written as its own test rather than folded into
    the one above because the removal has a live consumer contract attached:
    `config/feedback/feedback_policy.v1.yaml` lists `resource_pressure` in both
    `pressure_channels` and `positive_delta_channels`, so if a later patch
    "restores" the routing, feedback starts crediting actions for a CPU
    cooling off. Measured cause: thermal won the max() merge on 91.76% of
    28,735 real ticks, collapsing this dimension from 1,895 distinct values to
    128.
    """
    snap = extract_field_pressure_snapshot(
        _field({"node:test": {"thermal_pressure": 0.9}}),
        CHANNELS,
    )
    assert snap.get("resource_pressure", 0.0) == 0.0

    # ...and the surviving input still routes, so this is a removal of one
    # channel, not of the dimension.
    #
    # Asserted through `capability_vectors`, NOT `node_vectors`: `pressure` is a
    # capability-only channel in production (`capability_channels` in
    # config/field/orion_field_topology.v1.yaml, `level: [capability]` in the
    # glossary), written solely by the field-digester's apply_diffusion. An
    # earlier draft of this test put it in node_vectors, which exercised the
    # wrong branch of collect_field_channel_pressures() and would have kept
    # passing with the capability merge entirely broken.
    both = extract_field_pressure_snapshot(
        FieldStateV1(
            generated_at=NOW,
            tick_id="field.tick:test",
            node_vectors={"node:test": {"thermal_pressure": 0.9}},
            capability_vectors={"capability:orchestration": {"pressure": 0.3}},
        ),
        CHANNELS,
    )
    assert both.get("resource_pressure", 0.0) == 0.3


def test_resource_pressure_absent_when_diffusion_produces_no_pressure() -> None:
    """Documents a fragility Patch A introduced, so it is discovered by a
    failing assertion rather than by a silent dead dimension in production.

    After removing `thermal_pressure`, `resource_pressure`'s only remaining
    input is capability `pressure`, which exists *only* if the field-digester's
    apply_diffusion ran and wrote it. Previously `thermal_pressure` -- a node
    channel written directly by apply_perturbations -- guaranteed the key was
    present independently of the diffusion layer.

    So an empty `state.edges`, a pre-topology-reconcile persisted FieldStateV1,
    or an apply_diffusion failure now drops the dimension from
    `field_pressures()` entirely. Downstream that is not a 0.0 reading: it is
    absent, which makes `dimension_score` and `dimension_confidence` both 0.0
    AND causes update_dimension_precision_baseline to skip the dimension, so
    `dimension_precision_ewma_n` freezes and does not recover on its own once
    diffusion comes back.

    This test asserts the absent-not-zero behavior, which is the correct and
    intended contract (`map_channels_to_dimensions` only emits a dimension a
    real channel mapped to this tick -- never fabricating 0.0). It is here to
    make the coupling explicit and reviewable, not to claim the fragility is
    resolved.
    """
    snap = extract_field_pressure_snapshot(
        _field({"node:test": {"thermal_pressure": 0.9, "execution_pressure": 0.4}}),
        CHANNELS,
    )
    # extract_field_pressure_snapshot backfills requested channels with 0.0...
    assert snap.get("resource_pressure") == 0.0
    # ...but field_pressures() itself omits the key entirely, which is what the
    # precision-baseline updater keys off.
    from orion.field.pressure import field_pressures

    raw = field_pressures(
        _field({"node:test": {"thermal_pressure": 0.9, "execution_pressure": 0.4}})
    )
    assert "resource_pressure" not in raw
    assert raw.get("execution_pressure") == 0.4


def test_extract_snapshot_none_field() -> None:
    snap = extract_field_pressure_snapshot(None, CHANNELS)
    assert snap == {ch: 0.0 for ch in CHANNELS}


def test_pressure_delta() -> None:
    before = {"execution_pressure": 1.0, "reliability_pressure": 0.2}
    after = {"execution_pressure": 0.5, "reliability_pressure": 0.6}
    delta = pressure_delta(before, after)
    assert abs(delta["execution_pressure"] - (-0.5)) < 1e-6
    assert abs(delta["reliability_pressure"] - 0.4) < 1e-6


def test_classify_positive_negative() -> None:
    delta = {"reliability_pressure": 0.2, "execution_pressure": -0.3, "resource_pressure": 0.1}
    pos, neg = classify_pressure_deltas(
        delta,
        {
            "reliability_pressure": "decrease",
            "execution_pressure": "decrease",
            "resource_pressure": "decrease",
        },
    )
    assert "execution_pressure" in pos[0]
    assert "reliability_pressure" in neg[0]
    assert "resource_pressure" in neg[1]
