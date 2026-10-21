"""rest_range: a machine-readable calm band [lo, hi] on metric semantics.

Prose `rest` cannot be checked against a value; rest_range can. These tests
build each malformed shape and assert it is refused at every layer that
reads it (from_mapping, the node gate, the glossary loader), and that the
real glossary's ranges are valid and survive load_glossary as tuples.
"""
from __future__ import annotations

import math

import pytest

from orion.field.channel_glossary import load_glossary
from orion.metrics.definitions import fingerprint
from orion.metrics.lineage import MetricNode, resolve_field_channels
from orion.metrics.semantics import (
    MetricSemantics,
    check_node_semantics,
    rest_range_problem,
)


def _node(**kw) -> MetricNode:
    base = dict(
        urn="metric://inner_state/svc/sig#x",
        surface="inner_state",
        producer_service="svc",
        name="x",
        registry_source="orion/fake_registry.py",
        value_kind="level",
    )
    base.update(kw)
    return MetricNode(**base)


@pytest.mark.parametrize("good", [[0.0, 0.0], [0, 1], (0.01, 0.03), [-1.5, 2]])
def test_valid_range_accepted(good):
    assert rest_range_problem(good, "level") is None
    sem = MetricSemantics.from_mapping({"value_kind": "level", "rest_range": list(good)})
    assert sem.rest_range == (float(good[0]), float(good[1]))
    assert check_node_semantics([_node(rest_range=sem.rest_range)]) == []


def test_absent_range_is_fine():
    assert MetricSemantics.from_mapping({"value_kind": "level"}).rest_range is None


def test_lo_greater_than_hi_rejected():
    with pytest.raises(ValueError, match="lo > hi"):
        MetricSemantics.from_mapping({"rest_range": [0.3, 0.1]})
    assert check_node_semantics([_node(rest_range=(0.3, 0.1))])


@pytest.mark.parametrize(
    "bad", [["a", 1], [0, None], [True, 1], [math.nan, 1], [0, math.inf], [0, -math.inf]]
)
def test_non_numeric_or_non_finite_rejected(bad):
    with pytest.raises(ValueError, match="rest_range bounds"):
        MetricSemantics.from_mapping({"rest_range": bad})
    assert check_node_semantics([_node(rest_range=tuple(bad))])


@pytest.mark.parametrize("bad", [[], [0.1], [0, 0.1, 0.2], 0.1, "0-0.1", {"lo": 0, "hi": 1}])
def test_wrong_shape_rejected(bad):
    with pytest.raises(ValueError, match="2-element"):
        MetricSemantics.from_mapping({"rest_range": bad})


def test_rest_range_on_placeholder_rejected():
    with pytest.raises(ValueError, match="placeholder"):
        MetricSemantics.from_mapping({"value_kind": "placeholder", "rest_range": [0.65, 0.65]})
    failures = check_node_semantics([_node(value_kind="placeholder", rest_range=(0.0, 0.0))])
    assert failures and "placeholder" in failures[0]


def test_rest_range_on_bucket_allowed():
    assert MetricSemantics.from_mapping({"value_kind": "bucket", "rest_range": [0.3, 0.3]})


def test_rest_range_not_required_for_prompt():
    from orion.metrics.semantics import REQUIRED_FOR_PROMPT

    assert "rest_range" not in REQUIRED_FOR_PROMPT


def test_glossary_with_bad_rest_range_is_rejected(tmp_path):
    bad = tmp_path / "glossary.yaml"
    bad.write_text(
        "channels:\n"
        "  - channel: cpu_pressure\n"
        "    level: [node]\n"
        "    category: physical_substrate\n"
        "    meaning: x\n"
        "    semantics:\n"
        "      value_kind: level\n"
        "      rest_range: [0.5, 0.1]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lo > hi"):
        resolve_field_channels(bad)


def _real_ranges() -> dict[tuple[str | None, str], tuple]:
    out = {}
    for entry in load_glossary()["entries"]:
        sem = dict(entry.semantics)
        if "rest_range" in sem:
            out[(entry.node, entry.channel)] = (sem.get("value_kind"), sem["rest_range"])
    return out


def test_real_glossary_rest_ranges_validate_and_round_trip_as_tuples():
    ranges = _real_ranges()
    # 11 of the 12 semantics entries; harness_closure is a placeholder.
    # +1 2026-10-15: node:substrate.perception_walkway (curiosity reads it).
    # +1 2026-10-20: node:substrate.strain (strain news, a trigger).
    assert len(ranges) == 12
    assert ranges[("node:substrate.perception_walkway", "prediction_error")][1] == (0.0, 0.0)
    for key, (kind, rng) in ranges.items():
        assert isinstance(rng, tuple), key
        assert rest_range_problem(rng, kind) is None, key
    assert ranges[("node:substrate.bus_synaptic", "prediction_error")][1] == (0.0, 0.03)
    # The documented failure states must sit OUTSIDE the calm band.
    lo, hi = ranges[("node:substrate.bus_synaptic", "prediction_error")][1]
    assert not lo <= 0.27 <= hi  # historic permanent ~0.27 floor
    lo, hi = ranges[(None, "vision_frame_staleness")][1]
    assert not lo <= 1.0 <= hi  # camera dead / router silent


def test_placeholder_in_real_glossary_has_no_rest_range():
    for entry in load_glossary()["entries"]:
        sem = dict(entry.semantics)
        if sem.get("value_kind") == "placeholder":
            assert "rest_range" not in sem, (entry.node, entry.channel)


def test_rest_range_projects_onto_lineage_and_lock_in_order():
    nodes = {n.name: n for n in resolve_field_channels()}
    node = nodes["node:substrate.bus_synaptic.prediction_error"]
    assert node.rest_range == (0.0, 0.03)
    assert fingerprint(node)["rest_range"] == [0.0, 0.03]
    # Ordered, never sorted: a malformed hi<lo must not lock as valid-looking.
    flipped = MetricNode(**{**node.__dict__, "rest_range": (0.03, 0.0)})
    assert fingerprint(flipped)["rest_range"] == [0.03, 0.0]
