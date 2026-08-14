"""Tests for the definition-change alert (R4).

The load-bearing test is test_committed_lock_matches_current_registries: it is
the gate itself, so `pytest tests/test_metric_definition_drift.py` fails for an
un-re-locked definition edit even when nobody runs the CI workflow.

Every fixture below is hand-constructed with the expected classification worked
out by hand, and the suite was mutation-checked -- see
test_ambiguous_rename_group_is_not_paired's note for the one that mattered.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metrics.definitions import (  # noqa: E402
    ANNOTATION_FIELDS,
    DEFINITION_FIELDS,
    ROUTING_FIELDS,
    SEMANTIC_FIELDS,
    SEVERITY,
    DefinitionChange,
    build_lock,
    diff_locks,
    fingerprint,
    format_report,
)
from orion.metrics.lineage import MetricGraph, MetricNode  # noqa: E402

LOCK_PATH = REPO_ROOT / "config" / "metrics" / "metric_definitions.lock.json"


def _node(name: str, **kw) -> MetricNode:
    kw.setdefault("surface", "field_channel")
    kw.setdefault("producer_service", "orion-field-digester")
    kw.setdefault("registry_source", "config/field/field_channel_glossary.v1.yaml")
    return MetricNode(
        urn=f"metric://{kw['surface']}/{kw['producer_service']}/{name}",
        name=name,
        **kw,
    )


# ---------------------------------------------------------------- fingerprint


def test_fingerprint_keeps_only_definition_fields():
    node = _node("cpu_pressure", meaning="How loaded a node's CPU is.")
    fp = fingerprint(node)
    assert set(fp) <= set(DEFINITION_FIELDS)
    # urn/surface/producer/name are identity, never fingerprint content --
    # a change to any of them is a removal plus an addition by construction.
    assert "urn" not in fp and "name" not in fp and "surface" not in fp


def test_fingerprint_omits_empties_and_sorts_tuples():
    node = _node(
        "x",
        declared_consumers=("z-service", "a-service"),
        meaning=None,
        upstream=(),
    )
    fp = fingerprint(node)
    assert fp["declared_consumers"] == ["a-service", "z-service"]
    assert "meaning" not in fp  # None omitted
    assert "upstream" not in fp  # empty tuple omitted


def test_tuple_order_is_not_a_change():
    """A reordered YAML list must not read as a definition change -- this is
    what keeps a 99KB channels.yaml from generating cosmetic alerts."""
    a = fingerprint(_node("x", declared_consumers=("a", "b")))
    b = fingerprint(_node("x", declared_consumers=("b", "a")))
    assert a == b


def test_build_lock_is_sorted_by_urn():
    graph = MetricGraph()
    for name in ("zeta", "alpha", "mid"):
        node = _node(name)
        graph.nodes[node.urn] = node
    assert list(build_lock(graph)) == sorted(build_lock(graph))


# ---------------------------------------------------------------------- diff


def test_identical_locks_produce_no_changes():
    lock = build_lock_of(_node("a", meaning="m"))
    assert not diff_locks(lock, lock)


def build_lock_of(*nodes: MetricNode) -> dict:
    graph = MetricGraph()
    for node in nodes:
        graph.nodes[node.urn] = node
    return build_lock(graph)


def test_removal_is_high_and_addition_is_medium():
    before = build_lock_of(_node("gone", meaning="a"))
    after = build_lock_of(_node("fresh", meaning="b"))
    changes = diff_locks(before, after).changes
    kinds = {c.kind: c for c in changes}
    assert set(kinds) == {"removed", "added"}
    assert kinds["removed"].severity == "high"
    assert kinds["added"].severity == "medium"
    assert kinds["removed"].urn.endswith("/gone")
    assert kinds["added"].urn.endswith("/fresh")


def test_exact_rename_is_paired_into_one_change():
    before = build_lock_of(_node("execution_load", meaning="step load"))
    after = build_lock_of(_node("cortex_exec_step_load", meaning="step load"))
    diff = diff_locks(before, after)
    assert len(diff.changes) == 1
    change = diff.changes[0]
    assert change.kind == "renamed"
    assert change.severity == "high"
    assert change.previous_urn.endswith("/execution_load")
    assert change.urn.endswith("/cortex_exec_step_load")


def test_rename_that_also_edits_prose_degrades_to_remove_plus_add():
    """Documented behaviour, asserted so it cannot silently become worse.

    A real rename usually rewrites the meaning too, so it will NOT pair. That
    is safe by design: both halves are still reported, and `removed` is the
    highest-signal kind there is. The pairing pass only ever makes output
    quieter, so it is deliberately conservative.
    """
    before = build_lock_of(_node("execution_load", meaning="old prose"))
    after = build_lock_of(_node("cortex_exec_step_load", meaning="new prose"))
    kinds = sorted(c.kind for c in diff_locks(before, after).changes)
    assert kinds == ["added", "removed"]


def test_ambiguous_rename_group_is_not_paired():
    """Two removals and two additions sharing one definition stay unpaired.

    Mutation-checked: dropping the `len(...) == 1` guard in _pair_renames makes
    this test fail with 2 changes instead of 4, which is the point -- an
    arbitrary pairing would name a rename that never happened.
    """
    before = build_lock_of(_node("old_a", meaning="same"), _node("old_b", meaning="same"))
    after = build_lock_of(_node("new_a", meaning="same"), _node("new_b", meaning="same"))
    changes = diff_locks(before, after).changes
    assert len(changes) == 4
    assert sorted(c.kind for c in changes) == ["added", "added", "removed", "removed"]


def test_semantics_and_routing_changes_are_reported_separately():
    """One commit touching both must not hide the routing edit behind the prose
    edit -- they are different failure classes with different blast radius."""
    before = build_lock_of(
        _node("x", meaning="old", declared_consumers=("a",))
    )
    after = build_lock_of(
        _node("x", meaning="new", declared_consumers=("a", "b"))
    )
    changes = diff_locks(before, after).changes
    assert sorted(c.kind for c in changes) == ["routing_changed", "semantics_changed"]
    sem = next(c for c in changes if c.kind == "semantics_changed")
    assert sem.fields == {"meaning": ("old", "new")}
    routing = next(c for c in changes if c.kind == "routing_changed")
    assert routing.fields == {"declared_consumers": (["a"], ["a", "b"])}


def test_routing_change_is_high_severity():
    """The `execution_load` class of defect is a routing edit: a consumer
    quietly stops being listed and nothing announces it. Asserted explicitly
    because a downgrade to medium would be invisible in every other test."""
    before = build_lock_of(_node("x", declared_consumers=("a", "b")))
    after = build_lock_of(_node("x", declared_consumers=("a",)))
    changes = diff_locks(before, after).changes
    assert [c.kind for c in changes] == ["routing_changed"]
    assert changes[0].severity == "high"


def test_surface_is_parsed_from_the_urn():
    before = build_lock_of(_node("x", meaning="m"))
    after: dict = {}
    change = diff_locks(before, after).changes[0]
    assert change.surface == "field_channel"


def test_identical_definitions_on_different_surfaces_do_not_pair():
    """Surface is part of the pairing key, so a field channel disappearing and
    an organ signal appearing is never one rename."""
    before = build_lock_of(_node("thing", surface="field_channel"))
    after = build_lock_of(_node("thing2", surface="organ_signal"))
    kinds = sorted(c.kind for c in diff_locks(before, after).changes)
    assert kinds == ["added", "removed"]


def test_lock_field_retired_from_the_format_degrades_to_annotation():
    """A committed lock predating a change to DEFINITION_FIELDS still diffs.

    This is what the `_FIELD_CLASS.get(..., "annotation")` default is for: an
    old lock can name a field the current format no longer emits, and the gate
    must report it, not crash. Bucketing it as annotation is deliberate -- a
    field the format has retired cannot be a live routing or semantics edit.
    """
    before = {"metric://field_channel/p/x": {"a_retired_field": "old"}}
    after = {"metric://field_channel/p/x": {"meaning": "new"}}
    kinds = sorted(c.kind for c in diff_locks(before, after).changes)
    assert kinds == ["annotation_changed", "semantics_changed"]


def test_registry_source_move_is_annotation_not_semantics():
    before = build_lock_of(_node("x", registry_source="orion/a.py"))
    after = build_lock_of(_node("x", registry_source="orion/b.py"))
    changes = diff_locks(before, after).changes
    assert [c.kind for c in changes] == ["annotation_changed"]
    assert changes[0].severity == "medium"


@pytest.mark.parametrize("field_name", ROUTING_FIELDS)
def test_every_routing_field_classifies_as_routing(field_name):
    """Guards the _FIELD_CLASS map against a field being added to
    ROUTING_FIELDS but landing in the annotation bucket by default."""
    before = build_lock_of(_node("x", **{field_name: ("a",)}))
    after = build_lock_of(_node("x", **{field_name: ("a", "b")}))
    assert [c.kind for c in diff_locks(before, after).changes] == ["routing_changed"]


@pytest.mark.parametrize("field_name", SEMANTIC_FIELDS)
def test_every_semantic_field_classifies_as_semantics(field_name):
    before = build_lock_of(_node("x", **{field_name: "before"}))
    after = build_lock_of(_node("x", **{field_name: "after"}))
    assert [c.kind for c in diff_locks(before, after).changes] == ["semantics_changed"]


def test_high_severity_property_filters_correctly():
    # `gone` and `new` carry DIFFERENT meanings on purpose. Given identical
    # ones the pairer would -- correctly -- call this a rename; see
    # test_thin_definitions_can_mispair_and_that_is_survivable.
    before = build_lock_of(_node("gone", meaning="g"), _node("stay", notes="n1"))
    after = build_lock_of(_node("stay", notes="n2"), _node("new", meaning="w"))
    diff = diff_locks(before, after)
    assert len(diff.changes) == 3  # removed(high) + annotation(medium) + added(medium)
    assert [c.kind for c in diff.high] == ["removed"]


def test_high_severity_changes_sort_first():
    before = build_lock_of(_node("gone", meaning="g"), _node("stay", notes="n1"))
    after = build_lock_of(_node("stay", notes="n2"))
    changes = diff_locks(before, after).changes
    assert changes[0].severity == "high"


def test_thin_definitions_can_mispair_and_that_is_survivable():
    """Two unrelated metrics with identical definition content DO pair.

    49.7% of the live graph shares a fingerprint with another node (see
    _pair_renames). This asserts the consequence rather than pretending it
    away: an unrelated delete + add reads as "renamed", and the alert still
    names both URNs at `high`, which is the only property that matters.
    """
    before = build_lock_of(_node("deleted_thing"))
    after = build_lock_of(_node("unrelated_new_thing"))
    changes = diff_locks(before, after).changes
    assert [c.kind for c in changes] == ["renamed"]
    assert changes[0].severity == SEVERITY["removed"] == "high"
    described = changes[0].describe()
    assert "deleted_thing" in described and "unrelated_new_thing" in described


def test_field_classes_are_disjoint_and_cover_definition_fields():
    assert set(SEMANTIC_FIELDS) | set(ROUTING_FIELDS) | set(ANNOTATION_FIELDS) == set(
        DEFINITION_FIELDS
    )
    assert len(SEMANTIC_FIELDS) + len(ROUTING_FIELDS) + len(ANNOTATION_FIELDS) == len(
        DEFINITION_FIELDS
    )


def test_every_change_kind_has_a_severity():
    kinds = {"added", "removed", "renamed"} | {
        f"{k}_changed" for k in ("semantics", "routing", "annotation")
    }
    assert kinds == set(SEVERITY)
    assert set(SEVERITY.values()) <= {"high", "medium"}


def test_describe_names_the_urn_and_kind():
    change = DefinitionChange(kind="removed", urn="metric://x/y/z", surface="x")
    assert "removed" in change.describe() and "metric://x/y/z" in change.describe()


def test_format_report_says_so_when_empty():
    assert list(format_report(diff_locks({}, {}))) == ["no definition changes"]


# ------------------------------------------------------------------ the gate


def test_committed_lock_matches_current_registries():
    """The gate, as a test.

    If this fails, a metric definition was edited without re-locking. Fix:

        python scripts/check_definition_drift.py --update

    and commit the lock -- the resulting diff is the alert Juniper reads.
    """
    from orion.metrics.lineage import build_graph

    assert LOCK_PATH.exists(), f"{LOCK_PATH} missing -- run --update"
    locked = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["definitions"]
    diff = diff_locks(locked, build_lock(build_graph()))
    assert not diff.changes, "\n".join(format_report(diff))


def test_lock_metric_count_matches_its_own_definitions():
    data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert data["metric_count"] == len(data["definitions"])


def test_cli_gate_passes_on_a_clean_tree():
    """End-to-end: the actual command CI runs, not a re-implementation of it."""
    result = subprocess.run(
        [sys.executable, "scripts/check_definition_drift.py", "--gate"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "definition drift gate: PASS" in result.stdout
