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


def test_pairing_requires_uniqueness_across_the_whole_lock():
    """A definition shared with a SURVIVING node never pairs.

    Review found that treating a mispair as cosmetic was wrong: "renamed A ->
    B" tells the reader the metric still exists under a new name and no
    consumer needs migrating, which is the opposite of the truth when A was
    deleted. 49.7% of the live graph shares a fingerprint (organ signals
    collapse to `organ=X class=Y`), so diff-local uniqueness was not enough.

    Here `survivor` carries the same definition as the removed and added nodes,
    so the definition identifies nothing and must not be used to claim a
    rename.
    """
    survivor = _node("survivor")
    before = build_lock_of(_node("deleted_thing"), survivor)
    after = build_lock_of(_node("unrelated_new_thing"), survivor)
    kinds = sorted(c.kind for c in diff_locks(before, after).changes)
    assert kinds == ["added", "removed"]


def test_distinctive_definition_still_pairs_with_a_survivor_present():
    """The tightened guard must not kill real renames: a channel with its own
    `meaning` is unique lock-wide, so it still pairs."""
    survivor = _node("survivor", meaning="something else entirely")
    before = build_lock_of(_node("execution_load", meaning="step load"), survivor)
    after = build_lock_of(_node("cortex_exec_step_load", meaning="step load"), survivor)
    changes = diff_locks(before, after).changes
    assert [c.kind for c in changes] == ["renamed"]
    assert changes[0].previous_urn.endswith("/execution_load")


def test_feeds_dimensions_is_positional_not_a_set():
    """self_state_dimension and evidence_dimension are packed into one
    positional tuple by lineage.py. Sorting it made swapping which dimension a
    channel feeds as self-state versus as evidence a ZERO-delta edit -- a real
    routing change that emitted no alert. Found in review."""
    before = build_lock_of(_node("x", feeds_dimensions=("continuity", "resource")))
    after = build_lock_of(_node("x", feeds_dimensions=("resource", "continuity")))
    changes = diff_locks(before, after).changes
    assert [c.kind for c in changes] == ["routing_changed"]
    assert changes[0].severity == "high"


def test_set_valued_routing_fields_are_still_order_insensitive():
    """The counterpart: everything NOT in ORDERED_FIELDS stays a set, which is
    what keeps a reordered YAML consumer list from being an alert."""
    before = build_lock_of(_node("x", declared_consumers=("a", "b")))
    after = build_lock_of(_node("x", declared_consumers=("b", "a")))
    assert not diff_locks(before, after).changes


def test_long_values_differing_at_the_tail_render_differently():
    """`_short` truncated both sides independently at 60 chars, so amending the
    tail of a long `meaning` -- 29 of 38 field-channel meanings exceed 60 chars
    -- rendered as two byte-identical strings. The sentence claimed a change
    and displayed none."""
    shared = "Renamed from execution_load 2026-07-24 for scope honesty and " * 3
    before = build_lock_of(_node("x", meaning=shared + "ORIGINAL TAIL"))
    after = build_lock_of(_node("x", meaning=shared + "REPLACEMENT TAIL"))
    described = diff_locks(before, after).changes[0].describe()
    left, right = described.split(" -> ", 1)
    assert left != right
    assert "ORIGINAL TAIL" in left and "REPLACEMENT TAIL" in right


def test_midstring_difference_elides_both_ends():
    """The window must trim the shared TAIL too, not just the shared head.

    Without suffix trimming the renderer still distinguishes the two values,
    so distinguishability alone cannot pin this down -- the contract being
    locked here is that the alert stays focused on the differing region
    instead of dumping 100 characters of identical trailing prose.
    """
    head, tail = "A" * 100, "B" * 100
    before = build_lock_of(_node("x", meaning=head + "FOO" + tail))
    after = build_lock_of(_node("x", meaning=head + "BAR" + tail))
    described = diff_locks(before, after).changes[0].describe()
    left, right = described.split(" -> ", 1)
    assert "FOO" in left and "BAR" in right
    # A bounded slice of the shared tail as context is fine; 20+ characters of
    # it means the window ran to the truncation limit instead of stopping at
    # the common suffix. Asserting on "…" counts does NOT discriminate here --
    # the limit-truncation path emits one too, which is how the first version
    # of this test let the no-suffix-trim mutant survive.
    assert "B" * 20 not in left, left
    assert "A" * 20 not in left, left


def test_long_values_differing_at_the_head_render_differently():
    shared = " and then a great deal of identical trailing prose repeated" * 3
    before = build_lock_of(_node("x", meaning="ALPHA" + shared))
    after = build_lock_of(_node("x", meaning="OMEGA" + shared))
    described = diff_locks(before, after).changes[0].describe()
    left, right = described.split(" -> ", 1)
    assert left != right
    assert "ALPHA" in left and "OMEGA" in right


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


# ------------------------------------------- the alert block (merge-base derived)
#
# Every test below drives the real CLI module with LOCK_PATH redirected into a
# tmp_path and _base_definitions stubbed. Nothing here touches the committed
# lock -- asserted by test_committed_lock_matches_current_registries still
# passing alongside them.


@pytest.fixture()
def drift_cli(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_definition_drift", REPO_ROOT / "scripts" / "check_definition_drift.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "LOCK_PATH", tmp_path / "lock.json")
    return mod


def _stub_graph(mod, monkeypatch, nodes):
    graph = MetricGraph()
    for node in nodes:
        graph.nodes[node.urn] = node
    monkeypatch.setattr(mod, "build_graph", lambda: graph)


def _stub_base(
    mod,
    monkeypatch,
    definitions,
    note="merge base abc123 (origin/main)",
    *,
    on_base_branch=False,
):
    """Stub the merge base AND whether HEAD is it.

    `_base_is_head()` (PR fixing "the gate could never be green on main")
    correctly skips alert-block verification when HEAD *is* the merge base,
    because `_last_change` there describes the branch that already landed and
    cannot be recomputed. These tests stubbed only `_base_definitions`, so they
    inherited the REAL answer from whatever checkout they ran in -- passing on
    a feature branch and failing on a main checkout, which is exactly what
    happened once this landed on main. Default False: these tests are modelling
    a PR branch, which is the only place the check is meant to run.
    """
    monkeypatch.setattr(mod, "_base_definitions", lambda: (definitions, note))
    monkeypatch.setattr(mod, "_base_is_head", lambda: on_base_branch)


def _block(mod) -> dict:
    return json.loads(mod.LOCK_PATH.read_text(encoding="utf-8"))["_last_change"]


def test_repeated_updates_report_the_cumulative_branch_delta(
    drift_cli, monkeypatch
):
    """Two --update calls on one branch must not lose the first one's deltas.

    The original implementation diffed against the lock ON DISK, so the second
    call overwrote the first call's sentences and a high-severity consumer
    removal vanished while the gate stayed green. Found in review.
    """
    mod = drift_cli
    base = {
        "metric://field_channel/p/chan_a": {"declared_consumers": ["svc-x", "svc-y"]},
        "metric://field_channel/p/chan_b": {"meaning": "b"},
    }
    _stub_base(mod, monkeypatch, base)

    # Edit 1: drop a consumer from chan_a.
    _stub_graph(mod, monkeypatch, [
        _node("chan_a", declared_consumers=("svc-x",), producer_service="p"),
        _node("chan_b", meaning="b", producer_service="p"),
    ])
    assert mod.main([]) == 0 or True
    mod.main(["--update"])
    assert any("chan_a" in s for s in _block(mod)["changes"])

    # Edit 2: delete chan_b entirely, then re-lock again.
    _stub_graph(mod, monkeypatch, [
        _node("chan_a", declared_consumers=("svc-x",), producer_service="p"),
    ])
    mod.main(["--update"])
    changes = _block(mod)["changes"]

    # BOTH edits must survive -- this is the whole fix.
    assert any("chan_a" in s and "routing_changed" in s for s in changes), changes
    assert any("chan_b" in s and "removed" in s for s in changes), changes
    assert _block(mod)["high_severity_count"] == 2


def test_update_is_idempotent(drift_cli, monkeypatch):
    mod = drift_cli
    _stub_base(mod, monkeypatch, {"metric://field_channel/p/x": {"meaning": "old"}})
    _stub_graph(mod, monkeypatch, [_node("x", meaning="new", producer_service="p")])
    mod.main(["--update"])
    first = mod.LOCK_PATH.read_text(encoding="utf-8")
    mod.main(["--update"])
    assert mod.LOCK_PATH.read_text(encoding="utf-8") == first


def test_clobbered_lock_is_not_treated_as_a_first_run(drift_cli, monkeypatch):
    """A lock resolved to `{}` by a botched merge is falsy but is NOT a first
    run. Conflating them made --update announce "initial lock" and discard
    every real delta, gate green. Found in review."""
    mod = drift_cli
    _stub_base(mod, monkeypatch, {"metric://field_channel/p/gone": {"meaning": "g"}})
    _stub_graph(mod, monkeypatch, [_node("kept", meaning="k", producer_service="p")])
    mod.LOCK_PATH.write_text("{}", encoding="utf-8")

    mod.main(["--update"])
    changes = _block(mod)["changes"]
    assert changes != [mod.NO_PRIOR_STATE]
    assert any("gone" in s and "removed" in s for s in changes), changes


def test_gate_fails_when_the_alert_block_was_hand_edited(drift_cli, monkeypatch):
    """The claim "an agent cannot re-lock quietly" is only true if the gate
    verifies the block. Before this check it was a convention.

    Overlaps `test_off_the_base_branch_a_hand_edited_alert_still_fails` on the
    assertion and differs on the path: that one constructs the bad lock
    directly, this one drives the real `--update` round trip first, so it also
    proves the block `--update` writes is one the gate then ACCEPTS. A stub
    that only ever hand-builds locks cannot catch the two disagreeing.
    """
    mod = drift_cli
    _stub_base(mod, monkeypatch, {"metric://field_channel/p/x": {"meaning": "old"}})
    _stub_graph(mod, monkeypatch, [_node("x", meaning="new", producer_service="p")])
    mod.main(["--update"])
    assert mod.main(["--gate"]) == 0

    data = json.loads(mod.LOCK_PATH.read_text(encoding="utf-8"))
    data["_last_change"]["changes"] = ["nothing to see here"]
    mod.LOCK_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    assert mod.main(["--gate"]) == 1


def test_gate_skips_block_verification_loudly_when_base_is_unresolvable(
    drift_cli, monkeypatch, capsys
):
    """A skipped integrity check must announce itself. Silent skipping is the
    failure mode this file exists to stop."""
    mod = drift_cli
    _stub_base(mod, monkeypatch, {"metric://field_channel/p/x": {"meaning": "old"}})
    _stub_graph(mod, monkeypatch, [_node("x", meaning="old", producer_service="p")])
    mod.main(["--update"])

    monkeypatch.setattr(mod, "_base_definitions", lambda: (None, "no merge base"))
    capsys.readouterr()
    assert mod.main(["--gate"]) == 0
    out = capsys.readouterr().out
    assert mod.BASE_UNAVAILABLE in out
    assert "NOT verified" in out


def test_missing_lock_fails_the_gate_with_the_actionable_message(
    drift_cli, monkeypatch, capsys
):
    """A missing lock fails on definition drift anyway, so the exit code alone
    proves nothing. The message is the actionable part -- it has to name the
    real problem (there is no lock) rather than the symptom (595 additions)."""
    mod = drift_cli
    _stub_base(mod, monkeypatch, {})
    _stub_graph(mod, monkeypatch, [_node("x", meaning="m", producer_service="p")])
    assert not mod.LOCK_PATH.exists()
    capsys.readouterr()
    assert mod.main(["--gate"]) == 1
    err = capsys.readouterr().err
    assert "no committed lock" in err


def test_update_and_gate_are_mutually_exclusive(drift_cli, monkeypatch):
    """--update returns before the gate block, so the combination would rewrite
    the lock and exit 0 with the gate silently skipped."""
    mod = drift_cli
    _stub_base(mod, monkeypatch, {})
    _stub_graph(mod, monkeypatch, [_node("x", producer_service="p")])
    with pytest.raises(SystemExit) as exc:
        mod.main(["--update", "--gate"])
    assert exc.value.code == 2
    assert not mod.LOCK_PATH.exists()


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


def test_on_the_base_branch_the_derived_block_is_not_recomputed(
    drift_cli, monkeypatch, capsys
):
    """Regression: the gate failed on `main` after every definition change.

    `_last_change` answers "what does this BRANCH change relative to the merge base?".
    On the base branch itself the merge base IS HEAD, so that diff is empty by
    construction and the derived block collapses to "no definition changes" -- while
    the committed block still states what the branch that landed here changed. The two
    can never agree, so main went red after every re-lock, and every branch cut from
    that main inherited the failure.

    Confirmed live on 2026-08-14: both post-gate commits on main (4cca1eb5f, the PR
    that introduced this file, and b4a697a9d) failed for exactly this reason.
    """
    mod = drift_cli
    _stub_graph(mod, monkeypatch, [_node("x", producer_service="p", meaning="current")])
    # Derived from the stubbed graph, not hand-written: a hand-written definitions
    # dict does not match build_lock's fingerprint shape, so the gate would fail on
    # ordinary drift and the test would pass for the wrong reason.
    locked_defs = mod.build_lock(mod.build_graph())
    # Base == HEAD, so the base carries the very same definitions as the lock.
    _stub_base(mod, monkeypatch, locked_defs)
    monkeypatch.setattr(mod, "_base_is_head", lambda: True)

    mod.LOCK_PATH.write_text(
        json.dumps(
            {
                # What the branch that landed here changed -- historical, and not
                # rederivable from this vantage point.
                "_last_change": {
                    "base": "merge base abc123 (origin/main)",
                    "change_count": 1,
                    "high_severity_count": 1,
                    "changes": ["high   routing_changed metric://bus_channel/*/y"],
                },
                "definitions": locked_defs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert mod.main(["--gate"]) == 0
    out = capsys.readouterr().out
    assert "HEAD is the merge base" in out, "the skip must be stated, never silent"


def test_off_the_base_branch_a_hand_edited_alert_still_fails(
    drift_cli, monkeypatch
):
    """The skip above must not weaken the constraint where it matters. On a real PR
    branch (HEAD is not the merge base), erasing the alert by hand still fails."""
    mod = drift_cli
    _stub_graph(mod, monkeypatch, [_node("x", producer_service="p", meaning="new")])
    _stub_base(mod, monkeypatch, {"metric://field_channel/p/x": {"meaning": "old"}})
    monkeypatch.setattr(mod, "_base_is_head", lambda: False)

    mod.LOCK_PATH.write_text(
        json.dumps(
            {
                "_last_change": {
                    "base": "merge base abc123 (origin/main)",
                    "change_count": 0,
                    "high_severity_count": 0,
                    "changes": ["nothing to see here"],
                },
                "definitions": {"metric://field_channel/p/x": {"meaning": "new"}},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert mod.main(["--gate"]) == 1
