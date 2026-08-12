"""Gate tests for the metric semantic layer (phases 1+2).

Fixture expectations here are hand-computed against the literal source text in
each fixture, not read back from the implementation.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from orion.metrics.consumers import (
    HIGH_CONFIDENCE_KINDS,
    KIND_ATTRIBUTE,
    KIND_COLLECTION,
    KIND_COMPARE,
    KIND_DICT_KEY,
    KIND_GET,
    KIND_LITERAL,
    KIND_SUBSCRIPT,
    iter_source_files,
    scan_python,
    scan_repo,
)
from orion.metrics.lineage import (
    _DIMENSION_FIELDS,
    build_graph,
    resolve_bus_channels,
    resolve_field_channels,
    resolve_inner_state,
    resolve_organ_signals,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- resolver


def test_every_registry_resolves_nonempty():
    """A resolver returning [] means the registry rotted or the join broke.

    Lower bounds are the counts verified on 2026-08-12; these registries only
    grow in practice, and an exact-equality assert would fail on every
    legitimate addition.
    """
    assert len(resolve_field_channels()) >= 38
    assert len(resolve_bus_channels()) >= 261
    assert len(resolve_organ_signals()) >= 252
    # 13 signals + their enumerable scalar fields
    assert len(resolve_inner_state()) >= 13


def test_registry_import_failure_propagates(monkeypatch):
    """A registry that no longer imports must raise, not return []."""
    import orion.metrics.lineage as lineage

    monkeypatch.delitem(__import__("sys").modules, "orion.inner_state_registry", raising=False)

    def _boom(*_a, **_k):
        raise ImportError("registry gone")

    monkeypatch.setattr(lineage, "resolve_inner_state", _boom)
    with pytest.raises(ImportError):
        lineage.resolve_inner_state()


def test_urns_are_unique_and_well_formed():
    graph = build_graph()
    assert len(graph.nodes) >= 587
    for urn, node in graph.nodes.items():
        assert urn.startswith("metric://"), urn
        assert urn == node.urn
        # surface/producer/name -- at least 3 segments after the scheme
        assert len(urn[len("metric://") :].split("/")) >= 3, urn


def test_canonical_dimension_names_are_not_scan_tokens():
    """`level`/`confidence` are the #field half of a URN.

    Scanning for them would match nearly every dict access in the repo and
    make blast radius meaningless.
    """
    tokens = build_graph().scan_tokens()
    for dim in _DIMENSION_FIELDS:
        assert dim not in tokens, f"{dim} leaked into the scan token set"


def test_organ_signal_urn_carries_dimension_field():
    nodes = {n.urn: n for n in resolve_organ_signals()}
    urn = "metric://organ_signal/biometrics/gpu_load#level"
    assert urn in nodes
    assert nodes[urn].name == "gpu_load"
    assert nodes[urn].metric_field == "level"
    assert nodes[urn].producer_service == "orion-biometrics"


def test_field_channel_meaning_comes_from_glossary():
    nodes = {n.name: n for n in resolve_field_channels()}
    assert "cpu_pressure" in nodes
    assert nodes["cpu_pressure"].meaning
    assert nodes["cpu_pressure"].registry_source.endswith("field_channel_glossary.v1.yaml")


# ------------------------------------------------------- AST access kinds


FIXTURE = textwrap.dedent(
    '''\
    """Docstring mentioning cpu_pressure which is not a read."""
    # comment mentioning cpu_pressure, also not a read
    LABEL = "cpu_pressure"
    value = vector["cpu_pressure"]
    other = vector.get("cpu_pressure", 0.0)
    bag = {"cpu_pressure": 0.3}
    if key == "cpu_pressure":
        pass
    reading = frame.cpu_pressure
    CHANNELS = ("cpu_pressure", "gpu_pressure")
    '''
)
# Hand-computed against the literal text above. Matching is EXACT-equality on
# the string constant, so prose that merely contains the token never matches:
#   L1 docstring   -> no hit  (value is the whole sentence, != "cpu_pressure")
#   L2 comment     -> no hit  (invisible to the AST entirely)
#   L3 assignment  -> literal (bare constant, exact match)
#   L4 subscript   -> subscript
#   L5 .get        -> get
#   L6 dict key    -> dict_key
#   L7 comparison  -> compare
#   L9 attribute   -> attribute
#   L10 tuple      -> collection_member
EXPECTED = {
    (3, KIND_LITERAL),
    (4, KIND_SUBSCRIPT),
    (5, KIND_GET),
    (6, KIND_DICT_KEY),
    (7, KIND_COMPARE),
    (9, KIND_ATTRIBUTE),
    (10, KIND_COLLECTION),
}


def test_access_kinds_classified_by_line(tmp_path):
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE, encoding="utf-8")
    hits = scan_python(path, frozenset({"cpu_pressure"}), "sample.py")
    assert {(h.line, h.kind) for h in hits} == EXPECTED


def test_comment_mention_is_never_a_consumer(tmp_path):
    """Line 2 is a comment. No hit may be reported on it at all."""
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE, encoding="utf-8")
    hits = scan_python(path, frozenset({"cpu_pressure"}), "sample.py")
    assert 2 not in {h.line for h in hits}


def test_prose_mention_produces_no_hit_at_all(tmp_path):
    """Matching is exact-equality, so a docstring that merely contains the
    token yields nothing -- neither a high-confidence hit nor a literal."""
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE, encoding="utf-8")
    hits = scan_python(path, frozenset({"cpu_pressure"}), "sample.py")
    assert 1 not in {h.line for h in hits}


def test_collection_member_is_high_confidence(tmp_path):
    """A metric named in a tuple is registered into a generic consumer.

    This is the access kind humans miss -- the loop that iterates the tuple
    never names the metric again. Regression for the real case at
    orion/field/pressure.py:35.
    """
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE, encoding="utf-8")
    hits = [h for h in scan_python(path, frozenset({"cpu_pressure"}), "sample.py") if h.line == 10]
    assert len(hits) == 1
    assert hits[0].kind == KIND_COLLECTION
    assert hits[0].kind in HIGH_CONFIDENCE_KINDS


def test_subscript_constant_not_double_counted_as_literal(tmp_path):
    """Line 4 must yield exactly one hit, not a subscript plus a literal."""
    path = tmp_path / "sample.py"
    path.write_text(FIXTURE, encoding="utf-8")
    hits = [h for h in scan_python(path, frozenset({"cpu_pressure"}), "sample.py") if h.line == 4]
    assert len(hits) == 1
    assert hits[0].kind == KIND_SUBSCRIPT


def test_unparsed_file_is_reported_not_swallowed(tmp_path):
    (tmp_path / "orion").mkdir()
    bad = tmp_path / "orion" / "broken.py"
    bad.write_text("def f(:\n", encoding="utf-8")
    result = scan_repo(["cpu_pressure"], roots=["orion"], repo_root=tmp_path)
    assert "orion/broken.py" in result.unparsed


def test_worktrees_are_excluded_from_scan(tmp_path):
    """.worktrees/ holds other branches' code -- not this repo's blast radius."""
    wt = tmp_path / "orion" / ".worktrees" / "other"
    wt.mkdir(parents=True)
    (wt / "leak.py").write_text('x = v["cpu_pressure"]\n', encoding="utf-8")
    (tmp_path / "orion" / "real.py").write_text('y = v["cpu_pressure"]\n', encoding="utf-8")

    result = scan_repo(["cpu_pressure"], roots=["orion"], repo_root=tmp_path)
    paths = {h.path for h in result.hits}
    assert "orion/real.py" in paths
    assert not any(".worktrees" in p for p in paths)


def test_test_paths_are_tagged_not_dropped(tmp_path):
    (tmp_path / "orion" / "tests").mkdir(parents=True)
    (tmp_path / "orion" / "tests" / "test_x.py").write_text(
        'y = v["cpu_pressure"]\n', encoding="utf-8"
    )
    result = scan_repo(["cpu_pressure"], roots=["orion"], repo_root=tmp_path)
    assert len(result.hits) == 1
    assert result.hits[0].is_test is True
    # ...and excluded from the default blast-radius view
    assert result.consumers_for("cpu_pressure") == []


# ------------------------------------------------------------- regression


def test_prediction_error_blast_radius_includes_endogenous_curiosity():
    """Regression for the transport_prediction_error retirement miss.

    CLAUDE.md 0A records that transport_prediction_error was excluded from
    attention_self_model.py's ACTIVE_INFERENCE_DOMAINS as known-dead, while
    its node kept winning real budget slots in endogenous_curiosity.py --
    a generic consumer nobody checked. The whole point of mechanical blast
    radius is that both files surface together, so retiring a metric by
    editing one consumer is visibly incomplete.
    """
    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    consumers = {h.path for h in scan.consumers_for("prediction_error")}
    assert "orion/substrate/endogenous_curiosity.py" in consumers
    assert "orion/substrate/attention_self_model.py" in consumers


def test_cpu_pressure_generic_consumers_are_discovered():
    """cpu_pressure has NO explicit non-test key read anywhere in the repo.

    Every real production consumer registers it into a collection that some
    loop iterates. Verified live 2026-08-12: 81 raw hits, zero of them a
    subscript/get outside tests. If collection_member were not high
    confidence, this metric's blast radius would read as empty -- which is
    exactly how a "retired" metric keeps silently feeding a live consumer.
    """
    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    consumers = {h.path for h in scan.consumers_for("cpu_pressure")}
    assert "orion/field/pressure.py" in consumers
    assert "orion/field_coherence.py" in consumers


def test_repo_scan_finds_real_consumers_across_all_surfaces():
    """Lifecycle check: real registries -> real tokens -> real hits."""
    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    assert scan.files_scanned > 1000
    assert len(scan.hits) > 1000
    # a field channel, an organ signal kind, and a bus channel each land
    assert scan.consumers_for("cpu_pressure")
    assert scan.consumers_for("gpu_load", high_confidence_only=False)
    assert scan.consumers_for("orion:substrate:brain_frame", high_confidence_only=False)
