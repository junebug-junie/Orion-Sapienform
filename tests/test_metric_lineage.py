"""Gate tests for the metric semantic layer (phases 1+2).

Fixture expectations here are hand-computed against the literal source text in
each fixture, not read back from the implementation.
"""
from __future__ import annotations

import textwrap
import typing
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
    """A registry that no longer imports must raise, not return [].

    Exercises the REAL resolve_inner_state(). An earlier version of this test
    monkeypatched resolve_inner_state itself to a raising stub and then
    asserted the stub raised -- it asserted on its own mock and would have
    passed even if the function swallowed every error.

    Setting sys.modules[name] = None makes `from name import X` raise
    ImportError, so the guarantee under test is the real import statement.
    """
    import sys

    import orion.metrics.lineage as lineage

    monkeypatch.setitem(sys.modules, "orion.inner_state_registry", None)
    with pytest.raises(ImportError):
        lineage.resolve_inner_state()


def test_organ_registry_import_failure_propagates(monkeypatch):
    import sys

    import orion.metrics.lineage as lineage

    monkeypatch.setitem(sys.modules, "orion.signals.registry", None)
    with pytest.raises(ImportError):
        lineage.resolve_organ_signals()


# --------------------------------------------------- review-finding gates


def test_pep604_optional_float_fields_are_resolved():
    """`float | None` must resolve, not just Optional[float].

    str(types.UnionType) is "<class 'types.UnionType'>", so a string
    comparison silently never fired and dropped every `X | None` metric --
    including FieldStateV1.recent_perturbation_zscore.
    """
    from orion.metrics.lineage import _is_float_like

    assert _is_float_like(float) is True
    assert _is_float_like(typing.Optional[float]) is True
    assert _is_float_like(float | None) is True
    assert _is_float_like(str | None) is False
    assert _is_float_like(int) is False


def test_recent_perturbation_zscore_is_in_the_urn_space():
    """The concrete metric the float|None bug dropped."""
    names = {n.name for n in resolve_inner_state()}
    assert "recent_perturbation_zscore" in names


def test_no_dangling_upstream_urns():
    """Every upstream URN must resolve to a real node.

    Synthesised organ parents (metric://organ_signal/<p>/<p>) could never
    match a real node and left 14 permanently-dangling edges.
    """
    graph = build_graph()
    known = set(graph.nodes)
    dangling = {u for n in graph.nodes.values() for u in n.upstream if u not in known}
    assert dangling == set()


def test_organ_parents_are_recorded_as_organ_ids():
    nodes = {n.urn: n for n in resolve_organ_signals()}
    equilibrium = nodes["metric://organ_signal/equilibrium/mesh_health#level"]
    assert equilibrium.upstream_organs == ("biometrics",)
    assert equilibrium.upstream == ()


def test_real_metrics_named_like_dimensions_are_not_dropped():
    """`confidence` is a genuine glossary channel.

    A name-based blocklist deleted it and 5 real inner-state scalars from the
    token set, making them invisible in BOTH blast radius and orphan output.
    """
    graph = build_graph()
    tokens = graph.scan_tokens()
    assert any(n.name == "confidence" and n.surface == "field_channel" for n in graph.nodes.values())
    assert "confidence" in tokens


def test_multi_producer_bus_channel_urn_is_order_independent():
    """41 channels have >1 producer; a cosmetic YAML reorder must not
    rename the URN, and the dropped producers must still be recorded."""
    multi = [n for n in resolve_bus_channels() if len(n.all_producers) > 1]
    assert multi, "expected at least one multi-producer channel"
    for node in multi:
        assert list(node.all_producers) == sorted(node.all_producers)
        assert node.producer_service == node.all_producers[0]


def test_field_channel_feeds_dimensions_not_declared_consumers():
    """Dimension names and service names must not share one field."""
    nodes = {n.name: n for n in resolve_field_channels()}
    cpu = nodes["cpu_pressure"]
    assert cpu.feeds_dimensions == ("resource_pressure",)
    assert cpu.declared_consumers == ()


def test_urns_are_unique_and_well_formed():
    graph = build_graph()
    assert len(graph.nodes) >= 587
    for urn, node in graph.nodes.items():
        assert urn.startswith("metric://"), urn
        assert urn == node.urn
        # surface/producer/name -- at least 3 segments after the scheme
        assert len(urn[len("metric://") :].split("/")) >= 3, urn


def test_scan_token_is_never_the_dimension_half_of_a_urn():
    """Structural guarantee that replaced the old name blocklist: a token is
    always `name`, never `metric_field`."""
    graph = build_graph()
    for node in graph.nodes.values():
        assert node.scan_token == node.name


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


@pytest.mark.parametrize("prefix", [".worktrees/x", ".claude/worktrees/agent-abc"])
def test_scan_works_when_checkout_itself_sits_under_an_excluded_name(tmp_path, prefix):
    """CLAUDE.md 2 documents two worktree conventions whose paths contain
    excluded names. Matching absolute path parts made the scan return zero
    files with no error -- a silent false negative in exactly the
    "retired metric looks dead" case this tool exists to prevent.
    """
    base = tmp_path / prefix
    (base / "orion").mkdir(parents=True)
    (base / "orion" / "real.py").write_text('x = v["cpu_pressure"]\n', encoding="utf-8")

    result = scan_repo(["cpu_pressure"], roots=["orion"], repo_root=base)
    assert result.files_scanned == 1
    assert [h.path for h in result.hits] == ["orion/real.py"]


def test_attribute_write_is_not_a_consumer(tmp_path):
    """`self.pressure = x` is a write; counting it inflated single-word
    tokens with unrelated assignments across the repo."""
    path = tmp_path / "s.py"
    path.write_text("self.pressure = 1.0\ny = obj.pressure\n", encoding="utf-8")
    hits = scan_python(path, frozenset({"pressure"}), "s.py")
    assert [(h.line, h.kind) for h in hits] == [(2, KIND_ATTRIBUTE)]


def test_config_scan_requires_whole_token(tmp_path):
    """Substring matching made `cpu_pressure` match `cpu_pressure_ewma`."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "c.yaml").write_text(
        "a: cpu_pressure_ewma\nb: cpu_pressure\n", encoding="utf-8"
    )
    result = scan_repo(["cpu_pressure"], roots=["config"], repo_root=tmp_path)
    assert [h.line for h in result.hits] == [2]


def test_null_byte_file_is_reported_not_crashing(tmp_path):
    """ast.parse raises ValueError (not SyntaxError) on embedded NULs; letting
    it escape aborted the entire scan."""
    (tmp_path / "orion").mkdir()
    (tmp_path / "orion" / "nul.py").write_bytes(b"x = 1\x00\n")
    (tmp_path / "orion" / "ok.py").write_text('y = v["cpu_pressure"]\n', encoding="utf-8")

    result = scan_repo(["cpu_pressure"], roots=["orion"], repo_root=tmp_path)
    assert "orion/nul.py" in result.unparsed
    assert [h.path for h in result.hits] == ["orion/ok.py"]


def test_registry_of_origin_is_excluded_from_its_own_blast_radius():
    """orion/signals/registry.py declaring an organ's signal_kinds is not a
    consumer of them. Counting it reported 'blast radius: 1' for 38 of 57
    organ tokens that have no real consumer at all -- the tool's central
    claim inverted.
    """
    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    sources = graph.registry_sources_for("gpu_load")
    assert "orion/signals/registry.py" in sources
    consumers = scan.consumers_for("gpu_load", exclude_paths=sources)
    assert all(h.path != "orion/signals/registry.py" for h in consumers)


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
