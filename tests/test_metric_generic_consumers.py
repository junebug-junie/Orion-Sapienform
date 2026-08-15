"""Tests for the two blind spots found on 2026-08-14.

Both are regressions against a real mistake, not hypotheticals:

1. `field_coherence_warning`'s blast radius reported five call sites. Two were
   its own PRODUCER (`worker.py:272/282`, subscript writes counted as reads)
   and none of the remaining three was a named reader -- while attention's
   `_current_pressure_proxy()` max()'d over it every 2-second tick. The
   verdict a reader would draw from that card ("nothing reads this") was wrong
   in both directions at once.
2. `expected_offline_suppression` has a real producer whose guard is false for
   every node in the fleet. The layer had no producer concept at all, so
   finding that took a manual trace.
"""
from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from orion.metrics.consumers import (  # noqa: E402
    HIGH_CONFIDENCE_KINDS,
    KIND_CHANNEL_KWARG,
    KIND_FIELD_KWARG,
    KIND_SUBSCRIPT,
    KIND_SUBSCRIPT_WRITE,
    WRITE_KINDS,

    _MetricVisitor,
)
from orion.metrics.generic_consumers import (  # noqa: E402
    CONFIRMED,
    LIKELY,
    _GenericVisitor,
    _is_channel_vector_annotation,
    module_touches_vectors,
    scan_generic_consumers,
    surfaces_at_risk,
)


def _kinds(source: str, tokens: set[str]) -> list[tuple[str, str]]:
    visitor = _MetricVisitor(frozenset(tokens))
    visitor.visit(ast.parse(textwrap.dedent(source)))
    return [(tok, kind) for tok, _line, kind, _callee in visitor.hits]


def _kinds_full(source: str, tokens: set[str]) -> list[tuple[str, str, str | None]]:
    visitor = _MetricVisitor(frozenset(tokens))
    visitor.visit(ast.parse(textwrap.dedent(source)))
    return [(tok, kind, callee) for tok, _line, kind, callee in visitor.hits]


# ------------------------------------------------- read vs write


def test_subscript_write_is_not_a_consumer():
    """`vec["m"] = x` is a producer. Counting it as blast radius is what made
    field_coherence_warning's own writer look like its consumer."""
    kinds = _kinds('vec["m"] = 1.0', {"m"})
    assert kinds == [("m", KIND_SUBSCRIPT_WRITE)]
    assert KIND_SUBSCRIPT_WRITE not in HIGH_CONFIDENCE_KINDS


def test_subscript_read_is_still_a_consumer():
    assert _kinds("x = vec[\"m\"]", {"m"}) == [("m", KIND_SUBSCRIPT)]
    assert KIND_SUBSCRIPT in HIGH_CONFIDENCE_KINDS


def test_augmented_and_deleted_subscripts_are_writes():
    assert _kinds('vec["m"] += 1.0', {"m"}) == [("m", KIND_SUBSCRIPT_WRITE)]
    assert _kinds('del vec["m"]', {"m"}) == [("m", KIND_SUBSCRIPT_WRITE)]


def test_nested_write_target_is_a_write():
    """The real shape at worker.py:282."""
    kinds = _kinds('state.stamps.setdefault(nid, {})["m"] = ts', {"m"})
    assert ("m", KIND_SUBSCRIPT_WRITE) in kinds


def test_channel_kwarg_is_a_write():
    """`Perturbation(channel="expected_offline_suppression", ...)`."""
    kinds = _kinds('out.append(Perturbation(node_id=n, channel="m", intensity=1.0))', {"m"})
    assert ("m", KIND_CHANNEL_KWARG) in kinds
    assert KIND_CHANNEL_KWARG in WRITE_KINDS


def test_field_kwarg_is_a_write():
    """`Model(reasoning_load=0.5)` -- the kwarg NAME is the metric."""
    assert ("m", KIND_FIELD_KWARG) in _kinds("Model(m=0.5)", {"m"})


def test_unrelated_kwarg_name_is_not_a_write():
    assert _kinds("Model(unrelated=0.5)", {"m"}) == []


def test_write_kinds_are_disjoint_from_high_confidence_reads():
    """A kind must never be both -- that is the bug this suite exists for."""
    assert not (WRITE_KINDS & HIGH_CONFIDENCE_KINDS)


# ------------------------------------------------- generic consumers


def _generic(source: str, *, touches: bool = True):
    visitor = _GenericVisitor("x.py", False, module_touches_vectors=touches)
    visitor.visit(ast.parse(textwrap.dedent(source)))
    return visitor.found


def test_enumerator_over_an_annotated_vector_is_a_generic_consumer():
    """`_current_pressure_proxy` in one line -- the exact miss.

    Named for the ENUMERATORS branch it actually exercises: `.items()` inside
    the max() matches before AGGREGATORS is reached. The aggregator path has
    its own test below; conflating them let AGGREGATORS go untested.
    """
    found = _generic(
        """
        def proxy(vector: dict[str, float]) -> float:
            return max(v for k, v in vector.items())
        """
    )
    assert len(found) == 1
    assert found[0].confidence == LIKELY
    assert "vector" in found[0].evidence


def test_direct_node_vectors_enumeration_is_confirmed():
    found = _generic("for nid, vec in field.node_vectors.items(): pass")
    assert [c.confidence for c in found] == [CONFIRMED]


def test_named_subscript_only_is_not_generic():
    """Reading two named channels is a NAMED consumer; the string scan already
    covers it and double-reporting would inflate the risk list."""
    found = _generic(
        """
        def named(vector: dict[str, float]) -> float:
            return vector["cpu_pressure"] - vector["gpu_pressure"]
        """
    )
    assert found == []


def test_module_not_touching_vectors_is_skipped():
    """`dict[str, float]` is a shape, not a meaning. Without this filter the
    first run reported topic shares and rank scores as channel consumers.

    Only asserts the flag is HONORED. `module_touches_vectors` is what decides
    the flag, and it is tested separately below -- mutation testing found that
    pinning it to True left all 27 tests green while the live output grew to
    every `dict[str, float]` parameter in the repo.
    """
    source = """
        def top_share(shares: dict[str, float]) -> float:
            return max(shares.values())
    """
    assert _generic(source, touches=True) != []
    assert _generic(source, touches=False) == []


def _touches(source: str) -> bool:
    return module_touches_vectors(ast.parse(textwrap.dedent(source)))


def test_module_scope_needs_a_real_reference_not_prose():
    """A docstring mentioning `node_vectors` is not a reference to one.

    The first version substring-matched the source, so 20 of 58 matching
    modules had zero AST reference -- including this detector's own module,
    and including both false positives in the `likely` tier.
    """
    assert not _touches('"""Talks about node_vectors and FieldStateV1."""')
    assert not _touches("# node_vectors appears only in this comment\nx = 1")


@pytest.mark.parametrize(
    "source",
    [
        "for nid, v in field.node_vectors.items(): pass",
        "x = state.capability_vectors",
        "from orion.schemas.field_state import FieldStateV1",
        "def f(s: FieldStateV1): pass",
    ],
)
def test_module_scope_accepts_real_references(source):
    assert _touches(source)


def test_aggregator_path_fires_without_an_enumerator():
    """`max(vector)` with no `.items()` anywhere.

    Mutation testing caught this: the previous max() test wrote
    `max(v for k, v in vector.items())`, which matches on the ENUMERATORS
    branch before AGGREGATORS is ever reached -- emptying AGGREGATORS left all
    27 tests green. All 11 aggregator builtins were untested.
    """
    found = _generic(
        """
        def proxy(vector: dict[str, float]) -> float:
            return max(vector)
        """
    )
    assert len(found) == 1
    assert "max()" in found[0].evidence


def test_capability_vectors_is_a_confirmed_root():
    """5 of the 13 live confirmed sites come from capability_vectors; dropping
    it from VECTOR_ATTRS previously broke no test."""
    found = _generic("for cid, v in field.capability_vectors.items(): pass")
    assert [c.confidence for c in found] == [CONFIRMED]
    assert "capability_vectors" in found[0].evidence


def test_both_vector_roots_appear_in_the_live_scan():
    found = scan_generic_consumers(REPO_ROOT)
    evidence = " ".join(c.evidence for c in found)
    assert "node_vectors" in evidence
    assert "capability_vectors" in evidence


def test_comprehension_reports_a_real_line_number():
    """ast.comprehension carries no lineno; reporting `file.py:0` made two real
    sites uncitable in the first run."""
    found = _generic(
        """
        def f(vector: dict[str, float]):
            return [v for v in vector]
        """
    )
    assert len(found) == 1
    assert found[0].line > 0


@pytest.mark.parametrize(
    "annotation,expected",
    [
        ("dict[str, float]", True),
        ("Dict[str, float]", True),
        ("Mapping[str, float]", True),
        ("dict[str, float] | None", True),
        ("Optional[dict[str, float]]", True),
        ("dict[str, Any]", False),
        ("dict", False),
        ("dict[str, str]", False),
        ("list[float]", False),
    ],
)
def test_vector_annotation_recognition(annotation, expected):
    assert _is_channel_vector_annotation(annotation) is expected


def test_surfaces_at_risk_requires_a_confirmed_site():
    """A `likely` heuristic must not silently suppress an orphan verdict --
    that would trade a false negative for a false positive."""
    from orion.metrics.generic_consumers import GenericConsumer

    likely = [GenericConsumer("a.py", 1, "f", "e", LIKELY, False)]
    confirmed = [GenericConsumer("a.py", 1, "f", "e", CONFIRMED, False)]
    assert surfaces_at_risk(likely) == frozenset()
    assert surfaces_at_risk(confirmed) == frozenset({"field_channel"})
    assert surfaces_at_risk([]) == frozenset()


# ------------------------------------------------- against the live repo


def test_the_attention_proxy_is_actually_found():
    """The regression that matters: the live call site that read
    field_coherence_warning while the blast radius said nothing did."""
    found = scan_generic_consumers(REPO_ROOT)
    hits = [
        c
        for c in found
        if c.path == "orion/attention/field_attention/selectors.py"
        and c.function == "_current_pressure_proxy"
    ]
    assert hits, "the max()-over-whole-vector consumer must be discoverable"


def test_live_repo_has_confirmed_generic_consumers():
    found = scan_generic_consumers(REPO_ROOT)
    assert any(c.confidence == CONFIRMED for c in found)
    assert surfaces_at_risk(found) == frozenset({"field_channel"})


def test_field_coherence_warning_producer_is_not_its_blast_radius():
    """End-to-end on the real graph: worker.py must appear as a WRITER and must
    NOT appear in the high-confidence consumer list."""
    from orion.metrics.consumers import scan_repo
    from orion.metrics.lineage import build_graph

    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    token = "field_coherence_warning"
    sources = graph.registry_sources_for(token)

    writers = scan.producers_for(token, exclude_paths=sources)
    readers = scan.consumers_for(token, exclude_paths=sources)

    writer_paths = {h.path for h in writers}
    reader_paths = {h.path for h in readers}
    assert "services/orion-field-digester/app/worker.py" in writer_paths
    assert "services/orion-field-digester/app/worker.py" not in reader_paths


def test_expected_offline_suppression_producer_is_discoverable():
    """The channel whose writer cost a manual trace."""
    from orion.metrics.consumers import scan_repo
    from orion.metrics.lineage import build_graph

    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    token = "expected_offline_suppression"
    writers = scan.producers_for(token, exclude_paths=graph.registry_sources_for(token))
    assert any(
        h.path == "services/orion-field-digester/app/ingest/state_deltas.py"
        for h in writers
    ), [h.path for h in writers]


def test_field_kwarg_counts_only_for_the_declaring_schema():
    """An unqualified `x=0.5` kwarg is not evidence that x was written.

    Live: `confidence=` appears 425 non-test times across 142 callees and NOT
    ONE targets a schema declaring `confidence`. Counting them unqualified
    produced a 434-row "WRITTEN BY" list with zero real writers, and held 19
    metrics off --unwritten on passthrough like `max_gap_sec=args.max_gap_sec`.
    """
    from orion.metrics.consumers import scan_repo
    from orion.metrics.lineage import build_graph

    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())
    sources = graph.registry_sources_for("confidence")
    schemas = {n.schema_id for n in graph.by_token("confidence") if n.schema_id}
    assert schemas, "confidence must resolve to schema-bearing nodes"

    gated = scan.producers_for("confidence", exclude_paths=sources, schema_ids=schemas)
    assert all(
        h.kind != KIND_FIELD_KWARG or h.callee in schemas for h in gated
    ), [(h.path, h.line, h.callee) for h in gated if h.kind == KIND_FIELD_KWARG]
    # The conservative default: no schema_ids means no field_kwarg evidence,
    # which can leave a metric ON the unwritten list but never silently off it.
    ungated = scan.producers_for("confidence", exclude_paths=sources)
    assert all(h.kind != KIND_FIELD_KWARG for h in ungated)


def test_callee_is_recorded_for_kwarg_hits():
    hits = _kinds_full("Model(m=0.5)", {"m"})
    assert hits == [("m", KIND_FIELD_KWARG, "Model")]
    hits = _kinds_full('P(channel="m")', {"m"})
    assert hits == [("m", KIND_CHANNEL_KWARG, "P")]


def test_dotted_callee_records_the_attribute_name():
    assert _kinds_full("mod.Model(m=0.5)", {"m"}) == [("m", KIND_FIELD_KWARG, "Model")]


def test_floor_warning_prints_for_a_multi_surface_token(capsys):
    """The FLOOR warning must key on ALL nodes for the token, not the last one.

    `cmd_metric` prints a per-node block in a `for node in nodes:` loop, then
    checked `node.surface` afterwards -- the leaked binding, i.e. the LAST
    node. `confidence`, `memory_pressure` and `repair_pressure` are the three
    multi-surface tokens (named in gate.py's own orphan_nodes docstring), and
    for all three the last node is organ_signal/inner_state, so the field
    channel's "do not retire on this alone" warning silently never printed --
    on exactly the tokens most likely to be misread.

    Mutation testing found this fix untested: reverting it to `node.surface`
    left every other test green, because nothing covered cmd_metric at all.
    """
    import importlib.util

    from orion.metrics.consumers import scan_repo
    from orion.metrics.lineage import build_graph

    spec = importlib.util.spec_from_file_location(
        "check_metric_lineage", REPO_ROOT / "scripts" / "check_metric_lineage.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    graph = build_graph()
    scan = scan_repo(graph.scan_tokens().keys())

    for token in ("confidence", "memory_pressure", "repair_pressure"):
        surfaces = {n.surface for n in graph.by_token(token)}
        assert "field_channel" in surfaces and len(surfaces) > 1, (token, surfaces)
        # The last node must NOT be the field channel, or the bug would have
        # been invisible for this token and the test would prove nothing.
        assert graph.by_token(token)[-1].surface != "field_channel", token

        capsys.readouterr()
        mod.cmd_metric(graph, scan, token)
        out = capsys.readouterr().out
        assert "FLOOR, not a total" in out, f"{token} lost its retirement warning"
