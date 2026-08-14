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
    scan_generic_consumers,
    surfaces_at_risk,
)


def _kinds(source: str, tokens: set[str]) -> list[tuple[str, str]]:
    visitor = _MetricVisitor(frozenset(tokens))
    visitor.visit(ast.parse(textwrap.dedent(source)))
    return [(tok, kind) for tok, _line, kind in visitor.hits]


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


def test_max_over_an_annotated_vector_is_a_generic_consumer():
    """`_current_pressure_proxy` in one line -- the exact miss."""
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
    first run reported topic shares and rank scores as channel consumers."""
    source = """
        def top_share(shares: dict[str, float]) -> float:
            return max(shares.values())
    """
    assert _generic(source, touches=True) != []
    assert _generic(source, touches=False) == []


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
