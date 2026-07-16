"""Tests for Phase 4 relation-classification decision logic.

Covers docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md
Phase 4: the three interchangeable threshold strategies (A count baseline, B
PMI, C decayed activation) and the `classify_relation` entry point that
decides whether to invoke a caller-injected classifier callable.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    NodeRefV1,
    SubstrateActivationV1,
    SubstrateEdgeV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
    SubstrateTemporalWindowV1,
)
from orion.substrate.relation_classification import (
    DEFAULT_COUNT_THRESHOLD,
    classify_relation,
    count_score,
    decayed_activation_score,
    is_worth_classifying,
    pmi_score,
    worth_classifying_activation,
    worth_classifying_count,
    worth_classifying_pmi,
)

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _provenance(producer: str = "test") -> SubstrateProvenanceV1:
    return SubstrateProvenanceV1(
        authority="local_inferred",
        source_kind="test",
        source_channel="test",
        producer=producer,
    )


def _concept(
    *,
    node_id: str,
    label: str,
    salience: float = 0.1,
    observed_at: datetime = NOW,
    activation: SubstrateActivationV1 | None = None,
) -> ConceptNodeV1:
    return ConceptNodeV1(
        node_id=node_id,
        anchor_scope="world",
        label=label,
        temporal=SubstrateTemporalWindowV1(observed_at=observed_at),
        provenance=_provenance(),
        signals=SubstrateSignalBundleV1(
            salience=salience,
            activation=activation or SubstrateActivationV1(),
        ),
    )


def _edge(
    *,
    source_id: str = "concept-a",
    target_id: str = "concept-b",
    co_occurrence_count: int | None = 10,
    salience: float = 0.5,
    confidence: float = 0.5,
    extra_metadata: dict | None = None,
) -> SubstrateEdgeV1:
    metadata = dict(extra_metadata or {})
    if co_occurrence_count is not None:
        metadata["co_occurrence_count"] = co_occurrence_count
    return SubstrateEdgeV1(
        source=NodeRefV1(node_id=source_id, node_kind="concept"),
        target=NodeRefV1(node_id=target_id, node_kind="concept"),
        predicate="co_occurs_with",
        temporal=SubstrateTemporalWindowV1(observed_at=NOW),
        confidence=confidence,
        salience=salience,
        provenance=_provenance(),
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Option A -- count baseline.
# --------------------------------------------------------------------------


def test_count_below_threshold_not_worth_classifying() -> None:
    edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD - 1)
    assert worth_classifying_count(edge) is False


def test_count_at_threshold_is_worth_classifying() -> None:
    edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD)
    assert worth_classifying_count(edge) is True


def test_count_above_threshold_is_worth_classifying() -> None:
    edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD + 5)
    assert worth_classifying_count(edge) is True


def test_count_custom_threshold() -> None:
    edge = _edge(co_occurrence_count=3)
    assert worth_classifying_count(edge, threshold=3) is True
    assert worth_classifying_count(edge, threshold=4) is False


def test_count_degrades_gracefully_on_missing_metadata() -> None:
    edge = _edge(co_occurrence_count=None)
    assert count_score(edge) is None
    assert worth_classifying_count(edge) is False


def test_count_degrades_gracefully_on_malformed_metadata() -> None:
    edge = _edge(co_occurrence_count=None, extra_metadata={"co_occurrence_count": "not-a-number"})
    assert count_score(edge) is None
    assert worth_classifying_count(edge) is False


def test_count_degrades_gracefully_on_none_edge() -> None:
    assert worth_classifying_count(None) is False


# --------------------------------------------------------------------------
# Option B -- PMI.
# --------------------------------------------------------------------------


def test_pmi_associated_pair_scores_higher_than_coincidental_pair() -> None:
    # Clearly associated: both concepts individually rare, but they co-occur
    # at a high normalized strength relative to the run's max pair count.
    associated_a = _concept(node_id="concept-a", label="rare-topic-a", salience=0.1)
    associated_b = _concept(node_id="concept-b", label="rare-topic-b", salience=0.1)
    associated_edge = _edge(salience=0.5, confidence=0.5)

    # Clearly coincidental: both concepts individually very common (high
    # salience -- they show up everywhere), but their joint co-occurrence
    # strength is low, i.e. no more than chance would predict.
    coincidental_a = _concept(node_id="concept-c", label="common-topic-c", salience=0.5)
    coincidental_b = _concept(node_id="concept-d", label="common-topic-d", salience=0.5)
    coincidental_edge = _edge(salience=0.05, confidence=0.05)

    associated_score = pmi_score(associated_a, associated_b, associated_edge)
    coincidental_score = pmi_score(coincidental_a, coincidental_b, coincidental_edge)

    assert associated_score is not None
    assert coincidental_score is not None
    assert associated_score > coincidental_score
    # The associated pair should also cross the default "worth it" cutoff
    # while the coincidental pair should not.
    assert worth_classifying_pmi(associated_a, associated_b, associated_edge) is True
    assert worth_classifying_pmi(coincidental_a, coincidental_b, coincidental_edge) is False


def test_pmi_degrades_gracefully_on_zero_salience() -> None:
    node_a = _concept(node_id="concept-a", label="a", salience=0.0)
    node_b = _concept(node_id="concept-b", label="b", salience=0.1)
    edge = _edge()
    assert pmi_score(node_a, node_b, edge) is None
    assert worth_classifying_pmi(node_a, node_b, edge) is False


def test_pmi_degrades_gracefully_on_zero_joint_term() -> None:
    node_a = _concept(node_id="concept-a", label="a", salience=0.1)
    node_b = _concept(node_id="concept-b", label="b", salience=0.1)
    edge = _edge(salience=0.0, confidence=0.0)
    assert pmi_score(node_a, node_b, edge) is None
    assert worth_classifying_pmi(node_a, node_b, edge) is False


def test_pmi_degrades_gracefully_on_none_node() -> None:
    edge = _edge()
    assert pmi_score(None, None, edge) is None
    assert worth_classifying_pmi(None, None, edge) is False


def test_pmi_falls_back_to_confidence_when_salience_unset() -> None:
    node_a = _concept(node_id="concept-a", label="a", salience=0.1)
    node_b = _concept(node_id="concept-b", label="b", salience=0.1)
    edge = _edge(salience=0.0, confidence=0.6)
    score = pmi_score(node_a, node_b, edge)
    assert score is not None


# --------------------------------------------------------------------------
# Option C -- decayed activation.
# --------------------------------------------------------------------------


def test_activation_recent_reinforced_crosses_threshold() -> None:
    reinforced = SubstrateActivationV1(
        activation=0.9,
        recency_score=0.9,
        decay_half_life_seconds=3600,
        decay_floor=0.1,
    )
    node_a = _concept(node_id="concept-a", label="a", observed_at=NOW, activation=reinforced)
    node_b = _concept(node_id="concept-b", label="b", observed_at=NOW, activation=reinforced)

    score_a = decayed_activation_score(node_a, now=NOW)
    assert score_a is not None
    assert score_a >= 0.3
    assert worth_classifying_activation(node_a, node_b, now=NOW) is True


def test_activation_stale_decayed_does_not_cross_threshold() -> None:
    # Short half-life, observed long ago -- heavily decayed by "now".
    stale_bundle = SubstrateActivationV1(
        activation=0.9,
        recency_score=0.0,
        decay_half_life_seconds=60,
        decay_floor=0.0,
    )
    long_ago = NOW - timedelta(hours=1)
    node_a = _concept(node_id="concept-a", label="a", observed_at=long_ago, activation=stale_bundle)
    node_b = _concept(node_id="concept-b", label="b", observed_at=long_ago, activation=stale_bundle)

    score_a = decayed_activation_score(node_a, now=NOW)
    assert score_a is not None
    assert score_a < 0.3
    assert worth_classifying_activation(node_a, node_b, now=NOW) is False


def test_activation_unset_degrades_gracefully() -> None:
    # Default SubstrateActivationV1() has activation=0.0 -- schema default,
    # indistinguishable from "never seeded"; must degrade, not raise/guess.
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")

    assert decayed_activation_score(node_a) is None
    assert worth_classifying_activation(node_a, node_b) is False


def test_activation_one_node_unset_still_degrades() -> None:
    reinforced = SubstrateActivationV1(
        activation=0.9, recency_score=0.9, decay_half_life_seconds=3600, decay_floor=0.1
    )
    node_a = _concept(node_id="concept-a", label="a", observed_at=NOW, activation=reinforced)
    node_b = _concept(node_id="concept-b", label="b")  # unset activation

    assert worth_classifying_activation(node_a, node_b, now=NOW) is False


def test_activation_degrades_gracefully_on_none_node() -> None:
    assert decayed_activation_score(None) is None
    assert worth_classifying_activation(None, None) is False


# --------------------------------------------------------------------------
# is_worth_classifying dispatch.
# --------------------------------------------------------------------------


def test_dispatch_unknown_strategy_returns_false() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    edge = _edge()
    assert is_worth_classifying(node_a, node_b, edge, strategy="not-a-real-strategy") is False  # type: ignore[arg-type]


def test_dispatch_routes_to_count() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    edge = _edge(co_occurrence_count=100)
    assert is_worth_classifying(node_a, node_b, edge, strategy="count") is True
    assert is_worth_classifying(node_a, node_b, edge, strategy="count", count_threshold=1000) is False


# --------------------------------------------------------------------------
# classify_relation entry point.
# --------------------------------------------------------------------------


def test_classify_relation_calls_classifier_only_when_worth_it() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    calls: list[tuple] = []

    def fake_classifier(a, b, e):
        calls.append((a.node_id, b.node_id, e.edge_id))
        return "supports"

    below_edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD - 1)
    result_below = classify_relation(node_a, node_b, below_edge, classifier=fake_classifier)
    assert result_below is None
    assert calls == []

    above_edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD)
    result_above = classify_relation(node_a, node_b, above_edge, classifier=fake_classifier)
    assert result_above is not None
    assert result_above.predicate == "supports"
    assert result_above.source.node_id == "concept-a"
    assert result_above.target.node_id == "concept-b"
    assert calls == [("concept-a", "concept-b", above_edge.edge_id)]


def test_classify_relation_none_result_produces_no_edge() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD)

    def none_classifier(a, b, e):
        return None

    assert classify_relation(node_a, node_b, edge, classifier=none_classifier) is None


def test_classify_relation_catches_classifier_exception() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD)

    def raising_classifier(a, b, e):
        raise RuntimeError("boom")

    result = classify_relation(node_a, node_b, edge, classifier=raising_classifier)
    assert result is None


def test_classify_relation_catches_invalid_predicate_from_classifier() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    edge = _edge(co_occurrence_count=DEFAULT_COUNT_THRESHOLD)

    def bogus_classifier(a, b, e):
        return "not-a-real-predicate"

    result = classify_relation(node_a, node_b, edge, classifier=bogus_classifier)
    assert result is None


def test_classify_relation_with_pmi_strategy() -> None:
    node_a = _concept(node_id="concept-a", label="a", salience=0.1)
    node_b = _concept(node_id="concept-b", label="b", salience=0.1)
    edge = _edge(salience=0.5, confidence=0.5)

    def fake_classifier(a, b, e):
        return "refines"

    result = classify_relation(node_a, node_b, edge, classifier=fake_classifier, strategy="pmi")
    assert result is not None
    assert result.predicate == "refines"
    assert result.metadata["strategy"] == "pmi"


def test_classify_relation_with_activation_strategy() -> None:
    reinforced = SubstrateActivationV1(
        activation=0.9, recency_score=0.9, decay_half_life_seconds=3600, decay_floor=0.1
    )
    node_a = _concept(node_id="concept-a", label="a", observed_at=NOW, activation=reinforced)
    node_b = _concept(node_id="concept-b", label="b", observed_at=NOW, activation=reinforced)
    edge = _edge()

    def fake_classifier(a, b, e):
        return "contradicts"

    result = classify_relation(
        node_a, node_b, edge, classifier=fake_classifier, strategy="activation", now=NOW
    )
    assert result is not None
    assert result.predicate == "contradicts"


def test_classify_relation_degrades_gracefully_never_raises() -> None:
    node_a = _concept(node_id="concept-a", label="a")
    node_b = _concept(node_id="concept-b", label="b")
    malformed_edge = _edge(co_occurrence_count=None, extra_metadata={"co_occurrence_count": object()})

    def fake_classifier(a, b, e):
        return "supports"

    # Should not raise even with a malformed edge metadata value.
    result = classify_relation(node_a, node_b, malformed_edge, classifier=fake_classifier)
    assert result is None
