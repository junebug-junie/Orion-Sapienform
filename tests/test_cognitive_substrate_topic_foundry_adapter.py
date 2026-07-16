from __future__ import annotations

from datetime import datetime, timezone

from orion.core.schemas.cognitive_substrate import SubstrateGraphRecordV1
from orion.substrate.adapters.topic_foundry import map_topic_foundry_run_to_substrate


def _topics() -> list[dict]:
    return [
        {"topic_id": -1, "count": 411, "outlier_pct": 1.0, "label": None},
        {"topic_id": 0, "count": 200, "outlier_pct": 0.0, "label": None},
        {"topic_id": 1, "count": 39, "outlier_pct": 0.0, "label": None},
        {"topic_id": 2, "count": 2, "outlier_pct": 0.0, "label": None},  # below min_doc_count floor
    ]


def _keywords_by_topic() -> dict[int, list[str]]:
    return {
        0: ["like", "meow", "just", "user", "assistant", "juniper", "let", "hi"],
        1: ["memory", "recall", "context", "graph"],
        2: ["stray", "noise"],
    }


def test_outlier_bucket_never_produces_a_node() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
    )
    concept_ids = {n.metadata.get("topic_id") for n in out.nodes if n.node_kind == "concept"}
    assert -1 not in concept_ids


def test_thin_topic_below_floor_produces_no_node() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
        min_doc_count=3,
    )
    concept_ids = {n.metadata.get("topic_id") for n in out.nodes if n.node_kind == "concept"}
    assert 2 not in concept_ids
    assert concept_ids == {0, 1}


def test_shared_segment_produces_co_occurs_with_edge() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
        segment_topic_map={
            "window-a": [0, 1],
            "window-b": [0],
        },
    )
    cooccur_edges = [e for e in out.edges if e.predicate == "co_occurs_with"]
    assert len(cooccur_edges) == 1
    edge = cooccur_edges[0]
    assert {edge.source.node_id, edge.target.node_id} == {
        "sub-concept-topicfoundry-run-1-0",
        "sub-concept-topicfoundry-run-1-1",
    }
    assert edge.metadata["co_occurrence_count"] == 1


def test_cooccurrence_excludes_outlier_and_below_floor_topics() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
        segment_topic_map={
            "window-a": [-1, 0, 2],  # outlier + below-floor topic must not form edges
        },
    )
    cooccur_edges = [e for e in out.edges if e.predicate == "co_occurs_with"]
    assert cooccur_edges == []


def test_every_concept_node_has_non_empty_evidence_ref() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
    )
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert concept_nodes
    for node in concept_nodes:
        assert len(node.provenance.evidence_refs) > 0
        assert all(ref for ref in node.provenance.evidence_refs)


def test_evidence_node_and_supports_edge_emitted_per_concept() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
    )
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    evidence_nodes = [n for n in out.nodes if n.node_kind == "evidence"]
    supports_edges = [e for e in out.edges if e.predicate == "supports"]
    assert len(evidence_nodes) == len(concept_nodes)
    assert len(supports_edges) == len(concept_nodes)


def test_promotion_state_proposed_and_anchor_scope_world_by_default() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
    )
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert concept_nodes
    for node in concept_nodes:
        assert node.promotion_state == "proposed"
        assert node.anchor_scope == "world"


def test_label_falls_back_to_keywords_when_label_is_null() -> None:
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=_topics(),
        keywords_by_topic=_keywords_by_topic(),
    )
    concept_nodes = {n.metadata["topic_id"]: n for n in out.nodes if n.node_kind == "concept"}
    assert concept_nodes[0].label == "like / meow / just"


def test_uses_label_field_when_non_null() -> None:
    topics = [{"topic_id": 5, "count": 10, "outlier_pct": 0.0, "label": "Explicit Label"}]
    out = map_topic_foundry_run_to_substrate(run_id="run-1", topics=topics, keywords_by_topic={})
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert concept_nodes[0].label == "Explicit Label"


def test_concept_embedding_stored_in_metadata_when_provided() -> None:
    topics = [{"topic_id": 7, "count": 10, "outlier_pct": 0.0, "label": "vec topic"}]
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=topics,
        keywords_by_topic={},
        topic_embeddings={7: [0.1, 0.2, 0.3]},
    )
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert concept_nodes[0].metadata["concept_embedding"] == [0.1, 0.2, 0.3]


def test_concept_embedding_omitted_when_not_provided() -> None:
    topics = [{"topic_id": 8, "count": 10, "outlier_pct": 0.0, "label": "no vec topic"}]
    out = map_topic_foundry_run_to_substrate(run_id="run-1", topics=topics, keywords_by_topic={})
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert "concept_embedding" not in concept_nodes[0].metadata


def test_empty_topics_returns_empty_but_valid_record() -> None:
    out = map_topic_foundry_run_to_substrate(run_id="run-1", topics=[])
    assert isinstance(out, SubstrateGraphRecordV1)
    assert out.nodes == []
    assert out.edges == []

    out_none = map_topic_foundry_run_to_substrate(run_id="run-1", topics=None)
    assert isinstance(out_none, SubstrateGraphRecordV1)
    assert out_none.nodes == []


def test_malformed_topic_entries_are_skipped_not_raised() -> None:
    malformed = [
        {"topic_id": "not-an-int", "count": 10},
        {"topic_id": None, "count": 10},
        {"count": 10},  # missing topic_id entirely
        object(),  # no attributes at all
        {"topic_id": 3, "count": "also-not-an-int"},  # count coerces to 0, below floor
    ]
    out = map_topic_foundry_run_to_substrate(run_id="run-1", topics=malformed)
    assert isinstance(out, SubstrateGraphRecordV1)
    assert out.nodes == []
    assert out.edges == []


def test_malformed_segment_topic_map_degrades_gracefully() -> None:
    topics = [{"topic_id": 0, "count": 10, "outlier_pct": 0.0, "label": "ok"}]
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=topics,
        keywords_by_topic={},
        segment_topic_map={"bad-window": "not-iterable-of-ints"},
    )
    assert isinstance(out, SubstrateGraphRecordV1)
    # The single concept node still forms; the malformed window is simply skipped.
    assert len([n for n in out.nodes if n.node_kind == "concept"]) == 1


def test_observed_at_and_subject_ref_propagate() -> None:
    ts = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)
    topics = [{"topic_id": 0, "count": 10, "outlier_pct": 0.0, "label": "ok"}]
    out = map_topic_foundry_run_to_substrate(
        run_id="run-1",
        topics=topics,
        observed_at=ts,
        subject_ref="project:orion_sapienform",
    )
    assert out.subject_ref == "project:orion_sapienform"
    concept_nodes = [n for n in out.nodes if n.node_kind == "concept"]
    assert concept_nodes[0].subject_ref == "project:orion_sapienform"
    assert concept_nodes[0].temporal.observed_at == ts
