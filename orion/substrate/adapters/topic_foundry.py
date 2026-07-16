"""Pure conversion of orion-topic-foundry run output into cognitive-substrate records.

Phase 2 of docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md.

This module intentionally mirrors the construction patterns used by the sibling
adapter `orion/substrate/adapters/concept_induction.py::map_concept_profile_to_substrate`
(provenance via `_common.make_provenance`, temporal via `_common.make_temporal`,
evidence-node + `supports`-edge pairing, `co_occurs_with` edges built from shared
context). It does not perform any HTTP calls or bus I/O — callers are responsible
for fetching topic-foundry's `/topics`, `/topics/{topic_id}/keywords`, and
`segments.jsonl`-derived data and passing it in as plain Python data (dicts or
objects with matching attributes).

Wiring this adapter into a live producer/consumer/registry is explicitly out of
scope for this phase — see the spec's Phase 6/7/8 for where that happens.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping as ABCMapping
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateSignalBundleV1,
)

from ._common import make_provenance, make_temporal

# --- Caps (this repo requires capped collections everywhere; see
# docs/superpowers/pr-reports/... incident where an uncapped evidence-id list
# grew unboundedly in a different subsystem). These are generous relative to a
# single topic-foundry run (typically tens, not hundreds, of real topics) but
# exist to bound worst-case work if malformed/adversarial data is passed. ---
MAX_TOPICS_PER_RUN = 500
MAX_KEYWORDS_PER_TOPIC = 20
MAX_SEGMENTS_FOR_COOCCURRENCE = 5000
MAX_TOPICS_PER_SEGMENT = 20
MAX_COOCCURRENCE_EDGES = 2000

# HDBSCAN's noise/outlier bucket. Never a real cluster — always excluded.
OUTLIER_TOPIC_ID = -1

# Topics with fewer than this many documents are treated as noise, not real
# concepts. This mirrors the effective floor concept_induction's adapter gets
# for free from requiring at least one evidence ref per concept, and keeps
# single/double-document stray clusters (common near HDBSCAN's
# min_cluster_size boundary) from polluting the substrate with spurious
# "concepts."
DEFAULT_MIN_DOC_COUNT = 3


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object, defensively."""
    if obj is None:
        return default
    if isinstance(obj, ABCMapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def map_topic_foundry_run_to_substrate(
    *,
    run_id: Any,
    topics: Optional[Sequence[Any]],
    keywords_by_topic: Optional[Mapping[int, Sequence[str]]] = None,
    segment_topic_map: Optional[Mapping[Any, Iterable[int]]] = None,
    topic_embeddings: Optional[Mapping[int, Sequence[float]]] = None,
    observed_at: Optional[datetime] = None,
    anchor_scope: str = "world",
    subject_ref: Optional[str] = None,
    min_doc_count: int = DEFAULT_MIN_DOC_COUNT,
) -> SubstrateGraphRecordV1:
    """Convert one topic-foundry run's topic/keyword/segment output into substrate records.

    Args:
        run_id: topic-foundry's run identifier (UUID or str). Used to namespace
            node ids and as an evidence anchor.
        topics: sequence of topic summary items, each dict-or-object-like with
            `topic_id` (int), `count` (int), `outlier_pct` (float|None), and
            `label` (str|None) — matching `GET /topics?run_id=...` items
            (`TopicSummaryItem` in services/orion-topic-foundry/app/models.py).
        keywords_by_topic: mapping of `topic_id -> keywords` (list of str),
            typically assembled by the caller from repeated
            `GET /topics/{topic_id}/keywords` calls (`TopicKeywordsResponse`).
        segment_topic_map: mapping of a chat-window/segment grouping key to the
            topic_ids observed together within that window — pure counting
            input for `co_occurs_with` edge construction (no inference). A
            topic-foundry `SegmentRecord` carries exactly one `topic_id`, so
            this map is expected to already represent whatever "shared chat
            window" grouping the caller cares about (e.g. segments grouped by
            conversation/session), not a literal 1:1 segment_id -> topic_id
            dump.
        topic_embeddings: optional mapping of `topic_id -> centroid embedding`
            (list of floats), if the caller has one available. If absent for a
            given topic, `concept_embedding` is simply omitted from that
            node's metadata — never fabricated.
        observed_at: timestamp for the run; defaults to "now" (via
            `_common.make_temporal`) if not supplied.
        anchor_scope: defaults to "world" per the spec — organically-clustered
            topics are not golden/seeded orion/juniper/relationship concepts.
        subject_ref: optional subject reference to attach to all emitted nodes.
        min_doc_count: topics with `count` below this floor are skipped as
            noise (default 3; see `DEFAULT_MIN_DOC_COUNT`).

    Returns:
        A `SubstrateGraphRecordV1` with one `ConceptNodeV1` (+ backing
        `EvidenceNodeV1` and `supports` edge) per real topic, and free
        `co_occurs_with` `SubstrateEdgeV1` records between topics that share a
        segment/window. Never raises — malformed or empty input degrades to an
        empty (but valid) record.
    """
    run_id_str = str(run_id) if run_id is not None else "unknown-run"
    graph_id = f"sub-graph-topicfoundry-{run_id_str}"

    empty_record = SubstrateGraphRecordV1(
        graph_id=graph_id,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        nodes=[],
        edges=[],
    )

    if not topics:
        return empty_record

    try:
        return _build(
            run_id_str=run_id_str,
            graph_id=graph_id,
            topics=topics,
            keywords_by_topic=keywords_by_topic or {},
            segment_topic_map=segment_topic_map or {},
            topic_embeddings=topic_embeddings or {},
            observed_at=observed_at,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            min_doc_count=min_doc_count,
        )
    except Exception:
        # Never raise — malformed topic-foundry data degrades to an empty,
        # still-schema-valid record rather than taking down the caller.
        return empty_record


def _derive_label(label: Optional[str], keywords: Sequence[str], topic_id: int) -> str:
    if label:
        return str(label)
    if keywords:
        return " / ".join(str(k) for k in keywords[:3])
    return f"topic_{topic_id}"


def _build(
    *,
    run_id_str: str,
    graph_id: str,
    topics: Sequence[Any],
    keywords_by_topic: Mapping[int, Sequence[str]],
    segment_topic_map: Mapping[Any, Iterable[int]],
    topic_embeddings: Mapping[int, Sequence[float]],
    observed_at: Optional[datetime],
    anchor_scope: str,
    subject_ref: Optional[str],
    min_doc_count: int,
) -> SubstrateGraphRecordV1:
    nodes: list = []
    edges: list = []

    temporal = make_temporal(observed_at=observed_at)
    source_channel = f"orion:topic_foundry:run:{run_id_str}"

    accepted: dict[int, dict[str, Any]] = {}

    for raw_topic in topics[:MAX_TOPICS_PER_RUN]:
        topic_id = _get(raw_topic, "topic_id")
        if topic_id is None:
            continue
        try:
            topic_id = int(topic_id)
        except (TypeError, ValueError):
            continue
        if topic_id == OUTLIER_TOPIC_ID:
            continue

        count = _get(raw_topic, "count", 0) or 0
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = 0
        if count < min_doc_count:
            continue

        outlier_pct = _get(raw_topic, "outlier_pct")
        label = _get(raw_topic, "label")
        keywords = list(keywords_by_topic.get(topic_id, []) or [])[:MAX_KEYWORDS_PER_TOPIC]

        accepted[topic_id] = {
            "count": count,
            "outlier_pct": outlier_pct,
            "label": _derive_label(label, keywords, topic_id),
            "keywords": keywords,
        }

    if not accepted:
        return SubstrateGraphRecordV1(
            graph_id=graph_id,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            nodes=nodes,
            edges=edges,
        )

    total_docs = sum(item["count"] for item in accepted.values()) or 1

    for topic_id, info in accepted.items():
        concept_node_id = f"sub-concept-topicfoundry-{run_id_str}-{topic_id}"
        evidence_node_id = f"sub-evidence-topicfoundry-{run_id_str}-{topic_id}"
        evidence_ref = f"{run_id_str}:topic:{topic_id}"

        outlier_pct = info["outlier_pct"]
        try:
            confidence = 1.0 - float(outlier_pct) if outlier_pct is not None else 0.5
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = min(1.0, max(0.0, confidence))
        salience = min(1.0, max(0.0, info["count"] / total_docs))

        metadata: dict[str, Any] = {
            "topic_id": topic_id,
            "run_id": run_id_str,
            "doc_count": info["count"],
            "keywords": info["keywords"],
            "source": "orion-topic-foundry",
        }
        embedding = topic_embeddings.get(topic_id)
        if embedding:
            metadata["concept_embedding"] = [float(x) for x in embedding]

        nodes.append(
            ConceptNodeV1(
                node_id=concept_node_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                promotion_state="proposed",
                temporal=temporal,
                provenance=make_provenance(
                    source_kind="topic_foundry.topic",
                    source_channel=source_channel,
                    producer="topic_foundry_adapter",
                    evidence_refs=[evidence_ref],
                ),
                label=info["label"],
                definition=None,
                taxonomy_path=[],
                signals=SubstrateSignalBundleV1(confidence=confidence, salience=salience),
                metadata=metadata,
            )
        )

        nodes.append(
            EvidenceNodeV1(
                node_id=evidence_node_id,
                anchor_scope=anchor_scope,
                subject_ref=subject_ref,
                temporal=temporal,
                provenance=make_provenance(
                    source_kind="topic_foundry.run_topic_ref",
                    source_channel=source_channel,
                    producer="topic_foundry_adapter",
                ),
                evidence_type="topic_foundry_run_topic",
                content_ref=evidence_ref,
                signals=SubstrateSignalBundleV1(confidence=confidence, salience=salience),
                metadata={"topic_id": topic_id, "run_id": run_id_str},
            )
        )

        edges.append(
            SubstrateEdgeV1(
                source=NodeRefV1(node_id=evidence_node_id, node_kind="evidence"),
                target=NodeRefV1(node_id=concept_node_id, node_kind="concept"),
                predicate="supports",
                temporal=temporal,
                confidence=confidence,
                salience=salience,
                provenance=make_provenance(
                    source_kind="topic_foundry.support",
                    source_channel=source_channel,
                    producer="topic_foundry_adapter",
                ),
            )
        )

    # Free co_occurs_with edges: pure counting of topic pairs that co-appear
    # within the same caller-supplied segment/window grouping. No inference.
    pair_counts: dict[tuple[int, int], int] = {}
    segment_items = list(segment_topic_map.items())[:MAX_SEGMENTS_FOR_COOCCURRENCE]
    for _segment_key, raw_topic_ids in segment_items:
        if len(pair_counts) >= MAX_COOCCURRENCE_EDGES:
            break
        try:
            topic_ids = {int(t) for t in raw_topic_ids}
        except (TypeError, ValueError):
            continue
        distinct = sorted(t for t in topic_ids if t in accepted)[:MAX_TOPICS_PER_SEGMENT]
        for a, b in itertools.combinations(distinct, 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1
            if len(pair_counts) >= MAX_COOCCURRENCE_EDGES:
                break

    max_count = max(pair_counts.values()) if pair_counts else 1
    for (topic_a, topic_b), count in pair_counts.items():
        strength = min(1.0, max(0.0, count / max_count))
        edges.append(
            SubstrateEdgeV1(
                source=NodeRefV1(node_id=f"sub-concept-topicfoundry-{run_id_str}-{topic_a}", node_kind="concept"),
                target=NodeRefV1(node_id=f"sub-concept-topicfoundry-{run_id_str}-{topic_b}", node_kind="concept"),
                predicate="co_occurs_with",
                temporal=temporal,
                confidence=strength,
                salience=strength,
                provenance=make_provenance(
                    source_kind="topic_foundry.co_occurrence",
                    source_channel=source_channel,
                    producer="topic_foundry_adapter",
                ),
                metadata={"co_occurrence_count": count, "run_id": run_id_str},
            )
        )

    return SubstrateGraphRecordV1(
        graph_id=graph_id,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        nodes=nodes,
        edges=edges,
    )
