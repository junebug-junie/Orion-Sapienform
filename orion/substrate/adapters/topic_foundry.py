"""Pure conversion of orion-topic-foundry run output into cognitive-substrate records.

Phase 2 of docs/superpowers/specs/2026-07-15-concept-atlas-graph-pipeline-design.md.
Mention-edge -> EntityNodeV1 mapping added 2026-07-28 (see that spec's dated
revision note) as part of retiring topic-foundry's dead `orion:kg:edge:ingest.v1`
bus publish (zero live consumers, orion-rdf-writer/orion-graphdb per
`orion/bus/channels.yaml`'s own former comment) in favor of routing the same
already-computed, LLM-enriched typed edges through this already-live Falkor
ingestion path instead of a second, unconsumed bus channel.

This module intentionally mirrors the construction patterns used by the sibling
adapter `orion/substrate/adapters/concept_induction.py::map_concept_profile_to_substrate`
(provenance via `_common.make_provenance`, temporal via `_common.make_temporal`,
evidence-node + `supports`-edge pairing, `co_occurs_with` edges built from shared
context). It does not perform any HTTP calls or bus I/O — callers are responsible
for fetching topic-foundry's `/topics`, `/topics/{topic_id}/keywords`,
`segments.jsonl`-derived data, and (as of the mention-edge addition) `/kg/edges`
data, and passing it in as plain Python data (dicts or objects with matching
attributes).

Wiring this adapter into a live producer/consumer/registry is explicitly out of
scope for this phase — see the spec's Phase 6/7/8 for where that happens.

Known limitation, not fixed here: `services/orion-hub/scripts/concept_atlas_routes.py`'s
typed-relation-classification view still filters to `node_kind == "concept"`
explicitly. The network view no longer does -- see the 2026-08-20 landmark
addition below and that route's own hydration pass.

Landmark connection added 2026-08-20 (see
`docs/superpowers/specs/2026-08-20-concept-graph-landmark-connection-design.md`):
`landmark_concept_ids` lets a caller pass the golden seed concepts' own
node_ids (Orion/Juniper/Claude, from `orion.substrate.seed`) so a mention that
exact-matches one of them gets an extra `associated_with` edge straight to
that seed node -- connecting the organically-discovered concept graph to the
three real originating speakers of the chat corpus it's mined from, instead
of leaving them permanently isolated.
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Mapping as ABCMapping
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional, Sequence

from orion.core.schemas.cognitive_substrate import (
    ConceptNodeV1,
    EntityNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateSignalBundleV1,
)

from ._common import make_activation, make_provenance, make_temporal

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
# Belt-and-suspenders: in practice this adapter never sees more than
# topic_foundry_client.py's own MAX_KG_EDGES_LIMIT (500, no pagination loop)
# mention edges per call -- that client-side cap is the real ceiling on a
# single ingestion call, set well below this one. Kept here anyway so this
# module's own worst-case bound doesn't silently depend on a caller-side
# constant it can't see.
MAX_MENTION_EDGES = 2000
MAX_ENTITY_LABEL_LENGTH = 120

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
    mention_edges: Optional[Sequence[Any]] = None,
    segment_topic_id_map: Optional[Mapping[str, int]] = None,
    observed_at: Optional[datetime] = None,
    anchor_scope: str = "world",
    subject_ref: Optional[str] = None,
    min_doc_count: int = DEFAULT_MIN_DOC_COUNT,
    landmark_concept_ids: Optional[Mapping[str, str]] = None,
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
        mention_edges: sequence of dict-or-object-like items with `segment_id`
            (str/UUID), `object` (str, the mentioned entity's raw text), and
            `confidence` (float) — matching `GET /kg/edges?predicate=mentions`
            items (`KgEdgeRecord` in
            services/orion-topic-foundry/app/models.py). Only `predicate ==
            "mentions"` edges are meaningful here; the caller is expected to
            have already filtered to that predicate (this function does not
            re-filter by predicate itself, since callers may pass an
            already-scoped list). Other kg_edges predicates
            (`asks_about`/`claims_about`/`next_step`) have no corresponding
            substrate node kind yet and are deliberately out of scope — see
            module docstring.
        segment_topic_id_map: mapping of `segment_id` (str) -> `topic_id`
            (int), used to resolve which topic concept a mention edge's
            source segment belongs to. Distinct from `segment_topic_map`
            above (which is keyed by an arbitrary window/bucket, not
            segment_id, and used only for `co_occurs_with` counting). A
            mention whose segment_id has no entry here, or whose resolved
            topic was filtered out (below `min_doc_count`, or the outlier
            bucket), is silently skipped — never fabricates a topic link.
        observed_at: timestamp for the run; defaults to "now" (via
            `_common.make_temporal`) if not supplied.
        anchor_scope: defaults to "world" per the spec — organically-clustered
            topics are not golden/seeded orion/juniper/relationship concepts.
        subject_ref: optional subject reference to attach to all emitted nodes.
        min_doc_count: topics with `count` below this floor are skipped as
            noise (default 3; see `DEFAULT_MIN_DOC_COUNT`).
        landmark_concept_ids: optional mapping of normalized-lowercase label
            -> an already-existing seed concept node_id (typically built by a
            caller from `orion.substrate.seed.load_seed_concept_nodes()`).
            When a mention's normalized entity label exact-matches a key
            here, one additional `associated_with` edge is emitted from that
            mention's `EntityNodeV1` straight to the landmark node_id -- in
            addition to, not instead of, the normal topic-owned mention edge.
            Exact match only, no fuzzy/alias matching. `None` (the default)
            is a complete no-op: zero behavior change for every existing
            caller that doesn't pass it.

    Returns:
        A `SubstrateGraphRecordV1` with one `ConceptNodeV1` (+ backing
        `EvidenceNodeV1` and `supports` edge) per real topic, free
        `co_occurs_with` `SubstrateEdgeV1` records between topics that share a
        segment/window, and (when `mention_edges` is supplied) one
        `EntityNodeV1` per distinct mentioned entity plus an `associated_with`
        edge from the owning topic's concept node (plus a second
        `associated_with` edge to a landmark node_id, when
        `landmark_concept_ids` is supplied and the entity's label matches).
        Never raises — malformed or empty input degrades to an empty (but
        valid) record.
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
            mention_edges=mention_edges or [],
            segment_topic_id_map=segment_topic_id_map or {},
            observed_at=observed_at,
            anchor_scope=anchor_scope,
            subject_ref=subject_ref,
            min_doc_count=min_doc_count,
            landmark_concept_ids=landmark_concept_ids or {},
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


def _entity_node_id(run_id_str: str, normalized_label: str) -> str:
    # Entity labels are freeform extracted text (arbitrary unicode, length,
    # punctuation) -- not safe to slugify directly into a node_id. A short
    # stable hash keeps ids bounded and collision-safe within one run without
    # needing to sanitize the label itself.
    digest = hashlib.sha1(normalized_label.encode("utf-8")).hexdigest()[:16]
    return f"sub-entity-topicfoundry-{run_id_str}-{digest}"


def _build(
    *,
    run_id_str: str,
    graph_id: str,
    topics: Sequence[Any],
    keywords_by_topic: Mapping[int, Sequence[str]],
    segment_topic_map: Mapping[Any, Iterable[int]],
    topic_embeddings: Mapping[int, Sequence[float]],
    mention_edges: Sequence[Any],
    segment_topic_id_map: Mapping[str, int],
    observed_at: Optional[datetime],
    anchor_scope: str,
    subject_ref: Optional[str],
    min_doc_count: int,
    landmark_concept_ids: Mapping[str, str],
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
                signals=SubstrateSignalBundleV1(
                    confidence=confidence,
                    salience=salience,
                    activation=make_activation(initial_activation=salience),
                ),
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

    # Mention edges: kg_edges.py's real, LLM-enriched "mentions" triples
    # (subject=model_name, predicate="mentions", object=entity text),
    # resolved to the topic concept their source segment belongs to. One
    # EntityNodeV1 per distinct normalized entity label (deduped within this
    # run), one `associated_with` edge per (topic, entity) pair. A mention
    # whose segment isn't in segment_topic_id_map, or whose resolved topic
    # was filtered out above (noise/outlier), is silently skipped -- never
    # fabricates a topic link that doesn't exist.
    entity_node_ids: dict[str, str] = {}  # normalized label -> node_id, dedup within this run
    mention_pairs_seen: set[tuple[int, str]] = set()  # (topic_id, normalized label), dedup edges
    landmark_edges_seen: set[str] = set()  # normalized label, dedup landmark edges (topic-independent)
    mention_items = list(mention_edges)[:MAX_MENTION_EDGES]
    for raw_mention in mention_items:
        segment_id = _get(raw_mention, "segment_id")
        entity_text = _get(raw_mention, "object")
        confidence_raw = _get(raw_mention, "confidence", 0.5)
        if segment_id is None or not entity_text:
            continue
        topic_id = segment_topic_id_map.get(str(segment_id))
        if topic_id is None or topic_id not in accepted:
            continue
        normalized_label = " ".join(str(entity_text).strip().split())[:MAX_ENTITY_LABEL_LENGTH]
        if not normalized_label:
            continue
        try:
            mention_confidence = min(1.0, max(0.0, float(confidence_raw)))
        except (TypeError, ValueError):
            mention_confidence = 0.5

        entity_key = normalized_label.lower()
        entity_node_id = entity_node_ids.get(entity_key)
        if entity_node_id is None:
            entity_node_id = _entity_node_id(run_id_str, entity_key)
            entity_node_ids[entity_key] = entity_node_id
            nodes.append(
                EntityNodeV1(
                    node_id=entity_node_id,
                    anchor_scope=anchor_scope,
                    subject_ref=subject_ref,
                    promotion_state="proposed",
                    temporal=temporal,
                    provenance=make_provenance(
                        source_kind="topic_foundry.mention",
                        source_channel=source_channel,
                        producer="topic_foundry_adapter",
                    ),
                    entity_type="unknown",
                    label=normalized_label,
                    signals=SubstrateSignalBundleV1(confidence=mention_confidence, salience=0.0),
                    metadata={"run_id": run_id_str, "source": "orion-topic-foundry"},
                )
            )

        # Landmark connection: if this mention's entity exact-matches a
        # known golden seed concept (Orion/Juniper/Claude), link the entity
        # node straight to that seed's real node_id -- topic-independent (one
        # per distinct entity per run, not one per topic pairing), since the
        # entity *is* the landmark regardless of which topic mentioned it.
        landmark_node_id = landmark_concept_ids.get(entity_key)
        if landmark_node_id and entity_key not in landmark_edges_seen:
            landmark_edges_seen.add(entity_key)
            edges.append(
                SubstrateEdgeV1(
                    source=NodeRefV1(node_id=entity_node_id, node_kind="entity"),
                    target=NodeRefV1(node_id=landmark_node_id, node_kind="concept"),
                    predicate="associated_with",
                    temporal=temporal,
                    confidence=mention_confidence,
                    salience=0.0,
                    provenance=make_provenance(
                        source_kind="topic_foundry.mention_landmark",
                        source_channel=source_channel,
                        producer="topic_foundry_adapter",
                        evidence_refs=[str(segment_id)],
                    ),
                    metadata={"run_id": run_id_str, "landmark_label": entity_key},
                )
            )

        pair_key = (topic_id, entity_key)
        if pair_key in mention_pairs_seen:
            continue
        mention_pairs_seen.add(pair_key)
        edges.append(
            SubstrateEdgeV1(
                source=NodeRefV1(
                    node_id=f"sub-concept-topicfoundry-{run_id_str}-{topic_id}", node_kind="concept"
                ),
                target=NodeRefV1(node_id=entity_node_id, node_kind="entity"),
                predicate="associated_with",
                temporal=temporal,
                confidence=mention_confidence,
                salience=0.0,
                provenance=make_provenance(
                    source_kind="topic_foundry.mention",
                    source_channel=source_channel,
                    producer="topic_foundry_adapter",
                    evidence_refs=[str(segment_id)],
                ),
                metadata={"run_id": run_id_str, "segment_id": str(segment_id)},
            )
        )

    return SubstrateGraphRecordV1(
        graph_id=graph_id,
        anchor_scope=anchor_scope,
        subject_ref=subject_ref,
        nodes=nodes,
        edges=edges,
    )
