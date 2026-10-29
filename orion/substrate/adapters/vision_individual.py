"""Juniper's name for a walkway individual -> one substrate EntityNodeV1.

docs/superpowers/specs/2026-09-22-walkway-camera-busy-world-design.md ideas 1
and 3. When Juniper answers "who is this?" for a ``vision_individual`` cluster,
``orion-substrate-runtime`` consumes ``orion:ask:answered`` and materializes
the record built here, so the individual becomes something the substrate can
hold, activate and link -- not only a label on a Postgres row.

Identity: one node per individual, never per label.
- ``node_id`` is derived from ``individual_id``, so re-answering (a rename)
  lands on the same node via the materializer's raw node-id fallback.
- ``subject_ref`` is the individual, so the resolver's label-keyed identity
  (``entity|world|<subject>|label:<label>``) never merges two different
  individuals who happen to get the same name, and never merges a walkway
  individual into a chat-mined topic-foundry entity of the same name.

Pure: no I/O. The caller supplies ``kind`` (``vision_individual.kind``) if it
could read it.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Iterable, Optional

from orion.core.schemas.cognitive_substrate import (
    EntityNodeV1,
    EvidenceNodeV1,
    NodeRefV1,
    SubstrateEdgeV1,
    SubstrateGraphRecordV1,
    SubstrateProvenanceV1,
    SubstrateSignalBundleV1,
)

from ._common import make_temporal

SOURCE_KIND = "vision_individual"
SOURCE_CHANNEL = "orion:ask:answered"
PRODUCER = "orion-substrate-runtime.ask_answered"
MAX_LABEL_LENGTH = 120

# Detector label class -> entity_type. Anything unlisted keeps the detector's
# own class name (it is already a concrete noun), and a missing kind is
# "unknown", the schema default every other producer uses.
_PERSON = {"person", "man", "woman", "child", "people"}
_ANIMAL = {"dog", "cat", "bird", "horse", "deer", "squirrel", "rabbit", "fox", "animal"}
_VEHICLE = {
    "car", "truck", "bus", "bicycle", "motorcycle", "van", "scooter",
    "stroller", "vehicle", "suv", "pickup",
}


def entity_type_for_kind(kind: Optional[str]) -> str:
    k = (kind or "").strip().lower()
    if not k:
        return "unknown"
    if k in _PERSON:
        return "person"
    if k in _ANIMAL:
        return "animal"
    if k in _VEHICLE:
        return "vehicle"
    return k[:64]


def entity_node_id_for_individual(individual_id: str) -> str:
    digest = hashlib.sha256(individual_id.encode("utf-8")).hexdigest()[:24]
    return f"entity:vision_individual:{digest}"


def normalize_label(label: str) -> str:
    return " ".join(str(label or "").split())[:MAX_LABEL_LENGTH]


def map_vision_individual_label_to_substrate(
    *,
    individual_id: str,
    label: str,
    kind: Optional[str],
    ask_id: str,
    answered_at: datetime,
    stream_id: Optional[str] = None,
) -> Optional[SubstrateGraphRecordV1]:
    """Build the one-node record, or None if the answer is empty after cleanup."""
    clean = normalize_label(label)
    if not clean or not individual_id:
        return None
    subject_ref = f"vision_individual:{individual_id}"
    metadata = {
        "source_kind": SOURCE_KIND,
        "individual_id": individual_id,
        "label_ask_id": ask_id,
        "detector_kind": (kind or None),
    }
    if stream_id:
        metadata["stream_id"] = stream_id
    node = EntityNodeV1(
        node_id=entity_node_id_for_individual(individual_id),
        anchor_scope="world",
        subject_ref=subject_ref,
        promotion_state="proposed",
        temporal=make_temporal(observed_at=answered_at),
        # Juniper named it: highest-confidence thing in this record.
        signals=SubstrateSignalBundleV1(confidence=0.9, salience=0.5),
        provenance=SubstrateProvenanceV1(
            authority="user_asserted",
            source_kind=SOURCE_KIND,
            source_channel=SOURCE_CHANNEL,
            producer=PRODUCER,
            correlation_id=ask_id,
            evidence_refs=[individual_id, f"orion_ask:{ask_id}"],
        ),
        metadata=metadata,
        entity_type=entity_type_for_kind(kind),
        label=clean,
    )
    return SubstrateGraphRecordV1(anchor_scope="world", subject_ref=subject_ref, nodes=[node])


# --- "seen on <day>" (walkway rollups, 2026-10-18) -------------------------------------------
#
# The named-people entity nodes above had 0 edges: a name, and nothing it connects to.
# From each COMPLETE walkway day rollup (vision_rollup grain=day, orion-sql-writer),
# orion-substrate-runtime (app/walkway_seen_loop.py) writes one EvidenceNodeV1 for the day
# and one ``observed_in`` edge from every named individual's existing entity node to it.
# Idempotent: the day node's id is derived from (stream, date) and an edge's identity is
# (source, predicate, target), so a re-run lands on the same node and edges. Entity nodes
# are never created here -- only Juniper's answer (above) makes one.
#
# Why evidence, not event (fixed 2026-10-18): FalkorSubstrateStore durably writes only
# DURABLE_NODE_KINDS (concept, evidence, entity, assertion); an EventNodeV1 day raised
# ValueError on every live pass. A day's rollup is a sensed observation record that points
# back at its source row (content_ref), which is what evidence is for; it is the same shape
# as orion-memory-consolidation's referent projector (entity -observed_in-> evidence
# "episode_memory:<id>"). A concept is an abstract label with seeded activation decay -- the
# wrong thing for one dated record. The day's line rides in metadata (in-process only; the
# durable codec keeps evidence_type/content_ref) -- the rollup row content_ref names holds it.

SEEN_SOURCE_KIND = "walkway_rollup"
SEEN_SOURCE_CHANNEL = "vision_rollup"
SEEN_PRODUCER = "orion-substrate-runtime.walkway_seen"
MAX_SUMMARY_LENGTH = 600


WALKWAY_DAY_EVIDENCE_TYPE = "walkway_day"


MAX_ALIASES = 20


def relabelled_individual_entity(existing: EntityNodeV1, node: EntityNodeV1) -> EntityNodeV1:
    """``existing`` cluster node updated to ``node``'s name, kind and newest ask.

    Explicit update, not the materializer: merge_node() keeps the existing label and
    lets existing metadata win, so a rename (even a capitalization-only one) and the
    newest label_ask_id would both be silently dropped. The old name is kept as an
    alias so recall by it still works. The caller upserts the result."""
    aliases: list[str] = []
    for alias in [*existing.aliases, existing.label]:
        if alias and alias != node.label and alias not in aliases:
            aliases.append(alias)
    metadata = {**existing.metadata}
    for key, value in node.metadata.items():
        if value is not None:
            metadata[key] = value
    return existing.model_copy(
        update={
            "label": node.label,
            "aliases": aliases[-MAX_ALIASES:],
            "entity_type": node.entity_type if node.entity_type != "unknown" else existing.entity_type,
            "provenance": node.provenance.model_copy(
                update={"evidence_refs": sorted({*existing.provenance.evidence_refs, *node.provenance.evidence_refs})}
            ),
            "temporal": existing.temporal.model_copy(
                update={"observed_at": max(existing.temporal.observed_at, node.temporal.observed_at)}
            ),
            "signals": existing.signals.model_copy(
                update={"confidence": max(existing.signals.confidence, node.signals.confidence)}
            ),
            "metadata": metadata,
        }
    )


def walkway_day_content_ref(stream_id: str, day: date) -> str:
    return f"vision_rollup:{stream_id}:day:{day.isoformat()}"


def walkway_day_node_id(stream_id: str, day: date) -> str:
    return f"evidence:walkway_day:{stream_id}:{day.isoformat()}"


def map_walkway_day_seen_to_substrate(
    *,
    stream_id: str,
    day: date,
    line: str,
    seen: Iterable[tuple[str, str, Optional[str]]],
    observed_at: datetime,
) -> Optional[SubstrateGraphRecordV1]:
    """``seen`` = (entity node_id, label, first time seen that day "HH:MM" or None) per named
    individual whose entity node exists. None when there is nobody to link."""
    pairs = sorted({(nid, label, at) for nid, label, at in seen if nid})
    if not pairs or not line.strip():
        return None
    day_id = walkway_day_node_id(stream_id, day)
    subject_ref = f"walkway_day:{stream_id}:{day.isoformat()}"
    provenance = SubstrateProvenanceV1(
        authority="sensed",
        source_kind=SEEN_SOURCE_KIND,
        source_channel=SEEN_SOURCE_CHANNEL,
        producer=SEEN_PRODUCER,
        correlation_id=subject_ref,
        evidence_refs=[walkway_day_content_ref(stream_id, day)],
    )
    node = EvidenceNodeV1(
        node_id=day_id,
        anchor_scope="world",
        subject_ref=subject_ref,
        promotion_state="proposed",
        temporal=make_temporal(observed_at=observed_at),
        signals=SubstrateSignalBundleV1(confidence=0.8, salience=0.3),
        provenance=provenance,
        metadata={"source_kind": SEEN_SOURCE_KIND, "stream_id": stream_id, "day": day.isoformat(),
                  "summary": " ".join(line.split())[:MAX_SUMMARY_LENGTH]},
        evidence_type=WALKWAY_DAY_EVIDENCE_TYPE,
        content_ref=walkway_day_content_ref(stream_id, day),
    )
    edges = []
    for nid, label, at in pairs:
        digest = hashlib.sha256(f"{nid}|observed_in|{day_id}".encode("utf-8")).hexdigest()[:24]
        edges.append(SubstrateEdgeV1(
            edge_id=f"sub-edge-walkway-seen-{digest}",
            source=NodeRefV1(node_id=nid, node_kind="entity"),
            target=NodeRefV1(node_id=day_id, node_kind="evidence"),
            predicate="observed_in",
            temporal=make_temporal(observed_at=observed_at),
            confidence=0.8,
            salience=0.3,
            provenance=provenance,
            metadata={"label": label, "first_seen_local": at, "day": day.isoformat(), "stream_id": stream_id},
        ))
    return SubstrateGraphRecordV1(anchor_scope="world", subject_ref=subject_ref, nodes=[node], edges=edges)
