import uuid
import hashlib
import logging
import json
import unicodedata
from datetime import datetime, timezone
from typing import Tuple, Optional, Any

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# Typed schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.rdf import RdfWriteRequest, RdfBuildRequest
from orion.schemas.social_chat import SocialRoomTurnStoredV1

from app.provenance import attach_provenance

ORION = Namespace("http://conjourney.net/orion#")
CM = Namespace("http://orion.ai/collapse#")
PROV = Namespace("http://www.w3.org/ns/prov#")

logger = logging.getLogger(__name__)

def _normalize_observer(observer: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(observer or "")).encode("ascii", "ignore").decode("ascii")
    return normalized.strip().lower()

def _is_juniper(observer: Any) -> bool:
    return _normalize_observer(observer) == "juniper"

def _is_orion(observer: Any) -> bool:
    return _normalize_observer(observer) == "orion"

def _is_dense(entry: CollapseMirrorEntry) -> bool:
    if entry.is_causally_dense:
        return True
    score = None
    if entry.causal_density and isinstance(entry.causal_density, dict):
        score = entry.causal_density.get("score")
    elif entry.causal_density is not None:
        score = getattr(entry.causal_density, "score", None)
    return score is not None and float(score) >= 0.70

def _sanitize_fragment(raw: Any) -> str:
    """
    Turn things like 'llm.brain' or 'dream.synthesize' into safe local names
    like 'llm_brain' or 'dream_synthesize' for use in IRIs.
    """
    return "".join(c if c.isalnum() else "_" for c in str(raw))


def _entity_uri(text: str) -> URIRef:
    slug = _sanitize_fragment(str(text).strip().lower())
    return URIRef(f"http://conjourney.net/orion/entity/{slug}")

def build_triples_from_envelope(env_kind: str, payload: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Main entry point for converting typed payloads into RDF.
    Returns (nt_content, graph_name).
    """
    g = Graph()
    g.bind("cm", CM)
    g.bind("orion", ORION)

    # Dispatch based on kind or payload type
    try:
        # 1. RdfWriteRequest (Direct/Raw)
        if env_kind == "rdf.write.request":
            if isinstance(payload, dict):
                req = RdfWriteRequest.model_validate(payload)
            else:
                req = payload
            return _handle_write_request(g, req)

        # 2. cortex.worker.rdf_build (orion:rdf:worker) deliberately not
        # handled as of 2026-07-18: zero real producers found anywhere in
        # the repo despite channels.yaml claiming orion-cortex-exec as
        # producer -- the one other hit was self_study.py referencing the
        # channel name in a self-knowledge catalog string, not a publish
        # call. _handle_cortex_build (CognitiveStepExecution) removed with
        # it. Do not re-add without a real producer.

        # 3. Collapse Mirror (Raw)
        elif env_kind == "collapse.mirror.entry":
            if isinstance(payload, dict):
                entry = CollapseMirrorEntry.model_validate(payload)
            else:
                entry = payload

            observer = _normalize_observer(entry.observer)
            dense = _is_dense(entry)
            allow_write = False
            if _is_juniper(observer):
                allow_write = True
            elif _is_orion(observer):
                allow_write = dense
            else:
                allow_write = dense
                logger.warning("Unknown observer %s; applying dense-only RDF gating.", observer or "unknown")

            logger.debug(
                "Collapse RDF gate observer=%s dense=%s action=%s",
                observer or "unknown",
                dense,
                "write" if allow_write else "skip",
            )

            if not allow_write:
                return None, None

            # Collapse ID is usually in the envelope or we generate it?
            # The schema doesn't have ID, it has 'observer' etc.
            # Actually, `services/orion-collapse-mirror/app/schemas.py` shows it has no ID field in Pydantic.
            # But `models.py` (SQL) has ID.
            # We need a stable ID. We'll use a hash or look for one in extra.

            # For now, generate stable ID from content if missing
            subject_uri = URIRef(f"http://conjourney.net/event/{uuid.uuid4().hex}")
            _build_raw_collapse_graph(g, entry, subject_uri)
            attach_provenance(g, subject_uri, entry.observer)
            return g.serialize(format="nt"), "orion:collapse"

        # 4. Meta Tags (Enriched)
        elif env_kind in ("telemetry.meta_tags", "tags.enriched"):
            if isinstance(payload, dict):
                meta = MetaTagsPayload.model_validate(payload)
            else:
                meta = payload

            # enrichment_type == "chat_tagging" (chat/social-turn tag+entity
            # enrichment, published on the now-unsubscribed
            # orion:tags:chat:enriched channel) deliberately no longer
            # reaches this branch as of 2026-07-18: that data now lands in
            # FalkorDB only (orion-meta-tags' Phase 2 writer,
            # services/orion-meta-tags/README.md). The Fuseki chatTurn copy
            # was a second, redundant materialization of the same
            # entities/sentiment. Only generic collapse-mirror tagging
            # (enrichment_type == "tagging") can reach here now.
            subject_uri = URIRef(f"http://conjourney.net/event/{meta.collapse_id or meta.id}")
            _build_enrichment_graph(g, meta, subject_uri)
            attach_provenance(g, subject_uri, meta.service_name)
            return g.serialize(format="nt"), "orion:enrichment"

        # 5. Phase 3 autonomy artifact materialization -- deliberately
        # removed as of 2026-07-18: live Fuseki query found zero graphs
        # matching autonomy/identity/goals ever recorded, despite a real
        # producer existing. identity_snapshots already has a real,
        # actively-pruned Postgres store (orion-self-state-runtime's
        # SelfStateRuntimeStore) and goal proposals are already consumed
        # live by orion-substrate-runtime's goal_context_listener.py. Same
        # shape as the memory.drives.audit.v1 kill (2026-07-15) below --
        # do not re-add either without a real Falkor/Postgres-gap reason.
        # (memory.drives.audit.v1 deliberately absent: drive audits are
        # Postgres-only via orion-sql-writer as of 2026-07-15 — do not re-add.)

        # 6. Cognition Trace / Metacognitive Trace deliberately not handled
        # as of 2026-07-22: see settings.py's subscribe-list comment --
        # both kinds are Postgres-only via orion-sql-writer now
        # (`cognition_traces`, `orion_metacognitive_trace`), the Fuseki copy
        # was pure redundancy (~750 writes/6h). Handlers removed below
        # (_handle_cognition_trace / _handle_metacognitive_trace).

        # 7. Core Events (Legacy Fallback or "targets": ["rdf"])
        elif env_kind == "orion.event" or "targets" in str(payload):
             # Legacy dict handling
             if isinstance(payload, dict) and "rdf" in payload.get("targets", []):
                 return _legacy_dict_build(g, payload)

        # 8. world.pulse.graph.upsert.v1 deliberately not handled as of
        # 2026-07-18: WORLD_PULSE_GRAPH_ENABLED was false in the live .env
        # (fully inert), graph shape was 3 flat literals per digest item
        # with no edges to anything else, and world-pulse's richer
        # claim/event/entity emit channels already reach Postgres via
        # orion-sql-writer. Do not re-add without a real reason.

        # 9/10. Chat History (turn- and message-level) deliberately absent
        # (2026-07-17): orion:chat graph was found to cover only ~11-18% of
        # actual chat volume (299 RDF ChatTurn nodes vs 2579 Postgres
        # chat_message rows / 1699 chat_history_log rows, live-checked), with
        # almost none of the richer fields (model/intent/topic/tokens) the
        # builder supported ever actually populated. Both kinds are already
        # fully durable via orion-sql-writer (chat_message, chat_history_log).
        # See docs/superpowers/specs/2026-07-16-cypher-native-substrate-postgres-bus-split-design.md
        # open-questions section for the still-unexplained coverage gap
        # (why the RDF write path only ever captured a sliver of real chat
        # traffic) -- worth root-causing independent of this removal.

        elif env_kind == "social.turn.stored.v1":
            data = payload if isinstance(payload, dict) else payload.model_dump()
            return _handle_social_room_turn(g, data)

        else:
            logger.debug(f"Unknown kind {env_kind} for RDF builder")
            return None, None

    except Exception as e:
        logger.error(f"Error building triples for {env_kind}: {e}", exc_info=True)
        return None, None

    return None, None

def _handle_write_request(g: Graph, req: RdfWriteRequest) -> Tuple[str, str]:
    if req.triples:
        return req.triples, req.graph or "orion:default"

    # If explicit payload provided (e.g. wrapped content)
    # This is a stub for future "convert this generic dict to RDF" logic
    return None, None

def _handle_social_room_turn(g: Graph, payload: dict[str, Any]) -> Tuple[str, str]:
    turn = SocialRoomTurnStoredV1.model_validate(payload)
    turn_uri = URIRef(f"http://conjourney.net/orion/socialTurn/{_sanitize_fragment(turn.turn_id)}")
    session_val = turn.session_id or "unknown"
    g.add((turn_uri, RDF.type, ORION.SocialRoomTurn))
    g.add((turn_uri, ORION.artifactId, Literal(turn.turn_id, datatype=XSD.string)))
    g.add((turn_uri, ORION.sessionId, Literal(session_val, datatype=XSD.string)))
    g.add((turn_uri, ORION.profile, Literal(turn.profile, datatype=XSD.string)))
    # promptText/responseText/textContent omitted: full text lives in Postgres; IRI is the join key
    if turn.trace_verb:
        g.add((turn_uri, ORION.traceVerb, Literal(turn.trace_verb, datatype=XSD.string)))
    if turn.recall_profile:
        g.add((turn_uri, ORION.recallProfile, Literal(turn.recall_profile, datatype=XSD.string)))
    g.add((turn_uri, ORION.redactionLevel, Literal(turn.redaction.redaction_level, datatype=XSD.string)))
    g.add((turn_uri, ORION.recallSafe, Literal(turn.redaction.recall_safe, datatype=XSD.boolean)))
    if turn.grounding_state.continuity_anchor:
        g.add((turn_uri, ORION.continuityAnchor, Literal(turn.grounding_state.continuity_anchor)))
    for tag in turn.tags:
        g.add((turn_uri, ORION.hasTag, Literal(tag)))
    for item in turn.concept_evidence:
        evidence_uri = URIRef(f"{turn_uri}/evidence/{_sanitize_fragment(item.ref_id)}")
        g.add((evidence_uri, RDF.type, ORION.SocialConceptEvidence))
        g.add((evidence_uri, ORION.sourceRef, Literal(item.ref_id, datatype=XSD.string)))
        g.add((evidence_uri, ORION.sourceKind, Literal(item.source_kind, datatype=XSD.string)))
        g.add((evidence_uri, ORION.summaryText, Literal(item.summary)))
        g.add((evidence_uri, ORION.confidence, Literal(item.confidence, datatype=XSD.float)))
        g.add((turn_uri, ORION.supportedByEvidence, evidence_uri))
    attach_provenance(g, turn_uri, turn.source)
    return g.serialize(format="nt"), "orion:chat:social"


# === Chat History ===
#
# _handle_chat_turn / _handle_chat_message deliberately removed (2026-07-17):
# see the dispatch-site comment above (kinds "chat.history" /
# "chat.history.message.v1") for why. Both durably live in Postgres via
# orion-sql-writer; the RDF copy covered only ~11-18% of real chat volume.


# ================================================================
# --- HELPERS (Adapted) ------------------------------------------
# ================================================================

def _build_raw_collapse_graph(g: Graph, entry: CollapseMirrorEntry, subject: URIRef):
    g.add((subject, RDF.type, CM.CollapseEvent))
    g.add((subject, CM.observer, Literal(entry.observer)))
    g.add((subject, CM.trigger, Literal(entry.trigger)))
    g.add((subject, CM.summary, Literal(entry.summary)))
    if entry.timestamp:
        g.add((subject, CM.timestamp, Literal(entry.timestamp)))

    # Handle list or string state
    state = entry.observer_state
    if isinstance(state, list):
        state = ",".join(str(s) for s in state)
    g.add((subject, CM.observerState, Literal(state)))

def _build_enrichment_graph(
    g: Graph,
    meta: MetaTagsPayload,
    subject: URIRef,
) -> None:
    # Link back to original event if known, else assume subject IS the event
    # Ideally, subject is the event URI.

    tags = meta.tags or []
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        g.add((subject, CM.hasTag, Literal(tag)))

    entities = meta.entities or []
    if isinstance(entities, str):
        entities = [entities]
    for ent in entities:
        entity_text = None
        entity_label = None
        if isinstance(ent, str):
            entity_text = ent
            entity_label = ent
        elif isinstance(ent, dict):
            val = ent.get("value") or ent.get("name") or ent.get("text")
            typ = ent.get("type") or ent.get("label")
            if val and typ:
                entity_text = str(val)
                entity_label = f"{val} ({typ})"
            elif val:
                entity_text = str(val)
                entity_label = str(val)

        if not entity_text:
            continue

        g.add((subject, CM.hasEntity, Literal(entity_label or entity_text)))

        entity_uri = _entity_uri(entity_text)
        g.add((entity_uri, RDF.type, ORION.Entity))
        g.add((entity_uri, RDFS.label, Literal(entity_label or entity_text)))

        timestamp_val = meta.timestamp or meta.ts
        subject_id = _sanitize_fragment(str(subject).rsplit("/", 1)[-1])
        mention_seed = f"{subject}|{entity_text}|{timestamp_val}"
        mention_hash = hashlib.sha256(mention_seed.encode("utf-8")).hexdigest()[:16]
        mention_uri = URIRef(f"http://conjourney.net/orion/mention/{subject_id}/{mention_hash}")
        g.add((mention_uri, RDF.type, ORION.Mention))
        g.add((mention_uri, ORION.subject, subject))
        g.add((mention_uri, ORION.entity, entity_uri))

        confidence = meta.salience if meta.salience is not None else 0.6
        g.add((mention_uri, ORION.confidence, Literal(confidence, datatype=XSD.float)))
        if timestamp_val:
            g.add((mention_uri, ORION.timestamp, Literal(str(timestamp_val), datatype=XSD.string)))
        g.add((mention_uri, PROV.wasGeneratedBy, Literal(meta.service_name)))

    # Provenance for enrichment
    enrichment_id = meta.id or meta.collapse_id or str(uuid.uuid4())
    enrich_id = URIRef(f"http://conjourney.net/enrichment/{enrichment_id}")
    g.add((enrich_id, RDF.type, ORION.Enrichment))
    g.add((enrich_id, ORION.enriches, subject))
    g.add((enrich_id, ORION.processedBy, Literal(meta.service_name)))
    if meta.service_version:
        g.add((enrich_id, ORION.serviceVersion, Literal(meta.service_version)))
    if meta.node:
        g.add((enrich_id, ORION.serviceNode, Literal(meta.node)))
    if meta.correlation_id:
        g.add((enrich_id, ORION.correlationId, Literal(meta.correlation_id)))
    if meta.source_message_id:
        g.add((enrich_id, ORION.sourceMessageId, Literal(meta.source_message_id)))
    if meta.enrichment_type:
        g.add((enrich_id, ORION.enrichmentType, Literal(meta.enrichment_type)))
    if meta.salience is not None:
        g.add((enrich_id, ORION.salience, Literal(meta.salience, datatype=XSD.float)))
    if meta.collapse_id or meta.id:
        g.add((enrich_id, ORION.collapseId, Literal(meta.collapse_id or meta.id)))
    timestamp_val = meta.ts or meta.timestamp
    if timestamp_val:
        g.add((enrich_id, ORION.timestamp, Literal(str(timestamp_val), datatype=XSD.string)))


def _legacy_dict_build(g: Graph, event: dict) -> Tuple[str, str]:
    # Keep the old logic for fallback
    event_id = event.get("id")
    if not event_id:
        return None, None
    subject = URIRef(f"http://conjourney.net/event/{event_id}")

    # Reuse helpers if keys match
    # ... (simplified for brevity, main flow uses typed above)
    return None, None
