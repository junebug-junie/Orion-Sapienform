import uuid
import logging
import json
import unicodedata
from datetime import datetime, timezone
from typing import Tuple, Optional, Any

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

# Typed schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.rdf import RdfWriteRequest, RdfBuildRequest
from orion.schemas.social_chat import SocialRoomTurnStoredV1

from app.provenance import attach_provenance

ORION = Namespace("http://conjourney.net/orion#")
CM = Namespace("http://orion.ai/collapse#")

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

        # 4. Meta Tags (Enriched) -- "telemetry.meta_tags" / "tags.enriched"
        # deliberately removed as of 2026-07-22: this was the Fuseki copy of
        # generic collapse-mirror tag/entity enrichment (chat/social-turn
        # tagging already stopped reaching here on 2026-07-18). Falkor write
        # shipped additively (PR #1271), 68/68 historical rows backfilled
        # (PR #1273), one real live event verified landing in
        # Postgres/Fuseki/Falkor simultaneously, then this branch and its
        # `_build_enrichment_graph` helper removed. orion-meta-tags still
        # publishes to orion:tags:enriched -- orion-sql-writer is also
        # subscribed and materializes it into Postgres `collapse_enrichment`,
        # which orion-recall and orion-dream genuinely query; only this
        # service's Fuseki subscription was redundant. See
        # docs/superpowers/specs/2026-07-22-tags-enriched-fuseki-kill-spec.md.
        # Do not re-add without a real Falkor-gap reason.

        # 5. Phase 3 autonomy artifact materialization -- deliberately
        # removed as of 2026-07-18: live Fuseki query found zero graphs
        # matching autonomy/identity/goals ever recorded, despite a real
        # producer existing. identity_snapshots had a real, actively-pruned
        # Postgres store (orion-self-state-runtime's SelfStateRuntimeStore)
        # at the time; that service is deleted as of 2026-07-22 (SelfStateV1
        # burn), leaving the table with zero producer (flagged as a
        # follow-up). Goal proposals are already consumed live by
        # orion-substrate-runtime's goal_context_listener.py. Same
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

def _legacy_dict_build(g: Graph, event: dict) -> Tuple[str, str]:
    # Keep the old logic for fallback
    event_id = event.get("id")
    if not event_id:
        return None, None
    subject = URIRef(f"http://conjourney.net/event/{event_id}")

    # Reuse helpers if keys match
    # ... (simplified for brevity, main flow uses typed above)
    return None, None
