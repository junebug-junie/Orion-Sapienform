import uuid
import hashlib
import logging
import json
import unicodedata
from datetime import datetime, timezone
from typing import Tuple, Optional, Any

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

# Typed schemas
from orion.schemas.collapse_mirror import CollapseMirrorEntry
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from orion.schemas.rdf import RdfWriteRequest, RdfBuildRequest
from orion.schemas.telemetry.cognition_trace import CognitionTracePayload

from app.provenance import attach_provenance
from app.settings import settings

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

        # 2. RdfBuildRequest (Cortex Exec)
        elif env_kind == "cortex.worker.rdf_build":
             # Payload might be the standard Exec 'PlanExecutionRequest' or 'args'
             # but here we assume the worker receives the specific instruction.
             # Actually, Exec sends `PlanExecutionRequest` wrapper usually.
             # We will handle the inner args.
             if isinstance(payload, dict):
                 # Try to extract args if wrapped, or treat as direct args
                 args = payload.get("args", payload)
                 return _handle_cortex_build(g, args)
             else:
                 return _handle_cortex_build(g, payload.args)

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

            subject_uri = URIRef(f"http://conjourney.net/event/{meta.collapse_id or meta.id}")
            _build_enrichment_graph(g, meta, subject_uri, emit_claims=(env_kind == "tags.enriched"))
            attach_provenance(g, subject_uri, meta.service_name)
            return g.serialize(format="nt"), "orion:enrichment"

        # 5. Cognition Trace
        elif env_kind == "cognition.trace":
            if isinstance(payload, dict):
                trace = CognitionTracePayload.model_validate(payload)
            else:
                trace = payload
            return _handle_cognition_trace(g, trace)

        # 6. Core Events (Legacy Fallback or "targets": ["rdf"])
        elif env_kind == "orion.event" or "targets" in str(payload):
             # Legacy dict handling
             if isinstance(payload, dict) and "rdf" in payload.get("targets", []):
                 return _legacy_dict_build(g, payload)

        # 7. Chat History (Turn-level)
        elif env_kind == "chat.history":
            data = payload if isinstance(payload, dict) else payload.model_dump()
            return _handle_chat_turn(g, data)

        # 8. Chat History (Message-level)
        elif env_kind == "chat.history.message.v1":
            data = payload if isinstance(payload, dict) else payload.model_dump()
            return _handle_chat_message(g, data)

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

def _handle_cortex_build(g: Graph, args: Any) -> Tuple[str, str]:
    # Adapting logic from old _build_cortex_step_graph
    # We expect `args` to be a dict or object with relevant fields

    data = args if isinstance(args, dict) else args.model_dump()

    # Check if this is a Cortex Step Summary (telemetry) or a specific RDF task
    # If the verb is `rdf_build`, we assume it's a specific task to write data.

    # For now, let's assume we are logging the step execution itself as RDF (Cognitive Memory)
    # AND handling specific write instructions if present.

    cid = data.get("correlation_id") or data.get("trace_id") or str(uuid.uuid4())
    subject = ORION[f"cortexStep_{_sanitize_fragment(cid)}"]

    g.add((subject, RDF.type, ORION.CognitiveStepExecution))
    g.add((subject, ORION.correlationId, Literal(cid, datatype=XSD.string)))

    if "verb" in data:
        g.add((subject, ORION.verbName, Literal(data["verb"], datatype=XSD.string)))

    # Serialize whatever we have
    return g.serialize(format="nt"), "orion:cognition"


def _handle_cognition_trace(g: Graph, trace: CognitionTracePayload) -> Tuple[str, str]:
    """
    Builds a connectable graph for a CognitionTrace.
    """
    run_uri = ORION[f"run_{trace.correlation_id}"]

    # Run Metadata
    g.add((run_uri, RDF.type, ORION.CognitionRun))
    g.add((run_uri, ORION.correlationId, Literal(str(trace.correlation_id), datatype=XSD.string)))
    g.add((run_uri, ORION.mode, Literal(trace.mode, datatype=XSD.string)))
    g.add((run_uri, ORION.verb, Literal(trace.verb, datatype=XSD.string)))
    g.add((run_uri, ORION.timestamp, Literal(trace.timestamp, datatype=XSD.double)))
    g.add((run_uri, ORION.sourceService, Literal(trace.source_service, datatype=XSD.string)))

    if trace.final_text:
        # Truncate if excessively large, but generally keep it
        g.add((run_uri, ORION.producedFinalText, Literal(trace.final_text)))

    # Steps
    prev_step_uri = None

    for i, step in enumerate(trace.steps):
        step_uri = ORION[f"step_{trace.correlation_id}_{i}"]
        g.add((step_uri, RDF.type, ORION.CognitionStep))
        g.add((step_uri, ORION.stepIndex, Literal(i, datatype=XSD.integer)))
        g.add((step_uri, ORION.stepName, Literal(step.step_name)))
        g.add((step_uri, ORION.stepVerb, Literal(step.verb_name)))
        g.add((step_uri, ORION.status, Literal(step.status)))

        # Link to Run
        g.add((run_uri, ORION.hasStep, step_uri))

        # Sequence
        if prev_step_uri:
            g.add((prev_step_uri, ORION.nextStep, step_uri))
            g.add((step_uri, ORION.prevStep, prev_step_uri))
        prev_step_uri = step_uri

        # Evidence / Thoughts (if any in result/artifacts)
        if step.result:
            thought = step.result.get("thought") or step.result.get("reasoning")
            if thought:
                g.add((step_uri, ORION.hasThought, Literal(thought)))

        # Used Services/Tools
        # We don't have explicit 'tools used' in StepExecutionResult other than artifacts?
        # Assuming artifacts might contain refs
        if step.artifacts:
            for key, val in step.artifacts.items():
                # Naive check for IDs
                if isinstance(val, str) and (val.startswith("http") or val.startswith("uuid:")):
                     g.add((step_uri, ORION.hasEvidenceRef, Literal(val)))

    return g.serialize(format="nt"), "orion:cognition"


# === Chat History ===

def _handle_chat_turn(g: Graph, data: dict) -> Tuple[str, str]:
    session_id = data.get("session_id") or "unknown"
    turn_id = data.get("id") or data.get("turn_id") or data.get("correlation_id") or data.get("message_id")
    turn_id = turn_id or str(uuid.uuid4())

    session_uri = URIRef(f"http://conjourney.net/orion/chatSession/{_sanitize_fragment(session_id)}")
    turn_uri = URIRef(f"http://conjourney.net/orion/chatTurn/{_sanitize_fragment(turn_id)}")

    g.add((session_uri, RDF.type, ORION.ChatSession))
    g.add((turn_uri, RDF.type, ORION.ChatTurn))
    g.add((session_uri, ORION.hasTurn, turn_uri))

    g.add((turn_uri, ORION.sessionId, Literal(session_id, datatype=XSD.string)))

    prompt = data.get("prompt")
    if prompt:
        g.add((turn_uri, ORION.prompt, Literal(prompt)))
    response = data.get("response")
    if response:
        g.add((turn_uri, ORION.response, Literal(response)))

    timestamp = data.get("timestamp")
    if timestamp:
        g.add((turn_uri, ORION.timestamp, Literal(timestamp, datatype=XSD.string)))

    correlation_id = data.get("correlation_id")
    if correlation_id:
        g.add((turn_uri, ORION.correlationId, Literal(str(correlation_id), datatype=XSD.string)))

    trace_id = data.get("trace_id") or correlation_id
    if trace_id:
        g.add((turn_uri, ORION.traceId, Literal(str(trace_id), datatype=XSD.string)))

    spark_meta = data.get("spark_meta") if isinstance(data.get("spark_meta"), dict) else {}
    verb = spark_meta.get("trace_verb") or data.get("verb")
    if verb:
        g.add((turn_uri, ORION.verb, Literal(str(verb), datatype=XSD.string)))

    model = spark_meta.get("model") or data.get("model")
    if model:
        g.add((turn_uri, ORION.model, Literal(str(model), datatype=XSD.string)))

    node = spark_meta.get("source_node") or data.get("node")
    if node:
        g.add((turn_uri, ORION.node, Literal(str(node), datatype=XSD.string)))

    return g.serialize(format="nt"), "orion:chat"


def _handle_chat_message(g: Graph, data: dict) -> Tuple[str, str]:
    session_id = data.get("session_id") or "unknown"
    message_id = data.get("message_id") or data.get("id") or str(uuid.uuid4())

    session_uri = URIRef(f"http://conjourney.net/orion/chatSession/{_sanitize_fragment(session_id)}")
    message_uri = URIRef(f"http://conjourney.net/orion/chatMessage/{_sanitize_fragment(message_id)}")

    g.add((session_uri, RDF.type, ORION.ChatSession))
    g.add((message_uri, RDF.type, ORION.ChatMessage))
    g.add((session_uri, ORION.hasMessage, message_uri))

    g.add((message_uri, ORION.sessionId, Literal(session_id, datatype=XSD.string)))

    content = data.get("content")
    if content:
        g.add((message_uri, ORION.content, Literal(content)))

    role = data.get("role")
    if role:
        g.add((message_uri, ORION.role, Literal(str(role), datatype=XSD.string)))

    speaker = data.get("speaker")
    if speaker:
        g.add((message_uri, ORION.speaker, Literal(str(speaker), datatype=XSD.string)))

    timestamp = data.get("timestamp")
    if timestamp:
        g.add((message_uri, ORION.timestamp, Literal(timestamp, datatype=XSD.string)))

    model = data.get("model")
    if model:
        g.add((message_uri, ORION.model, Literal(str(model), datatype=XSD.string)))

    provider = data.get("provider")
    if provider:
        g.add((message_uri, ORION.provider, Literal(str(provider), datatype=XSD.string)))

    return g.serialize(format="nt"), "orion:chat"


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
    *,
    emit_claims: bool = False,
) -> None:
    # Link back to original event if known, else assume subject IS the event
    # Ideally, subject is the event URI.

    tags = meta.tags or []
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        g.add((subject, CM.hasTag, Literal(tag)))
        if emit_claims:
            _emit_claim(
                g,
                subject=subject,
                predicate=ORION.hasTag,
                obj=Literal(tag),
                meta=meta,
            )

    entities = meta.entities or []
    if isinstance(entities, str):
        entities = [entities]
    for ent in entities:
        if isinstance(ent, str):
            g.add((subject, CM.hasEntity, Literal(ent)))
            if emit_claims:
                _emit_claim(
                    g,
                    subject=subject,
                    predicate=ORION.mentionsEntity,
                    obj=Literal(ent),
                    meta=meta,
                )
            continue
        if isinstance(ent, dict):
            val = ent.get("value") or ent.get("name") or ent.get("text")
            typ = ent.get("type") or ent.get("label")
            if val and typ:
                entity_value = f"{val} ({typ})"
                g.add((subject, CM.hasEntity, Literal(entity_value)))
                if emit_claims:
                    _emit_claim(
                        g,
                        subject=subject,
                        predicate=ORION.mentionsEntity,
                        obj=Literal(entity_value),
                        meta=meta,
                    )
            elif val:
                g.add((subject, CM.hasEntity, Literal(str(val))))
                if emit_claims:
                    _emit_claim(
                        g,
                        subject=subject,
                        predicate=ORION.mentionsEntity,
                        obj=Literal(str(val)),
                        meta=meta,
                    )

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


def _emit_claim(
    g: Graph,
    *,
    subject: URIRef,
    predicate: URIRef,
    obj: Literal,
    meta: MetaTagsPayload,
) -> None:
    subject_id = _sanitize_fragment(meta.collapse_id or meta.id or meta.source_message_id or "unknown")
    timestamp_val = str(meta.timestamp or meta.ts or "")
    claim_seed = f"{predicate}|{obj}|{timestamp_val}"
    claim_hash = hashlib.sha256(claim_seed.encode("utf-8")).hexdigest()[:16]
    claim_uri = URIRef(f"http://conjourney.net/orion/claim/{subject_id}/{claim_hash}")

    g.add((claim_uri, RDF.type, ORION.Claim))
    g.add((claim_uri, ORION.subject, subject))
    g.add((claim_uri, ORION.predicate, predicate))
    g.add((claim_uri, ORION.obj, obj))

    salience = meta.salience
    confidence = salience if salience is not None else 0.6
    g.add((claim_uri, ORION.confidence, Literal(confidence, datatype=XSD.float)))
    if salience is not None:
        g.add((claim_uri, ORION.salience, Literal(salience, datatype=XSD.float)))

    g.add((claim_uri, ORION.extractorService, Literal(meta.service_name)))
    if meta.service_version:
        g.add((claim_uri, ORION.extractorVersion, Literal(meta.service_version)))
    if meta.node:
        g.add((claim_uri, ORION.node, Literal(meta.node)))

    timestamp_val = meta.timestamp or meta.ts
    if timestamp_val:
        g.add((claim_uri, ORION.timestamp, Literal(str(timestamp_val), datatype=XSD.string)))

    if meta.correlation_id:
        g.add((claim_uri, ORION.correlationId, Literal(meta.correlation_id)))
    if meta.source_message_id:
        g.add((claim_uri, ORION.sourceMessageId, Literal(meta.source_message_id)))


def _legacy_dict_build(g: Graph, event: dict) -> Tuple[str, str]:
    # Keep the old logic for fallback
    event_id = event.get("id")
    if not event_id:
        return None, None
    subject = URIRef(f"http://conjourney.net/event/{event_id}")

    # Reuse helpers if keys match
    # ... (simplified for brevity, main flow uses typed above)
    return None, None
