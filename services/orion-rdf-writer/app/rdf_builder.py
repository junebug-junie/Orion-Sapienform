import uuid
import logging
import json
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
        elif env_kind == "telemetry.meta_tags":
            if isinstance(payload, dict):
                meta = MetaTagsPayload.model_validate(payload)
            else:
                meta = payload

            subject_uri = URIRef(f"http://conjourney.net/event/{meta.collapse_id or meta.id}")
            _build_enrichment_graph(g, meta, subject_uri)
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

def _build_enrichment_graph(g: Graph, meta: MetaTagsPayload, subject: URIRef):
    # Link back to original event if known, else assume subject IS the event
    # Ideally, subject is the event URI.

    for tag in meta.tags:
        g.add((subject, CM.hasTag, Literal(tag)))

    for ent in meta.entities:
        val = ent.get("value")
        typ = ent.get("type")
        if val and typ:
             g.add((subject, CM.hasEntity, Literal(f"{val} ({typ})")))

    # Provenance for enrichment
    enrich_id = URIRef(f"http://conjourney.net/enrichment/{meta.id}")
    g.add((enrich_id, RDF.type, ORION.Enrichment))
    g.add((enrich_id, ORION.enriches, subject))
    g.add((enrich_id, ORION.processedBy, Literal(meta.service_name)))
    g.add((enrich_id, ORION.salience, Literal(meta.salience, datatype=XSD.float)))


def _legacy_dict_build(g: Graph, event: dict) -> Tuple[str, str]:
    # Keep the old logic for fallback
    event_id = event.get("id")
    if not event_id:
        return None, None
    subject = URIRef(f"http://conjourney.net/event/{event_id}")

    # Reuse helpers if keys match
    # ... (simplified for brevity, main flow uses typed above)
    return None, None
