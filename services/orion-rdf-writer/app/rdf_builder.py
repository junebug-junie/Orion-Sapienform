import uuid
import logging
import json
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from app.provenance import attach_provenance
from app.settings import settings

ORION = Namespace("http://conjourney.net/orion#")
CM = Namespace("http://orion.ai/collapse#")


def _sanitize_fragment(raw: str) -> str:
    """
    Turn things like 'llm.brain' or 'dream.synthesize' into safe local names
    like 'llm_brain' or 'dream_synthesize' for use in IRIs.
    """
    return "".join(c if c.isalnum() else "_" for c in str(raw))


def build_triples(event: dict) -> tuple[str | None, str | None]:
    g = Graph()
    g.bind("cm", CM)
    g.bind("orion", ORION)

    event_type = event.get("event")
    subject_uri = None

    if event_type == "cortex_step_summary":
        g, subject_uri = _build_cortex_step_graph(g, event)
    else:
        event_id = event.get("id")
        if not event_id:
            logging.warning("Event is missing an 'id', cannot generate collapse/enrichment triples.")
            return None, None

        subject_uri = URIRef(f"http://conjourney.net/event/{event_id}")

        if "enrichment_type" in event or "processed_by" in event:
            g = _build_enrichment_graph(g, event, subject_uri)
        else:
            g = _build_raw_collapse_graph(g, event, subject_uri)

    if subject_uri is None or not len(g):
        logging.warning(f"No triples were generated for event type={event_type!r}.")
        return None, None

    observer = event.get("observer") or event.get("node") or "system"
    graph_name = attach_provenance(g, subject_uri, observer)

    return g.serialize(format="nt"), graph_name


# ================================================================
# --- 5. ENRICHMENT GRAPH (UNCHANGED) ----------------------------
# ================================================================
def _build_enrichment_graph(g: Graph, event: dict, subject: URIRef) -> Graph:
    """Builds triples for an enrichment event, linking them to the original collapse."""

    # Add tags
    for tag in event.get("tags", []):
        g.add((subject, CM.hasTag, Literal(tag, datatype=XSD.string)))

    # Add entities
    for entity in event.get("entities", []):
        if entity.get("value") and entity.get("type"):
            g.add(
                (
                    subject,
                    CM.hasEntity,
                    Literal(f"{entity['value']} ({entity['type']})", datatype=XSD.string),
                )
            )

    # Separate Enrichment node
    enrichment_id = URIRef(f"http://conjourney.net/enrichment/{uuid.uuid4().hex}")
    g.add((enrichment_id, RDF.type, ORION.Enrichment))

    # Links the new enrichment node (enrichment_id) to the one, true event node (subject)
    g.add((enrichment_id, ORION.enriches, subject))
    g.add((enrichment_id, ORION.processedBy, Literal(event.get("processed_by"))))

    return g


# ================================================================
# --- 6. CORTEX STEP EXECUTION GRAPH (NEW) -----------------------
# ================================================================
def _build_cortex_step_graph(g: Graph, event: dict) -> tuple[Graph, URIRef | None]:
    cid = event.get("correlation_id")
    if not cid:
        logging.warning("Cortex event is missing 'correlation_id'; cannot generate triples.")
        return g, None

    subject = ORION[f"cortexStep_{_sanitize_fragment(cid)}"]

    g.add((subject, RDF.type, ORION.CognitiveStepExecution))
    g.add((subject, ORION.correlationId, Literal(cid, datatype=XSD.string)))

    verb = event.get("verb")
    if verb:
        vf = _sanitize_fragment(verb)
        g.add((subject, ORION.verb, ORION[f"verb_{vf}"]))
        g.add((subject, ORION.verbName, Literal(verb, datatype=XSD.string)))

    step = event.get("step")
    if step:
        sf = _sanitize_fragment(step)
        g.add((subject, ORION.step, ORION[f"step_{sf}"]))
        g.add((subject, ORION.stepName, Literal(step, datatype=XSD.string)))

    for svc in event.get("services", []):
        sf = _sanitize_fragment(svc)
        g.add((subject, ORION.service, ORION[f"service_{sf}"]))
        g.add((subject, ORION.serviceName, Literal(str(svc), datatype=XSD.string)))

    node = event.get("node")
    if node:
        nf = _sanitize_fragment(node)
        g.add((subject, ORION.originNode, ORION[f"node_{nf}"]))
        g.add((subject, ORION.originNodeName, Literal(str(node), datatype=XSD.string)))

    status = event.get("status")
    if status:
        g.add((subject, ORION.status, Literal(status, datatype=XSD.string)))

    latency = event.get("latency_ms")
    if latency is not None:
        try:
            g.add((subject, ORION.latencyMs, Literal(int(latency), datatype=XSD.int)))
        except Exception:
            pass

    ts = event.get("timestamp")
    if ts is not None:
        try:
            g.add((subject, ORION.timestampEpoch, Literal(float(ts), datatype=XSD.double)))
        except Exception:
            pass

    args = event.get("args") or {}
    ctx = event.get("context") or {}
    preview = event.get("result_preview") or {}

    g.add((subject, ORION.argsJson, Literal(json.dumps(args, separators=(",", ":")), datatype=XSD.string)))
    g.add((subject, ORION.contextJson, Literal(json.dumps(ctx, separators=(",", ":")), datatype=XSD.string)))
    g.add((subject, ORION.resultPreviewJson, Literal(json.dumps(preview, separators=(",", ":")), datatype=XSD.string)))

    return g, subject
