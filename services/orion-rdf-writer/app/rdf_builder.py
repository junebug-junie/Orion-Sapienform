import uuid
import logging
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, XSD

from app.provenance import attach_provenance
from app.settings import settings

ORION = Namespace("http://conjourney.net/orion#")
CM = Namespace("http://orion.ai/collapse#") # For collapse-specific terms

def build_triples(event: dict) -> tuple[str | None, str | None]:
    """
    Intelligently builds RDF triples based on the structure of the incoming event.
    It routes to a different builder function depending on the event type.
    """
    g = Graph()
    g.bind("cm", CM)
    g.bind("orion", ORION)
    
    event_id = event.get("id")
    if not event_id:
        logging.warning("Event is missing an 'id', cannot generate triples.")
        return None, None

    # --- Routing Logic ---
    # Check if this is an enrichment event from the meta-writer/meta-tags
    if "enrichment_type" in event or "processed_by" in event:
        g = _build_enrichment_graph(g, event)
    # Otherwise, assume it's a raw collapse event
    else:
        g = _build_raw_collapse_graph(g, event)

    if not len(g):
        logging.warning(f"No triples were generated for event {event_id}. Check event structure and builder logic.")
        return None, None

    # Attach provenance to the final graph
    subject_uri = URIRef(f"http://conjourney.net/event/{event_id}")
    observer = event.get("observer", "system")
    graph_name = attach_provenance(g, subject_uri, observer)

    return g.serialize(format="nt"), graph_name


def _build_raw_collapse_graph(g: Graph, event: dict) -> Graph:
    """Builds triples for a base collapse event."""
    subject = URIRef(f"{CM}{event.get('id')}")
    g.add((subject, RDF.type, CM.Collapse))
    g.add((subject, CM.id, Literal(event.get('id'), datatype=XSD.string)))

    # Add all key-value pairs from the raw collapse event
    for key, val in event.items():
        if val is None or key in ['id', 'service_name']:
            continue
        if isinstance(val, list):
            val = ", ".join(map(str, val))
        g.add((subject, URIRef(str(CM) + key), Literal(str(val), datatype=XSD.string)))
    
    return g


def _build_enrichment_graph(g: Graph, event: dict) -> Graph:
    """Builds triples for an enrichment event, linking them to the original collapse."""
    collapse_id = event.get("collapse_id") or event.get("id")
    subject = URIRef(f"{CM}{collapse_id}")

    # Add tags
    for tag in event.get("tags", []):
        g.add((subject, CM.hasTag, Literal(tag, datatype=XSD.string)))

    # Add entities
    for entity in event.get("entities", []):
        if entity.get("value") and entity.get("type"):
            g.add((subject, CM.hasEntity, Literal(f"{entity['value']} ({entity['type']})", datatype=XSD.string)))
            
    # You could also add provenance about the enrichment service itself
    enrichment_id = URIRef(f"http://conjourney.net/enrichment/{uuid.uuid4().hex}")
    g.add((enrichment_id, RDF.type, ORION.Enrichment))
    g.add((enrichment_id, ORION.enriches, subject))
    g.add((enrichment_id, ORION.processedBy, Literal(event.get("processed_by"))))
            
    return g
