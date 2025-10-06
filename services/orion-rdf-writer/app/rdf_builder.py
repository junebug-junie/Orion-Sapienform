from rdflib import Graph, Namespace, URIRef

import uuid
import logging

from app.provenance import attach_provenance


ORION = Namespace("http://conjourney.net/orion#")


def build_triples(event: dict) -> tuple[str | None, str | None]:
    g = Graph()
    event_id = URIRef(f"http://conjourney.net/event/{event.get('id', uuid.uuid4())}")

    for entity in event.get("mentions", []):
        g.add((event_id, ORION.mentions, URIRef(f"http://conjourney.net/entity/{entity}")))

    for topic in event.get("relatesTo", []):
        g.add((event_id, ORION.relatesTo, URIRef(f"http://conjourney.net/topic/{topic}")))

    # --- ADD THIS CHECK ---
    if not len(g):
        logging.warning(f"No triples generated for event {event.get('id')}. Check event structure.")
        return None, None
    # --- END CHECK ---

    observer = event.get("observer", "system")
    graph_name = attach_provenance(g, event_id, observer)

    return g.serialize(format="nt"), graph_name
