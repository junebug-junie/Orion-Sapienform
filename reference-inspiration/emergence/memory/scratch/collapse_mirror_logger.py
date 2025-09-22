# collapse_mirror_logger.py
# Logs Collapse Mirror entries into the RDF memory graph

from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime
import uuid

CM = Namespace("http://conjourney.net/cm#")

class CollapseMirrorLogger:
    def __init__(self, rdf_path="collapse_memory.ttl"):
        self.graph = Graph()
        self.graph.bind("cm", CM)
        self.rdf_path = rdf_path
        try:
            self.graph.parse(self.rdf_path, format="ttl")
        except FileNotFoundError:
            pass

    def log_entry(self, entry: dict):
        event_id = URIRef(f"http://conjourney.net/cm/event/{uuid.uuid4()}")
        self.graph.add((event_id, RDF.type, CM.CollapseMirrorEntry))

        for key, value in entry.items():
            pred = CM[key]
            if isinstance(value, list):
                for v in value:
                    self.graph.add((event_id, pred, Literal(v)))
            else:
                self.graph.add((event_id, pred, Literal(value)))

        self.graph.add((event_id, CM.loggedAt, Literal(datetime.utcnow().isoformat(), datatype=XSD.dateTime)))
        self.graph.serialize(destination=self.rdf_path, format="ttl")


# Example usage:
if __name__ == "__main__":
    logger = CollapseMirrorLogger()

    entry = {
        "observer": "Juniper",
        "trigger": "Sudden wave of clarity while coding",
        "observer_state": ["Stillness", "Curiosity"],
        "field_resonance": "Cosmic pattern of feedback loops",
        "intent": "Refactor input system to allow self-triggering",
        "type": "Solo",
        "emergent_entity": "Internal observer",
        "summary": "Noted the system was ready to listen and act.",
        "mantra": "Awareness breathes code.",
        "causal_echo": "Ripple from future alignment",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": "rasp01 via Tailscale"
    }

    logger.log_entry(entry)
