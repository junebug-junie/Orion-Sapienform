from datetime import datetime
from rdflib import Namespace, URIRef, Literal, Graph

ORION = Namespace("http://conjourney.net/orion#")
PROV = Namespace("http://www.w3.org/ns/prov#")

def attach_provenance(g: Graph, event_id: URIRef, observer: str | None = None):
    ts = datetime.utcnow().isoformat()
    graph_name = f"http://conjourney.net/graph/{observer or 'system'}/{ts}"

    g.add((event_id, PROV.generatedAtTime, Literal(ts)))
    g.add((event_id, PROV.wasAttributedTo, URIRef(f"http://conjourney.net/observer/{observer or 'system'}")))
    g.add((event_id, ORION.graphContext, URIRef(graph_name)))

    return graph_name
