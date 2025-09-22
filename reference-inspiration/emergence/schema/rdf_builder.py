# schema/rdf_builder.py
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF
from datetime import datetime
import hashlib
from schema.context import MemoryContext

CJ = Namespace("http://conjourney.net/schema#")

class RDFBuilder:
    def __init__(self, base_uri="http://conjourney.net/event"):
        self.base_uri = base_uri

    def build_graph(self, entry: dict, agent_id="unknown", context: MemoryContext = None) -> Graph:
        """
        Convert a memory entry into an RDF graph.
        Includes metadata like source (agent or sensor), timestamps,
        memory_id, and contextual RDF if provided.
        """
        g = Graph()
        g.bind("cj", CJ)

        uid = entry.get("memory_id") or self._entry_id(entry)
        subject = URIRef(f"{self.base_uri}/{uid}")

        g.add((subject, RDF.type, CJ.MemoryEvent))
        g.add((subject, CJ.memory_id, Literal(uid)))
        g.add((subject, CJ.timestamp, Literal(entry.get("timestamp", datetime.utcnow().isoformat()))))
        g.add((subject, CJ.source, Literal(entry.get("observer", agent_id))))

        for key, value in entry.items():
            if key in ["timestamp", "observer", "memory_id", "context"]:
                continue
            g.add((subject, CJ[key], Literal(value)))

        if context:
            for key, val in context.to_dict().items():
                g.add((subject, CJ[key], Literal(val)))

        return g

    def _entry_id(self, entry: dict) -> str:
        base = entry.get("observer", "") + entry.get("timestamp", "") + entry.get("summary", "")
        return hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]

