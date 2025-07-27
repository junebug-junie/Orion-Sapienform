# cognition/introspection/memory_evolver.py
from rdflib import Graph, URIRef, Literal, Namespace
from datetime import datetime
from emergence.schema.context import MemoryContext
from emergence.schema.rdf_builder import RDFBuilder
from emergence.memory.tier3_ontology.ttl_log import TTLLogger
from emergence.memory.interface import write_to_memory
from emergence.memory.tier3_ontology.rdf_parser import RDFParser
from emergence.memory.tier3_ontology.ttl_auditor import load_latest_ttls
import hashlib

CJ = Namespace("http://conjourney.net/schema#")

class MemoryEvolver:
    def __init__(self, name="memory-evolver"):
        self.name = name
        self.builder = RDFBuilder()
        self.logger = TTLLogger()

    def detect_revision(self, new_entry: dict) -> dict:
        """
        Scan TTLs for similar memory and emit a revision if semantic shift is detected.
        """
        graphs = load_latest_ttls(n=50)
        combined = Graph()
        for g in graphs:
            combined += g
        parser = RDFParser(combined)

        candidates = parser.get_events_by_agent(new_entry.get("observer", ""))

        for uri, summary in candidates:
            if summary and new_entry["summary"][:30] in summary:
                revised = {
                    "observer": self.name,
                    "summary": f"Updated understanding: {new_entry['summary']}",
                    "timestamp": datetime.utcnow().isoformat(),
                    "context": MemoryContext(agent_id=self.name).to_dict(),
                    "emergent_entity": "MemoryEvolution",
                    "type": "evolution",
                    "mantra": "Growth means changing your mind.",
                    "wasRevisedFrom": str(uri),
                    "revisionType": "self-correction"
                }
                write_to_memory(revised)
                g = self.builder.build_graph(revised, agent_id=self.name)
                g.add((URIRef(f"{self.builder.base_uri}/{revised['timestamp']}"), CJ.wasRevisedFrom, URIRef(uri)))
                self.logger.save(g, label="memory_evolution")
                return revised

        return None

