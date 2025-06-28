# memory/tier3_ontology/rdf_parser.py
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

CJ = Namespace("http://conjourney.net/schema#")

class RDFParser:
    def __init__(self, graph: Graph):
        self.graph = graph

    def get_event_metadata(self, event_uri: URIRef) -> dict:
        """
        Extract core metadata fields from a single RDF memory event.
        """
        data = {}
        for p, o in self.graph.predicate_objects(subject=event_uri):
            key = p.split("#")[-1]
            data[key] = str(o)
        return data

    def get_all_events(self) -> list:
        """
        Iterate all cj:MemoryEvent entries and return their parsed metadata.
        """
        events = []
        for subject in self.graph.subjects(RDF.type, CJ.MemoryEvent):
            meta = self.get_event_metadata(subject)
            meta["uri"] = str(subject)
            events.append(meta)
        return events

    def trace_lineage(self, uri: URIRef, depth=3) -> list:
        """
        Follow cj:generatedBy chain backwards to reconstruct causal lineage.
        """
        lineage = [str(uri)]
        current = uri
        for _ in range(depth):
            parent = self.graph.value(subject=current, predicate=CJ.generatedBy)
            if parent:
                lineage.append(str(parent))
                current = parent
            else:
                break
        return lineage

