from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal

class SchemaBuilder:
    def __init__(self):
        self.graph = Graph()
        self.NS = Namespace("http://conjourney.net/schema#")
        self.graph.bind("cj", self.NS)

    def define_classes(self):
        classes = [
            "Agent", "Observation", "Emotion", "Introspection", "MemoryEntry", 
            "CollapseMirror", "Perception", "Event", "Stimulus"
        ]
        for cls in classes:
            uri = self.NS[cls]
            self.graph.add((uri, RDF.type, RDFS.Class))
            self.graph.add((uri, RDFS.label, Literal(cls)))

    def define_properties(self):
        properties = {
            "hasObservation": ("Agent", "Observation"),
            "hasEmotion": ("Observation", "Emotion"),
            "hasMemoryEntry": ("Agent", "MemoryEntry"),
            "hasStimulus": ("Perception", "Stimulus"),
            "hasCollapse": ("Agent", "CollapseMirror"),
            "leadsTo": ("Stimulus", "Introspection"),
        }
        for prop, (domain, range_) in properties.items():
            uri = self.NS[prop]
            self.graph.add((uri, RDF.type, RDF.Property))
            self.graph.add((uri, RDFS.domain, self.NS[domain]))
            self.graph.add((uri, RDFS.range, self.NS[range_]))

    def build(self):
        self.define_classes()
        self.define_properties()
        return self.graph

    def serialize(self, format="turtle"):
        return self.graph.serialize(format=format).decode("utf-8")

if __name__ == "__main__":
    builder = SchemaBuilder()
    builder.build()
    print(builder.serialize())