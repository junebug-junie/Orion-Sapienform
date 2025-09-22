from rdflib import Graph, URIRef, Literal, Namespace

class RDFBuilder:
    def __init__(self, memory):
        self.memory = memory
        self.ns = Namespace("http://conjourney.net/rdf#")

    def build(self):
        g = Graph()
        for k, v in self.memory.all().items():
            subject = URIRef(self.ns[k])
            g.add((subject, self.ns["hasContent"], Literal(str(v))))
        return g.serialize(format="turtle")