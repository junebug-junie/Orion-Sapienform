from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD
import uuid
from datetime import datetime

CONJ = Namespace("http://conjourney.net/schema#")

class RDFEncoder:
    def __init__(self):
        self.graph = Graph()
        self.graph.bind("conj", CONJ)

    def _generate_uri(self, entity_type: str):
        return URIRef(f"http://conjourney.net/{entity_type}/{uuid.uuid4()}")

    def encode_entry(self, entry: dict) -> Graph:
        """
        Encodes a CollapseMirror or observational entry into RDF format.
        Expected keys: observer, trigger, observer_state, field_resonance,
        intent, type, emergent_entity, summary, mantra, causal_echo, timestamp, environment
        """
        entry_uri = self._generate_uri("entry")
        self.graph.add((entry_uri, RDF.type, CONJ.Entry))

        def safe_literal(val):
            if val is None:
                return None
            elif isinstance(val, list):
                return [Literal(v) for v in val]
            elif isinstance(val, (int, float)):
                return Literal(val, datatype=XSD.float)
            else:
                return Literal(str(val))

        for key, value in entry.items():
            prop_uri = CONJ[key]
            lit_val = safe_literal(value)
            if lit_val is None:
                continue
            if isinstance(lit_val, list):
                for item in lit_val:
                    self.graph.add((entry_uri, prop_uri, item))
            else:
                self.graph.add((entry_uri, prop_uri, lit_val))

        return self.graph

    def encode_vision_result(self, subject: str, image_uri: str, label_scores: dict, timestamp=None):
        """
        Encodes a vision output into RDF triples.
        subject: the observing entity
        image_uri: URI or path to image
        label_scores: dict of label -> probability
        timestamp: optional ISO format
        """
        event_uri = self._generate_uri("vision")
        self.graph.add((event_uri, RDF.type, CONJ.VisionObservation))
        self.graph.add((event_uri, CONJ.seenBy, Literal(subject)))
        self.graph.add((event_uri, CONJ.image, Literal(image_uri)))
        self.graph.add((event_uri, CONJ.timestamp, Literal(timestamp or datetime.utcnow().isoformat())))

        for label, prob in label_scores.items():
            label_uri = self._generate_uri("label")
            self.graph.add((label_uri, RDF.type, CONJ.Label))
            self.graph.add((label_uri, CONJ.labelText, Literal(label)))
            self.graph.add((label_uri, CONJ.probability, Literal(prob, datatype=XSD.float)))
            self.graph.add((event_uri, CONJ.hasLabel, label_uri))

        return self.graph

    def serialize(self, format="turtle") -> str:
        return self.graph.serialize(format=format)

    def reset(self):
        self.graph = Graph()
        self.graph.bind("conj", CONJ)


def encode_emotion_rdf(emotion_vector, image_path):
    triples = []
    for emotion, score in emotion_vector.items():
        triples.append((image_path, "hasEmotion", f"{emotion}:{score:.2f}"))
    return triples
