# schema/ttl_auditor.py
from rdflib import Graph, Namespace, RDF, Literal
from pathlib import Path
from datetime import datetime
import glob

CJ = Namespace("http://conjourney.net/schema#")

def load_latest_ttls(n=5, ttl_dir=None):
    ttl_dir = Path(ttl_dir or Path.home() / "logs/ttl")
    files = sorted(ttl_dir.glob("*.ttl"), key=lambda f: f.stat().st_mtime, reverse=True)
    return [Graph().parse(f, format="turtle") for f in files[:n]]

def diff_graphs(g1: Graph, g2: Graph) -> Graph:
    diff = Graph()
    diff.bind("cj", CJ)
    removed = g1 - g2
    added = g2 - g1

    for s, p, o in removed:
        diff.add((s, CJ.diffType, Literal("removed")))
        diff.add((s, CJ.affectedPredicate, p))
        diff.add((s, CJ.affectedObject, o))

    for s, p, o in added:
        diff.add((s, CJ.diffType, Literal("added")))
        diff.add((s, CJ.affectedPredicate, p))
        diff.add((s, CJ.affectedObject, o))

    return diff

def validate_graph_schema(graph: Graph) -> Graph:
    required = [CJ.id, CJ.timestamp, CJ.source]
    audit = Graph()
    for subject in set(graph.subjects(RDF.type, CJ.MemoryEvent)):
        for req in required:
            if (subject, req, None) not in graph:
                audit.add((subject, CJ.diffType, Literal("missing")))
                audit.add((subject, CJ.affectedPredicate, req))
    return audit

def report_broken_threads(graph: Graph) -> Graph:
    """
    Find cj:generatedBy links where the target URI is not defined in the graph.
    """
    audit = Graph()
    audit.bind("cj", CJ)

    all_subjects = set(graph.subjects(RDF.type, CJ.MemoryEvent))

    for subject in all_subjects:
        for _, _, parent in graph.triples((subject, CJ.generatedBy, None)):
            if (parent, RDF.type, CJ.MemoryEvent) not in graph:
                audit.add((subject, CJ.diffType, Literal("broken_thread")))
                audit.add((subject, CJ.missingReference, parent))
                audit.add((subject, CJ.auditedEvent, subject))

    return audit
