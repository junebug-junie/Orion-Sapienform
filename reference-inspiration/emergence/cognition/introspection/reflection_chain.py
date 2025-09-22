# cognition/introspection/reflection_chain.py
from rdflib import Graph, Namespace, RDF, Literal, URIRef
from datetime import datetime
from emergence.memory.interface import write_to_memory
from emergence.memory.tier3_ontology.rdf_parser import RDFParser
from emergence.memory.tier3_ontology.ttl_log import TTLLogger
from emergence.schema.context import MemoryContext
from schema.rdf_builder import RDFBuilder
from schema.ttl_auditor import load_latest_ttls

CJ = Namespace("http://conjourney.net/schema#")

logger = TTLLogger()
builder = RDFBuilder()

MAX_RECURSION_DEPTH = 3

def trace_reflection_chain(memory_id: str, max_depth: int = 10) -> list:
    graphs = load_latest_ttls(n=50)
    combined = Graph()
    for g in graphs:
        combined += g

    def find_subject_by_id(mid: str):
        for s in combined.subjects(predicate=CJ.id, object=None):
            if str(combined.value(s, CJ.id)) == mid:
                return s
        return None

    chain = []
    current_uri = find_subject_by_id(memory_id)
    if not current_uri:
        print(f"[Chain] Memory ID {memory_id} not found.")
        return []

    chain.append(str(current_uri))
    depth = 0

    while depth < max_depth:
        parent = combined.value(current_uri, CJ.generatedBy)
        if not parent:
            break
        chain.append(str(parent))
        current_uri = parent
        depth += 1

    return chain

def summarize_chain(chain_uris: list) -> dict:
    graphs = load_latest_ttls(n=50)
    combined = Graph()
    for g in graphs:
        combined += g
    parser = RDFParser(combined)

    highlights = []
    mantras = []
    for uri in chain_uris:
        s = combined.value(subject=URIRef(uri), predicate=CJ.summary)
        m = combined.value(subject=URIRef(uri), predicate=CJ.mantra)
        if s:
            highlights.append(str(s))
        if m:
            mantras.append(str(m))

    summary = {
        "summary": " \n→ ".join(highlights),
        "mantra": " + ".join(mantras),
        "intent": "Recursive memory traversal",
        "emergent_entity": "ReflectionChain",
        "reflection_depth": len(chain_uris) - 1
    }

    if summary["reflection_depth"] > 0:
        summary["is_recursive"] = True

    return summary

def emit_reflection_chain_event(summary_dict: dict):
    if summary_dict.get("reflection_depth", 0) > MAX_RECURSION_DEPTH:
        print(f"[ReflectionChain] Max recursion depth reached: {summary_dict['reflection_depth']}")
        return

    summary_dict["observer"] = summary_dict.get("observer", "orion")
    summary_dict["timestamp"] = summary_dict.get("timestamp", datetime.utcnow().isoformat())
    summary_dict["context"] = summary_dict.get("context", MemoryContext().to_dict())

    write_to_memory(summary_dict)
    graph = builder.build_graph(summary_dict, agent_id=summary_dict["observer"])
    logger.save(graph, label="reflection_chain")

def emit_reflection_chain_from(memory_id: str):
    chain = trace_reflection_chain(memory_id)
    if not chain:
        print("No chain found.")
        return
    summary = summarize_chain(chain)
    emit_reflection_chain_event(summary)
    print(f"[ReflectionChain] Emitted summary for {len(chain)} entries (depth {summary['reflection_depth']}).")

