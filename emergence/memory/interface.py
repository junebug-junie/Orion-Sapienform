# emergence/memory/interface.py
import uuid
import os
from datetime import datetime
from pathlib import Path

from emergence.memory.tier1_literal.writer import write_literal
from emergence.memory.tier1_literal.literal_reader import read_recent_literal_entries
from emergence.memory.tier2_semantic.chroma_writer import write_to_chroma
from emergence.memory.tier2_semantic.chroma_reader import query_chroma
from emergence.memory.tier3_ontology.graph_writer import write_rdf_entry
from emergence.memory.tier3_ontology.rdf_parser import RDFParser
from emergence.memory.tier3_ontology.ttl_log import TTLLogger
from emergence.schema.context import MemoryContext
from rdflib import Graph

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "default")

# TTL storage setup
if MEMORY_BACKEND == "optane":
    ttl_logger = TTLLogger(log_dir=Path("/mnt/optane/conjourney/logs/ttl"))
else:
    ttl_logger = TTLLogger()

def write_to_memory(entry: dict, context: MemoryContext = None):
    """
    Unified entry point to write a memory event to all tiers.
    Automatically assigns memory_id and default context.
    """
    entry = dict(entry)  # Defensive copy
    entry["memory_id"] = entry.get("memory_id") or str(uuid.uuid4())
    entry["timestamp"] = entry.get("timestamp") or datetime.utcnow().isoformat()
    entry["context"] = entry.get("context") or (context.to_dict() if context else MemoryContext().to_dict())

    write_literal(entry)
    write_to_chroma(entry)
    write_rdf_entry(entry)
    ttl_logger.save(write_rdf_entry(entry), label="memory", memory_id=entry["memory_id"])

def read_from_memory(query: str, mode: str = "semantic", limit: int = 5):
    """
    Unified read interface across memory tiers.
    mode = "semantic" | "literal" | "ontological"
    """
    if mode == "semantic":
        return query_chroma(query, top_k=limit)

    elif mode == "literal":
        return read_recent_literal_entries(limit=limit)

    elif mode == "ontological":
        g = Graph()
        ttl_dir = ttl_logger.log_dir
        for f in sorted(ttl_dir.glob("*.ttl"), reverse=True):
            g.parse(f, format="turtle")
        parser = RDFParser(g)
        return parser.get_all_events()[:limit]

    else:
        raise ValueError(f"Unsupported memory read mode: {mode}")

