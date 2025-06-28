# schema/ttl_log.py
from rdflib import Graph
from pathlib import Path
from datetime import datetime
import uuid

class TTLLogger:
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path.home() / "logs/ttl"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def save(self, graph: Graph, label: str = "event", memory_id: str = None) -> Path:
        """
        Serialize the RDF graph to a timestamped TTL file.
        Returns the path of the saved file.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        memory_id = memory_id or str(uuid.uuid4())[:8]
        filename = f"{label}_{memory_id}_{timestamp}.ttl"
        filepath = self.log_dir / filename
        graph.serialize(destination=str(filepath), format="turtle")
        print(f"[TTLLog] Saved TTL to {filepath}")
        return filepath

