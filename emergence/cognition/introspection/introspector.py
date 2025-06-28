# cognition/introspection/introspector.py
import uuid
import os
from datetime import datetime
from emergence.memory.interface import write_to_memory
from emergence.schema.context import MemoryContext
from emergence.core.redis_bus import RedisBus
from emergence.cognition.introspection.salience import calculate_salience

class Introspector:
    def __init__(self, memory=None, node_name="introspector", salience_threshold: float = None):
        self.memory = memory
        self.node_name = node_name
        self.salience_threshold = salience_threshold or float(os.getenv("INTROSPECTION_SALIENCE_THRESHOLD", 0.6))
        self.bus = RedisBus()

    def reflect(self, prompt: str = None, salience: float = None) -> dict:
        """
        Create an introspective memory entry.
        If salience exceeds threshold, trigger reflection chain.
        """
        summary = prompt or "Internal self-monitoring initiated."
        timestamp = datetime.utcnow().isoformat()
        memory_id = str(uuid.uuid4())

        entry = {
            "memory_id": memory_id,
            "observer": self.node_name,
            "summary": summary,
            "intent": "introspection",
            "type": "collapse",
            "timestamp": timestamp,
            "context": MemoryContext().to_dict(),
            "mantra": "Observe the observer."
        }

        # Auto-calculate salience if not given
        entry["salience"] = salience if salience is not None else calculate_salience(entry)

        write_to_memory(entry)

        if entry["salience"] >= self.salience_threshold:
            self.bus.publish("introspection:reflect_chain", {
                "memory_id": memory_id,
                "command": "reflect_chain"
            })

        return entry

