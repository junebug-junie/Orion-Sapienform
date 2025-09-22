# memory/memory_store.py
from datetime import datetime

class MemoryStore:
    """
    Simple in-memory store with timestamp tracking.
    """

    def __init__(self):
        self.data = {}

    def store(self, kind: str, value, timestamp: str = None):
        """
        Store an item in memory with a timestamp.
        """
        ts = timestamp or datetime.utcnow().isoformat()
        self.data[kind] = {
            "value": value,
            "timestamp": ts
        }
        print(f"[Memory] Stored '{kind}' at {ts}")

    def get(self, kind: str):
        """
        Retrieve a memory item by kind.
        """
        return self.data.get(kind, None)

    def last_keys(self):
        """
        Return a list of stored keys.
        """
        return list(self.data.keys())

