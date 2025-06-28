# schema/context.py
from typing import Optional
from uuid import uuid4
from datetime import datetime

class MemoryContext:
    def __init__(self,
                 origin: str = "unknown",
                 stimulus: str = "unspecified",
                 thread_id: Optional[str] = None,
                 agent_state: str = "neutral",
                 location: str = "node:unknown",
                 purpose: str = "general"):

        self.origin = origin
        self.stimulus = stimulus
        self.thread_id = thread_id or self._generate_thread_id()
        self.agent_state = agent_state
        self.location = location
        self.purpose = purpose
        self.timestamp = datetime.utcnow().isoformat()

    def _generate_thread_id(self) -> str:
        return f"thread-{uuid4().hex[:8]}"

    def to_dict(self) -> dict:
        return {
            "origin": self.origin,
            "stimulus": self.stimulus,
            "thread_id": self.thread_id,
            "agent_state": self.agent_state,
            "location": self.location,
            "purpose": self.purpose,
            "timestamp": self.timestamp
        }

