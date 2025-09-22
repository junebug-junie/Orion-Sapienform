# cognition/introspection/agent_introspector.py
from emergence.cognition.introspection.introspector import Introspector
from emergence.schema.context import MemoryContext

class AgentIntrospector(Introspector):
    def __init__(self, name: str, memory=None, salience_threshold: float = None):
        super().__init__(memory=memory, node_name=name, salience_threshold=salience_threshold)
        self.agent_id = name

    def reflect(self, prompt: str = None, salience: float = None, extra: dict = None) -> dict:
        """
        Agent-scoped introspection. Adds agent context and optional fields.
        """
        entry = super().reflect(prompt=prompt, salience=salience)

        # Agent-aware context
        ctx = MemoryContext(agent_id=self.agent_id)
        entry["context"] = ctx.to_dict()

        if extra:
            entry.update(extra)

        return entry

