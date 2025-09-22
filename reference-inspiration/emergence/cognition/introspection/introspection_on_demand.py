# emergence.cognition/introspection/introspect_on_demand.py
import os
from emergence.core.redis_bus import RedisBus
from emergence.cognition.introspection.agent_loop import IntrospectiveAgent
from emergence.schema.collapse_builder import build_reflection_entry
from emergence.memory.chroma_writer import write_to_memory
from emergence.memory.chroma_reader import query_recent_memory

CHROMA_ENABLED = os.getenv("USE_CHROMA", "true").lower() == "true"
CHROMA_QUERY_TEXT = os.getenv("CHROMA_QUERY_TEXT", "What should I reflect on?")

class IntrospectionTrigger:
    def __init__(self):
        self.bus = RedisBus()
        self.agent = IntrospectiveAgent(memory_source=self.get_memory_source)
        self.bus.subscribe("cognition:introspection:trigger", self.handle_trigger)

    def get_memory_source(self):
        if CHROMA_ENABLED:
            return query_recent_memory(CHROMA_QUERY_TEXT) or "Empty memory response"
        return "Spontaneous curiosity signal"

    def handle_trigger(self, message):
        print("[Introspect] Trigger received via Redis.")
        obs = message.get("observation") or self.agent.memory_source()
        entry = build_reflection_entry(obs)
        write_to_memory(entry)
        self.bus.publish("cognition:introspection:event", entry)

    def listen(self):
        print("[Introspect] Listening for introspection triggers...")
        self.bus.listen_forever()

if __name__ == "__main__":
    listener = IntrospectionTrigger()
    listener.listen()
