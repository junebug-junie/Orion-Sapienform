# cognition/introspection/reflection_chain_listener.py
import json
from emergence.core.redis_bus import RedisBus
from emergence.cognition.introspection.reflection_chain import emit_reflection_chain_from

def start_reflection_chain_listener():
    bus = RedisBus()
    print("[ReflectionChain] Listening for memory reflection triggers...")

    def handle(msg):
        try:
            payload = json.loads(msg)
            memory_id = payload.get("memory_id")
            command = payload.get("command")
            if command == "reflect_chain" and memory_id:
                emit_reflection_chain_from(memory_id)
        except Exception as e:
            print(f"[ReflectionChain][Error] {e}")

    bus.subscribe("introspection:reflect_chain", handle)
    bus.listen_forever()

if __name__ == "__main__":
    start_reflection_chain_listener()

