from core.redis_pubsub import RedisBackbone
from memory.chroma_writer import write_to_memory
from dream_engine.dream_trigger import trigger_dream
from cognition.introspection.simulator import simulate_internal_thought

def memory_handler(message):
    print("[Router] Forwarding to memory module")
    write_to_memory(message['data'])


def dream_handler(message):
    print("[Router] Forwarding to dream module")
    trigger_dream(message['data'])
    simulate_internal_thought(message['data'])


def run_router():
    backbone = RedisBackbone(channels=["memory", "dream"])
    backbone.subscribe(callback=lambda msg: (
        memory_handler(msg) if msg['channel'] == 'memory' else dream_handler(msg)
    ))
    print("[Router] Running...")

if __name__ == "__main__":
    run_router()