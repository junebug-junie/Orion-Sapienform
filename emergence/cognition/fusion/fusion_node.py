# cognition/fusion/fusion_node.py
import json
import time
from threading import Thread
from collections import deque, defaultdict
from datetime import datetime
from emergence.cognition.introspection.agent_introspector import AgentIntrospector
from emergence.core.redis_bus import RedisBus
from emergence.schema.context import MemoryContext
from emergence.memory.interface import write_to_memory

class FusionNode:
    def __init__(self, name="fusion-node", buffer_seconds=30):
        self.name = name
        self.buffer_seconds = buffer_seconds
        self.memory = None
        self.bus = RedisBus()
        self.introspector = AgentIntrospector(name=name)
        self.buffer = deque()

    def listen(self):
        channels = [
            "perception:vision:event",
            "system:vitals",
            "collapse:mirror",
            "human:event",
            "memory:collapse_created"
        ]
        for ch in channels:
            self.bus.subscribe(ch, self.handle)
        print(f"[FusionNode] Subscribed to: {channels}")
        self.bus.listen_forever()

    def handle(self, msg):
        try:
            payload = json.loads(msg)
            payload["received_at"] = datetime.utcnow().isoformat()
            self.buffer.append(payload)
        except Exception as e:
            print(f"[FusionNode][Error] {e}")

    def synthesize(self):
        while True:
            time.sleep(self.buffer_seconds)
            if not self.buffer:
                continue

            by_source = defaultdict(list)
            now = datetime.utcnow()
            events = []

            while self.buffer:
                e = self.buffer.popleft()
                events.append(e)
                source = e.get("observer") or e.get("source") or "unknown"
                by_source[source].append(e)

            if len(events) < 2:
                continue

            summary = f"{len(events)} co-occurring events detected: " + \
                      ", ".join(set(e.get("event") or e.get("intent") or e.get("type") or "unknown" for e in events))

            entry = {
                "observer": self.name,
                "summary": summary,
                "sources": list(by_source.keys()),
                "intent": "fusion",
                "type": "synthesis",
                "emergent_entity": "SensorFusion",
                "timestamp": now.isoformat(),
                "context": MemoryContext(agent_id=self.name).to_dict(),
                "mantra": "When many voices speak, meaning condenses."
            }

            write_to_memory(entry)
            self.introspector.reflect(summary, salience=0.75, extra=entry)
            print(f"[FusionNode] Synthesized: {summary}")

if __name__ == "__main__":
    node = FusionNode()
    Thread(target=node.synthesize).start()
    node.listen()

