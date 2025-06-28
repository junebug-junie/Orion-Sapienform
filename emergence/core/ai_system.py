# core/ai_system.py
import uuid
import asyncio
from datetime import datetime
from emergence.memory.memory_store import MemoryStore
from emergence.perception.vision.vision_processor import VisionProcessor
from emergence.cognition.introspection import Introspector
from emergence.schema.rdf_builder import RDFBuilder
from emergence.core.redis_bus import RedisBus

class EmergentAISystem:
    """
    Orchestrates the mesh loop: sensory input → introspection → semantic encoding.
    """

    def __init__(self, memory=None):
        self.session_id = str(uuid.uuid4())
        self.memory = memory or MemoryStore()
        self.context = {}  # Shared between steps
        self.bus = RedisBus()
        self.vision = VisionProcessor(self.memory)
        self.introspector = Introspector(self.memory)
        self.rdf_builder = RDFBuilder(self.memory)
        self.on_step_complete = None

    async def step(self):
        """
        Run one async-aware cognitive cycle.
        """
        print(f"[Step] Running cycle for session {self.session_id}...")

        now = datetime.utcnow().isoformat()
        vision_input = self.vision.capture(self.context)
        self.memory.store("vision", vision_input, timestamp=now)

        introspection = self.introspector.reflect(self.context)
        self.memory.store("introspection", introspection, timestamp=now)

        rdf = self.rdf_builder.build(self.context)
        self.memory.store("rdf", rdf, timestamp=now)

        self.bus.publish("cognition:step_complete", {
            "session": self.session_id,
            "timestamp": now
        })

        if self.on_step_complete:
            self.on_step_complete()

        print(f"[Step] Complete.")

    async def run(self, iterations=5, delay=1):
        """
        Run a loop of async cycles.
        """
        print(f"[Run] Starting {iterations} cycles.")
        for i in range(iterations):
            print(f"[Run] Cycle {i + 1}/{iterations}")
            await self.step()
            await asyncio.sleep(delay)
        print(f"[Run] Finished.")
