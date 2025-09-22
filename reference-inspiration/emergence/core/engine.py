import uuid
import time
from memory.memory_store import MemoryStore
from perception.vision import VisionProcessor
from cognition.introspection import Introspector
from schema.rdf_builder import RDFBuilder

class EmergentAISystem:
    def __init__(self):
        self.memory = MemoryStore()
        self.vision = VisionProcessor(self.memory)
        self.introspector = Introspector(self.memory)
        self.rdf_builder = RDFBuilder(self.memory)

    def step(self):
        vision_input = self.vision.capture()
        self.memory.store("vision", vision_input)

        introspection = self.introspector.reflect()
        self.memory.store("introspection", introspection)

        rdf = self.rdf_builder.build()
        self.memory.store("rdf", rdf)

        print("Step complete: memory, introspection, RDF updated.")

    def run(self, iterations=5, delay=1):
        for _ in range(iterations):
            self.step()
            time.sleep(delay)