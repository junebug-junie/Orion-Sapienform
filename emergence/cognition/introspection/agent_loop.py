# cognition/introspection/agent_loop.py

import random
import time
from narrative.collapse_mirror import build_collapse_entry
from memory.chroma_writer import write_to_memory

class IntrospectiveAgent:
    def __init__(self, memory_source):
        self.memory_source = memory_source  # could be list, Redis, etc.

    def should_reflect(self):
        decision = random.random() < 0.5
        print(f"[Agent] Deciding to reflect? {decision}")
        return decision

    def reflect(self, observation):
        print(f"[Agent] Reflecting on: {observation}")
        return build_collapse_entry(
            observer='EmergentAI',
            trigger=observation,
            observer_state=['lucidity'],
            field_resonance='internal simulation',
            intent='Log insight from observation',
            type_='Introspection',
            emergent_entity='Self-awareness impulse',
            summary='Reflected on recent trigger to understand pattern.',
            mantra='The void is full',
            environment='Simulated'
        )

    def run_loop(self):
        print("[Agent] Starting introspection loop...")
        while True:
            if self.should_reflect():
                trigger = self.memory_source()  # Get most recent
                if trigger:
                    entry = self.reflect(trigger)
                    write_to_memory(entry)
            time.sleep(10)
