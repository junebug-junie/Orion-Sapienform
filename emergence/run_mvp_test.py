# run_mvp_test.py
from emergence.perception.vision.vision_processor import VisionProcessor
from emergence.perception.system.system_monitor import SystemMonitor
from emergence.cognition.introspection.agent_introspector import AgentIntrospector
from emergence.cognition.introspection.reflection_chain import emit_reflection_chain_from
from emergence.cognition.introspection.memory_evolver import MemoryEvolver
import time

print("\n[TEST] Orion MVP Self-Test Starting\n")

# 1. Vision snapshot + reflection
vp = VisionProcessor(node_name="test-vision")
vision_frame = vp.capture()
if vision_frame:
    print("[TEST] Vision frame captured.")
else:
    print("[TEST] Vision frame failed.")

time.sleep(2)

# 2. System vitals + reflection
sm = SystemMonitor(name="test-system")
system_entry = sm.check_vitals()
print("[TEST] System vitals checked.")

time.sleep(2)

# 3. Fusion (allow Redis listeners to fire)
print("[TEST] Waiting for FusionNode synthesis...")
time.sleep(5)

# 4. Introspect directly
ai = AgentIntrospector(name="test-introspector")
reflection = ai.reflect("Testing agent introspection on boot")
print("[TEST] Introspective memory emitted.")

# 5. Reflection Chain
if reflection and reflection.get("memory_id"):
    emit_reflection_chain_from(reflection["memory_id"])
    print("[TEST] Reflection chain emitted.")
else:
    print("[TEST] No reflection memory_id to thread.")

# 6. Memory Evolver (try self-revision)
me = MemoryEvolver()
revised = me.detect_revision(reflection)
if revised:
    print("[TEST] Memory evolution occurred.")
else:
    print("[TEST] No memory evolution detected.")

print("\n✅ Orion MVP self-test complete. Check dashboard and logs.\n")

