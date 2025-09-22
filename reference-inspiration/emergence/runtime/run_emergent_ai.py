# run_emergent_ai.py
import asyncio
from emergence.core.ai_system import EmergentAISystem

async def main(iterations=5, delay=2):
    system = EmergentAISystem()

    def on_step_complete():
        system.bus.publish("cognition:cycle_complete", {
            "event": "cycle_complete",
            "session": system.session_id
        })

    system.on_step_complete = on_step_complete
    await system.run(iterations=iterations, delay=delay)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--delay", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main(iterations=args.iterations, delay=args.delay))

