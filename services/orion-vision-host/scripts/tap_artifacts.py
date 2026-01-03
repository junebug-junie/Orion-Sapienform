import asyncio
import argparse
from orion.core.bus.async_service import OrionBusAsync
from orion.schemas.vision import VisionArtifactPayload

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--channel", default="orion:vision:artifacts", help="Artifacts channel")
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.url)
    await bus.connect()

    print(f"Listening on {args.channel}...")

    async with bus.subscribe(args.channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg.get("data")
            decoded = bus.codec.decode(data)

            if decoded.ok:
                env = decoded.envelope
                print(f"\n[ARTIFACT] {env.kind} ({env.correlation_id})")
                try:
                    if isinstance(env.payload, dict):
                        payload = VisionArtifactPayload(**env.payload)
                    else:
                        payload = env.payload

                    print(f"  Task: {payload.task_type}")
                    print(f"  Device: {payload.device}")
                    print(f"  Inputs: {payload.inputs}")

                    if payload.outputs.objects:
                        print(f"  Objects: {len(payload.outputs.objects)} detected")
                        for obj in payload.outputs.objects[:3]:
                            print(f"    - {obj.label} ({obj.score:.2f})")
                        if len(payload.outputs.objects) > 3:
                            print("    ...")

                    if payload.outputs.caption:
                        print(f"  Caption: {payload.outputs.caption.text}")

                    if payload.outputs.embedding:
                        print(f"  Embedding: {payload.outputs.embedding.ref}")

                except Exception as e:
                    print(f"  Payload validation error: {e}")
                    print(f"  Raw Payload: {env.payload}")
            else:
                print(f"Decode error: {decoded.error}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
