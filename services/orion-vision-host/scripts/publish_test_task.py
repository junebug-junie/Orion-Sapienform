import asyncio
import uuid
import sys
import argparse
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionTaskRequestPayload, VisionTaskResultPayload

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="retina_fast", help="Task type")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompts", default="person, cat", help="Comma sep prompts for detection")
    parser.add_argument("--url", default="redis://localhost:6379/0", help="Redis URL")
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.url)
    await bus.connect()

request_channel = "orion:exec:request:VisionHostService"
    reply_channel = f"orion:vision:reply:{uuid.uuid4()}"
    correlation_id = str(uuid.uuid4())

    payload = VisionTaskRequestPayload(
        task_type=args.task,
        request={
            "image_path": args.image,
            "prompts": [p.strip() for p in args.prompts.split(",")],
        },
        meta={"source": "cli_test"}
    )

    envelope = BaseEnvelope(
        schema_id="vision.task.request",
        schema_version="1.0.0",
        kind="vision.task.request",
        source="cli-tester",
        correlation_id=correlation_id,
        reply_to=reply_channel,
        payload=payload
    )

    print(f"Sending request to {request_channel}...")
    print(f"Reply channel: {reply_channel}")
    print(f"Payload: {payload}")

    try:
        # Subscribe first
        async with bus.subscribe(reply_channel) as pubsub:
            await bus.publish(request_channel, envelope)

            print("Waiting for reply...")
            async for msg in bus.iter_messages(pubsub):
                data = msg.get("data")
                decoded = bus.codec.decode(data)

                if decoded.ok:
                    res_env = decoded.envelope
                    print("\n--- RECEIVED REPLY ---")
                    print(f"Kind: {res_env.kind}")
                    print(f"CorrID: {res_env.correlation_id}")

                    if isinstance(res_env.payload, dict):
                         try:
                             res_payload = VisionTaskResultPayload(**res_env.payload)
                             print(f"Payload (Typed): {res_payload}")
                         except Exception as e:
                             print(f"Payload (Dict): {res_env.payload}")
                             print(f"Validation Error: {e}")
                    else:
                        print(f"Payload: {res_env.payload}")

                    break
                else:
                    print(f"Decode error: {decoded.error}")
                    break

    except asyncio.TimeoutError:
        print("Timed out waiting for reply.")
    finally:
        await bus.close()

if __name__ == "__main__":
    asyncio.run(main())
