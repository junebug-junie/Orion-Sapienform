import asyncio
import uuid
import argparse
import sys
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vision import VisionWindowRequestPayload, VisionArtifactPayload, VisionArtifactOutputs, VisionCaption, VisionWindowResultPayload

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="redis://localhost:6379/0", help="Redis URL")
    args = parser.parse_args()

    bus = OrionBusAsync(url=args.url)
    await bus.connect()

    # Test Window RPC
    req_channel = "orion:exec:request:VisionWindowService"
    reply_channel = f"orion:vision:test:{uuid.uuid4()}"
    corr_id = str(uuid.uuid4())

    # Fake artifact
    artifact = VisionArtifactPayload(
        artifact_id="art-test-1",
        correlation_id=corr_id,
        task_type="test",
        device="cpu",
        inputs={},
        outputs=VisionArtifactOutputs(
            caption=VisionCaption(text="A test image caption", confidence=0.99)
        ),
        timing={},
        model_fingerprints={}
    )

    req = VisionWindowRequestPayload(artifact=artifact)

    env = BaseEnvelope(
        schema_id="vision.window.request",
        schema_version="1.0.0",
        kind="vision.window.request",
        source="test-script",
        correlation_id=corr_id,
        reply_to=reply_channel,
        payload=req
    )

    print(f"Sending Window RPC to {req_channel}...")

    try:
        reply = await bus.rpc_request(req_channel, env, reply_channel=reply_channel, timeout_sec=5.0)
        decoded = bus.codec.decode(reply.get("data"))
        if decoded.ok:
            res_env = decoded.envelope
            print("Received Reply:")
            print(f"Kind: {res_env.kind}")
            if isinstance(res_env.payload, dict):
                 res = VisionWindowResultPayload(**res_env.payload)
                 print(f"Window ID: {res.window.window_id}")
                 print(f"Summary: {res.window.summary}")
            else:
                 print(f"Payload: {res_env.payload}")
        else:
            print(f"Decode failed: {decoded.error}")

    except Exception as e:
        print(f"RPC failed: {e}")
    finally:
        await bus.close()

if __name__ == "__main__":
    asyncio.run(main())
