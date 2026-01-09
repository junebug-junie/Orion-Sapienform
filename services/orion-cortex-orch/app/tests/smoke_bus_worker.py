"""
Smoke test worker for Orion Cortex Orchestrator.

Listens on EXEC_REQUEST_PREFIX:TestService and publishes results to
the given reply/result channel. Used to validate that:

- Cortex Orchestrator publishes exec_step to the bus
- This worker receives it
- This worker publishes a result back
- Orchestrator fan-ins the result and returns a structured response

Usage (from repo root):

    cd /mnt/scripts/Orion-Sapienform
    export ORION_BUS_URL=redis://orion-redis:6379/0         # if not already in env
    export EXEC_REQUEST_PREFIX=orion-exec:request           # must match orchestrator .env
    python3 services/orion-cortex-orch/tests/smoke_bus_worker.py
"""

import asyncio
import json
import os
import time
import uuid

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

BUS_URL = os.getenv("ORION_BUS_URL", "redis://orion-redis:6379/0")
EXEC_REQUEST_PREFIX = os.getenv("EXEC_REQUEST_PREFIX", "orion-exec:request")
SERVICE_NAME = os.getenv("SMOKE_SERVICE_NAME", "TestService")

REQUEST_CHANNEL = f"{EXEC_REQUEST_PREFIX}:{SERVICE_NAME}"


def _source() -> ServiceRef:
    return ServiceRef(name=SERVICE_NAME, node="smoke-worker", version="0.0.0")


async def main() -> None:
    print(f"[smoke-worker] Connecting to bus: {BUS_URL}")
    bus = OrionBusAsync(BUS_URL)
    await bus.connect()
    print(f"[smoke-worker] Subscribing to {REQUEST_CHANNEL}")

    async with bus.subscribe(REQUEST_CHANNEL) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            try:
                decoded = bus.codec.decode(msg.get("data", b""))
                data = decoded.envelope.payload if decoded.ok else decoded.raw
                if not isinstance(data, dict):
                    print(f"[smoke-worker] Unexpected payload: {data}")
                    continue
            except Exception as e:
                print(f"[smoke-worker] Failed to parse JSON: {e} raw={msg.get('data')!r}")
                continue

            # Orchestrator sends event=exec_step; we’ll just log and handle anything
            event = data.get("event")
            if event != "exec_step":
                print(f"[smoke-worker] Ignoring non-exec_step message: {data}")
                continue

            # Prefer trace_id, fall back to correlation_id, then generate one
            trace_id = data.get("trace_id") or data.get("correlation_id") or str(uuid.uuid4())

            # Support multiple naming conventions:
            # - reply_channel (what cortex-orch uses)
            # - result_channel / response_channel for future-proofing
            reply_channel = (
                data.get("reply_channel")
                or data.get("result_channel")
                or data.get("response_channel")
            )

            if not reply_channel:
                print(f"[smoke-worker] No reply_channel/result_channel in message: {data}")
                continue

            # Unwrap nested payload if present (as in your current smoke wiring)
            step_payload = data
            if isinstance(data.get("payload"), dict):
                step_payload = data["payload"]

            prompt = step_payload.get("prompt", "")

            print(
                f"[smoke-worker] Got exec_step trace_id={trace_id} "
                f"service={SERVICE_NAME} reply_channel={reply_channel}"
            )

            started = time.time()
            # simulate some work
            await asyncio.sleep(0.1)
            elapsed_ms = int((time.time() - started) * 1000)

            # Standard-ish exec_step_result shape
            payload = {
                "trace_id": trace_id,
                "service": SERVICE_NAME,
                "ok": True,
                "elapsed_ms": elapsed_ms,
                "note": "smoke test response from TestService",
                "prompt_preview": prompt[:200],
                "echo": step_payload,  # so you can see the original step contents
            }

            response = BaseEnvelope(
                kind="exec.step.result",
                source=_source(),
                payload=payload,
            )
            await bus.publish(reply_channel, response)
            print(
                f"[smoke-worker] Published result for trace_id={trace_id} "
                f"to {reply_channel} (elapsed={elapsed_ms} ms)"
            )


if __name__ == "__main__":
    asyncio.run(main())
