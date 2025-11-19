"""
Smoke test worker for Orion Cortex Orchestrator.

Listens on EXEC_REQUEST_PREFIX:TestService and publishes results to
the given result_channel. Used to validate that:

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

import json
import os
import time

import redis

BUS_URL = os.getenv("ORION_BUS_URL", "redis://orion-redis:6379/0")
EXEC_REQUEST_PREFIX = os.getenv("EXEC_REQUEST_PREFIX", "orion-exec:request")
SERVICE_NAME = os.getenv("SMOKE_SERVICE_NAME", "TestService")

REQUEST_CHANNEL = f"{EXEC_REQUEST_PREFIX}:{SERVICE_NAME}"


def main() -> None:
    print(f"[smoke-worker] Connecting to bus: {BUS_URL}")
    r = redis.Redis.from_url(BUS_URL, decode_responses=True)
    r.ping()
    print(f"[smoke-worker] Subscribing to {REQUEST_CHANNEL}")

    pubsub = r.pubsub()
    pubsub.subscribe(REQUEST_CHANNEL)

    for msg in pubsub.listen():
        if msg.get("type") != "message":
            continue

        try:
            data = json.loads(msg["data"])
        except Exception as e:
            print(f"[smoke-worker] Failed to parse JSON: {e} raw={msg['data']!r}")
            continue

        trace_id = data.get("trace_id")
        result_channel = data.get("result_channel")
        prompt = data.get("prompt", "")

        if not result_channel:
            print(f"[smoke-worker] No result_channel in message: {data}")
            continue

        print(
            f"[smoke-worker] Got exec_step trace_id={trace_id} "
            f"service={SERVICE_NAME} result_channel={result_channel}"
        )

        started = time.time()
        # simulate some work
        time.sleep(0.1)
        elapsed_ms = int((time.time() - started) * 1000)

        payload = {
            "trace_id": trace_id,
            "service": SERVICE_NAME,
            "ok": True,
            "elapsed_ms": elapsed_ms,
            "note": "smoke test response from TestService",
            "prompt_preview": prompt[:200],
        }

        r.publish(result_channel, json.dumps(payload))
        print(
            f"[smoke-worker] Published result for trace_id={trace_id} "
            f"to {result_channel} (elapsed={elapsed_ms} ms)"
        )


if __name__ == "__main__":
    main()
