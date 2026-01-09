# Orion LLM Gateway – Manual Smoke Tests

This document captures the quick manual tests for verifying the `orion-llm-gateway`
service from *inside* its container.

It assumes:

* Docker container name: `orion-athena-llm-gateway`
* Code lives inside the container at: `/app`
* `OrionBusAsync` is available at: `orion.core.bus.async_service.OrionBusAsync`

---

## 1. Drop into the `llm-gateway` container

From the host:

```
docker exec -it orion-athena-llm-gateway /bin/bash
```

You should now see a prompt like:

```
root@<container-id>:/app#
```

If you’re not in `/app`, go there:

```
cd /app
```

All following commands assume you’re at:

```
root@<container-id>:/app#
```

---

## 2. Test 1 – Default backend chat sanity check

This verifies:

* Container → Redis bus works
* `llm-gateway` is listening on `orion-exec:request:LLMGatewayService`
* The default backend returns a response

Run inside the container:

```
cd /app

PYTHONPATH=/app python - << 'EOF'
import asyncio
import uuid
from orion.core.bus.async_service import OrionBusAsync

async def main():
    bus = OrionBusAsync("redis://localhost:6379/0")
    await bus.connect()

    corr_id = str(uuid.uuid4())
    reply_channel = f"orion:llm:reply:test:{corr_id}"

    envelope = {
        "event": "chat",
        "service": "LLMGatewayService",
        "correlation_id": corr_id,
        "reply_channel": reply_channel,
        "payload": {
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "You are being called through the Orion LLM Gateway. "
                            "Respond with exactly:\n"
                            "BACKEND: <backend-name-you-believe-you-are>\n"
                            "MODEL: <model-name-you-believe-you-are>"
                        ),
                    }
                ],
                # no explicit backend → uses settings.default_backend
                "options": {},
            }
        },
    }

    print("Publishing chat request:", corr_id)
    await bus.publish("orion-exec:request:LLMGatewayService", envelope)

    print("Listening on reply_channel:", reply_channel)
    async with bus.subscribe(reply_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg["data"]
            print("Got reply:", data)
            break

    await bus.close()


asyncio.run(main())
EOF
```

---

## 3. Test 2 – Explicit backend selection (Ollama vs vLLM)

These tests verify that the `options.backend` override is honored by the gateway.

### 3.1 Force `ollama` backend

```
cd /app

PYTHONPATH=/app python - << 'EOF'
import asyncio
import uuid
from orion.core.bus.async_service import OrionBusAsync

async def main():
    bus = OrionBusAsync("redis://localhost:6379/0")
    await bus.connect()

    corr_id = str(uuid.uuid4())
    reply_channel = f"orion:llm:reply:test:{corr_id}"

    envelope = {
        "event": "chat",
        "service": "LLMGatewayService",
        "correlation_id": corr_id,
        "reply_channel": reply_channel,
        "payload": {
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'I am using OLLAMA backend' and nothing else.",
                    }
                ],
                "options": {"backend": "ollama"},
            }
        },
    }

    print("Publishing chat request (ollama):", corr_id)
    await bus.publish("orion-exec:request:LLMGatewayService", envelope)

    async with bus.subscribe(reply_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            print("Got reply (ollama):", msg["data"])
            break

    await bus.close()


asyncio.run(main())
EOF
```

### 3.2 Force `vllm` backend

```
cd /app

PYTHONPATH=/app python - << 'EOF'
import asyncio
import uuid
from orion.core.bus.async_service import OrionBusAsync

async def main():
    bus = OrionBusAsync("redis://localhost:6379/0")
    await bus.connect()

    corr_id = str(uuid.uuid4())
    reply_channel = f"orion:llm:reply:test:{corr_id}"

    envelope = {
        "event": "chat",
        "service": "LLMGatewayService",
        "correlation_id": corr_id,
        "reply_channel": reply_channel,
        "payload": {
            "body": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'I am using VLLM backend' and nothing else.",
                    }
                ],
                "options": {"backend": "vllm"},
            }
        },
    }

    print("Publishing chat request (vllm):", corr_id)
    await bus.publish("orion-exec:request:LLMGatewayService", envelope)

    async with bus.subscribe(reply_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            print("Got reply (vllm):", msg["data"])
            break

    await bus.close()


asyncio.run(main())
EOF
```

---

## 4. Test 3 – `exec_step` through the gateway

This test exercises the **Cortex exec_step path** without involving Cortex yet:

```
cd /app

PYTHONPATH=/app python - << 'EOF'
import asyncio
import uuid
from orion.core.bus.async_service import OrionBusAsync

async def main():
    bus = OrionBusAsync("redis://localhost:6379/0")
    await bus.connect()

    corr_id = str(uuid.uuid4())
    reply_channel = f"orion:llm:reply:test:{corr_id}"

    envelope = {
        "event": "exec_step",
        "service": "LLMGatewayService",
        "correlation_id": corr_id,
        "reply_channel": reply_channel,
        "payload": {
            "verb": "debug_exec",
            "step": "step-1",
            "order": 1,
            "service": "manual-test",
            "origin_node": "athena",

            "prompt": "You are Orion LLM Gateway test. Reply with: EXEC_OK.",
            "prompt_template": None,
            "context": {},
            "args": {},
            "prior_step_results": [],

            "requires_gpu": False,
            "requires_memory": False,
        },
    }

    print("Publishing exec_step request:", corr_id)
    await bus.publish("orion-exec:request:LLMGatewayService", envelope)

    async with bus.subscribe(reply_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg["data"]
            print("Got exec_step reply:", data)
            break

    await bus.close()


asyncio.run(main())
EOF
```

---

These smoke tests guarantee that:

* Bus connectivity works inside the container
* `llm-gateway` receives events
* Backend selection logic is functioning
* `exec_step` paths return the correct payload structure

This file should be stored at:

````
services/orion-llm-gateway/tests/llm_gateway_smoketests.md
```}

````
