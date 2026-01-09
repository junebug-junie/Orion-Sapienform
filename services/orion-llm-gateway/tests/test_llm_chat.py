# scripts/test_llm_gateway_chat.py

import asyncio
import uuid
import time

from orion.core.bus.async_service import OrionBusAsync


def main() -> None:
    asyncio.run(_main_async())


async def _main_async() -> None:
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
                # model is optional now; profile/model resolution will fill it
                "messages": [
                    {
                        "role": "user",
                        "content": "Say 'hello from llm-gateway' and tell me which backend you are using.",
                    }
                ],
                # You can leave these off to exercise legacy defaults:
                # "profile_name": "brain-ollama-7b",
                # "verb": "generic_chat",

                # Or force a backend explicitly:
                "options": {
                    # "backend": "ollama",
                    # "backend": "vllm",
                },
            }
        },
    }

    print(f"Publishing chat request with correlation_id={corr_id}")
    await bus.publish("orion-exec:request:LLMGatewayService", envelope)

    print(f"Subscribing to reply_channel={reply_channel}")
    async with bus.subscribe(reply_channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            data = msg["data"]
            print("Got reply:", data)
            break

    await bus.close()


if __name__ == "__main__":
    main()
