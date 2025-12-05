# scripts/test_llm_gateway_chat.py

import uuid
import time

from orion.core.bus.service import OrionBus


def main():
    bus = OrionBus()
    if not bus.enabled:
        print("Bus not enabled / not connected")
        return

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
    bus.publish("orion-exec:request:LLMGatewayService", envelope)

    print(f"Subscribing to reply_channel={reply_channel}")
    for msg in bus.raw_subscribe(reply_channel):
        data = msg["data"]
        print("Got reply:", data)
        break


if __name__ == "__main__":
    main()
