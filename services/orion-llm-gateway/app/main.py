# services/orion-llm-gateway/app/main.py

import logging

from orion.core.bus.service import OrionBus  # same as brain
from .settings import settings
from .models import ExecutionEnvelope, ChatBody, GenerateBody
from .llm_backend import run_llm_chat, run_llm_generate

logger = logging.getLogger("orion-llm-gateway")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[LLM-GW] %(levelname)s - %(name)s - %(message)s",
    )

    bus = OrionBus(
        url=settings.orion_bus_url,
        enabled=settings.orion_bus_enabled,
    )

    if not bus.enabled:
        logger.error("Orion bus is disabled; exiting.")
        return

    logger.info(
        f"Starting {settings.service_name} v{settings.service_version} "
        f"listening on {settings.channel_llm_intake}"
    )

    for msg in bus.subscribe(settings.channel_llm_intake):
        if msg.get("type") != "message":
            continue

        try:
            data = msg["data"]
            envelope = ExecutionEnvelope(**data)

            logger.info(
                f"Received event={envelope.event} "
                f"corr_id={envelope.correlation_id}"
            )

            body_dict = envelope.payload.get("body", {})

            if envelope.event == "chat":
                body = ChatBody(**body_dict)
                text = run_llm_chat(body)

            elif envelope.event == "generate":
                body = GenerateBody(**body_dict)
                text = run_llm_generate(body)

            else:
                logger.warning(f"Unknown event type: {envelope.event}")
                continue

            reply = {
                "event": f"{envelope.event}_result",
                "service": settings.llm_service_name,
                "correlation_id": envelope.correlation_id,
                "payload": {
                    "text": text,
                },
            }

            bus.publish(envelope.reply_channel, reply)

            logger.info(
                f"Published {envelope.event}_result corr_id={envelope.correlation_id} "
                f"reply_channel={envelope.reply_channel}"
            )

        except Exception as e:
            logger.exception(f"Error processing message on {msg.get('channel')}: {e}")


if __name__ == "__main__":
    main()
