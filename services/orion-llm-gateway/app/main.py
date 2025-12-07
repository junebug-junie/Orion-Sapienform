import logging
import time
import threading

from orion.core.bus.service import OrionBus

from .settings import settings
from .models import (
    ExecutionEnvelope,
    ChatBody,
    GenerateBody,
    ExecStepPayload,
    EmbeddingsBody,
)
from .llm_backend import (
    run_llm_chat,
    run_llm_generate,
    run_llm_exec_step,
    run_llm_embeddings,
)

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

    # Main LLM loop (chat / generate / exec_step / embeddings)
    for msg in bus.subscribe(settings.channel_llm_intake):
        if msg.get("type") != "message":
            continue

        try:
            data = msg["data"]
            envelope = ExecutionEnvelope(**data)

            logger.info(
                "Received event=%s service=%s corr_id=%s",
                envelope.event,
                envelope.service,
                envelope.correlation_id,
            )

            # -------------------------
            # CHAT
            # -------------------------
            if envelope.event == "chat":
                raw = envelope.payload.get("body", envelope.payload) or {}

                # Backwards-compat: allow simple prompt-only payloads
                if "messages" not in raw and "prompt" in raw:
                    prompt = raw.get("prompt") or ""
                    raw = {
                        "model": raw.get("model"),
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        "options": raw.get("options") or {},
                        "stream": raw.get("stream", False),
                        "return_json": raw.get("return_json", False),
                        "trace_id": raw.get("trace_id", envelope.correlation_id),
                        "user_id": raw.get("user_id"),
                        "session_id": raw.get("session_id"),
                        "source": raw.get("source", "brain-cortex"),
                    }

                body = ChatBody(**raw)
                result = run_llm_chat(body)

                # New vLLM path returns dict; legacy path returns string.
                if isinstance(result, dict):
                    text = result.get("text") or ""
                    spark_meta = result.get("spark_meta")
                    raw_llm = result.get("raw")
                else:
                    text = str(result)
                    spark_meta = None
                    raw_llm = None

                reply = {
                    "event": "chat_result",
                    "service": settings.llm_service_name,
                    "correlation_id": envelope.correlation_id,
                    "payload": {
                        "text": text,
                        "spark_meta": spark_meta,
                        "raw": raw_llm,
                    },
                }
                bus.publish(envelope.reply_channel, reply)

                logger.info(
                    "Published chat_result corr_id=%s reply_channel=%s",
                    envelope.correlation_id,
                    envelope.reply_channel,
                )

            # -------------------------
            # GENERATE
            # -------------------------
            elif envelope.event == "generate":
                body = GenerateBody(**envelope.payload.get("body", envelope.payload))
                text = run_llm_generate(body)

                reply = {
                    "event": "generate_result",
                    "service": settings.llm_service_name,
                    "correlation_id": envelope.correlation_id,
                    "payload": {
                        "text": text,
                    },
                }
                bus.publish(envelope.reply_channel, reply)

                logger.info(
                    "Published generate_result corr_id=%s reply_channel=%s",
                    envelope.correlation_id,
                    envelope.reply_channel,
                )

            # -------------------------
            # CORTEX EXEC STEP
            # -------------------------
            elif envelope.event == "exec_step":
                t0 = time.time()
                body = ExecStepPayload(**envelope.payload)

                result = run_llm_exec_step(body)
                elapsed_ms = int((time.time() - t0) * 1000)

                reply = {
                    "trace_id": envelope.correlation_id,
                    "service": envelope.service,
                    "ok": True,
                    "elapsed_ms": elapsed_ms,
                    "result": result,
                    "artifacts": {},
                    "status": "success",
                }

                bus.publish(envelope.reply_channel, reply)

                logger.info(
                    "Published exec_step result corr_id=%s reply_channel=%s elapsed_ms=%d",
                    envelope.correlation_id,
                    envelope.reply_channel,
                    elapsed_ms,
                )

            # -------------------------
            # EMBEDDINGS
            # -------------------------
            elif envelope.event == "embeddings":
                body = EmbeddingsBody(**envelope.payload.get("body", envelope.payload))
                data_out = run_llm_embeddings(body)

                reply = {
                    "event": "embeddings_result",
                    "service": settings.llm_service_name,
                    "correlation_id": envelope.correlation_id,
                    "payload": data_out,
                }
                bus.publish(envelope.reply_channel, reply)

                logger.info(
                    "Published embeddings_result corr_id=%s reply_channel=%s",
                    envelope.correlation_id,
                    envelope.reply_channel,
                )

            else:
                logger.warning("Unknown event type: %s", envelope.event)

        except Exception as e:
            logger.exception(
                "Error processing message on %s: %s",
                msg.get("channel"),
                e,
            )


if __name__ == "__main__":
    main()
