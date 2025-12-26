import asyncio
import logging
from typing import Any

from orion.core.bus.chassis import ServiceChassis
from orion.core.bus.schemas import BaseEnvelope
from .settings import settings
from .models import ChatBody, GenerateBody, ExecStepPayload, EmbeddingsBody
from .llm_backend import (
    run_llm_chat,
    run_llm_generate,
    run_llm_exec_step,
    run_llm_embeddings,
)

# Setup Logging
logging.basicConfig(level=logging.INFO, format="[LLM-GW] %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("orion-llm-gateway")

# -------------------------------------------------------------------------
# Handlers (The Logic)
# -------------------------------------------------------------------------

async def handle_request(envelope: BaseEnvelope) -> Any:
    """
    Main RPC handler that routes based on envelope.event.
    This effectively replaces the manual if/elif block.
    """
    event = envelope.event
    payload = envelope.payload
    
    # Payload comes as a dict from BaseEnvelope generic, so we access it directly.
    # We might need to handle the nested "body" structure if it persists,
    # but strictly speaking BaseEnvelope.payload should be the body.
    # For backward compatibility with existing senders:
    raw_body = payload.get("body", payload) if isinstance(payload, dict) else payload

    logger.info(f"Handling {event} for {envelope.correlation_id}")

    # Use asyncio.to_thread to run synchronous backend calls in a separate thread.
    # This prevents blocking the Chassis heartbeats and signal handling.
    if event == "chat":
        # Logic extracted from old main.py
        if isinstance(raw_body, dict) and "messages" not in raw_body and "prompt" in raw_body:
             # Backwards compat hack
             raw_body = {
                "model": raw_body.get("model"),
                "messages": [{"role": "user", "content": raw_body.get("prompt")}],
                "options": raw_body.get("options") or {},
                "stream": raw_body.get("stream", False),
                "return_json": raw_body.get("return_json", False),
                "trace_id": raw_body.get("trace_id", envelope.correlation_id),
             }

        body = ChatBody(**raw_body)
        result = await asyncio.to_thread(run_llm_chat, body)
        
        # Format reply
        if isinstance(result, dict):
            return {
                "text": result.get("text") or "",
                "spark_meta": result.get("spark_meta"),
                "raw": result.get("raw"),
            }
        return {"text": str(result)}

    elif event == "generate":
        body = GenerateBody(**raw_body)
        text = await asyncio.to_thread(run_llm_generate, body)
        return {"text": text}

    elif event == "exec_step":
        # Note: exec_step logic in old main.py calculated elapsed time manually.
        # Ideally the chassis or the backend tracks this, but we'll call the backend.
        body = ExecStepPayload(**raw_body)
        result = await asyncio.to_thread(run_llm_exec_step, body)
        # The reply format for exec_step was specific (trace_id, ok, etc.)
        # Ideally we return the result and the caller wraps it, but for now:
        return {
            "result": result,
            "status": "success",
            "ok": True
        }

    elif event == "embeddings":
        body = EmbeddingsBody(**raw_body)
        return await asyncio.to_thread(run_llm_embeddings, body)

    else:
        logger.warning(f"Unknown event: {event}")
        return None

# -------------------------------------------------------------------------
# Main Entry Point ("Hello World" Target)
# -------------------------------------------------------------------------

async def main():
    chassis = ServiceChassis(
        service_name=settings.llm_service_name,
        bus_url=settings.orion_bus_url
    )

    # Register the single Rabbit (RPC) handler for the intake channel
    chassis.register_rpc(settings.channel_llm_intake, handle_request)

    # Go!
    await chassis.run()

if __name__ == "__main__":
    asyncio.run(main())
