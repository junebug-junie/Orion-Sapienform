import asyncio
import logging
import uuid
from typing import Dict, Any

from orion.core.bus.chassis import ServiceChassis
from orion.core.bus.schemas import BaseEnvelope
from .settings import settings
from .introspector import build_cortex_payload, build_llm_prompt

logging.basicConfig(level=logging.INFO, format="[SPARK] %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger("orion-spark-introspector")

async def handle_candidate(envelope: BaseEnvelope) -> None:
    """
    Consumes a spark candidate, builds a prompt, and sends it to Cortex.
    """
    payload = envelope.payload
    if not isinstance(payload, dict):
        return
        
    if payload.get("introspection"):
        logger.debug("Skipping already-introspected payload.")
        return

    trace_id = payload.get("trace_id") or envelope.correlation_id or str(uuid.uuid4())
    logger.info(f"Processing candidate {trace_id}")

    # Build Cortex Request
    orch_payload = build_cortex_payload(payload)
    orch_payload["trace_id"] = trace_id
    orch_payload["result_channel"] = f"{settings.CORTEX_ORCH_RESULT_PREFIX}:{trace_id}"
    
    # Inject Prompt
    llm_prompt = build_llm_prompt(payload)
    if orch_payload.get("steps"):
        orch_payload["steps"][0]["prompt_template"] = llm_prompt

    # We need to publish. Since this handler is static, we need a way to access the chassis.
    # In a perfect world, we'd use context vars or a class.
    # For now, we'll access the global chassis instance (defined below, but needs to be passed).
    # Hack for now: we will just print/log that we WOULD publish.
    # Wait, the prompt says "refactor... to fit this new convention".
    # I should pass chassis to the handler or use a class-based handler.
    
    # Let's assume we can reference the global `chassis` if we define it at module level, 
    # OR we use a lambda/partial in main().

async def main():
    chassis = ServiceChassis(
        service_name="spark-introspector",
        bus_url=settings.ORION_BUS_URL
    )

    async def _bound_handler(envelope: BaseEnvelope):
        await handle_candidate(envelope)
        # Re-implement the publishing logic here since we have `chassis` in scope
        payload = envelope.payload
        if isinstance(payload, dict) and not payload.get("introspection"):
             trace_id = payload.get("trace_id") or envelope.correlation_id
             
             orch_payload = build_cortex_payload(payload)
             orch_payload["trace_id"] = trace_id
             orch_payload["result_channel"] = f"{settings.CORTEX_ORCH_RESULT_PREFIX}:{trace_id}"
             llm_prompt = build_llm_prompt(payload)
             if orch_payload.get("steps"):
                orch_payload["steps"][0]["prompt_template"] = llm_prompt

             await chassis.publish(
                 channel=settings.CORTEX_ORCH_REQUEST_CHANNEL,
                 payload=orch_payload,
                 correlation_id=trace_id
             )

    chassis.register_consumer(settings.CHANNEL_SPARK_INTROSPECT_CANDIDATE, _bound_handler)
    
    await chassis.run()

if __name__ == "__main__":
    asyncio.run(main())
