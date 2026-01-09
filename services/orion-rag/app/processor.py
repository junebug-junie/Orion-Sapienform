import asyncio
import logging
import uuid
import json

from orion.core.bus.async_service import OrionBusAsync
from .settings import settings
from .vector_store import vector_store

logger = logging.getLogger(settings.SERVICE_NAME)

# --- FIX: Remove 'bus: OrionBus' from arguments ---
def process_rag_request(original_message: dict) -> None:
# --- END FIX ---
    """
    Handles a single RAG request: retrieves context, delegates to the brain,
    and publishes the final response.
    """
    asyncio.run(_process_rag_request(original_message))


async def _process_rag_request(original_message: dict) -> None:
    query = original_message.get("query")
    response_channel = original_message.get("response_channel")
    n_results = original_message.get("n_results", 3)

    if not all([query, response_channel]):
        logger.warning("RAG request is missing 'query' or 'response_channel'. Skipping.")
        return

    context_docs = vector_store.search(query, n_results=n_results)
    context_str = "\n\n---\n\n".join(context_docs) if context_docs else "No specific context found."
    prompt_for_brain = (
        "Based on the following context, please answer the user's question.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION:\n{query}"
    )
    brain_response_channel = f"orion:rag:brain-response:{uuid.uuid4().hex}"
    brain_payload = {
        "prompt": prompt_for_brain,
        "response_channel": brain_response_channel,
    }

    bus = OrionBusAsync(url=settings.ORION_BUS_URL, enabled=True)
    await bus.connect()

    try:
        await bus.publish(settings.PUBLISH_CHANNEL_BRAIN_INTAKE, brain_payload)
        logger.info(f"Delegated query to brain. Waiting for response on {brain_response_channel}")

        async with bus.subscribe(brain_response_channel) as pubsub:
            async for brain_message in bus.iter_messages(pubsub):
                try:
                    brain_data = brain_message["data"]
                    if isinstance(brain_data, (bytes, bytearray)):
                        brain_data = brain_data.decode("utf-8", "ignore")
                    if isinstance(brain_data, str):
                        brain_data = json.loads(brain_data)
                    if not isinstance(brain_data, dict):
                        logger.warning("Received non-dict data from brain. Skipping.")
                        break

                    final_answer = brain_data.get("text", "The brain did not provide a valid response.")

                    final_payload = {
                        "query": query,
                        "answer": final_answer,
                        "retrieved_context": context_docs,
                    }

                    await bus.publish(response_channel, final_payload)
                    logger.info(f"Sent final RAG answer to {response_channel}")
                    break

                except Exception as e:
                    logger.error(f"Error processing brain response: {e}", exc_info=True)
                    break
    finally:
        await bus.close()
