import logging
import uuid
import json
from orion.core.bus.service import OrionBus
from .settings import settings
from .vector_store import vector_store

logger = logging.getLogger(settings.SERVICE_NAME)

# --- FIX: Remove 'bus: OrionBus' from arguments ---
def process_rag_request(original_message: dict):
# --- END FIX ---
    """
    Handles a single RAG request: retrieves context, delegates to the brain,
    and publishes the final response.
    """
    query = original_message.get("query")
    response_channel = original_message.get("response_channel")
    n_results = original_message.get("n_results", 3)
    
    if not all([query, response_channel]):
        logger.warning("RAG request is missing 'query' or 'response_channel'. Skipping.")
        return

    # ... (Vector search, prompt building, etc. is all fine) ...
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
        "response_channel": brain_response_channel
    }

    # --- FIX: Create a new, local bus for publishing ---
    try:
        pub_bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
        pub_bus.publish(settings.PUBLISH_CHANNEL_BRAIN_INTAKE, brain_payload)
    except Exception as e:
        logger.error(f"Failed to publish request to brain: {e}", exc_info=True)
        return
    # --- END FIX ---

    logger.info(f"Delegated query to brain. Waiting for response on {brain_response_channel}")
    
    # This part was already correct (it creates its own reply_bus)
    reply_bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    for brain_message in reply_bus.subscribe(brain_response_channel):
        
        if not isinstance(brain_message, dict) or brain_message.get('type') != 'message':
            continue

        try:
            brain_data = brain_message['data']
            if not isinstance(brain_data, dict):
                logger.warning("Received non-dict data from brain. Skipping.")
                break 

            final_answer = brain_data.get("text", "The brain did not provide a valid response.")
            
            final_payload = {
                "query": query,
                "answer": final_answer,
                "retrieved_context": context_docs
            }
            
            # --- FIX: Use the 'pub_bus' to send the final reply ---
            pub_bus.publish(response_channel, final_payload)
            # --- END FIX ---
            
            logger.info(f"Sent final RAG answer to {response_channel}")
            break 
            
        except Exception as e:
            logger.error(f"Error processing brain response: {e}", exc_info=True)
            break
