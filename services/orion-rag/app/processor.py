import logging
import uuid
from orion.core.bus.service import OrionBus
from .settings import settings
from .vector_store import vector_store

logger = logging.getLogger(settings.SERVICE_NAME)

def process_rag_request(bus: OrionBus, original_message: dict):
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

    # 1. Retrieve context from the vector store
    context_docs = vector_store.search(query, n_results=n_results)
    if not context_docs:
        logger.warning(f"No documents found for query: '{query[:50]}...'")
        context_str = "No specific context found."
    else:
        context_str = "\n\n---\n\n".join(context_docs)

    # 2. Construct a new prompt for the brain
    prompt_for_brain = (
        "Based on the following context, please answer the user's question.\n\n"
        f"CONTEXT:\n{context_str}\n\n"
        f"USER QUESTION:\n{query}"
    )
    
    # 3. Create a unique, temporary channel for the brain's response
    brain_response_channel = f"orion:rag:brain-response:{uuid.uuid4().hex}"
    
    # 4. Publish the request to the brain
    brain_payload = {
        "prompt": prompt_for_brain,
        "response_channel": brain_response_channel
    }
    bus.publish(settings.PUBLISH_CHANNEL_BRAIN_INTAKE, brain_payload)
    logger.info(f"Delegated query to brain. Waiting for response on {brain_response_channel}")
    
    # 5. Wait for the brain's response on the unique channel
    # This is a blocking call within this worker thread. A temporary bus instance
    # is used for the reply to ensure thread safety.
    reply_bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    for brain_message in reply_bus.subscribe(brain_response_channel):
        final_answer = brain_message.get("data", {}).get("text", "The brain did not provide a response.")
        
        # 6. Publish the final answer back to the original requester
        final_payload = {
            "query": query,
            "answer": final_answer,
            "retrieved_context": context_docs
        }
        bus.publish(response_channel, final_payload)
        logger.info(f"Sent final RAG answer to {response_channel}")
        break # Exit the loop after receiving one message
