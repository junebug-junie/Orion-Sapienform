import httpx
import uuid
import json
import logging
from orion.core.bus.service import OrionBus
from app.config import (
    CONNECT_TIMEOUT, READ_TIMEOUT, ORION_BUS_URL, ORION_BUS_ENABLED
)
from app.router import router_instance
from app.bus_helpers import emit_brain_event, emit_brain_output, emit_chat_history_log

logger = logging.getLogger(__name__)

def process_brain_request(payload: dict):
    """
    Handles a single request from the bus.
    Converts the prompt to the /api/chat format.
    """
    trace_id = payload.get("trace_id") or str(uuid.uuid4())
    prompt_text = payload.get("prompt", "No prompt provided.")
    logger.info(f"[{trace_id}] Processing bus request: {prompt_text[:50]}...")

    response_channel = payload.get("response_channel")
    if not response_channel:
        logger.warning(f"[{trace_id}] Bus request missing 'response_channel'. Discarding.")
        return

    backend = router_instance.pick()
    if not backend:
        logger.error(f"[{trace_id}] No healthy backends available. Cannot process request.")
        return

    emit_brain_event("route.selected", {"trace_id": trace_id, "backend": backend.url})

    ollama_payload = {
        "model": "mistral:instruct", # <-- UPDATE ME
        "messages": [
            {"role": "user", "content": payload.get("prompt")}
        ],
        "stream": False
    }

    url = f"{backend.url.rstrip('/')}/api/chat"

    data = {}

    try:
        with httpx.Client(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = client.post(url, json=ollama_payload)
            if r.status_code != 200:
                logger.error(f"[{trace_id}] Ollama backend error {r.status_code} from {url}. Response: {r.text}")
                return # Give up
            data = r.json()
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to contact backend {url}: {e}")
        return # Give up

    # --- Parse response (Chat format is different) ---
    text = ""
    if "message" in data and "content" in data["message"]:
        text = data["message"]["content"].strip()

        emit_chat_history_log({
            "trace_id": trace_id,
            "source": "bus",
            "prompt": payload.get("prompt"),
            "response": text
        })

    else:
        logger.warning(f"[{trace_id}] No 'message.content' in response: {data}")

    emit_brain_output({
        "trace_id": trace_id,
        "text": text or "(empty response)",
        "service": "orion-brain",
        "model": data.get("model"),
    })

    # --- Send the final reply back to the RAG service ---
    try:
        reply_bus = OrionBus(url=ORION_BUS_URL, enabled=ORION_BUS_ENABLED)
        reply_payload = {
            "trace_id": trace_id,
            "text": text,
            "meta": data
        }
        reply_bus.publish(response_channel, reply_payload)
        logger.info(f"[{trace_id}] Sent final reply to {response_channel}")
    except Exception as e:
        logger.error(f"[{trace_id}] Failed to publish reply to {response_channel}: {e}")
