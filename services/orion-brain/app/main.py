# app/main.py
import os
import json
import uuid
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import threading
import logging

# Ensure necessary imports are present
from app.config import BACKENDS, PORT, READ_TIMEOUT, CONNECT_TIMEOUT
from app.router import router_instance, health_loop, probe_backend, router as health_api_router
from app.bus_helpers import emit_brain_event, emit_brain_output, emit_chat_history_log
from app.models import GenerateBody, ChatBody # Make sure ChatBody is imported
from app import health
from app.health import wait_for_redis
from app.bus_listener import listener_worker

# Configure logging (This is correct)
logging.basicConfig(level=logging.INFO, format="[BRAIN_SVC] %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__) # Add logger instance if needed

# 🧠 Initialize router + app
app = FastAPI(title="Orion Brain Service")

# ✅ Register health API endpoints
app.include_router(health_api_router)

# ───────────────────────────────────────────────
# Startup sequence
# ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # 🕐 Wait until Redis (Orion Bus) is reachable
    await wait_for_redis()

    # 🚀 Run initial backend probe
    await asyncio.gather(*(probe_backend(b) for b in router_instance.backends.values()))

    # ✅ Launch background tasks
    asyncio.create_task(health_loop(router_instance))

    # 🚀 Start the bus listener thread (This is correct)
    logger.info("🚀 Starting bus listener thread...") # Use logger
    threading.Thread(target=listener_worker, daemon=True).start()

# ───────────────────────────────────────────────
# API Endpoints
# ───────────────────────────────────────────────
@app.get("/health")
async def health_summary():
    """Return router health summary."""
    return {"ok": True, "backends": router_instance.list()}

@app.post("/chat")
async def chat(body: ChatBody, request: Request):
    """
    Route chat request to a healthy Ollama backend.
    Handles both chat-style (messages[]) and generate-style (prompt) payloads.
    Emits telemetry via Orion Bus.
    """
    payload = body.model_dump()
    trace_id = payload.get("trace_id") or str(uuid.uuid4())

    backend = router_instance.pick()

    if not backend:
        raise HTTPException(status_code=503, detail="No healthy backends")

    # 🧭 Publish routing decision (Correct)
    emit_brain_event("route.selected", {"trace_id": trace_id, "backend": backend.url})

    # Pick the correct endpoint based on payload structure (Correct)
    endpoint = "chat" if "messages" in payload else "generate"
    url = f"{backend.url.rstrip('/')}/api/{endpoint}"

    data = {} # Initialize data
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                # Log exact backend response for debugging (Correct)
                raw_text = await r.aread()
                err_preview = raw_text[:400].decode(errors="ignore")
                logger.error(f"⚠️ Ollama backend error {r.status_code} from {url}\n{err_preview}") # Use logger
                raise HTTPException(status_code=500, detail=f"Ollama backend error {r.status_code}")

            # Parse JSON safely (Correct)
            try:
                data = r.json()
            except Exception as e:
                logger.error(f"⚠️ Failed to parse JSON from Ollama: {e}", exc_info=True) # Use logger
                raise HTTPException(status_code=500, detail="Invalid response from backend")

    except httpx.RequestError as e:
        logger.error(f"❌ Network error contacting {url}: {e}", exc_info=True) # Use logger
        raise HTTPException(status_code=502, detail=f"Network error contacting backend: {str(e)}")

    except httpx.TimeoutException:
        logger.error(f"⏱️ Timeout contacting {url}") # Use logger
        raise HTTPException(status_code=504, detail="Backend timeout")

    # Extract LLM response text (Correct)
    text = (
        data.get("response")
        or data.get("text")
        or (data.get("message", {}).get("content"))
        or ""
    ).strip()

    # ================================================================
    # --- THIS IS THE PATCH ---
    # ================================================================
    
    # Get the 'source' from the incoming request body.
    # Default to "http" if missing.
    request_source = body.source if hasattr(body, "source") and body.source else "http"
    logger.info(f"[{trace_id}] API /chat endpoint received source: '{request_source}'") # <-- ADD THIS
    # Only log to chat history if it's NOT a dream pre-processor task
    if request_source != "dream_preprocessor":
        emit_chat_history_log({
            "trace_id": trace_id,
            "source": request_source, # Use the dynamic source
            "prompt": body.messages[-1].get("content") if body.messages else body.prompt,
            "response": text,
            # Safely access user_id and session_id
            "user_id": body.user_id if hasattr(body, 'user_id') else None,
            "session_id": body.session_id if hasattr(body, 'session_id') else None
        })
    else:
        # Optional: Log that you skipped it
        logger.info(f"[{trace_id}] Skipping chat history log for source: {request_source}")
    # ================================================================
    # --- END OF PATCH ---
    # ================================================================

    # 🧠 Publish structured LLM output event (Correct)
    emit_brain_output({
        "trace_id": trace_id,
        "text": text or "(empty response)",
        "service": "orion-brain",
        "model": data.get("model"),
        "done_reason": data.get("done_reason", "n/a"),
        "latency_ms": backend.last_latency_ms,
    })

    return {
        "trace_id": trace_id,
        "backend": backend.url,
        "response": text,
        "meta": {
            "model": data.get("model"),
            "latency_ms": backend.last_latency_ms,
            "done_reason": data.get("done_reason", "n/a"),
        },
    }

# --- Optional: Add main block for direct execution if needed ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=PORT)
