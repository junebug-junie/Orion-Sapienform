# app/main.py
import os, json, uuid, asyncio, httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.config import BACKENDS, PORT, READ_TIMEOUT, CONNECT_TIMEOUT
from app.router import BrainRouter, health_loop, probe_backend
from app.bus_helpers import emit_brain_event, emit_brain_output
from app.models import GenerateBody, ChatBody
from app import health
from app.health import wait_for_redis

# 🧠 Initialize router + app
router = BrainRouter(BACKENDS)
app = FastAPI(title="Orion Brain Service")

# ✅ Register health API endpoints
app.include_router(health.router)

# ───────────────────────────────────────────────
# Startup sequence
# ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    # 🕐 Wait until Redis (Orion Bus) is reachable
    await wait_for_redis()

    # 🚀 Run initial backend probe
    await asyncio.gather(*(probe_backend(b) for b in router.backends.values()))

    # ✅ Launch background tasks
    asyncio.create_task(health.startup_checks())
    asyncio.create_task(health_loop(router))

# ───────────────────────────────────────────────
# API Endpoints
# ───────────────────────────────────────────────
@app.get("/health")
async def health_summary():
    """Return router health summary."""
    return {"ok": True, "backends": router.list()}


@app.post("/chat")
async def chat(body: ChatBody, request: Request):
    """
    Route chat request to a healthy Ollama backend.
    Handles both chat-style (messages[]) and generate-style (prompt) payloads.
    Emits telemetry via Orion Bus.
    """
    payload = body.model_dump()
    trace_id = payload.get("trace_id") or str(uuid.uuid4())

    backend = router.pick()
    if not backend:
        raise HTTPException(status_code=503, detail="No healthy backends")

    # 🧭 Publish routing decision
    await emit_brain_event("route.selected", {"trace_id": trace_id, "backend": backend.url})

    # Pick the correct endpoint based on payload structure
    endpoint = "chat" if "messages" in payload else "generate"
    url = f"{backend.url.rstrip('/')}/api/{endpoint}"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
            r = await client.post(url, json=payload)
            if r.status_code != 200:
                # Log exact backend response for debugging
                raw_text = await r.aread()
                err_preview = raw_text[:400].decode(errors="ignore")
                print(f"⚠️  Ollama backend error {r.status_code} from {url}\n{err_preview}")
                raise HTTPException(status_code=500, detail=f"Ollama backend error {r.status_code}")

            # Parse JSON safely
            try:
                data = r.json()
            except Exception as e:
                print(f"⚠️  Failed to parse JSON from Ollama: {e}")
                raise HTTPException(status_code=500, detail="Invalid response from backend")

    except httpx.RequestError as e:
        print(f"❌  Network error contacting {url}: {e}")
        raise HTTPException(status_code=502, detail=f"Network error contacting backend: {str(e)}")

    except httpx.TimeoutException:
        print(f"⏱️  Timeout contacting {url}")
        raise HTTPException(status_code=504, detail="Backend timeout")

    # Emit LLM response telemetry
    text = (
        data.get("response")
        or data.get("text")
        or (data.get("message", {}).get("content"))
        or ""
    ).strip()

    # 🧠 Publish structured LLM output event
    await emit_brain_output({
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
