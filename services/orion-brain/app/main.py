# app/main.py
import os, json, uuid, asyncio, httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from app.config import BACKENDS, PORT, READ_TIMEOUT, CONNECT_TIMEOUT
from app.router import BrainRouter, health_loop, probe_backend
from app.bus_helpers import emit_event, emit_bus
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
    """Route chat request to a healthy LLM backend."""
    payload = body.model_dump()
    trace_id = payload.get("trace_id") or str(uuid.uuid4())

    backend = router.pick()
    if not backend:
        raise HTTPException(503, "No healthy backends")

    await emit_event("route.selected", {"trace_id": trace_id, "backend": backend.url})

    async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
        r = await client.post(f"{backend.url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    await emit_bus("llm.response", {"trace_id": trace_id, "text": data.get("response")})
    return data
