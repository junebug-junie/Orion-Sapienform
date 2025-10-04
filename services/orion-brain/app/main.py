# app/main.py
import os, json, uuid, time, asyncio, httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from app.config import BACKENDS, PORT, READ_TIMEOUT, CONNECT_TIMEOUT
from app.router import BrainRouter, health_loop, probe_backend
from app.bus_helpers import emit_event, emit_bus
from app.models import GenerateBody, ChatBody
from app import health

router = BrainRouter(BACKENDS)
app = FastAPI(title="Orion Brain Service")

# ✅ this is the health router defined inside app/health.py
app.include_router(health.router)

@app.on_event("startup")
async def startup():
    # Run initial backend probe
    await asyncio.gather(*(probe_backend(b) for b in router.backends.values()))
    # ✅ launch health checks in background
    asyncio.create_task(health.startup_checks())
    asyncio.create_task(health_loop(router))

@app.get("/health")
async def health_summary():
    """Return router health summary"""
    return {"ok": True, "backends": router.list()}

@app.post("/chat")
async def chat(body: ChatBody, request: Request):
    payload = body.model_dump()
    trace_id = payload.get("trace_id") or str(uuid.uuid4())
    backend = router.pick()
    if not backend:
        raise HTTPException(503, "No healthy backends")

    await emit_event("route.selected", {"trace_id": trace_id, "backend": backend.url})
    async with httpx.AsyncClient(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
        r = await client.post(f"{backend.url}/api/chat", json=payload)
        data = r.json()

    await emit_bus("llm.response", {"trace_id": trace_id, "text": data.get("response")})
    return data
