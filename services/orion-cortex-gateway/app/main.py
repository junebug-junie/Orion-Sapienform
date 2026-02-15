from contextlib import asynccontextmanager
from typing import Optional

import logging
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientContext,
    RecallDirective,
    LLMMessage
)
from orion.schemas.cortex.gateway import CortexChatRequest

from .settings import get_settings
from .bus_client import BusClient

# Ensure logs show up in container
logging.basicConfig(level=logging.INFO)

settings = get_settings()
bus_client = BusClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await bus_client.connect()
    await bus_client.start_gateway_consumer()
    yield
    await bus_client.close()

app = FastAPI(title="Orion Cortex Gateway", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.service_name,
        "node": settings.node_name
    }

@app.post("/v1/cortex/chat")
async def chat(req: CortexChatRequest, response: Response):
    # Defaults
    if req.mode in {"agent", "council"}:
        verb = req.verb
    else:
        verb = req.verb or "chat_general"
    # If packs is not provided, default to executive_pack (harness default)
    packs = req.packs if req.packs is not None else ["executive_pack"]

    # Messages
    messages = [LLMMessage(role="user", content=req.prompt)]

    # Context
    context = CortexClientContext(
        messages=messages,
        raw_user_text=req.prompt,
        user_message=req.prompt,
        session_id=req.session_id or "gateway-session",
        user_id=req.user_id or "gateway-user",
        trace_id=req.trace_id,
        metadata=req.metadata or {}
    )

    # Recall
    # If provided, override defaults. If not, use RecallDirective defaults.
    if req.recall:
        # Filter keys to match RecallDirective fields
        valid_keys = RecallDirective.model_fields.keys()
        filtered_recall = {k: v for k, v in req.recall.items() if k in valid_keys}
        recall = RecallDirective(**filtered_recall)
    else:
        recall = RecallDirective() # defaults: enabled=True, etc.

    client_req = CortexClientRequest(
        mode=req.mode,
        verb=verb,
        packs=packs,
        options=req.options or {},
        recall=recall,
        context=context
    )

    try:
        # returns dict or dict-dump of CortexClientResult
        result = await bus_client.rpc_call_cortex_orch(client_req)

        # Try to extract correlation_id for header
        if isinstance(result, dict) and "correlation_id" in result:
             response.headers["X-Orion-Correlation-Id"] = str(result["correlation_id"])

        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Gateway Timeout: Cortex RPC timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
