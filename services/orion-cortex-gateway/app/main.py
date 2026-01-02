from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientContext,
    RecallDirective,
    LLMMessage
)

from .settings import get_settings
from .models import CortexChatRequest
from .bus_client import BusClient

settings = get_settings()
bus_client = BusClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await bus_client.connect()
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
    # If verb is not provided, default to chat_general
    verb = req.verb if req.verb else "chat_general"
    # If packs is not provided, default to executive_pack (harness default)
    packs = req.packs if req.packs is not None else ["executive_pack"]

    # Messages
    messages = [LLMMessage(role="user", content=req.prompt)]

    # Context
    context = CortexClientContext(
        messages=messages,
        session_id=req.session_id or "gateway-session",
        user_id=req.user_id or "gateway-user",
        trace_id=req.trace_id,
        metadata=req.metadata or {}
    )

    # Recall
    # If provided, override defaults. If not, use RecallDirective defaults.
    if req.recall:
        recall = RecallDirective(**req.recall)
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
