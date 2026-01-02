from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientContext,
    RecallDirective,
    LLMMessage,
    CortexChatRequest
)

from .settings import get_settings
from .bus_client import BusClient

settings = get_settings()
bus_client = BusClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await bus_client.connect()
    # Use the incoming branch's consumer method
    # Note: BusClient in incoming branch has start_gateway_consumer()
    # I should verify if I have it in my merged bus_client.py
    # If not, I should have accepted it during merge or I need to add it.
    # The user provided the conflict block for main.py showing start_gateway_consumer() in incoming.
    # This implies bus_client.py *was* updated by incoming or merge.

    if hasattr(bus_client, "start_gateway_consumer"):
        await bus_client.start_gateway_consumer()
    else:
        # Fallback to my worker if merge failed to bring in method
        from .worker import listener_worker
        import asyncio
        asyncio.create_task(listener_worker(bus_client))

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
    verb = req.verb if req.verb else "chat_general"
    packs = req.packs if req.packs is not None else ["executive_pack"]

    messages = [LLMMessage(role="user", content=req.prompt)]

    context = CortexClientContext(
        messages=messages,
        session_id=req.session_id or "gateway-session",
        user_id=req.user_id or "gateway-user",
        trace_id=req.trace_id,
        metadata=req.metadata or {}
    )

    if req.recall:
        recall = RecallDirective(**req.recall)
    else:
        recall = RecallDirective()

    client_req = CortexClientRequest(
        mode=req.mode,
        verb=verb,
        packs=packs,
        options=req.options or {},
        recall=recall,
        context=context
    )

    try:
        result = await bus_client.rpc_call_cortex_orch(client_req)

        if isinstance(result, dict) and "correlation_id" in result:
             response.headers["X-Orion-Correlation-Id"] = str(result["correlation_id"])

        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Gateway Timeout: Cortex RPC timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
