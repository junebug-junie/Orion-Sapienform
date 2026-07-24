from contextlib import asynccontextmanager
from typing import Optional

import logging
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from orion.bus.consumer_readiness import bus_consumer_readiness_v1, check_bus_consumer_readiness
from orion.core.bus.bus_service_chassis import ChassisConfig, HeartbeatOnly
from orion.schemas.telemetry.system_health import BusConsumerReadinessV1

from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientContext,
    RecallDirective,
)
from orion.schemas.cortex.gateway import CortexChatRequest

from .settings import get_settings
from .bus_client import BusClient, _context_messages_from_chat_request

# Ensure logs show up in container
logging.basicConfig(level=logging.INFO)

settings = get_settings()
bus_client = BusClient()
heartbeat_chassis: HeartbeatOnly | None = None


def build_heartbeat_chassis() -> HeartbeatOnly:
    """Own, independent bus connection publishing SystemHealthV1 to orion:system:health
    every heartbeat_interval_sec. Deliberately separate from `bus_client`'s own bus/RPC
    forks (see docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md)."""
    return HeartbeatOnly(
        ChassisConfig(
            service_name=settings.service_name,
            service_version=settings.service_version,
            node_name=settings.node_name,
            bus_url=settings.orion_bus_url,
            bus_enabled=True,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global heartbeat_chassis
    await bus_client.connect()
    await bus_client.start_gateway_consumer()
    try:
        heartbeat_chassis = build_heartbeat_chassis()
        await heartbeat_chassis.start_background()
        logging.getLogger("orion.cortex.gateway").info(
            "system_health_heartbeat_started service=%s interval_sec=%s",
            settings.service_name,
            settings.heartbeat_interval_sec,
        )
    except Exception as exc:
        logging.getLogger("orion.cortex.gateway").warning(
            "system_health_heartbeat_start_failed error=%s", exc
        )
        heartbeat_chassis = None
    yield
    if heartbeat_chassis is not None:
        try:
            await heartbeat_chassis.stop()
        except Exception as exc:
            logging.getLogger("orion.cortex.gateway").warning(
                "system_health_heartbeat_stop_error error=%s", exc
            )
        heartbeat_chassis = None
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


@app.get("/ready")
async def ready() -> JSONResponse:
    intake_bus = getattr(bus_client, "_intake_bus", None)
    if intake_bus is None or not getattr(intake_bus, "enabled", False):
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.channel_gateway_request,
            subscriber_count=0,
            dependency_status="unavailable",
            error="intake bus not connected",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    redis = getattr(intake_bus, "redis", None)
    if redis is None:
        body = BusConsumerReadinessV1(
            ok=False,
            http_alive=True,
            bus_consumer_ready=False,
            intake_channel=settings.channel_gateway_request,
            subscriber_count=0,
            dependency_status="unavailable",
            error="redis unavailable",
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=503)

    result = await check_bus_consumer_readiness(
        redis,
        intake_channel=settings.channel_gateway_request,
        service_name=settings.service_name,
        check_heartbeat=False,
    )
    body = bus_consumer_readiness_v1(result, http_alive=True)
    status_code = 200 if body.ok else 503
    return JSONResponse(body.model_dump(mode="json"), status_code=status_code)


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
    messages = _context_messages_from_chat_request(req)
    logging.getLogger("orion.cortex.gateway").info(
        "gateway_context_messages mode=%s count=%s roles=%s",
        req.mode,
        len(messages),
        [m.role for m in messages[:12]],
    )

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

    route_intent = "auto" if req.mode == "auto" else req.route_intent
    options = dict(req.options or {})
    if route_intent == "auto":
        options["route_intent"] = "auto"

    client_req = CortexClientRequest(
        mode=req.mode,
        route_intent=route_intent,
        verb=verb,
        packs=packs,
        options=options,
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
