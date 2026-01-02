# services/orion-cortex-gateway/app/worker.py

import asyncio
import logging
from typing import Any, Dict, Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import (
    CortexChatRequest,
    CortexClientRequest,
    CortexClientContext,
    RecallDirective,
    LLMMessage,
)
from .settings import get_settings
from .bus_client import BusClient

logger = logging.getLogger("orion-cortex-gateway.worker")


async def process_gateway_request(
    bus: OrionBusAsync,
    bus_client: BusClient,
    envelope: BaseEnvelope,
) -> None:
    """
    Handles a single Cortex Chat Request from the bus (e.g. from Hub).
    """
    settings = get_settings()

    # 1. Parse Request
    try:
        if envelope.kind != "cortex.gateway.request":
            # Just ignore unknown kinds if sharing a channel, or log warn
            return

        req = CortexChatRequest.model_validate(envelope.payload)
    except Exception as e:
        logger.warning(
            "[%s] Invalid payload for cortex.gateway.request: %s",
            envelope.correlation_id,
            e,
        )
        return

    reply_to = envelope.reply_to
    if not reply_to:
        logger.warning("[%s] No reply_to. Dropping.", envelope.correlation_id)
        return

    logger.info(
        "[%s] Processing Gateway Chat Request -> %s",
        envelope.correlation_id,
        reply_to,
    )

    # 2. Convert to CortexClientRequest (Logic mirrored from main.py)
    # Defaults
    verb = req.verb if req.verb else "chat_general"
    packs = req.packs if req.packs is not None else ["executive_pack"]

    messages = [LLMMessage(role="user", content=req.prompt)]

    context = CortexClientContext(
        messages=messages,
        session_id=req.session_id or "gateway-session-bus",
        user_id=req.user_id or "gateway-user-bus",
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

    # 3. Call Orchestrator (RPC)
    try:
        # result is typically a Dict (CortexClientResult dump)
        result_payload = await bus_client.rpc_call_cortex_orch(client_req)

        # 4. Reply to Caller (Hub)
        # We wrap the result in an envelope
        response_env = envelope.derive_child(
            kind="cortex.gateway.result",
            source=ServiceRef(
                name=settings.service_name,
                version=settings.service_version,
                node=settings.node_name
            ),
            payload=result_payload,
            reply_to=None
        )

        await bus.publish(reply_to, response_env)
        logger.info("[%s] Sent Gateway Reply", envelope.correlation_id)

    except Exception as e:
        logger.error(
            "[%s] Gateway RPC failed: %s",
            envelope.correlation_id,
            e,
            exc_info=True,
        )
        # Send error reply
        err_env = envelope.derive_child(
            kind="system.error",
            source=ServiceRef(
                name=settings.service_name,
                version=settings.service_version,
            ),
            payload={"error": str(e)},
        )
        await bus.publish(reply_to, err_env)


async def listener_worker(bus_client: BusClient) -> None:
    """
    Subscribes to CHANNEL_CORTEX_GATEWAY_REQUEST.
    """
    settings = get_settings()
    bus = bus_client.bus  # Use the underlying bus from the shared client

    if not bus or not bus.enabled:
        logger.error("Bus not enabled/connected in listener_worker")
        return

    channel = settings.channel_cortex_gateway_request
    logger.info("👂 Subscribing to Gateway Request Channel: %s", channel)

    async with bus.subscribe(channel) as pubsub:
        async for msg in bus.iter_messages(pubsub):
            try:
                data = msg.get("data")
                decoded = bus.codec.decode(data)

                if not decoded.ok:
                    logger.warning("Decode failed on %s: %s", channel, decoded.error)
                    continue

                # Process async
                asyncio.create_task(
                    process_gateway_request(bus, bus_client, decoded.envelope)
                )

            except Exception:
                logger.error("Error in gateway listener loop", exc_info=True)
