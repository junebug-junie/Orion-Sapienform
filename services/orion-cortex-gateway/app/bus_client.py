import asyncio
import logging
from uuid import uuid4
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef, CausalityLink
from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientResult,
    CortexClientContext,
    RecallDirective,
    LLMMessage
)
from orion.schemas.cortex.gateway import CortexChatRequest, CortexChatResult

from .settings import get_settings

logger = logging.getLogger(__name__)

class BusClient:
    def __init__(self):
        self.settings = get_settings()
        self.bus = OrionBusAsync(url=self.settings.orion_bus_url)
        self.reply_prefix = self.settings.channel_cortex_result_prefix

    async def connect(self):
        logger.info(f"Connecting to bus at {self.settings.orion_bus_url}")
        await self.bus.connect()

    async def close(self):
        logger.info("Closing bus connection")
        await self.bus.close()

    def _service_ref(self) -> ServiceRef:
        return ServiceRef(
            name=self.settings.service_name,
            version=self.settings.service_version,
            node=self.settings.node_name
        )

    async def rpc_call_cortex_orch(self, req: CortexClientRequest, correlation_id: Any = None, causality_chain: Any = None) -> Dict[str, Any]:
        corr = correlation_id or uuid4()
        reply_to = f"{self.reply_prefix}:{corr}"

        env = BaseEnvelope(
            kind="cortex.orch.request",
            source=self._service_ref(),
            correlation_id=corr,
            causality_chain=causality_chain,
            reply_to=reply_to,
            payload=req.model_dump(mode="json"),
        )

        logger.info(
            f"RPC Request channel={self.settings.channel_cortex_request} "
            f"correlation_id={corr} reply_to={reply_to}"
        )
        logger.debug(f"RPC Payload correlation_id={corr}: {env.payload}")

        try:
            msg = await self.bus.rpc_request(
                self.settings.channel_cortex_request,
                env,
                reply_channel=reply_to,
                timeout_sec=self.settings.gateway_rpc_timeout_sec
            )
        except TimeoutError as te:
            logger.error(f"RPC Timeout correlation_id={corr}")
            raise TimeoutError(f"RPC timed out after {self.settings.gateway_rpc_timeout_sec}s") from te

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
            logger.error(f"Decode failed correlation_id={corr} error={decoded.error}")
            raise RuntimeError(f"Decode failed: {decoded.error}")

        payload = decoded.envelope.payload

        # Log minimal success
        logger.info(f"RPC Success correlation_id={corr} kind={decoded.envelope.kind}")

        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")

        return payload  # Should be dict or primitive, or BaseEnvelope if unknown

    async def start_gateway_consumer(self):
        logger.info(f"Starting gateway consumer on {self.settings.channel_gateway_request}")
        # Start a background task for subscription
        asyncio.create_task(self._consume_gateway_request())

    async def _consume_gateway_request(self):
        logger.info(f"Gateway consumer loop started. Listening on channel: {self.settings.channel_gateway_request}")
        while True:
            try:
                async with self.bus.subscribe(self.settings.channel_gateway_request) as pubsub:
                    async for msg in self.bus.iter_messages(pubsub):
                        await self.handle_gateway_request(msg)
            except asyncio.CancelledError:
                logger.info("Gateway consumer cancelled")
                break
            except Exception as e:
                logger.error(f"Gateway consumer failed: {e}", exc_info=True)
                await asyncio.sleep(5.0)

    async def handle_gateway_request(self, message: Dict[str, Any]):
        # decode
        decoded = self.bus.codec.decode(message.get("data"))
        if not decoded.ok:
            logger.error(f"Gateway decode failed: {decoded.error}")
            return

        env = decoded.envelope
        if env.kind != "cortex.gateway.chat.request":
            logger.debug(f"Ignoring kind: {env.kind}")
            return

        try:
            logger.info(f"🔔 Received request from Hub: correlation_id={env.correlation_id} source={env.source}")
            logger.info(f"Processing gateway request correlation_id={env.correlation_id}")
            # Validate payload
            req = CortexChatRequest.model_validate(env.payload)

            # Logic similar to HTTP endpoint
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
                recall = RecallDirective() # defaults: enabled=True, etc.

            client_req = CortexClientRequest(
                mode=req.mode,
                verb=verb,
                packs=packs,
                options=req.options or {},
                recall=recall,
                context=context
            )

            # RPC call to Orch
            # Append causality chain
            parent_link = CausalityLink(
                correlation_id=env.correlation_id,
                kind=env.kind,
                source=env.source,
                created_at=env.created_at
            )
            chain = (env.causality_chain or []) + [parent_link]

            orch_result_dict = await self.rpc_call_cortex_orch(
                client_req,
                correlation_id=env.correlation_id,
                causality_chain=chain
            )

            # Wrap result
            cortex_res_obj = CortexClientResult.model_validate(orch_result_dict)

            res_payload = CortexChatResult(
                cortex_result=cortex_res_obj,
                final_text=cortex_res_obj.final_text
            )

            # Reply
            if env.reply_to:
                reply_env = BaseEnvelope(
                    kind="cortex.gateway.chat.result",
                    source=self._service_ref(),
                    correlation_id=env.correlation_id,
                    causality_chain=chain,
                    payload=res_payload.model_dump(mode="json")
                )
                await self.bus.publish(env.reply_to, reply_env)
                logger.info(f"Sent reply to {env.reply_to}")
            else:
                logger.warning(f"No reply_to in gateway request correlation_id={env.correlation_id}")

        except Exception as e:
            logger.error(f"Error handling gateway request: {e}", exc_info=True)
            # We could send an error reply here if we wanted
