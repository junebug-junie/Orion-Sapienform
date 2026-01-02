import logging
import asyncio
from uuid import uuid4
from typing import Any, Dict

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.cortex.contracts import (
    CortexClientRequest,
    CortexClientResult,
    CortexClientContext,
    RecallDirective,
    LLMMessage,
    CortexChatRequest
)
from orion.schemas.cortex.gateway import CortexChatResult

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
        """
        Starts the background task to listen for incoming Gateway requests (from Hub).
        """
        # Create a task so it runs in background
        asyncio.create_task(self._gateway_consumer_loop())

    async def _gateway_consumer_loop(self):
        channel = self.settings.channel_cortex_gateway_request
        logger.info(f"Starting gateway consumer on {channel}")

        if not self.bus.enabled:
             logger.warning("Bus disabled, skipping consumer loop")
             return

        async with self.bus.subscribe(channel) as pubsub:
            async for msg in self.bus.iter_messages(pubsub):
                try:
                    data = msg.get("data")
                    decoded = self.bus.codec.decode(data)

                    if not decoded.ok:
                        logger.warning(f"Gateway decode failed: {decoded.error}")
                        continue

                    # Process asynchronously
                    asyncio.create_task(self.handle_gateway_request(decoded.envelope))

                except Exception as e:
                     logger.error(f"Error in gateway consumer loop: {e}", exc_info=True)

    async def handle_gateway_request(self, env: BaseEnvelope):
        if env.kind != "cortex.gateway.request":
            logger.debug(f"Ignoring kind: {env.kind}")
            return

        try:
            logger.info(f"Processing gateway request correlation_id={env.correlation_id}")
            # Validate payload
            req = CortexChatRequest.model_validate(env.payload)

            # Logic similar to HTTP endpoint
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
            chain = env.causality_chain or []
            chain.append(f"{self.settings.service_name}:{env.correlation_id}")

            orch_result_dict = await self.rpc_call_cortex_orch(
                client_req,
                correlation_id=env.correlation_id,
                causality_chain=chain
            )

            # Wrap result - we can just return the raw dict in payload or typed result
            # The previous worker returned the raw dict from CortexClientResult dump
            # but incoming code wanted CortexChatResult. Let's support both or stick to one.
            # Hub expects a CortexClientResult-like dict.

            # The incoming code wanted to wrap it in `CortexChatResult`.
            # Let's see if we can use that.

            try:
                cortex_res_obj = CortexClientResult.model_validate(orch_result_dict)
                res_payload = CortexChatResult(
                    cortex_result=cortex_res_obj,
                    final_text=cortex_res_obj.final_text
                )
                final_payload = res_payload.model_dump(mode="json")
                reply_kind = "cortex.gateway.chat.result"
            except Exception:
                # Fallback to just returning the dict if validation fails (robustness)
                final_payload = orch_result_dict
                reply_kind = "cortex.gateway.result"

            # Reply
            if env.reply_to:
                reply_env = env.derive_child(
                    kind=reply_kind,
                    source=self._service_ref(),
                    payload=final_payload,
                    reply_to=None
                )
                await self.bus.publish(env.reply_to, reply_env)
                logger.info(f"Sent reply to {env.reply_to}")
            else:
                logger.warning(f"No reply_to in gateway request correlation_id={env.correlation_id}")

        except Exception as e:
            logger.error(f"Error handling gateway request: {e}", exc_info=True)
            # Send error
            if env.reply_to:
                err_env = env.derive_child(
                    kind="system.error",
                    source=self._service_ref(),
                    payload={"error": str(e)},
                    reply_to=None
                )
                await self.bus.publish(env.reply_to, err_env)
