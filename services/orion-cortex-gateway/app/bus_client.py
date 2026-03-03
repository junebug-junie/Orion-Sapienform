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

        async def _wait_for_matching_reply() -> Dict[str, Any]:
            async with self.bus.subscribe(reply_to) as pubsub:
                await self.bus.publish(
                    self.settings.channel_cortex_request,
                    env,
                )
                async for msg in self.bus.iter_messages(pubsub):
                    decoded = self.bus.codec.decode(msg.get("data"))
                    if not decoded.ok:
                        logger.warning(f"Decode failed correlation_id={corr} error={decoded.error}")
                        continue

                    payload = decoded.envelope.payload
                    payload_dict: Dict[str, Any] | None = None
                    if isinstance(payload, dict):
                        payload_dict = payload
                    elif hasattr(payload, "model_dump"):
                        payload_dict = payload.model_dump(mode="json")

                    if payload_dict is not None:
                        verb = payload_dict.get("verb") or payload_dict.get("verb_name")
                        if verb == "introspect_spark" and req.verb != "introspect_spark":
                            logger.info(
                                "RPC reply filtered corr=%s verb=%s expected=%s",
                                corr,
                                verb,
                                req.verb,
                            )
                            continue
                        logger.info(f"RPC Success correlation_id={corr} kind={decoded.envelope.kind}")
                        return payload_dict

                    logger.info(f"RPC Success correlation_id={corr} kind={decoded.envelope.kind}")
                    return payload  # Should be dict or primitive, or BaseEnvelope if unknown
            raise RuntimeError("RPC reply subscription closed without a match.")

        try:
            return await asyncio.wait_for(
                _wait_for_matching_reply(),
                timeout=self.settings.gateway_rpc_timeout_sec,
            )
        except asyncio.TimeoutError as te:
            logger.error(f"RPC Timeout correlation_id={corr}")
            raise TimeoutError(f"RPC timed out after {self.settings.gateway_rpc_timeout_sec}s") from te

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


    async def _publish_gateway_reply(
        self,
        *,
        reply_to: str | None,
        correlation_id: Any,
        causality_chain: list[Any] | None,
        payload: CortexChatResult,
    ) -> None:
        if not reply_to:
            logger.warning(f"No reply_to in gateway request correlation_id={correlation_id}")
            return

        reply_env = BaseEnvelope(
            kind="cortex.gateway.chat.result",
            source=self._service_ref(),
            correlation_id=correlation_id,
            causality_chain=causality_chain or [],
            payload=payload.model_dump(mode="json"),
        )
        await self.bus.publish(reply_to, reply_env)
        logger.info(f"Sent reply to {reply_to}")

    def _build_error_chat_result(
        self,
        *,
        correlation_id: Any,
        message: str,
        mode: str = "unknown",
        verb: str = "unknown",
        error_type: str = "gateway_error",
    ) -> CortexChatResult:
        cortex_result = CortexClientResult(
            ok=False,
            mode=mode or "unknown",
            verb=verb or "unknown",
            status="fail",
            final_text=f"Request failed: {message}",
            memory_used=False,
            steps=[],
            error={"message": message, "type": error_type},
            correlation_id=str(correlation_id) if correlation_id is not None else None,
            metadata={"source": self.settings.service_name},
        )
        return CortexChatResult(cortex_result=cortex_result, final_text=cortex_result.final_text)

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
            if req.mode in {"agent", "council"}:
                verb = req.verb
            else:
                verb = req.verb or "chat_general"
            packs = req.packs if req.packs is not None else ["executive_pack"]
            messages = [LLMMessage(role="user", content=req.prompt)]

            context = CortexClientContext(
                messages=messages,
                raw_user_text=req.prompt,
                user_message=req.prompt,
                session_id=req.session_id or "gateway-session",
                user_id=req.user_id or "gateway-user",
                trace_id=req.trace_id,
                metadata=req.metadata or {}
            )

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
            try:
                cortex_res_obj = CortexClientResult.model_validate(orch_result_dict)
            except Exception as parse_error:
                logger.error(
                    "CortexClientResult validation failed corr=%s; sending failure payload",
                    env.correlation_id,
                    exc_info=True,
                )
                failed_payload = self._build_error_chat_result(
                    correlation_id=env.correlation_id,
                    message=f"Invalid Cortex result payload: {parse_error}",
                    mode=req.mode,
                    verb=(req.verb or "chat_general"),
                    error_type="invalid_cortex_result",
                )
                await self._publish_gateway_reply(
                    reply_to=env.reply_to,
                    correlation_id=env.correlation_id,
                    causality_chain=chain,
                    payload=failed_payload,
                )
                logger.info("Sent gateway error reply corr=%s reason=invalid_cortex_result", env.correlation_id)
                return

            executed_verbs: list[str] = []
            if isinstance(cortex_res_obj.metadata, dict):
                raw_executed = cortex_res_obj.metadata.get("executed_verbs") or []
                if isinstance(raw_executed, list):
                    executed_verbs = [str(v) for v in raw_executed]
            selected_verb = (
                cortex_res_obj.metadata.get("trace_verb")
                if isinstance(cortex_res_obj.metadata, dict)
                else None
            ) or cortex_res_obj.verb
            spark_introspection_triggered = "introspect_spark" in executed_verbs

            res_payload = CortexChatResult(
                cortex_result=cortex_res_obj,
                final_text=cortex_res_obj.final_text
            )

            logger.info(
                "Selected cortex reply corr=%s selected_verb=%s spark_introspection_triggered=%s",
                env.correlation_id,
                selected_verb,
                spark_introspection_triggered,
            )

            await self._publish_gateway_reply(
                reply_to=env.reply_to,
                correlation_id=env.correlation_id,
                causality_chain=chain,
                payload=res_payload,
            )

        except Exception as e:
            logger.error(f"Error handling gateway request: {e}", exc_info=True)
            error_payload = self._build_error_chat_result(
                correlation_id=getattr(env, "correlation_id", None),
                message=str(e),
                mode=getattr(locals().get("req", None), "mode", "unknown"),
                verb=getattr(locals().get("req", None), "verb", "unknown"),
                error_type=type(e).__name__,
            )
            try:
                await self._publish_gateway_reply(
                    reply_to=getattr(env, "reply_to", None),
                    correlation_id=getattr(env, "correlation_id", None),
                    causality_chain=getattr(env, "causality_chain", None),
                    payload=error_payload,
                )
                logger.info("Sent gateway error reply corr=%s", getattr(env, "correlation_id", None))
            except Exception:
                logger.exception("Failed to publish gateway error reply corr=%s", getattr(env, "correlation_id", None))
