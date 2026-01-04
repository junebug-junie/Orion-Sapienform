import logging
import uuid
from typing import Optional

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload, STTRequestPayload, STTResultPayload
from scripts.settings import settings

logger = logging.getLogger("hub.bus.tts")

class TTSClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self._source = ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
        )

    async def speak(self, request: TTSRequestPayload) -> TTSResultPayload:
        """
        Sends a TTSRequestPayload to the TTS Service and waits for a TTSResultPayload.
        """
        correlation_id = str(uuid.uuid4())
        reply_to = f"{settings.TTS_RESULT_PREFIX}:{correlation_id}"

        envelope = BaseEnvelope(
            kind="tts.synthesize.request",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request,
        )

        logger.info(f"[{correlation_id}] Sending TTS request to {settings.TTS_REQUEST_CHANNEL}")

        try:
            msg = await self.bus.rpc_request(
                settings.TTS_REQUEST_CHANNEL,
                envelope,
                reply_channel=reply_to,
                timeout_sec=settings.TIMEOUT_SEC,
            )
        except TimeoutError:
             logger.error(f"[{correlation_id}] TTS Request timed out.")
             raise

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
             logger.error(f"[{correlation_id}] Decode failed: {decoded.error}")
             raise ValueError(f"Bus decode error: {decoded.error}")

        # Strict validation
        payload = decoded.envelope.payload

        if isinstance(payload, dict):
            try:
                return TTSResultPayload.model_validate(payload)
            except Exception as e:
                 logger.error(f"[{correlation_id}] TTS Response validation failed: {e}")
                 raise ValueError(f"Invalid response format from TTS: {e}")
        elif isinstance(payload, TTSResultPayload):
             return payload
        else:
             raise ValueError(f"Unexpected payload type: {type(payload)}")

    async def transcribe(self, request: STTRequestPayload) -> STTResultPayload:
        """
        Sends an STTRequestPayload to the Speech Service and waits for an STTResultPayload.
        """
        correlation_id = str(uuid.uuid4())
        reply_to = f"{settings.STT_RESULT_PREFIX}:{correlation_id}"

        envelope = BaseEnvelope(
            kind="stt.transcribe.request",
            source=self._source,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=request,
        )

        logger.info(f"[{correlation_id}] Sending STT request to {settings.STT_REQUEST_CHANNEL}")

        try:
            msg = await self.bus.rpc_request(
                settings.STT_REQUEST_CHANNEL,
                envelope,
                reply_channel=reply_to,
                timeout_sec=settings.TIMEOUT_SEC,
            )
        except TimeoutError:
             logger.error(f"[{correlation_id}] STT Request timed out.")
             raise

        decoded = self.bus.codec.decode(msg.get("data"))
        if not decoded.ok:
             logger.error(f"[{correlation_id}] Decode failed: {decoded.error}")
             raise ValueError(f"Bus decode error: {decoded.error}")

        # Strict validation
        payload = decoded.envelope.payload

        if isinstance(payload, dict):
            try:
                return STTResultPayload.model_validate(payload)
            except Exception as e:
                 logger.error(f"[{correlation_id}] STT Response validation failed: {e}")
                 raise ValueError(f"Invalid response format from STT: {e}")
        elif isinstance(payload, STTResultPayload):
             return payload
        else:
             raise ValueError(f"Unexpected payload type: {type(payload)}")
