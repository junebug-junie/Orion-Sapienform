# services/orion-hub/scripts/bus_clients/tts_client.py
from __future__ import annotations

import logging
from uuid import uuid4
from typing import Optional, Dict, Any

from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.tts import TTSRequestPayload, TTSResultPayload

from ..settings import settings

logger = logging.getLogger("orion-hub.tts-client")


class TTSClient:
    def __init__(self, bus: OrionBusAsync):
        self.bus = bus
        self.service_ref = ServiceRef(
            name=settings.SERVICE_NAME,
            version=settings.SERVICE_VERSION,
        )

    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: Optional[str] = "en",
        options: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Sends a typed TTSRequestPayload to orion-whisper-tts and awaits audio.
        """
        if not text:
            return {"error": "Empty text", "ok": False}

        correlation_id = str(uuid4())

        req_payload = TTSRequestPayload(
            text=text,
            voice_id=voice_id,
            language=language,
            options=options,
        )

        reply_to = f"{settings.TTS_RESULT_PREFIX}:{correlation_id}"

        envelope = BaseEnvelope(
            kind="tts.synthesize.request",
            source=self.service_ref,
            correlation_id=correlation_id,
            reply_to=reply_to,
            payload=req_payload.model_dump(mode="json"),
        )

        logger.info(
            "[%s] Sending TTS Request to %s (len=%d)",
            correlation_id,
            settings.TTS_REQUEST_CHANNEL,
            len(text),
        )

        try:
            msg = await self.bus.rpc_request(
                channel=settings.TTS_REQUEST_CHANNEL,
                envelope=envelope,
                reply_channel=reply_to,
                timeout_sec=timeout_sec,
            )

            decoded = self.bus.codec.decode(msg.get("data"))
            if not decoded.ok:
                raise ValueError(f"Decode failed: {decoded.error}")

            # Expect TTSResultPayload
            payload = decoded.envelope.payload

            # If payload is dict (likely from json decode), wrap or return
            if isinstance(payload, dict):
                 # Check for errors
                 if "error" in payload:
                     return payload
                 return payload # Should match TTSResultPayload keys

            if hasattr(payload, "model_dump"):
                return payload.model_dump(mode="json")

            return {"result": payload}

        except Exception as e:
            logger.error("[%s] TTS Request failed: %s", correlation_id, e, exc_info=True)
            return {"error": str(e), "ok": False}
