# scripts/llm_tts_handler.py
import logging
import asyncio
import time
from typing import Any, Optional

from scripts.settings import settings
from scripts.tts_rpc import TTSRPC

logger = logging.getLogger("voice-app.llm")


async def run_tts_only(
    text: str,
    tts_q: asyncio.Queue,
    bus: Optional[Any] = None,
    tts: Optional[Any] = None,   # kept for backward compat, unused
    disable_tts: bool = False,
) -> None:
    """
    Run TTS for the given text via Brain RPC and stream audio chunks into tts_q.

    Flow:
      Hub → Bus (CHANNEL_TTS_INTAKE) → Brain (GPU TTS) → Bus (orion:tts:rpc:<id>) → Hub
    """

    if disable_tts or not text:
        await tts_q.put({"state": "idle"})
        return

    if bus is None or not getattr(bus, "enabled", False):
        logger.warning("run_tts_only called but OrionBus is disabled; skipping TTS.")
        await tts_q.put({"state": "idle"})
        return

    try:
        await tts_q.put({"state": "speaking"})

        # Telemetry: high-level TTS request event on voice channel
        try:
            bus.publish(
                settings.CHANNEL_VOICE_TTS,
                {
                    "type": "tts_request",
                    "source": settings.SERVICE_NAME,
                    "text_preview": text[:256],
                },
            )
        except Exception as e:
            logger.warning(
                f"Failed to publish tts_request to voice TTS channel: {e}",
                exc_info=True,
            )

        tts_start = time.perf_counter()

        # --- GPU TTS via Brain RPC ---
        tts_rpc = TTSRPC(bus)
        reply = await tts_rpc.call_tts(text)
        audio_b64 = reply.get("audio_b64")

        tts_end = time.perf_counter()
        logger.info(
            f"GPU TTS generated 1 chunk in {tts_end - tts_start:.2f}s"
        )

        if audio_b64:
            await tts_q.put({"audio_response": audio_b64})

            try:
                bus.publish(
                    settings.CHANNEL_VOICE_TTS,
                    {
                        "type": "audio_response",
                        "size": len(audio_b64),
                        "source": settings.SERVICE_NAME,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to publish TTS audio_response to voice channel: {e}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"run_tts_only error: {e}", exc_info=True)
        error_message = "I seem to have encountered an error."
        try:
            await tts_q.put({"state": "speaking"})
            # NOTE: we could fall back to local 'tts' here if you want
            # but for now we'll just skip audio on error.
        except Exception:
            pass
    finally:
        await tts_q.put({"state": "idle"})
