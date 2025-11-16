# scripts/llm_tts_handler.py
import logging
import asyncio
import time
from typing import Any, Dict, List, Optional

from scripts.settings import settings

logger = logging.getLogger("voice-app.llm")


async def run_tts_only(
    text: str,
    tts_q: asyncio.Queue,
    bus: Optional[Any] = None,
    tts: Optional[Any] = None,
    disable_tts: bool = False,
) -> None:
    """
    Run TTS for the given text and stream audio chunks into tts_q.

    Bus-first pattern:
      - Publish a tts_request to CHANNEL_VOICE_TTS
      - For each audio chunk, publish an audio_response metadata event.

    Designed as fire-and-forget (asyncio.create_task).
    """

    if disable_tts or not tts or not text:
        await tts_q.put({"state": "idle"})
        return

    try:
        await tts_q.put({"state": "speaking"})

        # Telemetry: TTS request
        if bus is not None and getattr(bus, "enabled", False):
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
                    f"Failed to publish tts_request to bus: {e}",
                    exc_info=True,
                )

        # Synthesize on local Piper (for now)
        tts_start = time.perf_counter()
        chunks = tts.synthesize_chunks(text)
        tts_end = time.perf_counter()
        logger.info(
            f"TTS generated {len(chunks)} chunk(s) in {tts_end - tts_start:.2f}s"
        )

        # Stream chunks to websocket
        for chunk in chunks:
            await tts_q.put({"audio_response": chunk})

            if bus is not None and getattr(bus, "enabled", False):
                try:
                    bus.publish(
                        settings.CHANNEL_VOICE_TTS,
                        {
                            "type": "audio_response",
                            "size": len(chunk),
                            "source": settings.SERVICE_NAME,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to publish TTS audio_response to bus: {e}",
                        exc_info=True,
                    )

    except Exception as e:
        logger.error(f"run_tts_only error: {e}", exc_info=True)
        error_message = "I seem to have encountered an error."
        try:
            await tts_q.put({"state": "speaking"})
            err_chunks = tts.synthesize_chunks(error_message) if tts else []
            for chunk in err_chunks:
                await tts_q.put({"audio_response": chunk})
        except Exception:
            pass
    finally:
        await tts_q.put({"state": "idle"})
