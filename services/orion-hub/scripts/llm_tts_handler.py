import logging
import asyncio
import aiohttp
from .settings import settings
# The direct import from .main is removed.

logger = logging.getLogger("voice-app.llm")

async def run_llm_tts(history, temperature, llm_q: asyncio.Queue, tts_q: asyncio.Queue):
    """
    Calls the brain service, gets an LLM response, and synthesizes TTS audio.
    """
    # Import shared objects locally inside the function.
    from .main import bus, tts
    
    try:
        # --- Call Brain Service ---
        url = settings.BRAIN_URL.rstrip("/") + "/chat"
        payload = {
            "model": settings.LLM_MODEL, "messages": history,
            "temperature": temperature, "stream": False
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=settings.LLM_TIMEOUT_S) as r:
                r.raise_for_status()
                data = await r.json()
                logger.info(f"Brain raw response: {data}")

        # --- Process Response ---
        text = (
            data.get("response") or data.get("text") or
            (data.get("message", {}).get("content")) or ""
        ).strip()
        tokens = data.get("tokens") or data.get("eval_count") or len(text.split())

        await llm_q.put({"llm_response": text, "tokens": tokens})
        
        # --- Publish to Bus ---
        if bus and bus.enabled:
            bus.publish(settings.CHANNEL_VOICE_LLM, {"type": "llm_response", "content": text, "tokens": tokens})
            bus.publish(settings.CHANNEL_BRAIN_OUT, {"type": "brain_response", "content": text, "tokens": tokens})

        if not text:
            await llm_q.put({"state": "idle"})
            return

        # --- TTS Synthesis ---
        await llm_q.put({"state": "speaking"})
        if tts:
            for chunk in tts.synthesize_chunks(text):
                await tts_q.put({"audio_response": chunk})
                if bus and bus.enabled:
                    bus.publish(settings.CHANNEL_VOICE_TTS, {"type": "audio_response", "size": len(chunk)})
        
        await llm_q.put({"state": "idle"})

    except Exception as e:
        logger.error(f"run_llm_tts error: {e}", exc_info=True)
        await llm_q.put({"error": "LLM or TTS failed."})
        await llm_q.put({"state": "idle"})

