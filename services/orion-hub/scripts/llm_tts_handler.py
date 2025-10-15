import logging
import asyncio
import aiohttp
from scripts.settings import settings

logger = logging.getLogger("voice-app.llm")

async def run_llm_tts(history, temperature, tts_q: asyncio.Queue, disable_tts: bool = False):
    """
    Calls the brain, gets a response, conditionally synthesizes TTS, and returns
    the response text and token count.
    """
    from scripts.main import bus, tts
    
    text = ""
    tokens = 0
    
    try:
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

        text = (
            data.get("response") or data.get("text") or
            (data.get("message", {}).get("content")) or ""
        ).strip()
        tokens = data.get("tokens") or data.get("eval_count") or len(text.split())
        
        if bus and bus.enabled:
            bus.publish(settings.CHANNEL_VOICE_LLM, {"type": "llm_response", "content": text, "tokens": tokens})
            bus.publish(settings.CHANNEL_BRAIN_OUT, {"type": "brain_response", "content": text, "tokens": tokens})

        if not text:
            # Return a default response if the brain gives nothing back
            return "I'm not sure how to respond to that.", 0

        # Conditionally run TTS
        if not disable_tts and tts:
            await tts_q.put({"state": "speaking"}) # Signal TTS is starting
            for chunk in tts.synthesize_chunks(text):
                await tts_q.put({"audio_response": chunk})
                if bus and bus.enabled:
                    bus.publish(settings.CHANNEL_VOICE_TTS, {"type": "audio_response", "size": len(chunk)})
        
        return text, tokens

    except Exception as e:
        logger.error(f"run_llm_tts error: {e}", exc_info=True)
        error_message = "I seem to have encountered an error."
        # Still try to send the error message to the TTS queue if not disabled
        if not disable_tts and tts:
             await tts_q.put({"state": "speaking"})
             for chunk in tts.synthesize_chunks(error_message):
                await tts_q.put({"audio_response": chunk})

        return error_message, 0

