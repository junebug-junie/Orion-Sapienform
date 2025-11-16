# scripts/tts.py
import os
import requests
import base64
import logging

logger = logging.getLogger("tts")


class TTS:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("TTS_URL")

    def _call_tts(self, text: str) -> str | None:
        """
        Call the TTS service ONCE with the full text and return a base64-encoded
        audio string (or None on failure).
        """
        if not self.url or not text:
            return None

        try:
            resp = requests.get(self.url, params={"text": text}, timeout=60)
            resp.raise_for_status()
            # The TTS service returns raw audio bytes; we base64-encode them
            return base64.b64encode(resp.content).decode("utf-8")
        except requests.exceptions.RequestException as e:
            snippet = (text[:64] + "...") if len(text) > 64 else text
            logger.error(f"TTS error on full text '{snippet}': {e}")
            return None

    def synthesize_chunks(self, text: str):
        """
        Backwards-compatible interface used by the rest of the code.

        Previously:
          - Split into sentences
          - Called TTS once per sentence

        Now:
          - Single TTS call for the whole response
          - Returns a list with ONE base64 chunk (or empty list on failure)
        """
        audio_b64 = self._call_tts(text)
        if not audio_b64:
            return []
        return [audio_b64]
