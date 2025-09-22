import os, requests, base64, logging
from scripts.utils import split_sentences
logger = logging.getLogger("tts")

class TTS:
    def __init__(self, url: str | None = None):
        self.url = url or os.getenv("TTS_URL")

    def synthesize_chunks(self, text: str):
        if not self.url or not text:
            return []
        out = []
        for sentence in split_sentences(text):
            try:
                resp = requests.get(self.url, params={"text": sentence}, timeout=60)
                resp.raise_for_status()
                out.append(base64.b64encode(resp.content).decode("utf-8"))
            except requests.exceptions.RequestException as e:
                logger.error(f"TTS error on '{sentence[:64]}...': {e}")
                break
        return out
