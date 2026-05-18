# services/orion-whisper-tts/app/tts.py

# app/tts.py
import base64
import logging
import tempfile
from TTS.api import TTS

from .settings import settings

logger = logging.getLogger(__name__)

class TTSEngine:
    """
    Simple GPU-backed TTS engine wrapper.

    Uses Coqui TTS; GPU use is controlled by TTS_USE_GPU.
    """

    def __init__(self):
        model_name = settings.tts_model_name
        use_gpu = settings.tts_use_gpu
        logger.info("[TTS] Loading model=%s gpu=%s", model_name, use_gpu)
        self.tts = TTS(model_name, gpu=use_gpu)
        logger.info("[TTS] Model loaded.")

    def synthesize_to_b64(self, text: str) -> str:
        if not text:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            # basic path — no fancy speaker/style yet
            self.tts.tts_to_file(text=text, file_path=f.name)
            f.seek(0)
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
