# services/orion-whisper-tts/app/tts.py

# app/tts.py
import base64
import logging
import os
import tempfile
from TTS.api import TTS

logger = logging.getLogger(__name__)

class TTSEngine:
    """
    Simple GPU-backed TTS engine wrapper.

    Uses Coqui TTS; will run on CUDA if available.
    """

    def __init__(self):
        model_name = os.getenv("TTS_MODEL_NAME", "tts_models/en/ljspeech/vits")
        logger.info(f"[TTS_GPU] Loading TTS model: {model_name}")
        self.tts = TTS(model_name, gpu=True)
        logger.info("[TTS_GPU] TTS model loaded.")

    def synthesize_to_b64(self, text: str) -> str:
        if not text:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            # basic path — no fancy speaker/style yet
            self.tts.tts_to_file(text=text, file_path=f.name)
            f.seek(0)
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
