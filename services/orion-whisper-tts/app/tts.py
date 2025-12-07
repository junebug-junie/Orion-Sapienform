# services/orion-whisper-tts/app/tts.py

import base64
import logging
import tempfile

from TTS.api import TTS

from .settings import settings

logger = logging.getLogger("orion-whisper-tts.tts")


class TTSEngine:
    """
    Simple GPU-backed TTS engine wrapper.

    On Atlas this will use CUDA (TTS_USE_GPU=true).
    """

    def __init__(self, model_name: str | None = None):
        model = model_name or settings.tts_model_name
        use_gpu = settings.tts_use_gpu

        logger.info(f"[TTS] Loading TTS model: {model} (gpu={use_gpu})")
        self.tts = TTS(model, gpu=use_gpu)
        logger.info("[TTS] TTS model loaded.")

    def synthesize_to_b64(self, text: str) -> str:
        """
        Generate speech for `text` and return base64-encoded WAV bytes.
        """
        if not text:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.tts.tts_to_file(text=text, file_path=f.name)
            f.seek(0)
            audio_bytes = f.read()

        return base64.b64encode(audio_bytes).decode("utf-8")
