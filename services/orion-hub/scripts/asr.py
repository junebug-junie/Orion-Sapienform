import tempfile
from faster_whisper import WhisperModel
import logging

logger = logging.getLogger("asr")

class ASR:
    def __init__(self, size: str, device: str, compute_type: str):
        logger.info(f"Loading Whisper model '{size}' on {device}/{compute_type} ...")
        self.model = WhisperModel(size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded.")

    def transcribe_bytes(self, audio_bytes: bytes, beam_size: int = 5) -> str:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".ogg") as f:
            f.write(audio_bytes)
            f.flush()
            segments, _ = self.model.transcribe(f.name, beam_size=beam_size)
            return " ".join([s.text for s in segments]).strip()
