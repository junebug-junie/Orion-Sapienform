# services/orion-whisper-tts/app/stt.py

from __future__ import annotations

import base64
import tempfile

import torch
import whisper


class STTEngine:
    def __init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base", device=device)

    def transcribe(self, audio_b64: str, language: str = "en") -> str:
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio.flush()
            result = self.model.transcribe(temp_audio.name, language=language)
        return result.get("text", "")
