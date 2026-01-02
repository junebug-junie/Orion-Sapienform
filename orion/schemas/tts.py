from typing import Optional
from pydantic import BaseModel, Field

class TTSRequestPayload(BaseModel):
    """
    Payload for TTS synthesis request.
    Kind: tts.synthesize.request
    """
    text: str = Field(..., description="Text to synthesize")
    voice_id: Optional[str] = Field(None, description="Voice ID to use (if supported)")
    language: Optional[str] = Field(None, description="Language code (e.g. 'en')")
    options: Optional[dict] = Field(None, description="Additional options for the engine")

class TTSResultPayload(BaseModel):
    """
    Payload for TTS synthesis result.
    Kind: tts.synthesize.result
    """
    audio_b64: str = Field(..., description="Base64 encoded audio data")
    content_type: str = Field("audio/wav", description="MIME type of the audio")
    duration_sec: Optional[float] = Field(None, description="Duration of the audio in seconds")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
