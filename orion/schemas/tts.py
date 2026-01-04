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

class STTRequestPayload(BaseModel):
    """
    Payload for Speech-to-Text (ASR) request.
    Kind: stt.transcribe.request
    """
    audio_b64: str = Field(..., description="Base64 encoded audio data")
    language: Optional[str] = Field("en", description="Language code")
    format: Optional[str] = Field("wav", description="Audio format")
    options: Optional[dict] = Field(None, description="Additional options")

class STTResultPayload(BaseModel):
    """
    Payload for Speech-to-Text (ASR) result.
    Kind: stt.transcribe.result
    """
    text: str = Field(..., description="Transcribed text")
    language_detected: Optional[str] = Field(None, description="Detected language")
    confidence: Optional[float] = Field(None, description="Confidence score")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
