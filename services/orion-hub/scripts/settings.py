import os
from pydantic_settings import BaseSettings
from pydantic import Field

print(f"--- LOADING SETTINGS FROM: {__file__} ---", flush=True)


class Settings(BaseSettings):
    """
    Configuration for the Orion Hub service, loaded from environment variables.
    """

    # --- Core Service Identity ---
    PROJECT: str = Field(default="orion-janus", env="PROJECT")
    SERVICE_NAME: str = Field(default="hub", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.3.0", env="SERVICE_VERSION")
    HUB_PORT: int = Field(default=8080, env="HUB_PORT")

    # --- Whisper Transcription Settings ---
    WHISPER_MODEL_SIZE: str = Field(default="distil-medium.en", env="WHISPER_MODEL_SIZE")
    WHISPER_DEVICE: str = Field(default="cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(default="float16", env="WHISPER_COMPUTE_TYPE")

    # --- Orion Bus Integration ---
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    ORION_BUS_URL: str = Field(
        default=("redis://100.92.216.81:6379/0"),
        env="ORION_BUS_URL",
    )

    # --- Bus Channels (existing) ---
    CHANNEL_VOICE_TRANSCRIPT: str = Field(..., env="CHANNEL_VOICE_TRANSCRIPT")
    CHANNEL_VOICE_LLM: str = Field(..., env="CHANNEL_VOICE_LLM")
    CHANNEL_VOICE_TTS: str = Field(..., env="CHANNEL_VOICE_TTS")

    CHANNEL_COLLAPSE_INTAKE: str = Field(..., env="CHANNEL_COLLAPSE_INTAKE")
    CHANNEL_COLLAPSE_TRIAGE: str = Field(..., env="CHANNEL_COLLAPSE_TRIAGE")

    CHANNEL_CHAT_HISTORY_LOG: str = Field(
        "orion:chat:history:log",
        env="CHANNEL_CHAT_HISTORY_LOG",
    )

    # --- Cortex Gateway Integration (New Dumb Hub) ---
    CORTEX_GATEWAY_REQUEST_CHANNEL: str = Field(
        default="orion:cortex-gateway:request",
        env="CORTEX_GATEWAY_REQUEST_CHANNEL",
    )
    CORTEX_GATEWAY_RESULT_PREFIX: str = Field(
        default="orion:cortex-gateway:reply",
        env="CORTEX_GATEWAY_RESULT_PREFIX",
    )

    # --- TTS Integration ---
    TTS_REQUEST_CHANNEL: str = Field(
        default="orion:tts:intake",
        env="TTS_REQUEST_CHANNEL",
    )
    TTS_RESULT_PREFIX: str = Field(
        default="orion:tts:reply",
        env="TTS_RESULT_PREFIX",
    )

    # --- Runtimes ----
    TIMEOUT_SEC: int = Field(
        default=300,
        env="TIMEOUT_SEC"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
