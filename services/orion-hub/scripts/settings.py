from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """
    Centralized configuration for the Orion Hub service.
    Loads settings from environment variables, providing a single source of truth.
    """

    # === Core Identity ===
    PROJECT: str = Field(default="orion-janus", env="PROJECT")
    SERVICE_NAME: str = Field(default="hub", env="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.3.0", env="SERVICE_VERSION")
    HUB_PORT: int = Field(default=8080, env="HUB_PORT")

    # === Orion Bus ===
    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")

    # === Voice / LLM Channels ===
    CHANNEL_VOICE_TRANSCRIPT: str = Field(default="orion:voice:transcript", env="CHANNEL_VOICE_TRANSCRIPT")
    CHANNEL_VOICE_LLM: str = Field(default="orion:voice:llm", env="CHANNEL_VOICE_LLM")
    CHANNEL_VOICE_TTS: str = Field(default="orion:voice:tts", env="CHANNEL_VOICE_TTS")

    # === Collapse Event Channels ===
    CHANNEL_COLLAPSE_INTAKE: str = Field(default="orion:collapse:intake", env="CHANNEL_COLLAPSE_INTAKE")

    # === Brain Channels ===
    CHANNEL_BRAIN_INTAKE: str = Field(default="orion:brain:intake", env="CHANNEL_BRAIN_INTAKE")
    CHANNEL_BRAIN_OUT: str = Field(default="orion:brain:out", env="CHANNEL_BRAIN_OUT")

    # === Whisper Model Configuration ===
    WHISPER_MODEL_SIZE: str = Field(default="base.en", env="WHISPER_MODEL_SIZE")
    WHISPER_DEVICE: str = Field(default="cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(default="float16", env="WHISPER_COMPUTE_TYPE")

    # === Cognitive Backend (Brain) ===
    BRAIN_URL: str = Field(..., env="BRAIN_URL")
    LLM_TIMEOUT_S: int = Field(default=60, env="LLM_TIMEOUT_S")
    LLM_MODEL: str = Field(default="mistral:instruct", env="LLM_MODEL")

    class Config:
        # Pydantic will automatically look for a .env file in the execution path
        # when used within Docker, this is managed by the docker-compose command.
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
