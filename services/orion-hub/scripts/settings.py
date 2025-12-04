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

    # --- LLM Model Selection ---
    LLM_MODEL: str = Field(default="mistral:instruct", env="LLM_MODEL")

    # --- Cognitive Backends ---
    BRAIN_URL: str = Field(
        default=f"http://{os.getenv('PROJECT', 'orion-janus')}-brain:8088",
        env="BRAIN_URL",
    )
    LLM_TIMEOUT_S: int = Field(default=60, env="LLM_TIMEOUT_S")

    # --- Orion Bus Integration ---
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    ORION_BUS_URL: str = Field(
        default=(
            os.getenv("REDIS_URL")
            or f"redis://{os.getenv('PROJECT', 'orion-janus')}-bus-core:6379/0"
        ),
        env="ORION_BUS_URL",
    )

    # --- Bus Channels (existing) ---
    CHANNEL_VOICE_TRANSCRIPT: str = Field(..., env="CHANNEL_VOICE_TRANSCRIPT")
    CHANNEL_VOICE_LLM: str = Field(..., env="CHANNEL_VOICE_LLM")
    CHANNEL_VOICE_TTS: str = Field(..., env="CHANNEL_VOICE_TTS")
    CHANNEL_TTS_INTAKE: str = os.getenv("CHANNEL_TTS_INTAKE", "orion:tts:intake")

    CHANNEL_COLLAPSE_INTAKE: str = Field(..., env="CHANNEL_COLLAPSE_INTAKE")
    CHANNEL_COLLAPSE_TRIAGE: str = Field(..., env="CHANNEL_COLLAPSE_TRIAGE")

    CHANNEL_BRAIN_INTAKE: str = Field(..., env="CHANNEL_BRAIN_INTAKE")
    CHANNEL_BRAIN_OUT: str = Field(..., env="CHANNEL_BRAIN_OUT")

    # --- Agent Council Integration (new) ---
    CHANNEL_COUNCIL_INTAKE: str = Field(
        default="orion:agent-council:intake",
        env="CHANNEL_COUNCIL_INTAKE",
    )
    CHANNEL_COUNCIL_REPLY_PREFIX: str = Field(
        default="orion:agent-council:reply",
        env="CHANNEL_COUNCIL_REPLY_PREFIX",
    )

    # Which bus service name the gateway listens on
    LLM_GATEWAY_SERVICE_NAME = "LLMGatewayService"

    # Where Cortex + Hub send exec_step/chat/generate requests
    EXEC_REQUEST_PREFIX: str = Field(
        default="orion-exec:request",
        env="EXEC_REQUEST_PREFIX",
    )
    CHANNEL_LLM_REPLY_PREFIX: str = Field(
        default="orion:llm:reply",
        env="CHANNEL_LLM_REPLY_PREFIX"
    )

    # LLM Gateway (Hub → Gateway chat RPC)
    CHANNEL_LLM_GATEWAY_INTAKE: str = Field(
        default="orion:llm-gateway:chat:intake",
        env="CHANNEL_LLM_GATEWAY_INTAKE
    )
    CHANNEL_LLM_REPLY_PREFIX: str = Field(
        default="orion:llm-gateway:chat:reply",
        env="CHANNEL_LLM_REPLY_PREFIX
    )

    # --- Recall / RAG Integration (new) ---
    CHANNEL_RECALL_REQUEST: str = Field(
        default="orion:recall:request",
        env="CHANNEL_RECALL_REQUEST",
    )

    CHANNEL_RECALL_DEFAULT_REPLY_PREFIX: str = Field(
        default="orion:recall:reply",
        env="CHANNEL_RECALL_DEFAULT_REPLY_PREFIX",
    )

    RECALL_DEFAULT_MAX_ITEMS: int = Field(
        default=16,
        env="RECALL_DEFAULT_MAX_ITEMS",
    )
    RECALL_DEFAULT_TIME_WINDOW_DAYS: int = Field(
        default=30,
        env="RECALL_DEFAULT_TIME_WINDOW_DAYS",
    )
    RECALL_DEFAULT_MODE: str = Field(
        default="hybrid",  # short_term | deep | hybrid
        env="RECALL_DEFAULT_MODE",
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
