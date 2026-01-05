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
    NODE_NAME: str = Field(default="athena", env="ORION_NODE_NAME")
    SERVICE_VERSION: str = Field(default="0.3.0", env="SERVICE_VERSION")
    HUB_PORT: int = Field(default=8080, env="HUB_PORT")

    # --- Whisper Transcription Settings (LEGACY/UNUSED - Hub uses Bus RPC now) ---
    WHISPER_MODEL_SIZE: str = Field(default="distil-medium.en", env="WHISPER_MODEL_SIZE")
    WHISPER_DEVICE: str = Field(default="cuda", env="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(default="float16", env="WHISPER_COMPUTE_TYPE")

    # --- Orion Bus Integration ---
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    ORION_BUS_URL: str = Field(
        default=("redis://100.92.216.81:6379/0"),
        env="ORION_BUS_URL",
    )

    # --- NEW: Cortex Gateway Integration (Titanium) ---
    CORTEX_GATEWAY_REQUEST_CHANNEL: str = Field(
        default="orion-cortex-gateway:request",
        env="CORTEX_GATEWAY_REQUEST_CHANNEL",
    )
    CORTEX_GATEWAY_RESULT_PREFIX: str = Field(
        default="orion-cortex-gateway:result",
        env="CORTEX_GATEWAY_RESULT_PREFIX",
    )

    # --- NEW: TTS / STT Integration (Titanium) ---
    TTS_REQUEST_CHANNEL: str = Field(
        default="orion:tts:intake",
        env="TTS_REQUEST_CHANNEL",
    )
    TTS_RESULT_PREFIX: str = Field(
        default="orion:tts:result",
        env="TTS_RESULT_PREFIX",
    )

    # STT (ASR) - Can reuse TTS channel if service is unified, or separate
    # Defaulting to a likely channel based on pattern
    STT_REQUEST_CHANNEL: str = Field(
        default="orion:stt:intake",
        env="STT_REQUEST_CHANNEL",
    )
    STT_RESULT_PREFIX: str = Field(
        default="orion:stt:result",
        env="STT_RESULT_PREFIX",
    )

    # --- Legacy / Existing Channels (Preserved for Env Compat, but unused in new logic) ---
    CHANNEL_VOICE_TRANSCRIPT: str = Field(..., env="CHANNEL_VOICE_TRANSCRIPT")
    CHANNEL_VOICE_LLM: str = Field(..., env="CHANNEL_VOICE_LLM")
    CHANNEL_VOICE_TTS: str = Field(..., env="CHANNEL_VOICE_TTS")
    # Kept as alias to TTS_REQUEST_CHANNEL if needed, but we use TTS_REQUEST_CHANNEL now
    CHANNEL_TTS_INTAKE: str = os.getenv("CHANNEL_TTS_INTAKE", "orion:tts:intake")

    CHANNEL_COLLAPSE_INTAKE: str = Field(..., env="CHANNEL_COLLAPSE_INTAKE")
    CHANNEL_COLLAPSE_TRIAGE: str = Field(..., env="CHANNEL_COLLAPSE_TRIAGE")

    PUBLISH_CHAT_HISTORY_LOG: bool = Field(default=True, env="PUBLISH_CHAT_HISTORY_LOG")

    CHANNEL_CHAT_HISTORY_LOG: str = Field(
        "orion:chat:history:log",
        env="CHANNEL_CHAT_HISTORY_LOG",
    )

    # --- Legacy Agent Council Integration ---
    CHANNEL_COUNCIL_INTAKE: str = Field(
        default="orion:agent-council:intake",
        env="CHANNEL_COUNCIL_INTAKE",
    )
    CHANNEL_COUNCIL_REPLY_PREFIX: str = Field(
        default="orion:agent-council:reply",
        env="CHANNEL_COUNCIL_REPLY_PREFIX",
    )

    # --- Legacy LLM/Exec/Orch Integration ---
    LLM_GATEWAY_SERVICE_NAME: str = Field(
        default="LLMGatewayService",
        env="LLM_GATEWAY_SERVICE_NAME"
    )
    EXEC_REQUEST_PREFIX: str = Field(
        default="orion-exec:request",
        env="EXEC_REQUEST_PREFIX",
    )
    CHANNEL_LLM_INTAKE: str = Field(
        default="orion-exec:request:LLMGatewayService",
        env="CHANNEL_LLM_INTAKE",
    )
    CHANNEL_LLM_REPLY_PREFIX: str = Field(
        default="orion:llm:reply:",
        env="CHANNEL_LLM_REPLY_PREFIX",
    )
    CORTEX_ORCH_REQUEST_CHANNEL: str = Field(
        default="orion-cortex:request",
        env="CORTEX_REQUEST_CHANNEL",
    )
    CORTEX_ORCH_RESULT_PREFIX: str = Field(
        default="orion-cortex:result",
        env="CORTEX_RESULT_PREFIX",
    )
    CHANNEL_AGENT_CHAIN_INTAKE: str = Field(
        default="orion-exec:request:AgentChainService",
        env="AGENT_CHAIN_REQUEST_CHANNEL"
    )
    CHANNEL_AGENT_CHAIN_REPLY_PREFIX: str = Field(
        default="orion-exec:result:AgentChainService",
        env="AGENT_CHAIN_RESULT_PREFIX"
    )

    # --- Legacy Recall Integration ---
    CHANNEL_RECALL_REQUEST: str = Field(
        default="orion-exec:request:RecallService",
        env="CHANNEL_RECALL_REQUEST",
    )
    CHANNEL_RECALL_DEFAULT_REPLY_PREFIX: str = Field(
        default="orion-exec:result:RecallService",
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

    # --- Runtimes ----
    TIMEOUT_SEC: int = Field(
        default=300,
        env="TIMEOUT_SEC"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
