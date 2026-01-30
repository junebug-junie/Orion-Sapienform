from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the Orion Hub service, loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Core Service Identity ---
    PROJECT: str = Field(default="orion-janus", alias="PROJECT")
    SERVICE_NAME: str = Field(default="hub", alias="SERVICE_NAME")
    NODE_NAME: str = Field(default="athena", alias="ORION_NODE_NAME")
    SERVICE_VERSION: str = Field(default="0.3.0", alias="SERVICE_VERSION")
    HUB_PORT: int = Field(default=8080, alias="HUB_PORT")

    # --- Whisper Transcription Settings (LEGACY/UNUSED - Hub uses Bus RPC now) ---
    WHISPER_MODEL_SIZE: str = Field(default="distil-medium.en", alias="WHISPER_MODEL_SIZE")
    WHISPER_DEVICE: str = Field(default="cuda", alias="WHISPER_DEVICE")
    WHISPER_COMPUTE_TYPE: str = Field(default="float16", alias="WHISPER_COMPUTE_TYPE")

    # --- Orion Bus Integration ---
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, alias="ORION_BUS_ENFORCE_CATALOG")
    ORION_BUS_URL: str = Field(
        default="redis://100.92.216.81:6379/0",
        alias="ORION_BUS_URL",
    )

    # --- Landing Pad Integration (Topic Rail) ---
    LANDING_PAD_URL: str = Field(
        default="http://orion-landing-pad:8370",
        alias="LANDING_PAD_URL",
    )
    LANDING_PAD_TIMEOUT_SEC: float = Field(default=5.0, alias="LANDING_PAD_TIMEOUT_SEC")

    # --- Biometrics Cache (Hub) ---
    BIOMETRICS_ENABLED: bool = Field(default=True, alias="BIOMETRICS_ENABLED")
    BIOMETRICS_STALE_AFTER_SEC: float = Field(default=60.0, alias="BIOMETRICS_STALE_AFTER_SEC")
    BIOMETRICS_NO_SIGNAL_AFTER_SEC: float = Field(default=600.0, alias="BIOMETRICS_NO_SIGNAL_AFTER_SEC")
    BIOMETRICS_ROLE_WEIGHTS_JSON: str = Field(
        default='{"atlas":0.6,"athena":0.4}',
        alias="BIOMETRICS_ROLE_WEIGHTS_JSON",
    )
    BIOMETRICS_PUSH_INTERVAL_SEC: float = Field(default=5.0, alias="BIOMETRICS_PUSH_INTERVAL_SEC")

    # --- Cortex Gateway Integration (Titanium) ---
    CORTEX_GATEWAY_REQUEST_CHANNEL: str = Field(
        default="orion:cortex:gateway:request",
        alias="CORTEX_GATEWAY_REQUEST_CHANNEL",
    )
    CORTEX_GATEWAY_RESULT_PREFIX: str = Field(
        default="orion:cortex:gateway:result",
        alias="CORTEX_GATEWAY_RESULT_PREFIX",
    )

    # --- TTS / STT Integration (Titanium) ---
    TTS_REQUEST_CHANNEL: str = Field(default="orion:tts:intake", alias="TTS_REQUEST_CHANNEL")
    TTS_RESULT_PREFIX: str = Field(default="orion:tts:result", alias="TTS_RESULT_PREFIX")
    STT_REQUEST_CHANNEL: str = Field(default="orion:stt:intake", alias="STT_REQUEST_CHANNEL")
    STT_RESULT_PREFIX: str = Field(default="orion:stt:result", alias="STT_RESULT_PREFIX")

    # --- Legacy / Existing Channels (Preserved for Env Compat, but unused in new logic) ---
    CHANNEL_VOICE_TRANSCRIPT: str = Field(..., alias="CHANNEL_VOICE_TRANSCRIPT")
    CHANNEL_VOICE_LLM: str = Field(..., alias="CHANNEL_VOICE_LLM")
    CHANNEL_VOICE_TTS: str = Field(..., alias="CHANNEL_VOICE_TTS")
    CHANNEL_TTS_INTAKE: str = Field(
        default_factory=lambda: os.getenv("CHANNEL_TTS_INTAKE", "orion:tts:intake"),
        alias="CHANNEL_TTS_INTAKE",
    )

    CHANNEL_COLLAPSE_INTAKE: str = Field(..., alias="CHANNEL_COLLAPSE_INTAKE")
    CHANNEL_COLLAPSE_TRIAGE: str = Field(..., alias="CHANNEL_COLLAPSE_TRIAGE")

    PUBLISH_CHAT_HISTORY_LOG: bool = Field(default=True, alias="PUBLISH_CHAT_HISTORY_LOG")
    CHAT_HISTORY_LOG_CHANNEL: str | None = Field(default=None, alias="CHAT_HISTORY_LOG_CHANNEL")
    CHANNEL_CHAT_HISTORY_LOG: str = Field(default="orion:chat:history:log", alias="CHANNEL_CHAT_HISTORY_LOG")

    # Turn-level (prompt/response) logging for SQL (chat_history_log)
    CHAT_HISTORY_TURN_CHANNEL: str | None = Field(default=None, alias="CHAT_HISTORY_TURN_CHANNEL")
    CHANNEL_CHAT_HISTORY_TURN: str = Field(default="orion:chat:history:turn", alias="CHANNEL_CHAT_HISTORY_TURN")

    # Spark introspection candidate channel (drives spark-introspector UI)
    CHANNEL_SPARK_INTROSPECT_CANDIDATE: str = Field(
        default="orion:spark:introspect:candidate:log",
        alias="CHANNEL_SPARK_INTROSPECT_CANDIDATE",
    )

    @property
    def chat_history_channel(self) -> str:
        return self.CHAT_HISTORY_LOG_CHANNEL or self.CHANNEL_CHAT_HISTORY_LOG

    @property
    def chat_history_turn_channel(self) -> str:
        return self.CHAT_HISTORY_TURN_CHANNEL or self.CHANNEL_CHAT_HISTORY_TURN

    # --- Legacy Agent Council Integration ---
    CHANNEL_COUNCIL_INTAKE: str = Field(default="orion:agent-council:intake", alias="CHANNEL_COUNCIL_INTAKE")
    CHANNEL_COUNCIL_REPLY_PREFIX: str = Field(default="orion:agent-council:reply", alias="CHANNEL_COUNCIL_REPLY_PREFIX")

    # --- Legacy LLM/Exec/Orch Integration ---
    LLM_GATEWAY_SERVICE_NAME: str = Field(default="LLMGatewayService", alias="LLM_GATEWAY_SERVICE_NAME")
    EXEC_REQUEST_PREFIX: str = Field(default="orion:exec:request", alias="EXEC_REQUEST_PREFIX")
    CHANNEL_LLM_INTAKE: str = Field(default="orion:exec:request:LLMGatewayService", alias="CHANNEL_LLM_INTAKE")
    CHANNEL_LLM_REPLY_PREFIX: str = Field(default="orion:llm:reply:", alias="CHANNEL_LLM_REPLY_PREFIX")
    CORTEX_ORCH_REQUEST_CHANNEL: str = Field(default="orion:cortex:request", alias="CORTEX_REQUEST_CHANNEL")
    CORTEX_ORCH_RESULT_PREFIX: str = Field(default="orion:cortex:result", alias="CORTEX_RESULT_PREFIX")
    CHANNEL_AGENT_CHAIN_INTAKE: str = Field(
        default="orion:exec:request:AgentChainService",
        alias="AGENT_CHAIN_REQUEST_CHANNEL",
    )
    CHANNEL_AGENT_CHAIN_REPLY_PREFIX: str = Field(
        default="orion:exec:result:AgentChainService",
        alias="AGENT_CHAIN_RESULT_PREFIX",
    )

    # --- Legacy Recall Integration ---
    CHANNEL_RECALL_REQUEST: str = Field(default="orion:exec:request:RecallService", alias="CHANNEL_RECALL_REQUEST")
    CHANNEL_RECALL_DEFAULT_REPLY_PREFIX: str = Field(
        default="orion:exec:result:RecallService",
        alias="CHANNEL_RECALL_DEFAULT_REPLY_PREFIX",
    )
    RECALL_DEFAULT_MAX_ITEMS: int = Field(default=16, alias="RECALL_DEFAULT_MAX_ITEMS")
    RECALL_DEFAULT_TIME_WINDOW_DAYS: int = Field(default=30, alias="RECALL_DEFAULT_TIME_WINDOW_DAYS")
    RECALL_DEFAULT_MODE: str = Field(default="hybrid", alias="RECALL_DEFAULT_MODE")

    # --- Runtimes ----
    TIMEOUT_SEC: int = Field(default=300, alias="TIMEOUT_SEC")


    # --- Hub Prompt Context (UI-side rolling history) ---
    # Number of *turns* (user+assistant pairs) to include as inline context
    HUB_CONTEXT_TURNS: int = Field(default=12, alias="HUB_CONTEXT_TURNS")
    # Hard cap to avoid runaway prompts when users paste long text
    HUB_CONTEXT_MAX_CHARS: int = Field(default=12000, alias="HUB_CONTEXT_MAX_CHARS")

    # --- Recall Debugging ---
    HUB_DEBUG_RECALL: bool = Field(default=False, alias="HUB_DEBUG_RECALL")

    # --- No-Write Debug Mode (skip publishing chat history) ---
    HUB_DEFAULT_NO_WRITE: bool = Field(default=False, alias="HUB_DEFAULT_NO_WRITE")



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
