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
    HUB_API_BASE_OVERRIDE: str = Field(default="", alias="HUB_API_BASE_OVERRIDE")
    HUB_WS_BASE_OVERRIDE: str = Field(default="", alias="HUB_WS_BASE_OVERRIDE")

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

    # --- Landing Pad ---
    LANDING_PAD_URL: str = Field(
        default="http://orion-landing-pad:8370",
        alias="LANDING_PAD_URL",
    )
    LANDING_PAD_TIMEOUT_SEC: float = Field(default=5.0, alias="LANDING_PAD_TIMEOUT_SEC")

    # --- Topic Foundry Integration ---
    TOPIC_FOUNDRY_BASE_URL: str = Field(
        default="http://orion-topic-foundry:8615",
        alias="TOPIC_FOUNDRY_BASE_URL",
    )
    WORLD_PULSE_BASE_URL: str = Field(
        default="http://orion-world-pulse:8628",
        alias="WORLD_PULSE_BASE_URL",
    )
    WORLD_PULSE_PROXY_TIMEOUT_SEC: float = Field(
        default=10.0,
        alias="WORLD_PULSE_PROXY_TIMEOUT_SEC",
    )
    KNOWLEDGE_FORGE_BASE_URL: str = Field(
        default="http://orion-knowledge-forge:8630",
        alias="KNOWLEDGE_FORGE_BASE_URL",
    )
    KNOWLEDGE_FORGE_PROXY_TIMEOUT_SEC: float = Field(
        default=15.0,
        alias="KNOWLEDGE_FORGE_PROXY_TIMEOUT_SEC",
    )

    # --- Proposal review API (read-only Hub attention surface) ---
    HUB_PROPOSAL_REVIEW_ENABLED: bool = Field(default=False, alias="HUB_PROPOSAL_REVIEW_ENABLED")
    HUB_PROPOSAL_REVIEW_API_URL: str = Field(
        default="http://orion-context-exec:8096",
        alias="HUB_PROPOSAL_REVIEW_API_URL",
    )
    HUB_PROPOSAL_REVIEW_TIMEOUT_SEC: float = Field(
        default=10.0,
        alias="HUB_PROPOSAL_REVIEW_TIMEOUT_SEC",
    )

    # --- LLM gateway route catalog (compute override) ---
    HUB_LLM_GATEWAY_URL: str = Field(
        default="http://orion-llm-gateway:8210",
        alias="HUB_LLM_GATEWAY_URL",
    )
    HUB_LLM_GATEWAY_TIMEOUT_SEC: float = Field(
        default=5.0,
        alias="HUB_LLM_GATEWAY_TIMEOUT_SEC",
    )

    # --- Context-exec agent lane (Hub Agent mode) ---
    HUB_AGENT_CONTEXT_EXEC_ENABLED: bool = Field(
        default=True,
        alias="HUB_AGENT_CONTEXT_EXEC_ENABLED",
    )
    HUB_CONTEXT_EXEC_API_URL: str = Field(
        default="http://orion-context-exec:8096",
        alias="HUB_CONTEXT_EXEC_API_URL",
    )
    HUB_CONTEXT_EXEC_TIMEOUT_SEC: float = Field(
        default=600.0,
        alias="HUB_CONTEXT_EXEC_TIMEOUT_SEC",
    )
    HUB_AGENT_REPL_ENABLED: bool = Field(
        default=True,
        alias="HUB_AGENT_REPL_ENABLED",
    )
    # --- Hub Agent Claude (FCC harness in chat) ---
    HUB_AGENT_CLAUDE_ENABLED: bool = Field(
        default=False,
        alias="HUB_AGENT_CLAUDE_ENABLED",
    )
    HUB_FCC_ENV_PATH: str = Field(
        default="~/.fcc/.env",
        alias="HUB_FCC_ENV_PATH",
    )
    HUB_FCC_SERVER_URL: str = Field(
        default="http://127.0.0.1:8082",
        alias="HUB_FCC_SERVER_URL",
    )
    HUB_FCC_AUTH_TOKEN: str = Field(
        default="",
        alias="HUB_FCC_AUTH_TOKEN",
    )
    HUB_AGENT_CLAUDE_BIN: str = Field(
        default="claude",
        alias="HUB_AGENT_CLAUDE_BIN",
    )
    HUB_AGENT_CLAUDE_WORKSPACE: str = Field(
        default="/mnt/scripts/Orion-Sapienform",
        alias="HUB_AGENT_CLAUDE_WORKSPACE",
    )
    HUB_AGENT_CLAUDE_TIMEOUT_SEC: float = Field(
        default=900.0,
        alias="HUB_AGENT_CLAUDE_TIMEOUT_SEC",
    )
    HUB_AGENT_CLAUDE_MAX_CONCURRENT: int = Field(
        default=1,
        alias="HUB_AGENT_CLAUDE_MAX_CONCURRENT",
    )
    HUB_AGENT_CLAUDE_STREAM_READ_LIMIT: int = Field(
        default=8 * 1024 * 1024,
        alias="HUB_AGENT_CLAUDE_STREAM_READ_LIMIT",
        description="asyncio StreamReader limit for one claude stream-json line (bytes). Default 8MB.",
    )
    HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS: int = Field(
        default=65536,
        alias="HUB_AGENT_CLAUDE_MAX_CONTEXT_TOKENS",
        description="Passed to claude as CLAUDE_CODE_MAX_CONTEXT_TOKENS for llamacpp parity.",
    )
    HUB_AGENT_CLAUDE_FILE_READ_MAX_TOKENS: int = Field(
        default=8192,
        alias="HUB_AGENT_CLAUDE_FILE_READ_MAX_TOKENS",
        description="Passed to claude as CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS to cap single Read payloads.",
    )
    HUB_CONTEXT_EXEC_EVENT_CHANNEL: str = Field(
        default="orion:context_exec:event",
        alias="HUB_CONTEXT_EXEC_EVENT_CHANNEL",
    )
    CONTEXT_EXEC_INVESTIGATION_V2_ENABLED: bool = Field(
        default=False,
        alias="CONTEXT_EXEC_INVESTIGATION_V2_ENABLED",
    )

    # --- Self-observability (hub presence + curiosity focus hint) ---
    HUB_PRESENCE_WRITER_ENABLED: bool = Field(
        default=True,
        alias="HUB_PRESENCE_WRITER_ENABLED",
    )
    HUB_AGENT_CURIOSITY_HINT_ENABLED: bool = Field(
        default=False,
        alias="HUB_AGENT_CURIOSITY_HINT_ENABLED",
    )

    WORLD_PULSE_UI_FIXTURE_RUN_ENABLED: bool = Field(
        default=False,
        alias="WORLD_PULSE_UI_FIXTURE_RUN_ENABLED",
    )
    SOCIAL_MEMORY_BASE_URL: str = Field(
        default="http://orion-social-memory:8765",
        alias="SOCIAL_MEMORY_BASE_URL",
    )
    SELF_EXPERIMENTS_BASE_URL: str = Field(
        default="http://orion-self-experiments:7172",
        alias="SELF_EXPERIMENTS_BASE_URL",
    )
    SELF_EXPERIMENTS_TIMEOUT_SEC: float = Field(default=6.0, alias="SELF_EXPERIMENTS_TIMEOUT_SEC")

    # --- Biometrics Cache (Hub) ---
    BIOMETRICS_ENABLED: bool = Field(default=True, alias="BIOMETRICS_ENABLED")
    BIOMETRICS_STALE_AFTER_SEC: float = Field(default=60.0, alias="BIOMETRICS_STALE_AFTER_SEC")
    BIOMETRICS_NO_SIGNAL_AFTER_SEC: float = Field(default=600.0, alias="BIOMETRICS_NO_SIGNAL_AFTER_SEC")
    BIOMETRICS_ROLE_WEIGHTS_JSON: str = Field(
        default='{"atlas":0.6,"athena":0.4}',
        alias="BIOMETRICS_ROLE_WEIGHTS_JSON",
    )
    BIOMETRICS_PUSH_INTERVAL_SEC: float = Field(default=5.0, alias="BIOMETRICS_PUSH_INTERVAL_SEC")

    # --- Organ signal gateway inspect (Phase 2b Hub) ---
    SIGNALS_INSPECT_ENABLED: bool = Field(default=True, alias="SIGNALS_INSPECT_ENABLED")
    SIGNALS_INSPECT_SUBSCRIBE_PATTERN: str = Field(
        default="orion:signals:*",
        alias="SIGNALS_INSPECT_SUBSCRIBE_PATTERN",
    )
    SIGNALS_INSPECT_WINDOW_SEC: float = Field(default=45.0, alias="SIGNALS_INSPECT_WINDOW_SEC")
    SIGNALS_TRACE_CACHE_ENABLED: bool = Field(default=True, alias="SIGNALS_TRACE_CACHE_ENABLED")
    TRACE_CACHE_MAX_TRACES: int = Field(default=200, alias="TRACE_CACHE_MAX_TRACES")
    TRACE_CACHE_TTL_SEC: float = Field(default=300.0, alias="TRACE_CACHE_TTL_SEC")
    TRACE_CACHE_MAX_SIGNALS_PER_TRACE: int = Field(default=64, alias="TRACE_CACHE_MAX_SIGNALS_PER_TRACE")

    # --- Cognition trace cache (Runtime Trace Nexus A4) ---
    COGNITION_TRACE_CACHE_ENABLED: bool = Field(default=True, alias="COGNITION_TRACE_CACHE_ENABLED")
    COGNITION_TRACE_CACHE_MAX: int = Field(default=200, alias="COGNITION_TRACE_CACHE_MAX")
    COGNITION_TRACE_CACHE_TTL_SEC: float = Field(default=300.0, alias="COGNITION_TRACE_CACHE_TTL_SEC")
    COGNITION_TRACE_SUBSCRIBE_CHANNEL: str = Field(
        default="orion:cognition:trace",
        alias="COGNITION_TRACE_SUBSCRIBE_CHANNEL",
    )
    COGNITION_TRACE_API_DEBUG: bool = Field(default=False, alias="COGNITION_TRACE_API_DEBUG")

    # --- OpenTelemetry / Grafana (operator deep links; spec Phase 1) ---
    HUB_OTEL_GRAFANA_BASE_URL: str = Field(
        default="",
        alias="HUB_OTEL_GRAFANA_BASE_URL",
        description="Grafana base URL (e.g. http://127.0.0.1:3001) for Tempo Explore links from Hub.",
    )
    HUB_OTEL_GRAFANA_DATASOURCE_UID: str = Field(
        default="tempo",
        alias="HUB_OTEL_GRAFANA_DATASOURCE_UID",
        description="Grafana Tempo datasource uid (must match Explore link target).",
    )
    HUB_OTEL_GRAFANA_ORG_ID: int = Field(
        default=1,
        ge=1,
        alias="HUB_OTEL_GRAFANA_ORG_ID",
        description="Grafana organization id for Explore URLs (default single-org).",
    )

    # --- In-app Notifications (Hub) ---
    NOTIFY_IN_APP_ENABLED: bool = Field(default=True, alias="NOTIFY_IN_APP_ENABLED")
    NOTIFY_IN_APP_CHANNEL: str = Field(default="orion:notify:in_app", alias="NOTIFY_IN_APP_CHANNEL")
    NOTIFY_IN_APP_MAX: int = Field(default=200, alias="NOTIFY_IN_APP_MAX")
    NOTIFY_TOAST_SECONDS: int = Field(default=8, alias="NOTIFY_TOAST_SECONDS")
    NOTIFY_BASE_URL: str = Field(default="http://orion-notify:7140", alias="NOTIFY_BASE_URL")
    NOTIFY_API_TOKEN: str = Field(default="", alias="NOTIFY_API_TOKEN")

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
    HUB_RECALL_SERVICE_URL: str = Field(default="", alias="HUB_RECALL_SERVICE_URL")
    RECALL_SERVICE_URL: str = Field(default="http://orion-recall:8090", alias="RECALL_SERVICE_URL")
    HUB_RECALL_SHADOW_EVAL_TIMEOUT_SEC: float = Field(default=20.0, alias="HUB_RECALL_SHADOW_EVAL_TIMEOUT_SEC")
    HUB_RECALL_SHADOW_EVAL_MAX_ROWS_PER_RUN: int = Field(default=128, alias="HUB_RECALL_SHADOW_EVAL_MAX_ROWS_PER_RUN")
    HUB_RECALL_SHADOW_EVAL_DEFAULT_CORPUS_LIMIT: int = Field(default=24, alias="HUB_RECALL_SHADOW_EVAL_DEFAULT_CORPUS_LIMIT")

    # --- Memory cards (Postgres, same DSN as recall conjourney DB) ---
    RECALL_PG_DSN: str = Field(default="", alias="RECALL_PG_DSN")

    # --- Memory crystallization Graphiti/FalkorDB (additive temporal projection) ---
    GRAPHITI_ENABLED: bool = Field(default=False, alias="GRAPHITI_ENABLED")
    GRAPHITI_URL: str = Field(default="", alias="GRAPHITI_URL")
    FALKORDB_URI: str = Field(default="", alias="FALKORDB_URI")

    # --- Memory crystallization projections (Chroma via vector bus) ---
    CRYSTALLIZER_VECTOR_COLLECTION: str = Field(
        default="orion_memory_crystallizations", alias="CRYSTALLIZER_VECTOR_COLLECTION"
    )
    CRYSTALLIZER_EMBED_HOST_URL: str = Field(default="", alias="CRYSTALLIZER_EMBED_HOST_URL")
    CRYSTALLIZER_EMBED_MODE: str = Field(default="http", alias="CRYSTALLIZER_EMBED_MODE")
    CRYSTALLIZER_EMBED_TIMEOUT_MS: int = Field(default=8000, alias="CRYSTALLIZER_EMBED_TIMEOUT_MS")
    CRYSTALLIZER_AUTO_PROJECT_ON_APPROVE: bool = Field(default=True, alias="CRYSTALLIZER_AUTO_PROJECT_ON_APPROVE")

    CHROMA_HOST: str = Field(default="", alias="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, alias="CHROMA_PORT")
    GRAPHITI_ADAPTER_URL: str = Field(default="", alias="GRAPHITI_ADAPTER_URL")

    # --- Runtimes ----
    TIMEOUT_SEC: int = Field(
        default=400,
        alias="TIMEOUT_SEC",
        description="Bus RPC wait (cortex gateway, TTS, …). Should exceed gateway orch timeout for long LLM runs.",
    )
    HUB_STT_TIMEOUT_SEC: float = Field(
        default=120.0,
        alias="HUB_STT_TIMEOUT_SEC",
        description="Hub bus RPC wait for speech-to-text (should exceed whisper STT worker timeout).",
    )
    HUB_TTS_TIMEOUT_SEC: float = Field(
        default=180.0,
        alias="HUB_TTS_TIMEOUT_SEC",
        description="Hub bus RPC wait for text-to-speech (should exceed whisper TTS synth worker timeout).",
    )


    # --- Hub Prompt Context (UI-side rolling history) ---
    # Number of *turns* (user+assistant pairs) to include as inline context
    HUB_CONTEXT_TURNS: int = Field(default=12, alias="HUB_CONTEXT_TURNS")
    # Hard cap to avoid runaway prompts when users paste long text
    HUB_CONTEXT_MAX_CHARS: int = Field(default=12000, alias="HUB_CONTEXT_MAX_CHARS")

    # --- Recall eval → mutation telemetry (manual operator ingest; default off) ---
    HUB_RECALL_EVAL_RECORDING_ENABLED: bool = Field(default=False, alias="HUB_RECALL_EVAL_RECORDING_ENABLED")

    # --- Recall Debugging ---
    HUB_DEBUG_RECALL: bool = Field(default=False, alias="HUB_DEBUG_RECALL")
    HUB_DEBUG_COUNCIL: bool = Field(default=False, alias="HUB_DEBUG_COUNCIL")
    HUB_SOCIAL_SKILLS_ENABLED: bool = Field(default=True, alias="HUB_SOCIAL_SKILLS_ENABLED")
    HUB_SOCIAL_SKILLS_ALLOWLIST: str = Field(
        default="social_artifact_dialogue,social_summarize_thread,social_safe_recall,social_self_ground,social_followup_question,social_room_reflection,social_exit_or_pause",
        alias="HUB_SOCIAL_SKILLS_ALLOWLIST",
    )
    HUB_SOCIAL_STYLE_ADAPTATION_ENABLED: bool = Field(default=True, alias="HUB_SOCIAL_STYLE_ADAPTATION_ENABLED")
    HUB_SOCIAL_ROOM_RITUALS_ENABLED: bool = Field(default=True, alias="HUB_SOCIAL_ROOM_RITUALS_ENABLED")
    HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR: float = Field(default=0.35, alias="HUB_SOCIAL_STYLE_CONFIDENCE_FLOOR")
    HUB_SOCIAL_ROOM_REDACTION_POSTURE: str = Field(default="strict", alias="HUB_SOCIAL_ROOM_REDACTION_POSTURE")
    ORION_PRESENCE_SESSION_TTL_SECONDS: int = Field(default=14400, alias="ORION_PRESENCE_SESSION_TTL_SECONDS")
    ORION_PRESENCE_DEFAULT_REQUESTOR: str = Field(default="Juniper", alias="ORION_PRESENCE_DEFAULT_REQUESTOR")
    ORION_PRESENCE_PERSIST_ALLOWED: bool = Field(default=False, alias="ORION_PRESENCE_PERSIST_ALLOWED")
    ORION_SITUATION_ENABLED: bool = Field(default=True, alias="ORION_SITUATION_ENABLED")
    ORION_SITUATION_TTL_SECONDS: int = Field(default=300, alias="ORION_SITUATION_TTL_SECONDS")
    ORION_SITUATION_TIMEZONE: str = Field(default="America/Denver", alias="ORION_SITUATION_TIMEZONE")
    ORION_SITUATION_WEATHER_PROVIDER: str = Field(default="stub", alias="ORION_SITUATION_WEATHER_PROVIDER")

    # --- No-Write Debug Mode (skip publishing chat history) ---
    HUB_DEFAULT_NO_WRITE: bool = Field(default=False, alias="HUB_DEFAULT_NO_WRITE")
    HUB_AUTO_DEFAULT_ENABLED: bool = Field(default=False, alias="HUB_AUTO_DEFAULT_ENABLED")
    ENABLE_REPAIR_PRESSURE_SPEECH_WIRING: bool = Field(
        default=True,
        alias="ENABLE_REPAIR_PRESSURE_SPEECH_WIRING",
    )
    ENABLE_PRE_TURN_APPRAISAL: bool = Field(default=False, alias="ENABLE_PRE_TURN_APPRAISAL")
    PRE_TURN_APPRAISAL_PARADIGMS: str = Field(default="repair_pressure", alias="PRE_TURN_APPRAISAL_PARADIGMS")
    PRE_TURN_APPRAISAL_TIMEOUT_MS: int = Field(default=60000, alias="PRE_TURN_APPRAISAL_TIMEOUT_MS")
    CHANNEL_PRE_TURN_APPRAISAL_REQUEST: str = Field(
        default="orion:cortex:pre_turn_appraisal:request",
        alias="CHANNEL_PRE_TURN_APPRAISAL_REQUEST",
    )
    CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX: str = Field(
        default="orion:cortex:pre_turn_appraisal:result",
        alias="CHANNEL_PRE_TURN_APPRAISAL_RESULT_PREFIX",
    )
    HUB_AUTONOMY_SUBJECT_DISPLAY: str = Field(default="two", alias="HUB_AUTONOMY_SUBJECT_DISPLAY")
    SUBSTRATE_AUTONOMY_ENABLED: bool = Field(default=False, alias="SUBSTRATE_AUTONOMY_ENABLED")
    SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED: bool = Field(default=True, alias="SUBSTRATE_AUTONOMY_PROPOSALS_ENABLED")
    SUBSTRATE_AUTONOMY_APPLY_ENABLED: bool = Field(default=False, alias="SUBSTRATE_AUTONOMY_APPLY_ENABLED")
    SUBSTRATE_AUTONOMY_MONITOR_ENABLED: bool = Field(default=True, alias="SUBSTRATE_AUTONOMY_MONITOR_ENABLED")
    SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED: bool = Field(
        default=True,
        alias="SUBSTRATE_AUTONOMY_ROUTING_PROPOSALS_ENABLED",
    )
    SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED: bool = Field(
        default=False,
        alias="SUBSTRATE_AUTONOMY_COGNITIVE_PROPOSALS_ENABLED",
    )
    SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED: bool = Field(
        default=False,
        alias="SUBSTRATE_AUTONOMY_ROUTING_APPLY_ENABLED",
    )
    SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD: float = Field(
        default=-0.05,
        alias="SUBSTRATE_AUTONOMY_ROUTING_ROLLBACK_DELTA_THRESHOLD",
    )
    SUBSTRATE_AUTONOMY_INTERVAL_SEC: float = Field(default=30.0, alias="SUBSTRATE_AUTONOMY_INTERVAL_SEC")

    # --- Global graph / RDF + substrate semantic store (see docs/architecture/rdf_store_v1_cutover.md) ---
    GRAPH_BACKEND: str = Field(default="fuseki", alias="GRAPH_BACKEND")
    RDF_STORE_BACKEND: str = Field(default="fuseki", alias="RDF_STORE_BACKEND")
    RDF_STORE_BASE_URL: str = Field(default="", alias="RDF_STORE_BASE_URL")
    RDF_STORE_DATASET: str = Field(default="orion", alias="RDF_STORE_DATASET")
    RDF_STORE_QUERY_URL: str = Field(default="", alias="RDF_STORE_QUERY_URL")
    SUBSTRATE_STORE_BACKEND: str = Field(default="sparql", alias="SUBSTRATE_STORE_BACKEND")
    SUBSTRATE_GRAPH_QUERY_URL: str = Field(default="", alias="SUBSTRATE_GRAPH_QUERY_URL")
    SUBSTRATE_GRAPH_UPDATE_URL: str = Field(default="", alias="SUBSTRATE_GRAPH_UPDATE_URL")
    SUBSTRATE_GRAPH_URI: str = Field(default="", alias="SUBSTRATE_GRAPH_URI")
    SUBSTRATE_GRAPH_TIMEOUT_SEC: float = Field(default=5.0, alias="SUBSTRATE_GRAPH_TIMEOUT_SEC")
    SUBSTRATE_GRAPH_USER: str = Field(default="", alias="SUBSTRATE_GRAPH_USER")
    SUBSTRATE_GRAPH_PASS: str = Field(default="", alias="SUBSTRATE_GRAPH_PASS")
    SUBSTRATE_GRAPHDB_ENDPOINT: str = Field(default="", alias="SUBSTRATE_GRAPHDB_ENDPOINT")
    SUBSTRATE_GRAPHDB_GRAPH_URI: str = Field(default="", alias="SUBSTRATE_GRAPHDB_GRAPH_URI")
    SUBSTRATE_GRAPHDB_TIMEOUT_SEC: float = Field(default=5.0, alias="SUBSTRATE_GRAPHDB_TIMEOUT_SEC")
    SUBSTRATE_GRAPHDB_USER: str = Field(default="", alias="SUBSTRATE_GRAPHDB_USER")
    SUBSTRATE_GRAPHDB_PASS: str = Field(default="", alias="SUBSTRATE_GRAPHDB_PASS")

    # --- Chat Grammar Substrate Lane ---
    PUBLISH_HUB_CHAT_GRAMMAR: bool = Field(default=False, alias="PUBLISH_HUB_CHAT_GRAMMAR")
    GRAMMAR_EVENT_CHANNEL: str = Field(default="orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    # --- Grammar Atlas (substrate trace/graph read API) ---
    GRAMMAR_ATLAS_ENABLED: bool = Field(default=True, alias="GRAMMAR_ATLAS_ENABLED")
    GRAMMAR_ATLAS_POLL_INTERVAL_MS: int = Field(
        default=3000,
        ge=500,
        le=60000,
        alias="GRAMMAR_ATLAS_POLL_INTERVAL_MS",
    )
    GRAMMAR_ATLAS_POSTGRES_URI: str = Field(
        default="",
        alias="GRAMMAR_ATLAS_POSTGRES_URI",
        description="Optional Postgres DSN for grammar_* tables; falls back to DATABASE_URL.",
    )

    # --- Substrate Lattice tuning console ---
    SUBSTRATE_LATTICE_FRESHNESS_THRESHOLD_SEC: int = Field(
        default=60,
        alias="SUBSTRATE_LATTICE_FRESHNESS_THRESHOLD_SEC",
        description="Age in seconds before a substrate layer reading is marked stale (default 60s).",
    )

    MEMORY_GRAPH_APPROVAL_BACKEND: str = Field(default="auto", alias="MEMORY_GRAPH_APPROVAL_BACKEND")
    RDF_STORE_GRAPH_STORE_URL: str = Field(default="", alias="RDF_STORE_GRAPH_STORE_URL")
    RDF_STORE_UPDATE_URL: str = Field(default="", alias="RDF_STORE_UPDATE_URL")
    RDF_STORE_USER: str = Field(default="", alias="RDF_STORE_USER")
    RDF_STORE_PASS: str = Field(default="", alias="RDF_STORE_PASS")
    FUSEKI_USER: str = Field(default="", alias="FUSEKI_USER")
    FUSEKI_PASS: str = Field(default="", alias="FUSEKI_PASS")

    # --- Memory graph annotator (GraphDB + dual-write) ---
    GRAPHDB_URL: str = Field(default="", alias="GRAPHDB_URL")
    GRAPHDB_REPO: str = Field(default="collapse", alias="GRAPHDB_REPO")
    GRAPHDB_USER: str = Field(default="", alias="GRAPHDB_USER")
    GRAPHDB_PASS: str = Field(default="", alias="GRAPHDB_PASS")
    MEMORY_GRAPH_DEFAULT_NAMED_GRAPH: str = Field(default="", alias="MEMORY_GRAPH_DEFAULT_NAMED_GRAPH")

    # POST /api/memory/graph/suggest — grounded Quick primary, Brain escalation on hard failures
    MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE: str = Field(default="quick", alias="MEMORY_GRAPH_SUGGEST_PRIMARY_ROUTE")
    MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE: str = Field(default="brain", alias="MEMORY_GRAPH_SUGGEST_ESCALATION_ROUTE")
    MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION: bool = Field(default=True, alias="MEMORY_GRAPH_SUGGEST_ENABLE_ESCALATION")
    MEMORY_GRAPH_SUGGEST_INCLUDE_GROUNDING: bool = Field(default=True, alias="MEMORY_GRAPH_SUGGEST_INCLUDE_GROUNDING")
    MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS: int = Field(
        default=180_000,
        alias="MEMORY_GRAPH_SUGGEST_VERB_TIMEOUT_MS",
        description="Verb budget; defaults match orion/cognition/verbs/memory_graph_suggest.yaml timeout_ms.",
    )
    MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC: float = Field(
        default=0.0,
        alias="MEMORY_GRAPH_SUGGEST_BRAIN_TIMEOUT_SEC",
        description="Per Brain-route hub wait; 0 = use full verb timeout (180s default).",
    )
    MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC: float = Field(
        default=0.0,
        alias="MEMORY_GRAPH_SUGGEST_QUICK_TIMEOUT_SEC",
        description=(
            "Per Quick-route hub wait; 0 = ~35% of verb budget when escalation is on "
            "(Quick+Brain share one verb budget), else ~40% of verb."
        ),
    )
    MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_TIMEOUT_MS: int = Field(
        default=0,
        alias="MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_TIMEOUT_MS",
        description=(
            "Hub UI fetch AbortController for POST /api/memory/graph/suggest; "
            "0 = auto (server budget + buffer, typically ~205s when verb=180s and escalation on)."
        ),
    )
    MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_BUFFER_SEC: float = Field(
        default=25.0,
        alias="MEMORY_GRAPH_SUGGEST_CLIENT_FETCH_BUFFER_SEC",
        description="Added to computed server budget for browser fetch timeout when client ms is 0.",
    )
    MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD: str = Field(
        default="json_object_schema",
        alias="MEMORY_GRAPH_SUGGEST_STRUCTURED_OUTPUT_METHOD",
        description=(
            "llama.cpp response_format method (probe-selected): json_object_schema, "
            "json_schema_schema, json_object_only, none, or auto. "
            "none is mapped to json_object_schema at runtime so suggest drafts stay JSON."
        ),
    )
    MEMORY_GRAPH_SUGGEST_MAX_TOKENS: int = Field(
        default=4096,
        alias="MEMORY_GRAPH_SUGGEST_MAX_TOKENS",
        description=(
            "Completion ceiling for suggest draft JSON (gateway max_tokens). "
            "Actual budget is min(this, ctx - prompt - overhead); see MEMORY_GRAPH_SUGGEST_CTX_TOKENS."
        ),
    )
    MEMORY_GRAPH_SUGGEST_CTX_TOKENS: int = Field(
        default=4096,
        alias="MEMORY_GRAPH_SUGGEST_CTX_TOKENS",
        description=(
            "Assumed llama.cpp ctx_size for the suggest route (align with config/llm_profiles.yaml "
            "atlas-fast / quick lane)."
        ),
    )
    MEMORY_GRAPH_SUGGEST_PROMPT_OVERHEAD_TOKENS: int = Field(
        default=1800,
        alias="MEMORY_GRAPH_SUGGEST_PROMPT_OVERHEAD_TOKENS",
        description="Reserved tokens for system prompt, schema, and template outside the user transcript.",
    )
    MEMORY_GRAPH_SUGGEST_MIN_COMPLETION_TOKENS: int = Field(
        default=768,
        alias="MEMORY_GRAPH_SUGGEST_MIN_COMPLETION_TOKENS",
        description="Floor for memory_graph_suggest completion budget after ctx subtraction.",
    )
    MEMORY_GRAPH_SUGGEST_CHARS_PER_TOKEN: int = Field(
        default=3,
        alias="MEMORY_GRAPH_SUGGEST_CHARS_PER_TOKEN",
        description="Rough chars-per-token estimate for transcript length when budgeting suggest output.",
    )
    MEMORY_GRAPH_SUGGEST_MIN_PROMPT_TOKENS_ESTIMATE: int = Field(
        default=400,
        alias="MEMORY_GRAPH_SUGGEST_MIN_PROMPT_TOKENS_ESTIMATE",
        description="Minimum assumed prompt tokens before transcript length is applied.",
    )
    MEMORY_GRAPH_SUGGEST_CONTEXT_TURNS: int = Field(
        default=3,
        alias="MEMORY_GRAPH_SUGGEST_CONTEXT_TURNS",
        description=(
            "History turns passed to memory_graph_suggest LLM call. "
            "Kept low (default 3) to avoid context-window overflow on grammar-constrained generation."
        ),
    )
    # Deprecated aliases (read by legacy env only)
    MEMORY_GRAPH_SUGGEST_FALLBACK_ROUTE: str = Field(default="brain", alias="MEMORY_GRAPH_SUGGEST_FALLBACK_ROUTE")
    MEMORY_GRAPH_SUGGEST_ENABLE_FALLBACK: bool = Field(default=True, alias="MEMORY_GRAPH_SUGGEST_ENABLE_FALLBACK")

    @property
    def recall_service_url(self) -> str:
        return str(self.HUB_RECALL_SERVICE_URL or self.RECALL_SERVICE_URL or "http://orion-recall:8090").strip().rstrip("/")



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
