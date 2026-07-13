from __future__ import annotations

import logging
from functools import lru_cache
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings

_logger = logging.getLogger("orion.cortex.orch.settings")


class Settings(BaseSettings):
    # Service identity
    service_name: str = Field("cortex-orch", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus config
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # RPC intake channel (hub -> cortex-orch)
    channel_cortex_request: str = Field(
        "orion:cortex:request",
        validation_alias=AliasChoices("CORTEX_REQUEST_CHANNEL", "ORCH_REQUEST_CHANNEL"),
    )
    channel_cortex_result_prefix: str = Field(
        "orion:cortex:result",
        validation_alias=AliasChoices("CORTEX_RESULT_PREFIX", "ORCH_RESULT_PREFIX"),
    )

    # RPC out to cortex-exec
    channel_exec_request: str = Field(
        "orion:cortex:exec:request",
        validation_alias=AliasChoices("CORTEX_EXEC_REQUEST_CHANNEL", "CHANNEL_EXEC_REQUEST"),
    )
    channel_exec_request_chat: str = Field(
        "orion:cortex:exec:request:chat",
        alias="CHANNEL_EXEC_REQUEST_CHAT",
    )
    channel_exec_request_spark: str = Field(
        "orion:cortex:exec:request:spark",
        alias="CHANNEL_EXEC_REQUEST_SPARK",
    )
    channel_exec_request_background: str = Field(
        "orion:cortex:exec:request:background",
        alias="CHANNEL_EXEC_REQUEST_BACKGROUND",
    )
    exec_lane_routing_enabled: bool = Field(True, alias="EXEC_LANE_ROUTING_ENABLED")
    channel_exec_result_prefix: str = Field(
        "orion:exec:result",
        validation_alias=AliasChoices("CORTEX_EXEC_RESULT_PREFIX", "EXEC_RESULT_PREFIX"),
    )
    diagnostic_mode: bool = Field(False, alias="DIAGNOSTIC_MODE")
    orion_verb_backdoor_enabled: bool = Field(False, alias="ORION_VERB_BACKDOOR_ENABLED")

    # Latest Orion state (Spark) read-model service
    orion_state_enabled: bool = Field(True, alias="ORION_STATE_ENABLED")
    state_request_channel: str = Field(
        "orion:state:request",
        validation_alias=AliasChoices("STATE_REQUEST_CHANNEL", "ORION_STATE_REQUEST_CHANNEL"),
    )
    state_result_prefix: str = Field(
        "orion:state:reply",
        validation_alias=AliasChoices("STATE_RESULT_PREFIX", "ORION_STATE_RESULT_PREFIX"),
    )
    state_timeout_sec: float = Field(2.0, alias="STATE_TIMEOUT_SEC")
    state_scope: str = Field("global", alias="STATE_SCOPE")  # global|node
    state_node: str = Field("", alias="STATE_NODE")

    # Equilibrium metacog triggers
    auto_router_llm_enabled: bool = Field(False, alias="AUTO_ROUTER_LLM_ENABLED")
    auto_router_llm_request_channel: str = Field(
        "orion:exec:request:LLMGatewayService",
        alias="AUTO_ROUTER_LLM_REQUEST_CHANNEL",
    )
    auto_router_llm_reply_prefix: str = Field("orion:llm:reply", alias="AUTO_ROUTER_LLM_REPLY_PREFIX")

    channel_metacog_trigger: str = Field(
        "orion:equilibrium:metacog:trigger",
        validation_alias=AliasChoices("CHANNEL_EQUILIBRIUM_METACOG_TRIGGER", "CHANNEL_METACOG_TRIGGER"),
    )
    channel_dream_trigger: str = Field(
        "orion:dream:trigger",
        validation_alias=AliasChoices("CHANNEL_DREAM_TRIGGER", "DREAM_TRIGGER_CHANNEL"),
    )

    # Orion Mind (HTTP control-plane; Orch is sole canonical caller for binding runs)
    orion_mind_base_url: str = Field("", alias="ORION_MIND_BASE_URL")
    orion_mind_timeout_sec: float = Field(210.0, alias="ORION_MIND_TIMEOUT_SEC")
    mind_n_loops_default: int = Field(1, alias="MIND_N_LOOPS_DEFAULT")
    mind_wall_ms_default: int = Field(180_000, alias="MIND_WALL_MS_DEFAULT")
    orion_mind_max_response_bytes: int = Field(2_000_000, alias="ORION_MIND_MAX_RESPONSE_BYTES")

    orion_substrate_telemetry_base_url: str = Field("", alias="ORION_SUBSTRATE_TELEMETRY_BASE_URL")
    orion_substrate_telemetry_timeout_sec: float = Field(2.0, alias="ORION_SUBSTRATE_TELEMETRY_TIMEOUT_SEC")
    orion_substrate_telemetry_read_token: str = Field("", alias="ORION_SUBSTRATE_TELEMETRY_READ_TOKEN")

    # Memory cards (always-on inject + optional auto-extractor)
    recall_pg_dsn: str = Field("", alias="RECALL_PG_DSN")
    channel_recall_intake: str = Field(
        "orion:exec:request:RecallService",
        alias="CHANNEL_RECALL_INTAKE",
    )
    channel_recall_reply_prefix: str = Field(
        "orion:exec:result:RecallService",
        validation_alias=AliasChoices(
            "CHANNEL_RECALL_REPLY_PREFIX",
            "CHANNEL_RECALL_DEFAULT_REPLY_PREFIX",
            "MIND_RECALL_REPLY_PREFIX",
        ),
    )
    mind_recall_prefetch_enabled: bool = Field(True, alias="MIND_RECALL_PREFETCH_ENABLED")
    mind_recall_prefetch_timeout_sec: float = Field(30.0, alias="MIND_RECALL_PREFETCH_TIMEOUT_SEC")
    orion_always_inject_token_budget: int = Field(300, alias="ORION_ALWAYS_INJECT_TOKEN_BUDGET")
    orion_always_inject_enabled: bool = Field(True, alias="ORION_ALWAYS_INJECT_ENABLED")
    orion_auto_extractor_enabled: bool = Field(False, alias="ORION_AUTO_EXTRACTOR_ENABLED")
    orion_auto_extractor_stage2_enabled: bool = Field(False, alias="ORION_AUTO_EXTRACTOR_STAGE2_ENABLED")
    orion_auto_extractor_auto_promote_threshold: int = Field(2, alias="ORION_AUTO_EXTRACTOR_AUTO_PROMOTE_THRESHOLD")

    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8072, alias="API_PORT")

    # Route arbitration grammar trace (shadow producer, off by default).
    # Publishes the same lane/mind/output_mode facts already surfaced on
    # VerbResultV1.output["_route_metadata"] as a GrammarEventV1 trace so the
    # (separately owned) route_loop reducer in orion/substrate/route_loop/
    # can materialize them. Fire-and-forget, never affects the chat response.
    publish_cortex_orch_grammar: bool = Field(True, alias="PUBLISH_CORTEX_ORCH_GRAMMAR")
    grammar_event_channel: str = Field("orion:grammar:event", alias="GRAMMAR_EVENT_CHANNEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def exec_request_channel_for_lane(self, lane: str) -> str:
        """Resolve Redis channel for PlanExecution RPC to cortex-exec (lane-isolated or legacy)."""
        if not self.exec_lane_routing_enabled:
            return self.channel_exec_request
        key = (lane or "background").strip().lower()
        if key == "chat":
            return self.channel_exec_request_chat
        if key == "spark":
            return self.channel_exec_request_spark
        if key == "background":
            return self.channel_exec_request_background
        _logger.warning(
            "exec_request_channel_for_lane_unknown lane=%r normalized=%r using_background_channel",
            lane,
            key,
        )
        return self.channel_exec_request_background


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
