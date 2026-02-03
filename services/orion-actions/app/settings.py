from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("orion-actions", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Bus
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(True, alias="ORION_BUS_ENFORCE_CATALOG")

    # Channels
    actions_subscribe_channel: str = Field("orion:collapse:triage", alias="ACTIONS_SUBSCRIBE_CHANNEL")
    recall_rpc_channel: str = Field("orion:exec:request:RecallService", alias="RECALL_RPC_CHANNEL")
    llm_rpc_channel: str = Field("orion:exec:request:LLMGatewayService", alias="LLM_RPC_CHANNEL")
    actions_audit_channel: str = Field("orion:actions:audit", alias="ACTIONS_AUDIT_CHANNEL")

    # Notify
    notify_url: str = Field("http://orion-notify:7140", alias="NOTIFY_URL")
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")

    # Action policy
    actions_recipient_group: str = Field("juniper_primary", alias="ACTIONS_RECIPIENT_GROUP")
    actions_session_id: str = Field("collapse_mirror", alias="ACTIONS_SESSION_ID")
    actions_dedupe_ttl_seconds: int = Field(86400, alias="ACTIONS_DEDUPE_TTL_SECONDS")
    actions_notify_dedupe_window_seconds: int = Field(86400, alias="ACTIONS_NOTIFY_DEDUPE_WINDOW_SECONDS")
    actions_max_concurrency: int = Field(2, alias="ACTIONS_MAX_CONCURRENCY")

    # Recall
    actions_recall_profile: str = Field("reflect.v1", alias="ACTIONS_RECALL_PROFILE")
    actions_recall_timeout_seconds: float = Field(60.0, alias="ACTIONS_RECALL_TIMEOUT_SECONDS")

    # LLM
    actions_llm_route: str | None = Field("chat", alias="ACTIONS_LLM_ROUTE")
    actions_llm_timeout_seconds: float = Field(200.0, alias="ACTIONS_LLM_TIMEOUT_SECONDS")

    # HTTP
    port: int = Field(7160, alias="ACTIONS_PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
