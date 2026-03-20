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
    actions_subscribe_channels: str = Field(
        "orion:collapse:triage,orion:actions:trigger:daily_pulse.v1,orion:actions:trigger:daily_metacog.v1",
        alias="ACTIONS_SUBSCRIBE_CHANNELS",
    )
    recall_rpc_channel: str = Field("orion:exec:request:RecallService", alias="RECALL_RPC_CHANNEL")
    llm_rpc_channel: str = Field("orion:exec:request:LLMGatewayService", alias="LLM_RPC_CHANNEL")
    actions_audit_channel: str = Field("orion:actions:audit", alias="ACTIONS_AUDIT_CHANNEL")
    cortex_exec_request_channel: str = Field("orion:cortex:exec:request", alias="CORTEX_EXEC_REQUEST_CHANNEL")

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

    # Daily execution via cortex-exec
    actions_exec_timeout_seconds: float = Field(240.0, alias="ACTIONS_EXEC_TIMEOUT_SECONDS")
    actions_daily_timezone: str = Field("America/Denver", alias="ACTIONS_DAILY_TIMEZONE")
    actions_daily_run_on_startup: bool = Field(False, alias="ACTIONS_DAILY_RUN_ON_STARTUP")
    actions_daily_run_once_date: str | None = Field(None, alias="ACTIONS_DAILY_RUN_ONCE_DATE")

    actions_daily_pulse_enabled: bool = Field(True, alias="ACTIONS_DAILY_PULSE_ENABLED")
    actions_daily_pulse_hour_local: int = Field(8, alias="ACTIONS_DAILY_PULSE_HOUR_LOCAL")
    actions_daily_pulse_minute_local: int = Field(30, alias="ACTIONS_DAILY_PULSE_MINUTE_LOCAL")

    actions_daily_metacog_enabled: bool = Field(True, alias="ACTIONS_DAILY_METACOG_ENABLED")
    actions_daily_metacog_hour_local: int = Field(20, alias="ACTIONS_DAILY_METACOG_HOUR_LOCAL")
    actions_daily_metacog_minute_local: int = Field(15, alias="ACTIONS_DAILY_METACOG_MINUTE_LOCAL")

    # HTTP
    port: int = Field(7160, alias="ACTIONS_PORT")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True

    def subscribe_patterns(self) -> list[str]:
        raw = self.actions_subscribe_channels or self.actions_subscribe_channel
        values = [v.strip() for v in str(raw).split(",") if v.strip()]
        if self.actions_subscribe_channel and self.actions_subscribe_channel not in values:
            values.append(self.actions_subscribe_channel)
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
