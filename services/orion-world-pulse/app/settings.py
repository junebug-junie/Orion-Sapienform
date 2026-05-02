from __future__ import annotations

from pydantic import AliasChoices
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field("orion-world-pulse", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("unknown", alias="NODE_NAME")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    port: int = Field(8628, alias="PORT")

    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    world_pulse_enabled: bool = Field(False, alias="WORLD_PULSE_ENABLED")
    world_pulse_dry_run: bool = Field(True, alias="WORLD_PULSE_DRY_RUN")
    world_pulse_fetch_enabled: bool = Field(True, alias="WORLD_PULSE_FETCH_ENABLED")
    world_pulse_sql_enabled: bool = Field(True, alias="WORLD_PULSE_SQL_ENABLED")
    world_pulse_graph_enabled: bool = Field(False, alias="WORLD_PULSE_GRAPH_ENABLED")
    world_pulse_graph_dry_run: bool = Field(True, alias="WORLD_PULSE_GRAPH_DRY_RUN")
    world_pulse_email_enabled: bool = Field(False, alias="WORLD_PULSE_EMAIL_ENABLED")
    world_pulse_email_dry_run: bool = Field(True, alias="WORLD_PULSE_EMAIL_DRY_RUN")
    world_pulse_hub_messages_enabled: bool = Field(True, alias="WORLD_PULSE_HUB_MESSAGES_ENABLED")
    world_pulse_stance_enabled: bool = Field(False, alias="WORLD_PULSE_STANCE_ENABLED")
    world_pulse_schedule_enabled: bool = Field(False, alias="WORLD_PULSE_SCHEDULE_ENABLED")

    world_pulse_timezone: str = Field("America/Denver", alias="WORLD_PULSE_TIMEZONE")
    world_pulse_locality: str = Field("Utah", alias="WORLD_PULSE_LOCALITY")
    world_pulse_max_articles_per_run: int = Field(100, alias="WORLD_PULSE_MAX_ARTICLES_PER_RUN")
    world_pulse_max_articles_per_source: int = Field(10, alias="WORLD_PULSE_MAX_ARTICLES_PER_SOURCE")
    world_pulse_fetch_timeout_seconds: int = Field(20, alias="WORLD_PULSE_FETCH_TIMEOUT_SECONDS")
    world_pulse_min_stance_confidence: float = Field(0.65, alias="WORLD_PULSE_MIN_STANCE_CONFIDENCE")
    world_pulse_stance_max_topics: int = Field(5, alias="WORLD_PULSE_STANCE_MAX_TOPICS")
    world_pulse_politics_stance_default: str = Field("only_when_requested", alias="WORLD_PULSE_POLITICS_STANCE_DEFAULT")

    world_pulse_sources_config_path: str = Field(
        "config/world_pulse/sources.yaml",
        alias="WORLD_PULSE_SOURCES_CONFIG_PATH",
    )

    world_pulse_run_request_channel: str = Field(
        "orion:world_pulse:run:request",
        alias="WORLD_PULSE_RUN_REQUEST_CHANNEL",
    )
    world_pulse_run_result_channel: str = Field(
        "orion:world_pulse:run:result",
        alias="WORLD_PULSE_RUN_RESULT_CHANNEL",
    )
    world_pulse_digest_created_channel: str = Field(
        "orion:world_pulse:digest:created",
        alias="WORLD_PULSE_DIGEST_CREATED_CHANNEL",
    )
    world_pulse_digest_published_channel: str = Field(
        "orion:world_pulse:digest:published",
        alias="WORLD_PULSE_DIGEST_PUBLISHED_CHANNEL",
    )
    world_context_daily_capsule_channel: str = Field(
        "orion:world_context:daily_capsule",
        alias="WORLD_CONTEXT_DAILY_CAPSULE_CHANNEL",
    )
    world_pulse_hub_message_channel: str = Field(
        "orion:hub:messages:create",
        alias="WORLD_PULSE_HUB_MESSAGE_CHANNEL",
    )
    world_pulse_graph_channel: str = Field(
        "orion:world_pulse:graph:upsert",
        alias="WORLD_PULSE_GRAPH_CHANNEL",
    )
    world_pulse_claim_channel: str = Field(
        "orion:world_pulse:claim:emit",
        alias="WORLD_PULSE_CLAIM_CHANNEL",
    )
    world_pulse_event_channel: str = Field(
        "orion:world_pulse:event:emit",
        alias="WORLD_PULSE_EVENT_CHANNEL",
    )
    world_pulse_entity_channel: str = Field(
        "orion:world_pulse:entity:emit",
        alias="WORLD_PULSE_ENTITY_CHANNEL",
    )
    world_pulse_learning_channel: str = Field(
        "orion:world_pulse:learning:emit",
        alias="WORLD_PULSE_LEARNING_CHANNEL",
    )
    world_pulse_situation_brief_channel: str = Field(
        "orion:world_pulse:situation:brief:upsert",
        alias="WORLD_PULSE_SITUATION_BRIEF_CHANNEL",
    )
    world_pulse_situation_change_channel: str = Field(
        "orion:world_pulse:situation:change:emit",
        alias="WORLD_PULSE_SITUATION_CHANGE_CHANNEL",
    )

    notify_url: str = Field("http://orion-notify:7140", validation_alias=AliasChoices("WORLD_PULSE_NOTIFY_URL", "NOTIFY_URL"))
    notify_api_token: str | None = Field(None, alias="NOTIFY_API_TOKEN")
    actions_url: str = Field("http://orion-actions:8110", alias="ACTIONS_URL")


settings = Settings()
