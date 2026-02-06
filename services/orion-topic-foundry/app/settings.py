from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field("orion-topic-foundry", validation_alias=AliasChoices("SERVICE_NAME"))
    service_version: str = Field("0.1.0", validation_alias=AliasChoices("SERVICE_VERSION"))
    node_name: str = Field("unknown", validation_alias=AliasChoices("NODE_NAME", "HOSTNAME"))
    log_level: str = Field("INFO", validation_alias=AliasChoices("LOG_LEVEL"))
    port: int = Field(8615, validation_alias=AliasChoices("PORT"))

    topic_foundry_pg_dsn: str = Field(
        "postgresql://postgres:postgres@orion-athena-sql-db:5432/conjourney",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_PG_DSN", "POSTGRES_URI", "POSTGRES_DSN"),
    )
    topic_foundry_embedding_url: str = Field(
        "http://orion-vector-host:8320/embedding",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_EMBEDDING_URL"),
    )
    topic_foundry_model_dir: str = Field(
        "/mnt/telemetry/models/topic-foundry",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_MODEL_DIR"),
    )
    topic_foundry_llm_base_url: str = Field(
        "http://orion-llm-gateway:8011/v1",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_BASE_URL"),
    )
    topic_foundry_llm_model: str = Field(
        "gpt-4o-mini",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_MODEL"),
    )
    topic_foundry_llm_route: str = Field(
        "",
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_ROUTE"),
    )
    topic_foundry_llm_timeout_secs: int = Field(
        60,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_TIMEOUT_SECS"),
    )
    topic_foundry_llm_max_concurrency: int = Field(
        4,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_MAX_CONCURRENCY"),
    )
    topic_foundry_llm_enable: bool = Field(
        False,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_LLM_ENABLE"),
    )

    orion_bus_enabled: bool = Field(
        True,
        validation_alias=AliasChoices("ORION_BUS_ENABLED"),
    )
    orion_bus_url: str = Field(
        "redis://100.92.216.81:6379/0",
        validation_alias=AliasChoices("ORION_BUS_URL"),
    )

    topic_foundry_drift_daemon: bool = Field(
        False,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_DRIFT_DAEMON"),
    )
    topic_foundry_drift_poll_seconds: int = Field(
        900,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_DRIFT_POLL_SECONDS"),
    )
    topic_foundry_drift_window_hours: int = Field(
        24,
        validation_alias=AliasChoices("TOPIC_FOUNDRY_DRIFT_WINDOW_HOURS"),
    )


settings = Settings()
