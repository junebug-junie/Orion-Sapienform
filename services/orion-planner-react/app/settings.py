# services/orion-planner-react/app/settings.py

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Core identity
    project: Optional[str] = Field(default=None, alias="PROJECT")
    service_name: str = Field("planner-react", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    planner_port: int = Field(8090, alias="PLANNER_PORT")

    # Orion Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")

    # LLM Gateway (Exec bus)
    llm_gateway_service_name: str = Field(
        "LLMGatewayService",
        alias="LLM_GATEWAY_SERVICE_NAME",
        description="Service label used on exec bus (e.g. LLMGatewayService)",
    )
    exec_request_prefix: str = Field(
        "orion-exec:request",
        alias="EXEC_REQUEST_PREFIX",
        description="Exec bus intake prefix for all cognitive services",
    )
    llm_reply_prefix: str = Field(
        "orion:llm:reply",
        alias="CHANNEL_LLM_REPLY_PREFIX",
        description="Reply prefix for LLM Gateway RPC (Hub pattern)",
    )

    # Cortex Orchestrator (orch bus)
    cortex_request_channel: str = Field(
        "orion-cortex:request",
        alias="CORTEX_REQUEST_CHANNEL",
        description="Bus channel where Cortex-Orch listens for orchestrate_verb",
    )
    cortex_result_prefix: str = Field(
        "orion-cortex:result",
        alias="CORTEX_RESULT_PREFIX",
        description="Result prefix for Cortex-Orch RPC replies",
    )

    # Service Request Channels
    planner_request_channel: str = Field(
        "orion-planner:request",
        alias="PLANNER_REQUEST_CHANNEL",
    )
    planner_result_prefix: str = Field(
        "orion-planner:result",
        alias="PLANNER_RESULT_PREFIX",
    )

    # Logging
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
