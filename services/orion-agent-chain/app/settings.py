# services/orion-agent-chain/app/settings.py

from __future__ import annotations

import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("orion-agent-chain.settings")


class AgentChainSettings(BaseSettings):
    """
    Pydantic Settings for orion-agent-chain.

    Mirrors the planner-react pattern:
    - SERVICE_NAME / SERVICE_VERSION
    - AGENT_CHAIN_PORT
    - ORION_BUS_* for Redis
    - Planner + Cortex channels so we can talk on the bus
    """

    # --- Service identity ---
    service_name: str = Field("agent-chain", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")

    # --- HTTP port ---
    port: int = Field(8092, alias="AGENT_CHAIN_PORT")

    # --- Orion bus / Redis ---
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field(
        "redis://orion-redis:6379/0",
        alias="ORION_BUS_URL",
    )

    # --- Planner / Cortex integration ---
    # If agent-chain will call planner-react as a tool orchestrator:
    planner_request_channel: str = Field(
        "orion-exec:request:PlannerReactService",
        alias="CHANNEL_PLANNER_REQUEST",
    )
    planner_reply_prefix: str = Field(
        "orion:planner:reply",
        alias="CHANNEL_PLANNER_REPLY_PREFIX",
    )


settings = AgentChainSettings()
logger.info(
    "Loaded orion-agent-chain settings: service=%s v%s port=%d bus=%s enabled=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
    settings.orion_bus_url,
    settings.orion_bus_enabled,
)
