# services/orion-agent-chain/app/settings.py

from __future__ import annotations

import logging
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from orion.core.bus.contracts import CHANNELS

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
    planner_request_channel: str = Field(
        "orion-planner:request",
        alias="PLANNER_REQUEST_CHANNEL",
    )

    planner_result_prefix: str = Field(
        "orion-planner:result",
        alias="PLANNER_RESULT_PREFIX",
    )

    # ─────────────────────────────────────────────────────────────
    # UPSTREAM CHANNELS (From UI/Client)
    # ─────────────────────────────────────────────────────────────
    # The UI will publish here, and we will listen (once we add the listener)
    agent_chain_request_channel: str = Field(
        CHANNELS.agent_chain_intake,
        alias="AGENT_CHAIN_REQUEST_CHANNEL"
    )

    # We will publish the final answer here
    agent_chain_result_prefix: str = Field(
        "orion-agent-chain:result",
        alias="AGENT_CHAIN_RESULT_PREFIX"
    )

    # Defaults for the PlannerRequest
    default_max_steps: int = Field(6, alias="AGENT_CHAIN_MAX_STEPS")
    default_timeout_seconds: int = Field(90, alias="AGENT_CHAIN_TIMEOUT_SECONDS")

settings = AgentChainSettings()
logger.info(
    "Loaded orion-agent-chain settings: service=%s v%s port=%d bus=%s enabled=%s",
    settings.service_name,
    settings.service_version,
    settings.port,
    settings.orion_bus_url,
    settings.orion_bus_enabled,
)
