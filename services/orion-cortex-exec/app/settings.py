from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings

from orion.core.bus.contracts import CHANNELS


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("cortex-exec", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus
    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Intake channel (hub or orch -> exec)
    channel_exec_request: str = Field(CHANNELS.exec_request, alias="CHANNEL_EXEC_REQUEST")

    # Downstream routing (exec -> step services)
    exec_request_prefix: str = Field("orion-exec:request", alias="EXEC_REQUEST_PREFIX")
    
    # CHANGED: 8000 -> 60000 (60s). LLMs need time.
    step_timeout_ms: int = Field(60000, alias="STEP_TIMEOUT_MS")

    # Worker intake channels (canonical, enforced)
    channel_llm_intake: str = Field(CHANNELS.llm_intake, alias="CHANNEL_LLM_INTAKE")
    channel_recall_intake: str = Field(CHANNELS.recall_intake, alias="CHANNEL_RECALL_INTAKE")
    channel_agent_chain_intake: str = Field(CHANNELS.agent_chain_intake, alias="CHANNEL_AGENT_CHAIN_INTAKE")
    channel_planner_intake: str = Field(CHANNELS.planner_intake, alias="CHANNEL_PLANNER_INTAKE")
    channel_council_intake: str = Field(CHANNELS.council_intake, alias="CHANNEL_COUNCIL_INTAKE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
