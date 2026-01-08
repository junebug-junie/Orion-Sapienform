from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Identity
    service_name: str = Field("cortex-exec", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    node_name: str = Field("athena", alias="NODE_NAME")

    # Bus
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_enforce_catalog: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Intake channel (hub or orch -> exec)
    channel_exec_request: str = Field("orion-cortex-exec:request", alias="CHANNEL_EXEC_REQUEST")

    # Downstream routing (exec -> step services)
    exec_request_prefix: str = Field("orion-exec:request", alias="EXEC_REQUEST_PREFIX")
    exec_result_prefix: str = Field("orion-exec:result", alias="EXEC_RESULT_PREFIX")

    # CHANGED: 8000 -> 60000 (60s). LLMs need time.
    step_timeout_ms: int = Field(60000, alias="STEP_TIMEOUT_MS")

    # CHANGED: "orion-llm:intake" -> "orion-exec:request:LLMGatewayService"
    channel_llm_intake: str = Field("orion-exec:request:LLMGatewayService", alias="CHANNEL_LLM_INTAKE")
    channel_recall_intake: str = Field("recall.query.v1", alias="CHANNEL_RECALL_INTAKE")
    channel_agent_chain_intake: str = Field("orion-exec:request:AgentChainService", alias="CHANNEL_AGENT_CHAIN_INTAKE")
    channel_planner_intake: str = Field("orion-exec:request:PlannerReactService", alias="CHANNEL_PLANNER_INTAKE")
    channel_council_intake: str = Field("orion:agent-council:intake", alias="CHANNEL_COUNCIL_INTAKE")
    channel_council_reply_prefix: str = Field("orion:council:reply", alias="CHANNEL_COUNCIL_REPLY_PREFIX")
    channel_cognition_trace_pub: str = Field("orion:cognition:trace", alias="CHANNEL_COGNITION_TRACE_PUB")

    diagnostic_mode: bool = Field(False, alias="DIAGNOSTIC_MODE")
    diagnostic_recall_timeout_sec: float = Field(5.0, alias="DIAGNOSTIC_RECALL_TIMEOUT_SEC")
    diagnostic_agent_timeout_sec: float = Field(15.0, alias="DIAGNOSTIC_AGENT_TIMEOUT_SEC")
    orion_verb_backdoor_enabled: bool = Field(False, alias="ORION_VERB_BACKDOOR_ENABLED")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
