from __future__ import annotations

from functools import lru_cache
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
