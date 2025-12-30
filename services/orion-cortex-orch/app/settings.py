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

    # Chassis
    heartbeat_interval_sec: float = Field(10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # RPC intake channel (hub -> cortex-orch)
    channel_cortex_request: str = Field(
        "orion-cortex-orch:request",
        validation_alias=AliasChoices("CORTEX_REQUEST_CHANNEL", "ORCH_REQUEST_CHANNEL"),
    )

    # RPC out to cortex-exec
    channel_exec_request: str = Field(
        "orion-cortex-exec:request",
        validation_alias=AliasChoices("CORTEX_EXEC_REQUEST_CHANNEL", "CHANNEL_EXEC_REQUEST"),
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
