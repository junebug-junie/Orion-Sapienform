from __future__ import annotations

from functools import lru_cache
from typing import List, Union
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service identity
    service_name: str = Field("cortex-gateway", alias="SERVICE_NAME")
    service_version: str = Field("0.1.0", alias="SERVICE_VERSION")
    node_name: str = Field("gateway-node", alias="NODE_NAME")

    # Bus config
    orion_bus_url: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    # RPC config (Gateway -> Orch)
    channel_cortex_request: str = Field(
        "orion-cortex:request",
        validation_alias=AliasChoices("CORTEX_REQUEST_CHANNEL", "ORCH_REQUEST_CHANNEL"),
    )
    channel_cortex_result_prefix: str = Field(
        "orion-cortex:result",
        validation_alias=AliasChoices("CORTEX_RESULT_PREFIX", "ORCH_RESULT_PREFIX"),
    )

    gateway_rpc_timeout_sec: float = Field(
        120.0,
        alias="CORTEX_GATEWAY_RPC_TIMEOUT_SEC"
    )

    # API Config
    cors_allow_origins: Union[List[str], str] = Field(
        ["*"],
        alias="CORS_ALLOW_ORIGINS"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
