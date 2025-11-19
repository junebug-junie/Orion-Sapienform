from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Node identity
    node_name: str = Field("athena-cortex-orchestrator", alias="NODE_NAME")

    # Redis / bus config
    orion_bus_url: str = Field("redis://orion-redis:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")

    # Exec routing
    exec_request_prefix: str = Field("orion-exec:request", alias="EXEC_REQUEST_PREFIX")
    exec_result_prefix: str = Field("orion-exec:result", alias="EXEC_RESULT_PREFIX")

    # Optional orchestrator-level routing
    cortex_orch_request_channel: str = Field(
        "orion-cortex:request", alias="ORCH_REQUEST_CHANNEL"
    )
    cortex_orch_result_prefix: str = Field(
        "orion-cortex:result", alias="ORCH_RESULT_PREFIX"
    )

    # Timeouts
    cortex_step_timeout_ms: int = Field(8000, alias="ORION_CORTEX_STEP_TIMEOUT_MS")

    # HTTP service config
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8072, alias="API_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
