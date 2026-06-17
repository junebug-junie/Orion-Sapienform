from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_name: str = Field("orion-self-experiments", alias="SERVICE_NAME")
    service_version: str = Field("0.2.0", alias="SERVICE_VERSION")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    experiments_store_path: str = Field(
        "/tmp/orion-self-experiments/experiments.sqlite3",
        alias="EXPERIMENTS_STORE_PATH",
    )
    experiments_allow_non_read_only: bool = Field(False, alias="EXPERIMENTS_ALLOW_NON_READ_ONLY")
    port: int = Field(7172, alias="PORT")

    orion_bus_url: str = Field("redis://127.0.0.1:6379/0", alias="ORION_BUS_URL")
    orion_bus_enabled: bool = Field(False, alias="ORION_BUS_ENABLED")

    self_experiments_dispatch_enabled: bool = Field(False, alias="SELF_EXPERIMENTS_DISPATCH_ENABLED")
    self_experiments_context_exec_dispatch_transport: str = Field(
        "bus",
        alias="SELF_EXPERIMENTS_CONTEXT_EXEC_DISPATCH_TRANSPORT",
    )
    self_experiments_context_exec_url: str = Field(
        "http://orion-context-exec:8096",
        alias="SELF_EXPERIMENTS_CONTEXT_EXEC_URL",
    )
    self_experiments_context_exec_request_channel: str = Field(
        "orion:exec:request:ContextExecService",
        alias="SELF_EXPERIMENTS_CONTEXT_EXEC_REQUEST_CHANNEL",
    )
    self_experiments_context_exec_timeout_seconds: float = Field(
        90.0,
        alias="SELF_EXPERIMENTS_CONTEXT_EXEC_TIMEOUT_SECONDS",
    )
    self_experiments_max_dispatch_attempts: int = Field(2, alias="SELF_EXPERIMENTS_MAX_DISPATCH_ATTEMPTS")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()
