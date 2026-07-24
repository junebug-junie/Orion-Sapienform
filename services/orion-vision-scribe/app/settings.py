from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-scribe"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "athena"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False
    # Bus-native SystemHealthV1 heartbeat cadence (orion:system:health). See
    # docs/superpowers/specs/2026-07-24-service-heartbeat-node-telemetry-design.md.
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Channels
    CHANNEL_SCRIBE_INTAKE: str = "orion:vision:events"
    CHANNEL_SCRIBE_PUB: str = "orion:vision:scribe:pub"

    # Cortex Exec
    CHANNEL_SCRIBE_REQUEST: str = "orion:exec:request:VisionScribeService"

    CHANNEL_SQL_WRITE: str = "orion:vision:events:sql-write"
    CHANNEL_VECTOR_WRITE: str = "orion:vector:write"
