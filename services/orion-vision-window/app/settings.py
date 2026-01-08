from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-window"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    # Channels
    CHANNEL_WINDOW_INTAKE: str = "orion:vision:artifacts"
    CHANNEL_WINDOW_PUB: str = "orion:vision:windows"

    # Cortex Exec
    CHANNEL_WINDOW_REQUEST: str = "orion-exec:request:VisionWindowService"

    # Config
    WINDOW_SIZE_SEC: float = 30.0
