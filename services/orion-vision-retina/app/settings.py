from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-retina"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"

    # Channels
    CHANNEL_RETINA_PUB: str = "orion:vision:frames"

    # Capture
    RETINA_SOURCE_TYPE: str = "mock" # mock, folder
    RETINA_SOURCE_PATH: str = "/mnt/telemetry/vision/intake"
    RETINA_FPS: float = 0.2
