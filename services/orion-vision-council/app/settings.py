from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "vision-council"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    ORION_BUS_ENFORCE_CATALOG: bool = False

    # Channels
    CHANNEL_COUNCIL_INTAKE: str = "orion:vision:windows"
    CHANNEL_COUNCIL_PUB: str = "orion:vision:events"

    # Cortex Exec
    CHANNEL_COUNCIL_REQUEST: str = "orion:exec:request:VisionCouncilService"

    CHANNEL_LLM_REQUEST: str = "orion:exec:request:LLMGatewayService"
    CHANNEL_LLM_REPLY_PREFIX: str = "orion:council:reply"

    # Config
    COUNCIL_MODEL: str = "llama-3-8b-instruct-q4_k_m"
    COUNCIL_LLM_ROUTE: str = "metacog"
    COUNCIL_LLM_MAX_TOKENS: int = 1024
    COUNCIL_LLM_TIMEOUT_SEC: float = 90.0
    COUNCIL_STRUCTURED_OUTPUT_METHOD: str = "json_object_schema"
