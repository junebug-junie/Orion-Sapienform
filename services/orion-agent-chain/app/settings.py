from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    service_name: str = Field("agent-chain", alias="SERVICE_NAME")

    # Bus
    orion_bus_enabled: bool = Field(True, alias="ORION_BUS_ENABLED")
    orion_bus_url: str = Field(
        "redis://orion-redis:6379/0",
        alias="ORION_BUS_URL",
    )

    # Planner channels
    planner_request_channel: str = Field(
        "orion-planner:request",
        alias="PLANNER_REQUEST_CHANNEL",
    )
    planner_result_prefix: str = Field(
        "orion-planner:result",
        alias="PLANNER_RESULT_PREFIX",
    )


settings = Settings()
