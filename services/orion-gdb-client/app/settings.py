from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    project: str = Field(default="orion", env="PROJECT")
    service_name: str = Field(default="orion-meta-writer", env="SERVICE_NAME")
    service_version: str = Field(default="1.0.0", env="SERVICE_VERSION")

    ORION_BUS_URL: str = Field(..., env="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, env="ORION_BUS_ENABLED")
    SUBSCRIBE_CHANNEL: str = Field(default="orion:rdf-collapse:enqueue", env="SUBSCRIBE_CHANNEL")

    graphdb_url: str = Field(..., env="GRAPHDB_URL")
    graphdb_repo: str = Field(default="collapse", env="GRAPHDB_REPO")

    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    port: int = Field(default=8010, env="PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
