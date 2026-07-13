from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-graphiti-adapter", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    POSTGRES_URI: str = Field(default="", alias="POSTGRES_URI")
    GRAPHITI_AUTO_APPLY_SCHEMA: bool = Field(default=True, alias="GRAPHITI_AUTO_APPLY_SCHEMA")

    FALKORDB_URI: str = Field(default="", alias="FALKORDB_URI")
    FALKORDB_GRAPH: str = Field(default="graphiti_temporal", alias="FALKORDB_GRAPH")
    FALKORDB_ENABLED: bool = Field(default=False, alias="FALKORDB_ENABLED")

    GRAPHITI_BACKEND: Literal["orion_postgres", "graphiti_core"] = Field(
        default="orion_postgres", alias="GRAPHITI_BACKEND"
    )
    CRYSTALLIZER_EMBED_HOST_URL: str = Field(default="", alias="CRYSTALLIZER_EMBED_HOST_URL")
    # Bootstraps graphiti-core's RELATES_TO fulltext+range indices once at startup when the
    # graphiti_core backend + FalkorDB are both active. See
    # app/backends/graphiti_core.py::ensure_graphiti_indices for the idempotency handling
    # (FalkorDB's CREATE FULLTEXT INDEX has no IF NOT EXISTS guard).
    GRAPHITI_AUTO_BUILD_INDICES: bool = Field(default=True, alias="GRAPHITI_AUTO_BUILD_INDICES")

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()
