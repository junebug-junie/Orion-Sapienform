# ==================================================
# settings.py
# ==================================================
import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- Metadata ---
    PROJECT: str = Field(default="orion-janus")
    SERVICE_NAME: str = Field(default="orion-dream")
    SERVICE_VERSION: str = Field(default="1.0.0")
    ENVIRONMENT: str = Field(default="prod")
    PORT: int = Field(default=8620)

    # --- Redis ---
    ORION_BUS_URL: str = Field(default="redis://redis:6379/0")
    ORION_BUS_ENABLED: bool = Field(default=True)
    CHANNEL_DREAM_TRIGGER: str = Field(default="orion:dream:trigger")
    CHANNEL_DREAM_BUFFER: str = Field(default="orion:dream:buffer")
    CHANNEL_DREAM_COMPLETE: str = Field(default="orion:dream:complete")
    CHANNEL_DREAM_STATUS: str = Field(default="orion:dream:status")

    # --- Memory streams ---
    CHANNEL_COLLAPSE_SQL_PUBLISH: str = Field(default="orion:collapse:sql-write")
    CHANNEL_COLLAPSE_TAGS_PUBLISH: str = Field(default="orion:tags:enriched")
    CHANNEL_TELEMETRY_PUBLISH: str = Field(default="orion:biometrics:telemetry")
    CHANNEL_CHAT: str = Field(default="orion:chat:history:log")

    # --- Stores ---
    POSTGRES_URI: str = Field(default="postgresql://postgres:postgres@postgres:5432/conjourney")
    VECTOR_DB_HOST: str = Field(default="vector-db")
    VECTOR_DB_PORT: int = Field(default=8000)
    VECTOR_DB_COLLECTION: str = Field(default="orion_main_store")

    GRAPHDB_URL: str = Field(default="http://graphdb:7200")
    GRAPHDB_REPO: str = Field(default="collapse")
    GRAPHDB_USER: str = Field(default="admin")
    GRAPHDB_PASS: str = Field(default="admin")

    # --- Brain ---
    BRAIN_URL: str = Field(default="http://brain:8088")
    LLM_MODEL: str = Field(default="mistral:instruct")

settings = Settings()
