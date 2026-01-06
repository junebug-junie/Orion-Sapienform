import json
from typing import List, Union, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-chat-memory", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    # Bus
    ORION_BUS_URL: str = Field(..., alias="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    HEALTH_CHANNEL: str = "system.health"
    ERROR_CHANNEL: str = "system.error"

    # Inbound channels
    CHAT_MEMORY_INPUT_CHANNELS: Union[str, List[str]] = Field(
        default='["orion:chat:history:log"]',
        alias="CHAT_MEMORY_INPUT_CHANNELS",
    )
    CHAT_MEMORY_COLLECTION: str = Field(default="orion_chat", alias="CHAT_MEMORY_COLLECTION")
    CHAT_MEMORY_UPSERT_CHANNEL: str = Field(
        default="orion:memory:vector:upsert",
        alias="CHAT_MEMORY_UPSERT_CHANNEL",
    )
    CHAT_MEMORY_CHUNK_INTERVAL: int = Field(
        default=0,
        alias="CHAT_MEMORY_CHUNK_INTERVAL",
        description="Embed every Nth chunk message when >0; otherwise only on finalization.",
    )

    # Embedding config
    CHAT_MEMORY_EMBED_ENABLE: bool = Field(default=True, alias="CHAT_MEMORY_EMBED_ENABLE")
    CHAT_MEMORY_EMBED_MODE: str = Field(default="bus", alias="CHAT_MEMORY_EMBED_MODE")
    CHAT_MEMORY_EMBED_REQUEST_CHANNEL: str = Field(
        default="orion:embedding:generate",
        alias="CHAT_MEMORY_EMBED_REQUEST_CHANNEL",
    )
    CHAT_MEMORY_EMBED_RESULT_CHANNEL: str = Field(
        default="orion:embedding:result",
        alias="CHAT_MEMORY_EMBED_RESULT_CHANNEL",
    )
    CHAT_MEMORY_EMBED_HOST_URL: Optional[str] = Field(default=None, alias="CHAT_MEMORY_EMBED_HOST_URL")
    CHAT_MEMORY_INCLUDE_LATENTS: bool = Field(default=False, alias="CHAT_MEMORY_INCLUDE_LATENTS")
    CHAT_MEMORY_EMBED_TIMEOUT_MS: int = Field(default=10000, alias="CHAT_MEMORY_EMBED_TIMEOUT_MS")
    CHAT_MEMORY_EMBED_PROFILE: str = Field(default="default", alias="CHAT_MEMORY_EMBED_PROFILE")

    @property
    def SUBSCRIBE_CHANNELS(self) -> List[str]:
        val = self.CHAT_MEMORY_INPUT_CHANNELS
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return [x.strip() for x in val.split(",") if x.strip()]

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()
