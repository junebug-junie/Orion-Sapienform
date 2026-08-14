from __future__ import annotations

import json
from typing import List, Optional, Union, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field(default="orion-vector-host", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field(default="0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field(default="athena", alias="NODE_NAME")
    LOG_LEVEL: str = Field(default="INFO", alias="LOG_LEVEL")

    # Bus
    ORION_BUS_URL: str = Field(..., alias="ORION_BUS_URL")
    ORION_BUS_ENABLED: bool = Field(default=True, alias="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False, alias="ORION_BUS_ENFORCE_CATALOG")
    ORION_HEALTH_CHANNEL: str = Field(default="orion:system:health", alias="ORION_HEALTH_CHANNEL")
    ERROR_CHANNEL: str = Field(default="orion:system:error", alias="ERROR_CHANNEL")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0, alias="HEARTBEAT_INTERVAL_SEC")

    # Channels
    VECTOR_HOST_CHAT_HISTORY_CHANNEL: str = Field(
        default="orion:chat:history:log",
        alias="VECTOR_HOST_CHAT_HISTORY_CHANNEL",
    )
    VECTOR_HOST_GPT_ENABLED: bool = Field(
        default=False,
        alias="VECTOR_HOST_GPT_ENABLED",
    )
    VECTOR_HOST_GPT_MESSAGE_CHANNEL: str = Field(
        default="orion:chat:gpt:log",
        alias="VECTOR_HOST_GPT_MESSAGE_CHANNEL",
    )
    VECTOR_HOST_GPT_TURN_CHANNEL: str = Field(
        default="orion:chat:gpt:turn",
        alias="VECTOR_HOST_GPT_TURN_CHANNEL",
    )
    VECTOR_HOST_CHAT_TURN_CHANNEL: str = Field(
        default="orion:chat:history:turn",
        alias="VECTOR_HOST_CHAT_TURN_CHANNEL",
    )
    VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL: str = Field(
        default="orion:embedding:generate",
        alias="VECTOR_HOST_EMBEDDING_REQUEST_CHANNEL",
    )
    VECTOR_HOST_EMBEDDING_RESULT_PREFIX: str = Field(
        default="orion:embedding:result:",
        alias="VECTOR_HOST_EMBEDDING_RESULT_PREFIX",
    )
    VECTOR_HOST_SEMANTIC_UPSERT_CHANNEL: str = Field(
        default="orion:vector:semantic:upsert",
        alias="VECTOR_HOST_SEMANTIC_UPSERT_CHANNEL",
    )
    VECTOR_HOST_SKIP_REJECTED: bool = Field(
        default=True,
        alias="VECTOR_HOST_SKIP_REJECTED",
    )
    VECTOR_HOST_EMBED_DURABLE_ONLY: bool = Field(
        default=False,
        alias="VECTOR_HOST_EMBED_DURABLE_ONLY",
    )

    # Embedding behavior
    VECTOR_HOST_EMBED_ROLES: Union[str, List[str]] = Field(
        default='["user","assistant"]',
        alias="VECTOR_HOST_EMBED_ROLES",
    )
    VECTOR_HOST_EMBED_BACKEND: Literal["hf", "vllm"] = Field(
        default="hf",
        alias="VECTOR_HOST_EMBED_BACKEND",
    )
    VECTOR_HOST_EMBEDDING_MODEL: str = Field(..., alias="VECTOR_HOST_EMBEDDING_MODEL")
    VECTOR_HOST_EMBEDDING_DEVICE: str = Field(
        default="cpu",
        alias="VECTOR_HOST_EMBEDDING_DEVICE",
    )
    VECTOR_HOST_SEMANTIC_COLLECTION: str = Field(
        default="orion_main_store",
        alias="VECTOR_HOST_SEMANTIC_COLLECTION",
    )
    # VECTOR_HOST_CHAT_MESSAGE_COLLECTION / VECTOR_HOST_CHAT_TURN_COLLECTION
    # removed 2026-08-14: they only targeted the now-dead orion_chat /
    # orion_chat_turns Chroma vector-upsert writes (see main.py's
    # _handle_chat_history / _handle_chat_turn docstrings). Chat/turn
    # embeddings still feed OrionTissue directly; they no longer target a
    # Chroma collection at all.
    VECTOR_HOST_GPT_MESSAGE_COLLECTION: str = Field(
        default="orion_chat_gpt",
        alias="VECTOR_HOST_GPT_MESSAGE_COLLECTION",
    )
    VECTOR_HOST_GPT_TURN_COLLECTION: str = Field(
        default="orion_chat_gpt_turns",
        alias="VECTOR_HOST_GPT_TURN_COLLECTION",
    )

    ORION_LLM_VLLM_URL: Optional[str] = Field(None, alias="ORION_LLM_VLLM_URL")
    ORION_LLM_LLAMA_COLA_URL: Optional[str] = Field(None, alias="ORION_LLM_LLAMA_COLA_URL")

    VECTOR_HOST_EMBED_CONNECT_TIMEOUT_SEC: float = Field(
        default=10.0,
        alias="VECTOR_HOST_EMBED_CONNECT_TIMEOUT_SEC",
    )
    VECTOR_HOST_EMBED_READ_TIMEOUT_SEC: float = Field(
        default=60.0,
        alias="VECTOR_HOST_EMBED_READ_TIMEOUT_SEC",
    )

    # OrionTissue physics feed (2026-07-28, extracted from the retired
    # orion-spark-introspector service -- see app/tissue_feed.py's module
    # docstring for the full formula this drives). ORION_TISSUE_SNAPSHOT_PATH
    # is read directly from os.environ by orion.spark.orion_tissue.OrionTissue
    # itself (its own long-standing contract, unchanged here); declared below
    # only so .env_example documents it as part of this service's contract.
    ORION_TISSUE_SNAPSHOT_PATH: str = Field(
        default="/mnt/graphdb/orion/spark/tissue-brain.npy",
        alias="ORION_TISSUE_SNAPSHOT_PATH",
    )
    TISSUE_STIMULUS_EMA_ALPHA: float = Field(
        default=0.05,
        alias="TISSUE_STIMULUS_EMA_ALPHA",
    )
    TISSUE_STIMULUS_STEPS: int = Field(
        default=2,
        alias="TISSUE_STIMULUS_STEPS",
    )
    TISSUE_STIMULUS_LEARNING_RATE: float = Field(
        default=0.1,
        alias="TISSUE_STIMULUS_LEARNING_RATE",
    )

    @property
    def EMBED_ROLES(self) -> List[str]:
        val = self.VECTOR_HOST_EMBED_ROLES
        if isinstance(val, list):
            return [str(v) for v in val]
        try:
            loaded = json.loads(val)
            if isinstance(loaded, list):
                return [str(v) for v in loaded]
        except json.JSONDecodeError:
            pass
        return [v.strip() for v in str(val).split(",") if v.strip()]

    @field_validator("VECTOR_HOST_EMBEDDING_MODEL")
    @classmethod
    def _require_model(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("VECTOR_HOST_EMBEDDING_MODEL is required")
        return v

    class Config:
        env_file = ".env"
        extra = "ignore"
        populate_by_name = True


settings = Settings()
