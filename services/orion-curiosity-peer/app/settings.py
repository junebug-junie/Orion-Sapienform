from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Contractor peer service settings.

    Dual kill switch (both required for a live hire):
    - Hub: HUB_CURIOSITY_CONTRACTOR_PEER_ENABLED (enqueue gate; not read here)
    - This service: CURIOSITY_PEER_ENABLED (subscribe / process gate)
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Service identity ──────────────────────────────────────────────
    SERVICE_NAME: str = Field(default="curiosity-peer")
    SERVICE_VERSION: str = Field(default="0.1.0")
    CURIOSITY_PEER_NODE_NAME: str = Field(default_factory=lambda: os.uname().nodename)

    # ── Orion Bus config ──────────────────────────────────────────────
    ORION_BUS_ENABLED: bool = Field(default=True)
    ORION_BUS_ENFORCE_CATALOG: bool = Field(default=False)
    # ALWAYS use redis://<tailscale-node-ip>:6379/0 — never bus-core / redis hostname.
    ORION_BUS_URL: str = Field(default="redis://100.92.216.81:6379/0")
    HEARTBEAT_INTERVAL_SEC: float = Field(default=10.0)

    # ── Kill switch (service-side) ────────────────────────────────────
    # Typo alias CURIOUSITY_PEER_ENABLED accepted so a misspelling still
    # controls the same dial; default remains false either way.
    CURIOSITY_PEER_ENABLED: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "CURIOSITY_PEER_ENABLED",
            "CURIOUSITY_PEER_ENABLED",
        ),
    )

    # ── Cursor credential — lives HERE, never in Hub ──────────────────
    # Same isolation rationale as room-companion's Claude OAuth token:
    # Hub is root-equivalent with docker.sock; a key there is one Orion's
    # FCC turns can read. SecretStr so logs / model_dump do not leak it.
    CURSOR_API_KEY: Optional[SecretStr] = Field(default=None)

    @field_validator("CURSOR_API_KEY", mode="before")
    @classmethod
    def _strip_pasted_key(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    CURIOSITY_PEER_REPO_ROOT: str = Field(default="/repo")
    CURIOSITY_PEER_MODEL: str = Field(default="composer-2.5")

    # ── Worldview graph (PeerBrief dual-write target) ─────────────────
    ORION_CURIOSITY_GRAPH_HOST: str = Field(default="")
    ORION_CURIOSITY_GRAPH_PORT: str = Field(default="")
    ORION_CURIOSITY_GRAPH_USER: str = Field(default="")
    ORION_CURIOSITY_GRAPH_PASSWORD: Optional[SecretStr] = Field(default=None)
    ORION_CURIOSITY_GRAPH_OWN: str = Field(default="orion_worldview")

    # ── Channels ──────────────────────────────────────────────────────
    CHANNEL_HELP_REQUEST: str = Field(default="orion:curiosity:help:request")
    CHANNEL_PEER_BRIEF: str = Field(default="orion:curiosity:peer:brief")
    CHANNEL_ROOM_CLAUDE_REQUEST: str = Field(default="orion:room:claude:request")
    CHANNEL_ROOM_CLAUDE_UTTERANCE: str = Field(default="orion:room:claude:utterance")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
