from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr
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

    # Require an acknowledged immutable expectation before a real hire.
    CURIOSITY_PEER_EPISODES_ENABLED: bool = Field(default=False)

    # ── Claude usage meter (fail-closed) ──────────────────────────────
    # Claude (via orion-room-companion) is the only peer. Before each hire the
    # worker reads Claude Code transcripts under this path
    # (orion.dev_economics.rate_limit_events.observe) and refuses unless the
    # meter reads a fresh `clear`. Compose mounts the host tree read-only at
    # this SAME path so absolute symlinks inside it still resolve.
    CURIOSITY_PEER_CLAUDE_PROJECTS_PATH: str = Field(
        default_factory=lambda: os.path.expanduser("~/.claude/projects")
    )

    # ── Worldview graph (PeerBrief dual-write target) ─────────────────
    ORION_CURIOSITY_GRAPH_HOST: str = Field(default="")
    ORION_CURIOSITY_GRAPH_PORT: str = Field(default="")
    ORION_CURIOSITY_GRAPH_USER: str = Field(default="")
    ORION_CURIOSITY_GRAPH_PASSWORD: Optional[SecretStr] = Field(default=None)
    ORION_CURIOSITY_GRAPH_OWN: str = Field(default="orion_worldview")

    # ── Channels ──────────────────────────────────────────────────────
    CHANNEL_HELP_REQUEST: str = Field(default="orion:curiosity:help:request")
    CHANNEL_PEER_BRIEF: str = Field(default="orion:curiosity:peer:brief")
    CHANNEL_PEER_BRIEF_CONSUMED: str = Field(
        default="orion:curiosity:peer:brief:consumed"
    )
    CHANNEL_ROOM_CLAUDE_REQUEST: str = Field(default="orion:room:claude:request")
    CHANNEL_ROOM_CLAUDE_UTTERANCE: str = Field(default="orion:room:claude:utterance")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
