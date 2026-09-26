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

    # ── Cursor Agent CLI (desktop login) — lives HERE, never in Hub ───
    # Same isolation rationale as room-companion's Claude credential:
    # Hub is root-equivalent with docker.sock. Auth is host `agent login`
    # (~/.config/cursor/auth.json), not CURSOR_API_KEY / cursor-sdk.
    # Compose mounts the host install + config; default points at the
    # versioned binary under the share mount (wrapper needs sibling node).
    CURIOSITY_PEER_AGENT_BIN: str = Field(
        default="/opt/cursor-agent/versions/2026.05.20-2b5dd59/cursor-agent"
    )

    CURIOSITY_PEER_REPO_ROOT: str = Field(default="/repo")
    CURIOSITY_PEER_MODEL: str = Field(default="composer-2.5")
    # Require an acknowledged immutable expectation before a real hire.
    CURIOSITY_PEER_EPISODES_ENABLED: bool = Field(default=False)

    # ── Contested Cursor budget meter (fail-closed) ───────────────────
    # observe_cursor_limit() refuses hire until a real reading exists.
    # File wins over STATE. Empty both → unobserved (no hire).
    # STATE values: clear | limited | unknown
    # FILE: JSON {"state":"clear","observed_at":"<iso>"} or plain "clear".
    CURIOSITY_PEER_CURSOR_BUDGET_STATE: str = Field(default="")
    CURIOSITY_PEER_CURSOR_BUDGET_FILE: str = Field(default="")

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
