from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "percept-store"
    SERVICE_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"
    HTTP_HOST: str = "0.0.0.0"
    HTTP_PORT: int = 8000

    # Content-addressed blob root. Deliberately NOT Hub's
    # HUB_CHAT_ATTACHMENT_DIR: percepts are camera frames of a private home and
    # must not share a namespace, a lifetime, or a serving process with user
    # chat uploads. Hub also holds the docker socket.
    PERCEPT_STORE_DIR: str = "/mnt/orion-percepts"

    # Retention. Percepts are the shortest-lived artifact in the system: a
    # frame exists to be interpreted once, and the interpretation is what gets
    # kept. Anything longer is a surveillance archive nobody asked for.
    PERCEPT_RETENTION_SECONDS: int = 3600
    PERCEPT_SWEEP_INTERVAL_SECONDS: int = 300

    # Upload guards.
    PERCEPT_MAX_BYTES: int = 8 * 1024 * 1024
    # audio/wav and video/mp4 added 2026-08-22 for AffectGPT's short
    # face+audio capture clips (see sniff_mime in storage.py) -- still
    # comfortably under PERCEPT_MAX_BYTES for a clip of a few seconds at
    # modest resolution/bitrate; raise PERCEPT_MAX_BYTES if clip length or
    # quality grows.
    PERCEPT_ALLOWED_MIMES: str = "image/jpeg,image/png,image/webp,audio/wav,video/mp4"

    # Shared-secret gate. Empty disables the check, which is acceptable only on
    # a closed tailnet; set it wherever the port is reachable more widely.
    PERCEPT_STORE_TOKEN: str = ""


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
