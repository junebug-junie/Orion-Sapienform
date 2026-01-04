from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    SERVICE_NAME: str = Field("security-watcher", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field("0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field("unknown", alias="NODE_NAME")

    # Orion bus
    ORION_BUS_ENABLED: bool = Field(True, alias="ORION_BUS_ENABLED")
    ORION_BUS_URL: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    # Vision events
    VISION_EVENTS_SUBSCRIBE_RAW: str = Field("orion:vision:edge:raw", alias="VISION_EVENTS_SUBSCRIBE_RAW")
    CHANNEL_VISION_ARTIFACTS: str = Field("vision.artifacts", alias="CHANNEL_VISION_ARTIFACTS")

    # Guard Output Channels
    CHANNEL_VISION_GUARD_SIGNAL: str = Field("vision.guard.signal", alias="CHANNEL_VISION_GUARD_SIGNAL")
    CHANNEL_VISION_GUARD_ALERT: str = Field("vision.guard.alert", alias="CHANNEL_VISION_GUARD_ALERT")

    # Guard Logic Config
    GUARD_WINDOW_SECONDS: int = Field(30, alias="GUARD_WINDOW_SECONDS")
    GUARD_EMIT_EVERY_SECONDS: int = Field(5, alias="GUARD_EMIT_EVERY_SECONDS")
    GUARD_PERSON_MIN_CONF: float = Field(0.4, alias="GUARD_PERSON_MIN_CONF")
    GUARD_PERSON_MIN_COUNT: int = Field(1, alias="GUARD_PERSON_MIN_COUNT")
    GUARD_SUSTAIN_SECONDS: int = Field(2, alias="GUARD_SUSTAIN_SECONDS")
    GUARD_ALERT_COOLDOWN_SECONDS: int = Field(60, alias="GUARD_ALERT_COOLDOWN_SECONDS")

    # Legacy Security Config (Keep for existing logic compatibility if needed)
    CHANNEL_SECURITY_VISITS: str = Field("orion:security:visits", alias="CHANNEL_SECURITY_VISITS")
    CHANNEL_SECURITY_ALERTS: str = Field("orion:security:alerts", alias="CHANNEL_SECURITY_ALERTS")
    SECURITY_ENABLED: bool = Field(True, alias="SECURITY_ENABLED")
    SECURITY_DEFAULT_ARMED: bool = Field(False, alias="SECURITY_DEFAULT_ARMED")
    SECURITY_MODE: str = Field("vacation_strict", alias="SECURITY_MODE")
    SECURITY_GLOBAL_COOLDOWN_SEC: int = Field(60, alias="SECURITY_GLOBAL_COOLDOWN_SEC")
    SECURITY_IDENTITY_COOLDOWN_SEC: int = Field(300, alias="SECURITY_IDENTITY_COOLDOWN_SEC")
    SECURITY_MIN_YOLO_SCORE: float = Field(0.30, alias="SECURITY_MIN_YOLO_SCORE")

    # Snapshot behavior
    VISION_SNAPSHOT_URL: str = Field("http://100.92.216.81:7100/snapshot.jpg", alias="VISION_SNAPSHOT_URL")
    VISION_SNAPSHOT_PUBLIC_URL: str = Field("", alias="VISION_SNAPSHOT_PUBLIC_URL")
    SECURITY_SNAPSHOT_COUNT: int = Field(3, alias="SECURITY_SNAPSHOT_COUNT")
    SECURITY_SNAPSHOT_DIR: str = Field("/mnt/telemetry/orion-security/alerts", alias="SECURITY_SNAPSHOT_DIR")

    # Notification mode
    NOTIFY_MODE: str = Field("inline", alias="NOTIFY_MODE")

    # Email config
    NOTIFY_EMAIL_SMTP_HOST: str = Field("", alias="NOTIFY_EMAIL_SMTP_HOST")
    NOTIFY_EMAIL_SMTP_PORT: int = Field(587, alias="NOTIFY_EMAIL_SMTP_PORT")
    NOTIFY_EMAIL_SMTP_USERNAME: str = Field("", alias="NOTIFY_EMAIL_SMTP_USERNAME")
    NOTIFY_EMAIL_SMTP_PASSWORD: str = Field("", alias="NOTIFY_EMAIL_SMTP_PASSWORD")
    NOTIFY_EMAIL_USE_TLS: bool = Field(True, alias="NOTIFY_EMAIL_USE_TLS")
    NOTIFY_EMAIL_FROM: str = Field("orion-security@example.com", alias="NOTIFY_EMAIL_FROM")
    NOTIFY_EMAIL_TO: str = Field("", alias="NOTIFY_EMAIL_TO")

    # State store
    SECURITY_STATE_PATH: str = Field("/mnt/telemetry/orion-security/state.json", alias="SECURITY_STATE_PATH")

    @property
    def notify_email_to(self) -> List[str]:
        raw = self.NOTIFY_EMAIL_TO or ""
        return [e.strip() for e in raw.split(",") if e.strip()]


def get_settings() -> Settings:
    return Settings()
