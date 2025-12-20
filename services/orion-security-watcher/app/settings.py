from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = "security-watcher"
    SERVICE_VERSION: str = "0.1.0"

    # Orion bus
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_URL: str = "redis://100.92.216.81:6379/0"

    # Vision events
    VISION_EVENTS_SUBSCRIBE_RAW: str = "orion:vision:edge:raw"

    # Security event channels
    CHANNEL_SECURITY_VISITS: str = "orion:security:visits"
    CHANNEL_SECURITY_ALERTS: str = "orion:security:alerts"

    # Core security toggles
    SECURITY_ENABLED: bool = True
    SECURITY_DEFAULT_ARMED: bool = False
    SECURITY_MODE: str = "vacation_strict"  # "vacation_strict", "off"

    # Rate limiting
    SECURITY_GLOBAL_COOLDOWN_SEC: int = 60
    SECURITY_IDENTITY_COOLDOWN_SEC: int = 300

    # YOLO score threshold (edge default conf=0.25, so keep this low-ish)
    SECURITY_MIN_YOLO_SCORE: float = 0.30

    # Snapshot behavior
    # NOTE: this may contain credentials. We will redact it in logs/emails.
    VISION_SNAPSHOT_URL: str = "http://100.92.216.81:7100/snapshot.jpg"

    # Optional: a safe/public URL to include in emails (no creds)
    # e.g. "http://100.92.216.81:7100/snapshot.jpg" behind your tailscale ACL/proxy
    VISION_SNAPSHOT_PUBLIC_URL: str = ""

    SECURITY_SNAPSHOT_COUNT: int = 3
    SECURITY_SNAPSHOT_DIR: str = "/mnt/telemetry/orion-security/alerts"

    # Notification mode
    # "inline" = send email directly from this service
    # "off"    = don't send, just publish alerts on bus
    NOTIFY_MODE: str = "inline"

    # Email config (for inline mode)
    NOTIFY_EMAIL_SMTP_HOST: str = ""
    NOTIFY_EMAIL_SMTP_PORT: int = 587
    NOTIFY_EMAIL_SMTP_USERNAME: str = ""
    NOTIFY_EMAIL_SMTP_PASSWORD: str = ""
    NOTIFY_EMAIL_USE_TLS: bool = True
    NOTIFY_EMAIL_FROM: str = "orion-security@example.com"
    NOTIFY_EMAIL_TO: str = ""

    # State store
    SECURITY_STATE_PATH: str = "/mnt/telemetry/orion-security/state.json"

    @property
    def notify_email_to(self) -> List[str]:
        raw = self.NOTIFY_EMAIL_TO or ""
        return [e.strip() for e in raw.split(",") if e.strip()]


def get_settings() -> Settings:
    return Settings()
