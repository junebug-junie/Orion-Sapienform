# app/settings.py
from typing import List, Optional
from pydantic import Field
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
    SECURITY_DEFAULT_ARMED: bool = False
    SECURITY_ENABLED: bool = True
    SECURITY_DEFAULT_ARMED: bool = False
    SECURITY_MODE: str = "vacation_strict"  # future: family_only, off

    # Cameras of interest (stream_ids); empty = all
    SECURITY_CAMERA_IDS: str = ""  # comma-separated

    # Visit logic
    VISIT_IDLE_TIMEOUT_SEC: int = 5
    MIN_VISIT_DURATION_SEC: int = 2
    MIN_PERSON_FRAMES: int = 2

    # Detection gating
    HUMAN_KINDS: str = "face,yolo,presence"
    MOTION_REQUIRED: bool = True
    HUMAN_MIN_SCORE: float = 0.4
    MIN_BBOX_AREA_FRACTION: float = 0.01  # ignore tiny blobs (<1% frame)

    # Identity / whitelist (v2)
    ALLOWED_IDENTITIES: str = "juniper,wife,kidA,kidB"
    IDENTITY_CONFIDENCE_THRESHOLD: float = 0.6

    # Rate limiting
    SECURITY_GLOBAL_COOLDOWN_SEC: int = 60
    SECURITY_IDENTITY_COOLDOWN_SEC: int = 300

    # Snapshot behavior
    VISION_SNAPSHOT_URL: str = "http://orion-athena-vision-edge:7100/snapshot.jpg"
    SECURITY_SNAPSHOT_COUNT: int = 3
    SECURITY_SNAPSHOT_INTERVAL_SEC: int = 2
    SECURITY_SNAPSHOT_DIR: str = "/mnt/telemetry/orion-security/alerts"

    # Notification mode
    # "inline" = send email directly from this service
    # "none"   = don't send, just publish alerts on bus
    NOTIFY_MODE: str = "inline"

    # Email config (for inline mode)
    NOTIFY_EMAIL_ENABLED: bool = False
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
    def human_kinds(self) -> List[str]:
        return [k.strip() for k in self.HUMAN_KINDS.split(",") if k.strip()]

    @property
    def allowed_identities(self) -> List[str]:
        return [k.strip() for k in self.ALLOWED_IDENTITIES.split(",") if k.strip()]

    @property
    def camera_ids(self) -> List[str]:
        if not self.SECURITY_CAMERA_IDS.strip():
            return []
        return [c.strip() for c in self.SECURITY_CAMERA_IDS.split(",") if c.strip()]


def get_settings() -> Settings:
    return Settings()
