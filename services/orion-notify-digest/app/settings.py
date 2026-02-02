from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field("notify-digest", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field("0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field("unknown", alias="NODE_NAME")

    POSTGRES_URI: str = Field("sqlite:////data/notify.db", alias="POSTGRES_URI")

    NOTIFY_SERVICE_URL: str = Field("http://orion-notify:7140", alias="NOTIFY_SERVICE_URL")
    NOTIFY_API_TOKEN: Optional[str] = Field(None, alias="NOTIFY_API_TOKEN")

    DIGEST_ENABLED: bool = Field(True, alias="DIGEST_ENABLED")
    DIGEST_TIME_LOCAL: str = Field("07:30", alias="DIGEST_TIME_LOCAL")
    DIGEST_WINDOW_HOURS: int = Field(24, alias="DIGEST_WINDOW_HOURS")
    DIGEST_RECIPIENT_GROUP: str = Field("juniper_primary", alias="DIGEST_RECIPIENT_GROUP")
    DIGEST_RUN_ON_START: bool = Field(False, alias="DIGEST_RUN_ON_START")

    LANDING_PAD_URL: Optional[str] = Field(None, alias="LANDING_PAD_URL")
    TOPICS_WINDOW_MINUTES: Optional[int] = Field(None, alias="TOPICS_WINDOW_MINUTES")
    TOPICS_MAX_TOPICS: int = Field(20, alias="TOPICS_MAX_TOPICS")
    TOPICS_DRIFT_MIN_TURNS: int = Field(10, alias="TOPICS_DRIFT_MIN_TURNS")
    TOPICS_DRIFT_MAX_SESSIONS: int = Field(50, alias="TOPICS_DRIFT_MAX_SESSIONS")

    DRIFT_ALERTS_ENABLED: bool = Field(False, alias="DRIFT_ALERTS_ENABLED")
    DRIFT_CHECK_INTERVAL_SECONDS: int = Field(900, alias="DRIFT_CHECK_INTERVAL_SECONDS")
    DRIFT_ALERT_THRESHOLD: float = Field(0.5, alias="DRIFT_ALERT_THRESHOLD")
    DRIFT_ALERT_MAX_ITEMS: int = Field(5, alias="DRIFT_ALERT_MAX_ITEMS")
    DRIFT_ALERT_SEVERITY: str = Field("warning", alias="DRIFT_ALERT_SEVERITY")
    DRIFT_ALERT_EVENT_KIND: str = Field("orion.topics.drift", alias="DRIFT_ALERT_EVENT_KIND")
    DRIFT_ALERT_DEDUPE_WINDOW_SECONDS: int = Field(3600, alias="DRIFT_ALERT_DEDUPE_WINDOW_SECONDS")


settings = Settings()
