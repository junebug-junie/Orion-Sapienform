from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERVICE_NAME: str = Field("notify", alias="SERVICE_NAME")
    SERVICE_VERSION: str = Field("0.1.0", alias="SERVICE_VERSION")
    NODE_NAME: str = Field("unknown", alias="NODE_NAME")

    API_TOKEN: Optional[str] = Field(None, alias="API_TOKEN")

    ORION_BUS_ENABLED: bool = Field(True, alias="ORION_BUS_ENABLED")
    ORION_BUS_ENFORCE_CATALOG: bool = Field(False, alias="ORION_BUS_ENFORCE_CATALOG")
    ORION_BUS_URL: str = Field("redis://100.92.216.81:6379/0", alias="ORION_BUS_URL")

    NOTIFY_IN_APP_ENABLED: bool = Field(True, alias="NOTIFY_IN_APP_ENABLED")
    NOTIFY_IN_APP_CHANNEL: str = Field("orion:notify:in_app", alias="NOTIFY_IN_APP_CHANNEL")

    SQL_WRITER_API_URL: str = Field("http://orion-sql-writer:8220", alias="SQL_WRITER_API_URL")
    POLICY_RULES_PATH: str = Field("/app/app/policy/rules.yaml", alias="POLICY_RULES_PATH")

    NOTIFY_EMAIL_SMTP_HOST: str = Field("", alias="NOTIFY_EMAIL_SMTP_HOST")
    NOTIFY_EMAIL_SMTP_PORT: int = Field(587, alias="NOTIFY_EMAIL_SMTP_PORT")
    NOTIFY_EMAIL_SMTP_USERNAME: str = Field("", alias="NOTIFY_EMAIL_SMTP_USERNAME")
    NOTIFY_EMAIL_SMTP_PASSWORD: str = Field("", alias="NOTIFY_EMAIL_SMTP_PASSWORD")
    NOTIFY_EMAIL_USE_TLS: bool = Field(True, alias="NOTIFY_EMAIL_USE_TLS")
    NOTIFY_EMAIL_FROM: str = Field("orion-notify@example.com", alias="NOTIFY_EMAIL_FROM")
    NOTIFY_EMAIL_TO: str = Field("", alias="NOTIFY_EMAIL_TO")
    NOTIFY_ESCALATION_POLL_SECONDS: int = Field(60, alias="NOTIFY_ESCALATION_POLL_SECONDS")
    NOTIFY_PRESENCE_URL: str = Field("", alias="NOTIFY_PRESENCE_URL")
    NOTIFY_PRESENCE_TIMEOUT_SECONDS: int = Field(3, alias="NOTIFY_PRESENCE_TIMEOUT_SECONDS")
    NOTIFY_HUB_URL: str = Field("", alias="NOTIFY_HUB_URL")

    @property
    def notify_email_to(self) -> List[str]:
        raw = self.NOTIFY_EMAIL_TO or ""
        return [e.strip() for e in raw.split(",") if e.strip()]


settings = Settings()
