from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PortalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ENERGY_PORTAL_BASE_URL: str = Field(default="https://csapps.rockymountainpower.net")
    ENERGY_PORTAL_PROFILE_DIR: str = Field(default="/data/energy/portal/profile")
    ENERGY_PORTAL_STATUS_PATH: str = Field(default="/data/energy/portal/status.json")
    ENERGY_PORTAL_RAW_DIR: str = Field(default="/data/energy/portal/raw")
    ENERGY_INBOX_DIR: str = Field(default="/data/energy/inbox")
    ENERGY_BILL_INBOX_DIR: str = Field(default="/data/energy/bills/inbox")
    ENERGY_PORTAL_INTERVAL_HOURS: float = Field(default=24.0, gt=0)
    # Rolling re-fetch window; late AMI intervals arrive for ~3 days.
    ENERGY_PORTAL_BACKFILL_DAYS: int = Field(default=3, ge=1, le=730)
    ENERGY_PORTAL_TIMEOUT_SEC: float = Field(default=300.0, gt=0)
    # chmod 600 file with RMP_USERNAME= / RMP_PASSWORD=, in a dir bind-mounted read-only into
    # the portal container only; absent means manual reauth only.
    ENERGY_PORTAL_CREDENTIALS_PATH: str = Field(default="/run/secrets/rmp/credentials.env")
    # Billing-history selectors are unverified against the live portal; off = usage XML only.
    ENERGY_PORTAL_SCRAPE_BILLS: bool = Field(default=False)


@lru_cache
def get_portal_settings() -> PortalSettings:
    return PortalSettings()
