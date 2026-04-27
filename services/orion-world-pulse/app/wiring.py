from __future__ import annotations

from functools import lru_cache

from orion.notify.client import NotifyClient

from app.settings import settings


@lru_cache()
def notify_client() -> NotifyClient:
    return NotifyClient(base_url=settings.notify_url, api_token=settings.notify_api_token, timeout=12)
