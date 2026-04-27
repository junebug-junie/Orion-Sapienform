from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from xml.etree import ElementTree

import requests

from orion.schemas.world_pulse import WorldPulseSourceV1


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def fetch_rss_articles(source: WorldPulseSourceV1, timeout_seconds: int) -> list[dict[str, Any]]:
    if not source.url:
        return []
    resp = requests.get(source.url, timeout=timeout_seconds)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.text)
    out: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        out.append(
            {
                "title": title,
                "url": link,
                "published_at": _parse_date(item.findtext("pubDate")),
                "author": (item.findtext("author") or "").strip() or None,
                "summary": (item.findtext("description") or "").strip() or None,
            }
        )
    return out
