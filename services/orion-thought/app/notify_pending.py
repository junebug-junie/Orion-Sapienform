"""Which of this service's attention pages are still open in orion-notify.

Shared by the edge-triggered health monitors (resonance, reverie metacog,
visual chain). A page and its recovery note share one `reason` key and both
require ack, so "any pending item with this reason" is not "still open".
Live 2026-11-03: that reading made the resonance monitor re-bootstrap
recovered themes on every restart and send another recovery note (273 of
311 unacked items). A reason is open only if its NEWEST pending item is not
a recovery note. sql-writer returns the list newest first.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger("orion-thought.notify_pending")

SOURCE_SERVICE = "orion-thought"
RECOVERED_PREFIX = "[Orion reverie] recovered:"


def open_alert_reasons(settings_obj, *, log_label: str) -> list[str]:
    """Reasons whose newest pending orion-thought item is still a page.

    Fail-open: [] on any lookup failure, so a caller prefers a possibly
    duplicate page over silently missing a real incident."""
    headers = {}
    if settings_obj.notify_api_token:
        headers["X-Orion-Notify-Token"] = settings_obj.notify_api_token
    try:
        response = requests.get(
            f"{settings_obj.notify_base_url}/attention",
            params={"status": "pending", "limit": 200},
            headers=headers,
            timeout=10,
        )
        response.raise_for_status()
        items = response.json()
    except Exception:
        logger.exception("%s_pending_lookup_failed", log_label)
        return []
    if not isinstance(items, list):
        return []
    newest_is_open: dict[str, bool] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("source_service") != SOURCE_SERVICE:
            continue
        reason = item.get("reason")
        if not reason or str(reason) in newest_is_open:
            continue
        message = str(item.get("message") or "")
        newest_is_open[str(reason)] = not message.startswith(RECOVERED_PREFIX)
    return [reason for reason, is_open in newest_is_open.items() if is_open]
