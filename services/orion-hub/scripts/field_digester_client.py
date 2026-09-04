"""Hub client for orion-field-digester's `/health` endpoint.

Only one call today: relaying the `field_channel_anomaly` status block
(encoder version, live-enrichment field coverage -- see app/anomaly_scorer.py
::status() in that service) for the mood-arc-status operator page. Modeled on
`biometrics_node_client.py`'s aiohttp pattern: one controlled exception type,
no raw `aiohttp.ClientError` ever escapes to a route handler.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import aiohttp

from scripts.settings import settings

logger = logging.getLogger("orion-hub.field-digester-client")


class FieldDigesterClientError(Exception):
    """Controlled field-digester client failure (unreachable, timeout, HTTP error)."""


def _timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=float(settings.FIELD_DIGESTER_CLIENT_TIMEOUT_SEC))


async def fetch_health() -> Dict[str, Any]:
    """`GET /health` on orion-field-digester's own instance.

    Never raises for "the service is down" -- that is a real, expected
    operational state for a status page to show, not an exception a route
    handler needs to catch specially. Raises FieldDigesterClientError only
    for a genuinely malformed response (non-JSON, non-object), which would
    indicate the endpoint's own contract changed underneath this client.
    """
    base = str(settings.FIELD_DIGESTER_BASE_URL or "").strip().rstrip("/")
    if not base:
        raise FieldDigesterClientError("no FIELD_DIGESTER_BASE_URL configured")
    url = f"{base}/health"
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with session.get(url) as response:
                raw = await response.json()
                if response.status >= 400:
                    raise FieldDigesterClientError(f"GET {url} HTTP {response.status}: {raw!r}"[:400])
    except aiohttp.ClientError as exc:
        logger.warning("field-digester unreachable: %s", url)
        raise FieldDigesterClientError(f"{url} unreachable") from exc
    except ValueError as exc:
        # response.json() raises json.JSONDecodeError (a ValueError) for a
        # malformed/truncated body even with a correct content-type header --
        # review finding (2026-09-04): this was previously uncaught, turning
        # a crashed/truncated orion-field-digester response into an
        # unhandled 500 on the Hub route instead of the graceful degrade
        # this function's own docstring promises.
        logger.warning("field-digester returned malformed JSON: %s", url)
        raise FieldDigesterClientError(f"{url} returned malformed JSON") from exc
    if not isinstance(raw, dict):
        raise FieldDigesterClientError(f"GET {url} returned non-object payload")
    return raw
