"""Hub client for per-node `orion-biometrics` HTTP APIs (snapshot / raw/recent).

Only two nodes are live: athena (local to Hub's own host, `127.0.0.1:8100`) and
circe (cross-host, `settings.CIRCE_BIOMETRICS_BASE_URL`). A third node, `atlas`,
was decommissioned 2026-08-20 and is deliberately rejected here rather than
silently resolved to nothing or forwarded as a doomed HTTP call — "absent is
not zero" applies to node identity too, not just to a missing reading.

Modeled on `context_exec_client.py`'s aiohttp pattern: one controlled
exception type, no raw `aiohttp.ClientError` ever escapes to a route handler.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import aiohttp

from scripts.settings import settings

logger = logging.getLogger("orion-hub.biometrics-node-client")

#: The only two nodes this client will ever address. `atlas` is intentionally
#: absent — decommissioned 2026-08-20, per project_node_liveness memory and
#: node_catalog.yaml. Adding a node here means it has a live, reachable
#: orion-biometrics instance; do not add it speculatively.
LIVE_NODES: tuple[str, ...] = ("athena", "circe")

ATHENA_BASE_URL = "http://127.0.0.1:8100"


class BiometricsNodeClientError(Exception):
    """Controlled biometrics-node client failure (unknown node, timeout, HTTP error)."""


def _base_url(node: str) -> str:
    if node == "athena":
        return ATHENA_BASE_URL
    if node == "circe":
        return str(settings.CIRCE_BIOMETRICS_BASE_URL or "").strip().rstrip("/")
    raise BiometricsNodeClientError(
        f"unknown or decommissioned node {node!r}; only {LIVE_NODES} are live"
    )


def _timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=float(settings.BIOMETRICS_NODE_CLIENT_TIMEOUT_SEC))


async def _get_json(url: str) -> Dict[str, Any]:
    try:
        async with aiohttp.ClientSession(timeout=_timeout()) as session:
            async with session.get(url) as response:
                raw = await response.json()
                if response.status >= 400:
                    raise BiometricsNodeClientError(
                        f"GET {url} HTTP {response.status}: {raw!r}"[:400]
                    )
    except aiohttp.ClientError as exc:
        logger.warning("biometrics node unreachable: %s", url)
        raise BiometricsNodeClientError(f"{url} unreachable") from exc
    if not isinstance(raw, dict):
        raise BiometricsNodeClientError(f"GET {url} returned non-object payload")
    return raw


async def fetch_snapshot(node: str) -> Dict[str, Any]:
    """`GET /snapshot` on the given node's own orion-biometrics instance."""
    base = _base_url(node)
    if not base:
        raise BiometricsNodeClientError(f"no base URL configured for node {node!r}")
    return await _get_json(f"{base}/snapshot")


async def fetch_raw_recent(node: str, *, limit: int = 10) -> Dict[str, Any]:
    """`GET /raw/recent?node=&limit=` on the given node's own orion-biometrics instance.

    Each node's own API already filters by `node`, so this passes it through
    for symmetry/defense-in-depth rather than relying solely on which host
    answered the request.
    """
    base = _base_url(node)
    if not base:
        raise BiometricsNodeClientError(f"no base URL configured for node {node!r}")
    return await _get_json(f"{base}/raw/recent?node={node}&limit={int(limit)}")
