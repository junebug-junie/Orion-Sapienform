import asyncio
import hashlib
import httpx
import json
import logging
import time

from orion.core.bus.bus_schemas import BaseEnvelope

from app.rdf_builder import build_triples_from_envelope
from app.settings import settings

logger = logging.getLogger(settings.SERVICE_NAME)

_DEDUP_WINDOW_SEC = 2.0
_DEDUP_MAX_SIZE = 512
_dedupe_cache: dict[str, float] = {}


def _payload_fingerprint(payload: object) -> str:
    dumped = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def _should_dedupe(env: BaseEnvelope) -> bool:
    now = time.monotonic()
    expired = [key for key, ts in _dedupe_cache.items() if now - ts > _DEDUP_WINDOW_SEC]
    for key in expired:
        _dedupe_cache.pop(key, None)

    correlation_id = str(env.correlation_id or "")
    payload_hash = _payload_fingerprint(env.payload)
    key = f"{env.kind}|{correlation_id}|{payload_hash}"
    last_seen = _dedupe_cache.get(key)
    if last_seen is not None and now - last_seen <= _DEDUP_WINDOW_SEC:
        return True

    _dedupe_cache[key] = now
    if len(_dedupe_cache) > _DEDUP_MAX_SIZE:
        for old_key, _ in sorted(_dedupe_cache.items(), key=lambda item: item[1])[: len(_dedupe_cache) - _DEDUP_MAX_SIZE]:
            _dedupe_cache.pop(old_key, None)
    return False


async def _push_to_graphdb(turtle_content: str, graph_name: str = None):
    """
    Pushes NTriples/Turtle to GraphDB.
    """
    # Construct URL. If graph_name is provided, use context param.
    base_url = f"{settings.GRAPHDB_URL}/repositories/{settings.GRAPHDB_REPO}/statements"

    params = {}
    if graph_name:
        params["context"] = f"<{graph_name}>"

    headers = {"Content-Type": "text/plain"} # N-Triples usually text/plain, Turtle text/turtle

    auth = None
    if settings.GRAPHDB_USER and settings.GRAPHDB_PASS:
        auth = (settings.GRAPHDB_USER, settings.GRAPHDB_PASS)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(base_url, content=turtle_content, headers=headers, params=params, auth=auth, timeout=10.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error(f"GraphDB write failed: {e}")
            raise

async def handle_envelope(env: BaseEnvelope) -> None:
    """
    Bus handler: Converts incoming envelopes to RDF and pushes to GraphDB.
    """
    logger.debug(f"Received {env.kind} from {env.source}")
    if _should_dedupe(env):
        logger.info(f"dedupe skip kind={env.kind} correlation_id={env.correlation_id}")
        return

    try:
        # Normalize and build
        content, graph = build_triples_from_envelope(env.kind, env.payload)

        if content:
            await _push_to_graphdb(content, graph)
            logger.info(f"✅ Written RDF for {env.kind} to {graph or 'default'} ({len(content)} bytes)")

            # TODO: If this was an RPC call (has reply_to), send confirmation
            # This requires access to the bus instance which is in the hunter.
            # For now, we just log. The Architecture requires us to honor reply_to.
            # We'll rely on the Hunter/Chassis to potentially handle replies if we return a value,
            # but current chassis implementation requires manual publish for now.

        else:
            logger.debug(f"No triples generated for {env.kind}")

    except Exception as e:
        logger.error(f"Failed to process RDF for {env.kind}: {e}", exc_info=True)
