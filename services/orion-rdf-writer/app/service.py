import httpx
import logging
import asyncio
from orion.core.bus.bus_schemas import BaseEnvelope
from app.settings import settings
from app.rdf_builder import build_triples_from_envelope

logger = logging.getLogger(settings.SERVICE_NAME)

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
