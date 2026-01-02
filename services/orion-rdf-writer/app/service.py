import httpx
import logging
import asyncio
from app.settings import settings
from app.rdf_builder import build_triples
from orion.core.bus.bus_schemas import BaseEnvelope

logger = logging.getLogger(settings.SERVICE_NAME)

async def _push_to_graphdb(turtle_content: str):
    """
    Pushes Turtle-formatted RDF triples to GraphDB.
    """
    url = f"{settings.graphdb_url}/repositories/{settings.graphdb_repo}/statements"
    headers = {"Content-Type": "text/turtle"}

    # Optional: Add authentication if GraphDB requires it
    # auth = (settings.graphdb_user, settings.graphdb_pass) if settings.graphdb_user else None

    async with httpx.AsyncClient() as client:
        # resp = await client.post(url, content=turtle_content, headers=headers, auth=auth)
        resp = await client.post(url, content=turtle_content, headers=headers)
        resp.raise_for_status()

async def handle_envelope(env: BaseEnvelope) -> None:
    """
    Bus handler: Converts incoming envelopes to RDF and pushes to GraphDB.
    """
    logger.debug(f"Received {env.kind} for RDF processing")
    try:
        turtle = build_triples(env)
        if turtle:
            await _push_to_graphdb(turtle)
            logger.info(f"Written RDF for {env.kind} ({len(turtle)} bytes)")
        else:
            logger.debug(f"No triples generated for {env.kind}")
    except Exception as e:
        logger.error(f"Failed to process RDF for {env.kind}: {e}")
