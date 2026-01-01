import httpx
import logging
import asyncio
from app.settings import settings
from app.rdf_builder import build_triples
from orion.core.bus.consumer_worker import run_worker
from orion.core.bus.bus_schemas import BaseEnvelope

logger = logging.getLogger(settings.SERVICE_NAME)

async def _push_to_graphdb(nt_data: str, graph_name: str, envelope_id: str):
    """
    Pushes N-Triples data to GraphDB with a retry mechanism.
    """
    if not nt_data or not graph_name:
        logger.warning(f"Skipping push for event {envelope_id} due to empty RDF data.")
        return

    url = f"{settings.GRAPHDB_URL}/repositories/{settings.GRAPHDB_REPO}/statements?context=<{graph_name}>"
    headers = {"Content-Type": "application/n-triples"}

    for attempt in range(settings.RETRY_LIMIT):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.post(url, content=nt_data, headers=headers)
                res.raise_for_status()
                logger.info(f"✅ RDF inserted ({envelope_id}) → {graph_name}")
                return
        except httpx.HTTPStatusError as e:
            logger.warning(f"⚠️ Insert failed (attempt {attempt + 1}/{settings.RETRY_LIMIT}): {e.response.status_code} {e.response.text}")
        except Exception as e:
            logger.error(f"❌ GraphDB connection error (attempt {attempt + 1}/{settings.RETRY_LIMIT}): {e}")

        await asyncio.sleep(settings.RETRY_INTERVAL)

    logger.error(f"🚨 Failed to push event {envelope_id} after {settings.RETRY_LIMIT} attempts.")

async def message_handler(envelope: BaseEnvelope):
    """
    Handles incoming bus messages.
    """
    logger.debug(f"Processing envelope kind={envelope.kind}, id={envelope.id}")

    # 1. Check if we have a specific routing rule for this kind
    # If using explicit map:
    route_map = settings.route_map
    # If the kind is not in the map, and we want strict routing, we might skip.
    # However, existing logic seemed to try to build triples for anything.
    # We'll allow it to try build_triples.

    # Optional: Override graph context based on route_map
    # graph_context = route_map.get(envelope.kind)
    # The current build_triples logic generates graph name dynamically based on observer/provenance.
    # We might want to pass the suggestion down?
    # For now, let's keep existing logic but allow build_triples to decide.

    try:
        triples, graph_name = build_triples(envelope)

        if triples:
            # If route_map overrides the context:
            # if envelope.kind in route_map:
            #     # This is tricky because graph_name is a full URI in build_triples.
            #     # Let's trust build_triples for now as it handles provenance.
            #     pass

            await _push_to_graphdb(triples, graph_name, str(envelope.id))
        else:
            logger.debug(f"Ignored {envelope.kind} (no triples generated)")

    except Exception as e:
        logger.exception(f"Error processing {envelope.kind}: {e}")


def listener_worker():
    """
    Entrypoint for the worker thread/process.
    """
    if not settings.ORION_BUS_ENABLED:
        logger.warning("Bus disabled by config.")
        return

    # run_worker is async, so we need to run it in an event loop.
    # But wait, this is called from a thread in main.py.
    asyncio.run(run_worker(
        service_name=settings.SERVICE_NAME,
        bus_url=settings.ORION_BUS_URL,
        channels=settings.RDF_WRITER_SUBSCRIBE_CHANNELS,
        handler=message_handler
    ))
