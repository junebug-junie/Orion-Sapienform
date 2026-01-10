import asyncio
import logging
import traceback
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
import spacy

# 1. Use the existing Chassis components
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter, Rabbit
from .settings import settings
from .models import EventIn, Enrichment

# Setup Logging
logging.basicConfig(level=logging.getLevelName(settings.LOG_LEVEL))
logger = logging.getLogger(settings.SERVICE_NAME)

# Initialize NLP
try:
    nlp = spacy.load(settings.SPA_MODEL)
except OSError:
    from spacy.cli import download
    logger.info(f"Downloading spaCy model {settings.SPA_MODEL}...")
    download(settings.SPA_MODEL)
    nlp = spacy.load(settings.SPA_MODEL)

# Global Chassis Reference
meta_tagger: Hunter = None
meta_tagger_rpc: Rabbit = None


async def handle_meta_tags_rpc(env: BaseEnvelope) -> BaseEnvelope:
    raw = env.payload if isinstance(env.payload, dict) else {}
    # reuse EventIn to normalize text
    in_payload = EventIn(**raw)

    doc = nlp(in_payload.text or "")
    tags = [ent.text for ent in doc.ents]
    entities = [ent.label_ for ent in doc.ents]

    out = {
        "id": in_payload.id,
        "tags": tags,
        "entities": entities,
    }

    return BaseEnvelope(
        kind="meta_tags.result.v1",
        source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
        correlation_id=env.correlation_id,
        payload=out,
    )


async def handle_triage_event(envelope: BaseEnvelope):
    """
    Handler for 'orion:collapse:triage'.
    Uses EventIn to normalize text, enriches with NLP, and publishes tags.
    """
    # Use global chassis reference to publish results
    global meta_tagger
    
    try:
        # 1. VALIDATION & TEXT EXTRACTION
        # EventIn validator 'prepare_and_hydrate_text' automatically 
        # finds the best text source (prompt/response dialogue or summary)
        in_payload = EventIn(**envelope.payload)
        
        logger.info(f"📨 Processing {in_payload.id} (Text len: {len(in_payload.text)})")

        # 2. NLP PROCESSING
        doc = nlp(in_payload.text)
        
        # Extract Entities (Tags)
        tags = [ent.text for ent in doc.ents]
        
        # Basic Sentiment Logic (Heuristic)
        # We append sentiment as a tag since Enrichment model uses a list of strings
        sentiment_tag = "sentiment:neutral"
        positive_keywords = {"triumphant", "relief", "capable", "good", "success"}
        negative_keywords = {"anxious", "fear", "fail", "bad", "panic"}
        
        tokens = set(in_payload.text.lower().split())
        if tokens & positive_keywords:
            sentiment_tag = "sentiment:positive"
        elif tokens & negative_keywords:
            sentiment_tag = "sentiment:negative"
        
        tags.append(sentiment_tag)

        # 3. RESULT GENERATION (Enrichment Model)
        # [FIX] Logic to ensure collapse_id is not None
        # If the input event IS the collapse, then in_payload.id is the collapse_id.
        # This prevents the NotNullViolation in the SQL writer.
        target_collapse_id = in_payload.collapse_id or in_payload.id

        if not target_collapse_id:
             logger.warning(f"⚠️  Missing collapse_id for event {in_payload.id}, sql-writer may fail.")

        enrichment = Enrichment(
            id=in_payload.id,
            collapse_id=target_collapse_id,
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            enrichment_type="tagging", # [FIX] Explicitly set type to satisfy NOT NULL constraints
            tags=tags,
            entities=[], # entities could be mapped here if needed
            salience=0.0,
            ts=datetime.now(timezone.utc).isoformat(),
            
            # [FIX] Populate Lineage Fields required by Titanium Contract / MetaTagsPayload
            node=settings.NODE_NAME,
            correlation_id=str(envelope.correlation_id) if envelope.correlation_id else str(in_payload.id),
            source_message_id=str(envelope.id) # Link back to the trigger message
        )

        # 4. PUBLISH (Enriched)
        # Wrap in standard envelope
        out_env = envelope.derive_child(
            kind="tags.enriched",
            source=ServiceRef(
                name=settings.SERVICE_NAME, 
                version=settings.SERVICE_VERSION, 
                node=settings.NODE_NAME
            ),
            payload=enrichment # Validated, typed object
        )

        await meta_tagger.bus.publish(settings.CHANNEL_EVENTS_TAGGED, out_env)
        logger.info(f"✅ Published tags for {in_payload.id} -> {settings.CHANNEL_EVENTS_TAGGED}")

    except Exception as e:
        logger.error(f"❌ Error processing event: {e}")
        # Hunter chassis will also catch exceptions and emit system.error, 
        # but we log here for immediate visibility.
        traceback.print_exc()

# --- Service Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_tagger
    
    # Configure Chassis
    config = ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
    )
    
    # Initialize Hunter (Subscribe pattern)
    meta_tagger = Hunter(
        cfg=config,
        handler=handle_triage_event,
        pattern=settings.CHANNEL_EVENTS_TRIAGE
    )

    meta_tagger_rpc = Rabbit(
        config,
        request_channel="orion:exec:request:MetaTagsService",
        handler=handle_meta_tags_rpc,
    )
    await meta_tagger_rpc.start_background()
    await meta_tagger.start_background()
    logger.info("Tagger starting...")

    yield

    logger.info("Shutting down...")
    await meta_tagger.stop()
    await meta_tagger_rpc.stop()

app = FastAPI(
    title="Orion Meta-Tags",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

@app.get("/health")
def health():
    status = "disconnected"
    if meta_tagger and meta_tagger.bus and meta_tagger.bus.redis:
        status = "connected"
    return {"status": "ok", "bus": status, "model": settings.SPA_MODEL}
