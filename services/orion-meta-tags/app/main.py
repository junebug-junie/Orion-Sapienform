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
from .settings import settings
# Use MetaTagsPayload from shared schema (aliased as Enrichment in models.py if kept, or import directly)
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
        # Note: Enrichment is now an alias for MetaTagsPayload
        enrichment = Enrichment(
            id=in_payload.id,
            collapse_id=in_payload.collapse_id,
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            # enrichment_type defaults to "tagging" in the shared schema
            tags=tags,
            # entities could be mapped here if needed
            ts=datetime.now(timezone.utc).isoformat()
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
            payload=enrichment
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

    logger.info("Starting Meta-Tags Service...")
    await meta_tagger.start_background()
    
    yield
    
    logger.info("Shutting down...")
    await meta_tagger.stop()

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
