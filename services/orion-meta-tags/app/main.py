import asyncio
import json
import traceback
import logging
from datetime import datetime
from fastapi import FastAPI
from contextlib import asynccontextmanager
import spacy

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef
from orion.schemas.telemetry.meta_tags import MetaTagsPayload
from .settings import settings
from .models import EventIn, Enrichment

# ───────────────────────────────────────────────────────────────
# 🔧 Setup Logger
# ───────────────────────────────────────────────────────────────
logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(settings.SERVICE_NAME)

# 📦 Load SpaCy model
try:
    nlp = spacy.load(settings.SPA_MODEL)
except Exception as e:
    logger.warning(f"⚠️  Failed to load {settings.SPA_MODEL}, falling back to en_core_web_sm ({e})")
    nlp = spacy.load("en_core_web_sm")


# ───────────────────────────────────────────────────────────────
# 🧠 Core Processing Logic
# ───────────────────────────────────────────────────────────────
def process_text(event: EventIn) -> Enrichment:
    doc = nlp(event.text)

    entities = [{"type": ent.label_, "value": ent.text} for ent in doc.ents]
    tags = list(
        set([t.lemma_ for t in doc if t.pos_ in ("NOUN", "VERB") and not t.is_stop])
    )
    salience = min(1.0, (len(tags) + len(entities)) / 10)

    return Enrichment(
        id=event.id,
        collapse_id=event.collapse_id,
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        enrichment_type="tagging",
        tags=tags,
        entities=entities,
        salience=salience,
        ts=datetime.utcnow().isoformat()
    )

def chassis_cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
        health_channel=settings.HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
        shutdown_timeout_sec=settings.SHUTDOWN_GRACE_SEC,
    )

# ───────────────────────────────────────────────────────────────
# 🔁 Tagging Hunter (Subclass)
# ───────────────────────────────────────────────────────────────
class TaggingHunter(Hunter):
    """
    Specialized Hunter that consumes triage events and publishes enrichments
    using the same bus connection.
    """
    def __init__(self, cfg: ChassisConfig, *, pattern: str):
        # We pass self.handle_event as the handler to the parent Hunter
        super().__init__(cfg, pattern=pattern, handler=self.handle_event)

    async def handle_event(self, env: BaseEnvelope):
        logger.info(f"Received triage event from {env.source}")
        try:
            # Check if we should process this kind.
            # We assume the subscription pattern matches, but we can double check kind if needed.

            try:
                ev = EventIn.model_validate(env.payload)
            except Exception as e:
                # This might happen if we subscribe to a wildcard and get unrelated messages
                logger.debug(f"Skipping incompatible payload: {e}")
                return

            tagged = process_text(ev)

            # Convert to MetaTagsPayload for canonical typed publishing
            out_payload = MetaTagsPayload(
                id=tagged.id,
                collapse_id=tagged.collapse_id,
                service_name=tagged.service_name,
                service_version=tagged.service_version,
                enrichment_type=tagged.enrichment_type,
                tags=tagged.tags,
                entities=tagged.entities,
                salience=tagged.salience,
                ts=tagged.ts
            )

            # Create response envelope
            out_env = env.derive_child(
                kind="telemetry.meta_tags",
                source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
                payload=out_payload.model_dump(mode="json")
            )

            # Use the chassis's existing bus connection
            await self.bus.publish(settings.CHANNEL_EVENTS_TAGGED, out_env)
            logger.info(f"✅ Published enrichment for {ev.id}")

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            traceback.print_exc()

# ───────────────────────────────────────────────────────────────
# compass Lifespan & App
# ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")

    if settings.STARTUP_DELAY > 0:
        await asyncio.sleep(settings.STARTUP_DELAY)

    # Initialize our specialized Hunter
    hunter = TaggingHunter(
        chassis_cfg(),
        pattern=settings.CHANNEL_EVENTS_TRIAGE
    )
    await hunter.start_background()

    # Store reference to prevent garbage collection if needed (though start_background stores tasks)
    # and to stop it later.
    app.state.hunter = hunter

    yield

    await hunter.stop()
    logger.info("🛑 Shutting down listener")

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, lifespan=lifespan)

@app.post("/tag")
async def tag_event(event: EventIn):
    """Manual API entrypoint for testing."""
    tagged = process_text(event)

    # For manual API, we might still need a transient connection or use the hunter's bus if running
    # Using app.state.hunter if available is cleaner.

    if hasattr(app.state, "hunter") and app.state.hunter._started:
        hunter: TaggingHunter = app.state.hunter

        out_payload = MetaTagsPayload(
            id=tagged.id,
            collapse_id=tagged.collapse_id,
            service_name=tagged.service_name,
            service_version=tagged.service_version,
            enrichment_type=tagged.enrichment_type,
            tags=tagged.tags,
            entities=tagged.entities,
            salience=tagged.salience,
            ts=tagged.ts
        )
        env = BaseEnvelope(
            kind="telemetry.meta_tags",
            source=ServiceRef(name=settings.SERVICE_NAME, version=settings.SERVICE_VERSION, node=settings.NODE_NAME),
            payload=out_payload.model_dump(mode="json")
        )
        await hunter.bus.publish(settings.CHANNEL_EVENTS_TAGGED, env)

    return tagged


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "listen_channel": settings.CHANNEL_EVENTS_TRIAGE,
        "publish_channel": settings.CHANNEL_EVENTS_TAGGED,
        "bus_enabled": settings.ORION_BUS_ENABLED,
    }
