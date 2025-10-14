import asyncio
import json
import traceback
from datetime import datetime
from fastapi import FastAPI
import spacy

from orion.core.bus.service import OrionBus
from .settings import settings
from .models import EventIn, Enrichment


# ───────────────────────────────────────────────────────────────
# 🚀 FastAPI App Initialization
# ───────────────────────────────────────────────────────────────
app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# 📦 Load SpaCy model
try:
    nlp = spacy.load(settings.SPA_MODEL)
except Exception as e:
    print(f"⚠️  Failed to load {settings.SPA_MODEL}, falling back to en_core_web_sm ({e})")
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
    )


# ───────────────────────────────────────────────────────────────
# 🌐 API Endpoints
# ───────────────────────────────────────────────────────────────
@app.post("/tag")
def tag_event(event: EventIn):
    """Manual API entrypoint for testing."""
    # Create a temporary bus instance for this single request
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    if not bus.enabled:
        return {"error": "Bus is not connected or enabled"}
    tagged = process_text(event)
    bus.publish(settings.CHANNEL_EVENTS_TAGGED, tagged.model_dump())
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


# ───────────────────────────────────────────────────────────────
# 🔁 Background Listener
# ───────────────────────────────────────────────────────────────
def listener_worker():
    """
    Creates its own thread-local bus connection and enters a blocking
    listen loop. This is a thread-safe pattern.
    """
    # Create the OrionBus instance INSIDE the thread.
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    if not bus.enabled:
        print("❌ Listener could not connect to bus. Thread exiting.")
        return

    listen_channel = settings.CHANNEL_EVENTS_TRIAGE
    publish_channel = settings.CHANNEL_EVENTS_TAGGED

    print(f"👂 Listening on {listen_channel}")
    print(f"📨 Publishing tagged events → {publish_channel}")

    try:
        # Iterate directly over the generator from bus.subscribe()
        for data in bus.subscribe(listen_channel):
            print(f"Received message on {listen_channel}")
            try:
                ev = EventIn.model_validate(data)
                tagged = process_text(ev)
                bus.publish(publish_channel, tagged.model_dump())
                print(f"✅ Published enrichment for {ev.id}")
            except Exception as e:
                print(f"❌ Listener error while processing message: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"💥 Listener fatal error: {e}")
        traceback.print_exc()


# ───────────────────────────────────────────────────────────────
# 🧭 Startup Hook
# ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    delay = settings.STARTUP_DELAY
    print(f"⏳ Waiting for {delay} seconds before starting listener...")
    await asyncio.sleep(delay)
    
    if settings.ORION_BUS_ENABLED:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, listener_worker)
        print(f"✅ {settings.SERVICE_NAME} listener started in background")
    else:
        print("⚠️ Bus is disabled, listener will not be started.")

