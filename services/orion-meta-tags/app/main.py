import asyncio, json
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import spacy

from orionbus import OrionBus
from .settings import settings
from .models import EventIn, Enrichment

# 🚀 FastAPI app
app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)
print('hi')

# 🚌 Bus
bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)

# 📦 Load upgraded SpaCy pipeline (transformer-based)
try:
    nlp = spacy.load(settings.SPA_MODEL)
except Exception:
    nlp = spacy.load("en_core_web_sm")  # fallback

class TaggedEvent(BaseModel):
    id: str
    tags: List[str]
    entities: List[Dict[str, str]]
    salience: float
    ts: str

def process_text(event: EventIn) -> Enrichment:
    doc = nlp(event.text)
    entities = [{"type": ent.label_, "value": ent.text} for ent in doc.ents]
    tags = list(set([t.lemma_ for t in doc if t.pos_ in ("NOUN", "VERB") and not t.is_stop]))
    salience = min(1.0, (len(tags) + len(entities)) / 10)

    return Enrichment(
        id=event.id,
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        enrichment_type="tagging",
        tags=tags,
        entities=entities,
        salience=salience
    )

@app.post("/tag", response_model=Enrichment)
def tag_event(event: EventIn):
    tagged = process_text(event)
    bus.publish(settings.PUBLISH_CHANNEL, tagged.dict())
    return tagged

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "subscribe_channel": settings.SUBSCRIBE_CHANNEL,
        "publish_channel": settings.PUBLISH_CHANNEL,
    }

# Listener
def listener_worker():
    sub = bus.subscribe(settings.SUBSCRIBE_CHANNEL)
    for message in sub:
        try:
            print("📥 Raw message from bus:", message)
            # If bus already gives dicts, no need for json.loads
            ev = EventIn(**message)
            tagged = process_text(ev)
            bus.publish(settings.PUBLISH_CHANNEL, tagged.dict())
            print("✅ Published enrichment for", ev.id)
        except Exception as e:
            print("❌ Listener error:", e)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, listener_worker)  # runs in a background thread
    print("✅ Listener running in background, startup complete")
