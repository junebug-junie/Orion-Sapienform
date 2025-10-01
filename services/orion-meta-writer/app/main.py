import os, json, asyncio
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from uuid import uuid4

from .models import Base, CollapseEnrichmentSQL
from .settings import settings
from orionbus import OrionBus

app = FastAPI(title="Enrichment Writer")
bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)

# --- DB setup ---
engine = create_engine(settings.POSTGRES_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Worker ---
def write_enrichment(db: Session, enrichment: dict):
    entry = CollapseEnrichmentSQL(
        id=f"enrichment_{uuid4().hex}",
        collapse_id=enrichment["id"],
        service_name=enrichment["service_name"],
        service_version=enrichment["service_version"],
        enrichment_type=enrichment.get("enrichment_type", "tagging"),
        tags=json.dumps(enrichment.get("tags", [])),
        entities=json.dumps(enrichment.get("entities", [])),
        salience=enrichment.get("salience"),
    )
    db.add(entry)
    db.commit()
    return entry

def listener_worker():
    sub = bus.subscribe(settings.SUBSCRIBE_CHANNEL)
    for message in sub:
        try:
            enrichment = json.loads(message)
            with SessionLocal() as db:
                row = write_enrichment(db, enrichment)
                print(f"✅ Enrichment {row.id} saved for collapse {row.collapse_id}")
        except Exception as e:
            print("❌ Listener error:", e)

@app.on_event("startup")
async def startup_event():
    asyncio.get_event_loop().run_in_executor(None, listener_worker)

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "subscribe_channel": settings.SUBSCRIBE_CHANNEL,
    }

