import logging
import uuid
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.settings import settings
from app.db import Base, engine, get_db
from app.models import BiometricRecord
from orion.core.bus.service import OrionBus

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# Global bus object, initialized on startup
bus: OrionBus | None = None

@app.on_event("startup")
def startup_event():
    """
    Initializes database schema and connects to the Orion Bus.
    """
    global bus
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")

    try:
        # This will create the table defined in models.py if it doesn't exist
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database schema verified.")
    except Exception as e:
        logger.critical(f"🚨 Failed to connect to or initialize database: {e}", exc_info=True)
    
    if settings.ORION_BUS_ENABLED:
        logger.info(f"Connecting to OrionBus at {settings.ORION_BUS_URL}")
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    else:
        logger.warning("OrionBus is disabled.")


@app.get("/health")
def health():
    """Provides a simple health check endpoint."""
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": settings.ORION_BUS_ENABLED,
    }

# Pydantic model for creating new records via the API
class BiometricRecordCreate(BaseModel):
    user_id: str
    record_type: str
    data_hash: str

# --- API Endpoints ---

@app.post("/records", status_code=201)
def create_biometric_record(record: BiometricRecordCreate, db: Session = Depends(get_db)):
    """
    Creates a new biometric record, saves it to the database, and publishes
    a telemetry event to the Orion Bus.
    """
    record_id = f"bio_{uuid.uuid4().hex}"
    db_record = BiometricRecord(
        id=record_id,
        **record.model_dump()
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
    logger.info(f"✅ Created biometric record {record_id} for user {record.user_id}")
    
    # Publish telemetry event to the bus
    if bus and bus.enabled:
        payload = {
            "id": record_id,
            "user_id": record.user_id,
            "record_type": record.record_type,
            "status": "created"
        }
        bus.publish(settings.PUBLISH_CHANNEL_BIOMETRICS_NEW, payload)
        logger.info(f"📡 Published event to {settings.PUBLISH_CHANNEL_BIOMETRICS_NEW}")
        
    return db_record

@app.get("/records/{user_id}")
def get_user_records(user_id: str, db: Session = Depends(get_db)):
    """
    Fetches all biometric records for a given user ID.
    """
    records = db.query(BiometricRecord).filter(BiometricRecord.user_id == user_id).all()
    return records
