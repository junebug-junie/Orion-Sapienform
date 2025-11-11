import logging
from fastapi import FastAPI

from app.settings import settings
from app.db import Base, engine
from app.worker import start_listeners

logging.basicConfig(
    level=logging.INFO,
    format="[SQL_WRITER] %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

@app.on_event("startup")
def startup_event():
    # Ensure schema exists
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("🛠️  Ensured DB schema is present")
    except Exception as e:
        logger.warning(f"Schema init warning: {e}")

    # Start bus listeners
    if not settings.ORION_BUS_ENABLED:
        logger.warning("Bus disabled; writer will be idle.")
        return

    logger.info(f"🚀 {settings.SERVICE_NAME} starting listeners")
    start_listeners()

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "channels": settings.get_all_subscribe_channels(),
        "bus_url": settings.ORION_BUS_URL,
    }
