import json
import threading
import logging
from fastapi import FastAPI

from app.settings import settings
from app.db import Base, engine, get_session, remove_session
from app.models import (
    EnrichmentInput, MirrorInput, ChatHistoryInput,
    DreamInput, Dream,
    CollapseEnrichment, CollapseMirror, ChatHistoryLogSQL
)
from orion.core.bus.service import OrionBus

logging.basicConfig(level=logging.INFO, format="[SQL_WRITER] %(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.SERVICE_NAME)


def _normalize_message(raw_msg):
    if isinstance(raw_msg, (bytes, bytearray)):
        raw_msg = raw_msg.decode("utf-8", errors="replace")
    if isinstance(raw_msg, str):
        try:
            return json.loads(raw_msg)
        except Exception:
            logger.warning(f"Skipping non-JSON string payload: {raw_msg[:120]!r}")
            return None
    if isinstance(raw_msg, dict):
        return raw_msg
    logger.warning(f"Skipping non-object payload type: {type(raw_msg)}")
    return None


def _process_one(session, channel: str, msg: dict):
    """
    Routes a message to the correct model and stages it for commit.
    This function does NOT commit or rollback.
    """
    table = settings.get_table_for_channel(channel)
    if not table:
        logger.warning(f"No table mapping for channel '{channel}', skipping")
        return None

    log_id = "unknown"

    try:
        if table == "collapse_enrichment":
            payload = EnrichmentInput.model_validate(msg).normalize()
            db_data = payload.model_dump()
            log_id = payload.id
            session.merge(CollapseEnrichment(**db_data))

        elif table == "collapse_mirror":
            payload = MirrorInput.model_validate(msg).normalize()
            db_data = payload.model_dump()
            log_id = payload.id
            session.merge(CollapseMirror(**db_data))

        elif table == "chat_history_log":
            payload = ChatHistoryInput.model_validate(msg).normalize()
            db_data = payload.model_dump()
            log_id = payload.id
            session.merge(ChatHistoryLogSQL(**db_data))

        elif table == "dreams":
            payload = DreamInput.model_validate(msg).normalize()
            db_data = payload.model_dump(mode='json', exclude_unset=True) 
            log_id = payload.dream_date

            # 1. Look for the existing record
            existing = session.query(Dream).filter(
                Dream.dream_date == payload.dream_date
            ).first()

            if existing:
                logger.info(f"Updating {table} for date: {log_id}")

                # Transfer data directly to the existing object in the session
                for key, value in db_data.items():
                    setattr(existing, key, value)

            else:
                logger.info(f"Inserting new {table} for date: {log_id}")
                db_model = Dream(**db_data)
                session.add(db_model)

        return log_id

    except Exception as e:
        raise e

def _channel_worker(channel: str):
    """
    Manages the session lifecycle for each message.
    """
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        logger.error(f"Bus connection failed for {channel}")
        return

    logger.info(f"👂 Subscribed (worker) for channel: {channel}")

    for message in bus.subscribe(channel):
        session = None
        log_id = "unknown"

        try:
            data = message.get("data")
            if isinstance(data, dict):
                normalized = data

            session = get_session()
            log_id = _process_one(session, channel, normalized)

            if log_id:
                logger.info(f"🏁 Attempting COMMIT for {log_id} on channel '{channel}'...")
                session.commit()
                logger.info(f"✅ COMMIT successful for {log_id} on channel '{channel}'")

        except Exception as e:
            logger.error(f"❌ ROLLBACK triggered on {channel} (payload {log_id}): {e}", exc_info=True)
            if session:
                session.rollback()
        finally:
            if session:
                remove_session()

@app.on_event("startup")
def startup_event():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("🛠️  Ensured DB schema is present")
    except Exception as e:
        logger.warning(f"Schema init warning: {e}")

    if not settings.ORION_BUS_ENABLED:
        logger.warning("Bus disabled; writer will be idle.")
        return

    channels = settings.get_all_subscribe_channels()
    logger.info(f"🚀 {settings.SERVICE_NAME} starting listeners for channels: {channels}")

    for ch in channels:
        t = threading.Thread(target=_channel_worker, args=(ch,), daemon=False)
        t.start()


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "channels": settings.get_all_subscribe_channels(),
        "bus_url": settings.ORION_BUS_URL,
    }
