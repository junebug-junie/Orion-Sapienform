import json
import threading
import logging
from typing import Optional

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    CollapseEnrichment, CollapseMirror, ChatHistoryLogSQL, Dream, BiometricsTelemetry
)
from app.schemas import (
    EnrichmentInput, MirrorInput, ChatHistoryInput, DreamInput, BiometricsInput
)
from orion.core.bus.service import OrionBus

logger = logging.getLogger(__name__)

def _normalize_message(raw_msg) -> Optional[dict]:
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

def _process_one(session, channel: str, msg: dict) -> Optional[str]:
    table = settings.get_table_for_channel(channel)
    if not table:
        logger.warning(f"No table mapping for channel '{channel}', skipping")
        return None

    log_id = "unknown"

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
        db_data = payload.model_dump(mode="json", exclude_unset=True)
        log_id = str(db_data.get("dream_date"))

        existing = session.query(Dream).filter(
            Dream.dream_date == db_data["dream_date"]
        ).first()

        if existing:
            logger.info(f"Updating {table} for date: {log_id}")
            for k, v in db_data.items():
                setattr(existing, k, v)
        else:
            logger.info(f"Inserting new {table} for date: {log_id}")
            session.add(Dream(**db_data))

    elif table == "biometrics_telemetry":
        payload = BiometricsInput.model_validate(msg)
        db_data = payload.model_dump()
        session.add(BiometricsTelemetry(**db_data))
        log_id = db_data.get("timestamp", "unknown")

    return log_id

def channel_worker(channel: str):
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        logger.error(f"Bus connection failed for {channel}")
        return

    logger.info(f"👂 Subscribed (worker) for channel: {channel}")

    for message in bus.subscribe(channel):
        session = None
        log_id = "unknown"
        try:
            normalized = _normalize_message(message.get("data"))
            if normalized is None:
                continue

            session = get_session()
            log_id = _process_one(session, channel, normalized)

            if log_id:
                logger.info(f"🏁 COMMIT for {log_id} on '{channel}'")
                session.commit()

        except Exception as e:
            logger.error(f"❌ ROLLBACK on {channel} (payload {log_id}): {e}", exc_info=True)
            if session:
                session.rollback()
        finally:
            if session:
                remove_session()

def start_listeners():
    channels = settings.get_all_subscribe_channels()
    logger.info(f"🚀 Starting listeners: {channels}")
    for ch in channels:
        t = threading.Thread(target=channel_worker, args=(ch,), daemon=False)
        t.start()
