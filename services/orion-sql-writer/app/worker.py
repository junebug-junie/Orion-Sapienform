import json
import threading
import logging
from typing import Optional

from app.settings import settings
from app.db import get_session, remove_session
from app.models import (
    CollapseEnrichment, CollapseMirror, ChatHistoryLogSQL, Dream, BiometricsTelemetry, SparkIntrospectionLogSQL
)
from app.schemas import (
    EnrichmentInput, MirrorInput, ChatHistoryInput, DreamInput, BiometricsInput, SparkIntrospectionInput
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

        kind = msg.get("kind")
        if kind == "warm_start":
            logger.info("Skipping warm_start event for chat_history_log")
            return None

        payload = ChatHistoryInput.model_validate(msg).normalize()
        db_data = payload.model_dump()
        log_id = payload.id
        session.merge(ChatHistoryLogSQL(**db_data))

    elif table == "dreams":
        # 1. Filter msg to only include fields known by DreamInput
        known_fields = DreamInput.model_fields.keys()
        filtered_msg = {k: v for k, v in msg.items() if k in known_fields}

        # 2. Validate the CLEANED message
        payload = DreamInput.model_validate(filtered_msg).normalize()

        # 3. Get log_id and query using the PAYLOAD OBJECT (Fixes KeyError)
        log_id = str(payload.dream_date)

        existing = session.query(Dream).filter(
            Dream.dream_date == payload.dream_date
        ).first()

        if existing:
            logger.info(f"Updating {table} for date: {log_id}")
            # For updates, exclude_unset=True is correct
            db_data = payload.model_dump(mode="json", exclude_unset=True)
            for k, v in db_data.items():
                setattr(existing, k, v)
        else:
            logger.info(f"Inserting new {table} for date: {log_id}")
            # For inserts, use the full model (no exclude_unset)
            db_data = payload.model_dump(mode="json")
            session.add(Dream(**db_data))

    elif table == "orion_biometrics":
        payload = BiometricsInput.model_validate(msg)
        db_data = payload.model_dump()
        session.add(BiometricsTelemetry(**db_data))
        log_id = db_data.get("timestamp", "unknown")

    elif table == "spark_introspection_log":
        payload = SparkIntrospectionInput.model_validate(msg).normalize()
        db_data = payload.model_dump()

        # Ensure spark_meta is stored as a JSON string
        spark_meta_val = db_data.get("spark_meta")
        if isinstance(spark_meta_val, (dict, list)):
            db_data["spark_meta"] = json.dumps(spark_meta_val)
        elif spark_meta_val is None:
            db_data["spark_meta"] = None

        log_id = payload.id
        logger.info(f"Upserting Spark introspection log id={log_id} trace_id={payload.trace_id}")

        session.merge(SparkIntrospectionLogSQL(**db_data))
        return log_id

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
