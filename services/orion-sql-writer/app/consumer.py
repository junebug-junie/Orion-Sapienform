import json
import time
import redis
import logging
from sqlalchemy import text
from app.settings import settings
from app.db import engine, ensure_table_exists
from app.models import MessageModel

logger = logging.getLogger(settings.SERVICE_NAME)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def connect_redis():
    return redis.from_url(settings.ORION_BUS_URL, decode_responses=True)

def parse_message(raw):
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Invalid JSON: {e}")
        return {"_raw": raw}

def upsert_payload(payload: dict):
    with engine.begin() as conn:
        sql = text(f"""
            INSERT INTO {settings.POSTGRES_TABLE} (id, payload, created_at, updated_at)
            VALUES (:id, :payload, now(), now())
            ON CONFLICT (id) DO UPDATE
            SET payload = EXCLUDED.payload,
                updated_at = now();
        """)
        conn.execute(sql, {"id": payload.get("id"), "payload": json.dumps(payload)})

def consume_forever():
    ensure_table_exists()
    r = connect_redis()
    channels = [c.strip() for c in settings.SUBSCRIBE_CHANNELS.split(",") if c.strip()]
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(*channels)

    logger.info(f"📡 Subscribed to: {channels}")
    while True:
        msg = pubsub.get_message(timeout=settings.POLL_TIMEOUT)
        if not msg:
            time.sleep(0.1)
            continue
        payload = parse_message(msg.get("data"))
        if not isinstance(payload, dict):
            continue
        upsert_payload(payload)
        logger.info(f"🟢 Upserted id={payload.get('id')} from {msg.get('channel')}")
