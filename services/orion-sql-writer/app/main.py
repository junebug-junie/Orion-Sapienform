import json
import threading
from fastapi import FastAPI

from app.settings import settings
from app.db import Base, engine
from app.models import EnrichmentInput, MirrorInput
from app.db_utils import upsert_record
from orion.core.bus.service import OrionBus


app = FastAPI(title=settings.SERVICE_NAME)


def _normalize_message(raw_msg):
    """
    Ensure messages are dicts (Redis can deliver already-encoded JSON strings).
    """
    if isinstance(raw_msg, (bytes, bytearray)):
        raw_msg = raw_msg.decode("utf-8", errors="replace")
    if isinstance(raw_msg, str):
        try:
            return json.loads(raw_msg)
        except Exception:
            print(f"⚠️  Skipping non-JSON string payload: {raw_msg[:120]!r}")
            return None
    if isinstance(raw_msg, dict):
        return raw_msg
    # Anything else (lists, numbers) — skip
    print(f"⚠️  Skipping non-object payload type: {type(raw_msg)}")
    return None


def _process_one(channel: str, msg: dict):
    """
    Route by channel → table; validate; normalize; upsert.
    """
    table = settings.get_table_for_channel(channel)

    try:
        if table == "collapse_enrichment":
            payload = EnrichmentInput.model_validate(msg).normalize()
        elif table == "collapse_mirror":
            payload = MirrorInput.model_validate(msg).normalize()
        else:
            # If you add new tables, register them in MODEL_MAP + add a branch here
            print(f"⚠️  No model branch for table '{table}', skipping")
            return

        upsert_record(table, payload.model_dump())

    except Exception as e:
        print(f"❌ Listener error on '{channel}' → table '{table}': {e}")


def _channel_worker(channel: str, bus: OrionBus):
    """
    Dedicated worker per channel because bus.subscribe() is a blocking generator.
    """
    print(f"👂 Subscribed (worker) for channel: {channel}")
    for msg in bus.subscribe(channel):
        data = _normalize_message(msg)
        if data is None:
            continue
        _process_one(channel, data)


@app.on_event("startup")
def startup_event():
    # Optional: auto-create tables if not present (safe idempotent)
    try:
        Base.metadata.create_all(bind=engine)
        print("🛠️  Ensured DB schema is present")
    except Exception as e:
        print(f"⚠️  Schema init warning: {e}")

    if not settings.ORION_BUS_ENABLED:
        print("⚠️  Bus disabled; writer idle")
        return

    # Create a single OrionBus instance and spawn a thread per channel.
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)

    channels = [
        settings.CHANNEL_TAGS_RAW,
        settings.CHANNEL_TAGS_ENRICHED,
        settings.CHANNEL_COLLAPSE_INTAKE,
    ]
    print(f"🚀 {settings.SERVICE_NAME} starting; channels={channels}")

    for ch in channels:
        t = threading.Thread(target=_channel_worker, args=(ch, bus), daemon=True)
        t.start()


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "channels": [c.strip() for c in settings.SUBSCRIBE_CHANNELS.split(",") if c.strip()],
        "bus_url": settings.ORION_BUS_URL,
    }
