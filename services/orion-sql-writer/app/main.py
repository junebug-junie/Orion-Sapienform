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
    Ensure messages are consistently represented as dictionaries.
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
    print(f"⚠️  Skipping non-object payload type: {type(raw_msg)}")
    return None


def _process_one(channel: str, msg: dict):
    """
    Routes a message to the correct Pydantic model and table based on its
    source channel, then validates and upserts it.
    """
    table = settings.get_table_for_channel(channel)
    if not table:
        print(f"⚠️  No table mapping for channel '{channel}', skipping")
        return

    try:
        if table == "collapse_enrichment":
            payload = EnrichmentInput.model_validate(msg).normalize()
        elif table == "collapse_mirror":
            payload = MirrorInput.model_validate(msg).normalize()
        else:
            print(f"⚠️  No model branch for table '{table}', skipping")
            return

        upsert_record(table, payload.model_dump())
        print(f"✅ Upserted id={payload.id} from '{channel}' → {table}")

    except Exception as e:
        print(f"❌ Listener error on '{channel}' → table '{table}': {e}")


def _channel_worker(channel: str):
    """Dedicated worker per channel."""
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=True)
    if not bus.enabled:
        print(f"❌ Bus connection failed for {channel}", flush=True)
        return

    print(f"👂 Subscribed (worker) for channel: {channel}", flush=True)
    for message in bus.subscribe(channel):
        try:
            print(f"🧠 RAW bus message on {channel}: {message}", flush=True)
            data = message.get("data")
            if not data:
                print(f"⚠️ Empty data on {channel}", flush=True)
                continue
            if isinstance(data, dict):
                normalized = data
            elif isinstance(data, str):
                try:
                    normalized = json.loads(data)
                except Exception as e:
                    print(f"⚠️ JSON parse error: {e}", flush=True)
                    continue
            else:
                print(f"⚠️ Unknown payload type: {type(data)}", flush=True)
                continue

            print(f"📥 Normalized payload on {channel}: {normalized}", flush=True)
            _process_one(channel, normalized)

        except Exception as e:
            print(f"❌ Worker error on {channel}: {e}", flush=True)


@app.on_event("startup")
def startup_event():
    """
    Ensures the DB schema is present and starts a listener thread for each channel.
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("🛠️  Ensured DB schema is present")
    except Exception as e:
        print(f"⚠️  Schema init warning: {e}")

    if not settings.ORION_BUS_ENABLED:
        print("⚠️  Bus disabled; writer will be idle.")
        return

    channels = settings.get_all_subscribe_channels()
    print(f"🚀 {settings.SERVICE_NAME} starting listeners for channels: {channels}")

    for ch in channels:
        # Pass only the channel name to the worker. The worker will create its own bus.
        t = threading.Thread(target=_channel_worker, args=(ch,), daemon=True)
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

