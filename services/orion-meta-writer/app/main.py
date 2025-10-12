import json
import threading
import logging
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import ValidationError

from app.settings import settings
from app.models import EnrichmentInput, EnrichmentOutput, translate_payload
from orion.core.bus.service import OrionBus

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("orion-meta-writer")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------
# App + globals
# -----------------------
app = FastAPI(title=settings.SERVICE_NAME)
bus: OrionBus | None = None

BAD_PAYLOAD_LOG = "/tmp/meta_writer_bad_messages.log"


def _persist_bad_payload(payload, reason: str) -> None:
    """Persist bad payloads to a local file for offline inspection."""
    try:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "reason": reason, "payload": payload}
        with open(BAD_PAYLOAD_LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        logger.exception("Failed to write bad payload to disk")


def listener_worker():
    """
    Listens for messages, translates them into Pydantic models, enriches and republishes.
    Defensive: validation or publish errors are logged and stored; the loop continues.
    """
    if not bus:
        logger.error("Bus is not initialized. Listener cannot start.")
        return

    logger.info("Listening on Redis channel: %s", settings.SUBSCRIBE_CHANNEL)
    sub = bus.subscribe(settings.SUBSCRIBE_CHANNEL)

    for message in sub:
        try:
            # normalize incoming message to a dict
            if isinstance(message, (str, bytes, bytearray)):
                try:
                    data = json.loads(message)
                except Exception as e:
                    logger.error("Invalid JSON payload received: %s", e)
                    _persist_bad_payload({"raw": str(message)}, f"invalid_json: {e}")
                    continue
            else:
                data = message

            # --- 1. Translate & Validate (may raise pydantic.ValidationError) ---
            try:
                validated_input = translate_payload(data)
                logger.info("Validated enrichment id=%s", getattr(validated_input, "id", "<no-id>"))
            except ValidationError as e:
                logger.error("Schema validation failed for incoming message: %s", e)
                _persist_bad_payload(data if isinstance(data, dict) else {"raw": str(data)}, f"validation_error: {e}")
                # continue processing next message (do not crash)
                continue

            # --- 2. Enrich Data and Create Output Schema ---
            try:
                enriched = EnrichmentOutput(
                    **validated_input.model_dump(),
                    processed_by=settings.SERVICE_NAME,
                    processed_version=settings.SERVICE_VERSION,
                    processed_at=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.exception("Failed to build EnrichmentOutput for id=%s: %s", getattr(validated_input, "id", "<no-id>"), e)
                _persist_bad_payload(validated_input.model_dump() if hasattr(validated_input, "model_dump") else {}, f"output_build_error: {e}")
                continue

            # --- 3. Re-publish Downstream ---
            try:
                bus.publish(settings.PUBLISH_CHANNEL, enriched.model_dump_json())
                logger.info("Published enriched data id=%s -> %s", getattr(validated_input, "id", "<no-id>"), settings.PUBLISH_CHANNEL)
            except Exception as e:
                logger.exception("Failed to publish enriched data id=%s: %s", getattr(validated_input, "id", "<no-id>"), e)
                _persist_bad_payload(enriched.model_dump() if hasattr(enriched, "model_dump") else {}, f"publish_error: {e}")
                continue

        except Exception:
            # Top-level protection — keep listener alive
            logger.exception("Unexpected listener error; continuing")
            continue


@app.on_event("startup")
def startup_event():
    """
    Initialize the OrionBus connection and start the listener thread.
    """
    global bus
    if settings.ORION_BUS_ENABLED:
        logger.info("Initializing OrionBus connection to %s", settings.ORION_BUS_URL)
        bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
        threading.Thread(target=listener_worker, daemon=True, name="meta-writer-listener").start()
        logger.info("Meta-writer listener thread started")
    else:
        logger.warning("Meta-writer listener is disabled (ORION_BUS_ENABLED=false)")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": settings.ORION_BUS_ENABLED,
        "subscribe_channel": settings.SUBSCRIBE_CHANNEL,
        "publish_channel": settings.PUBLISH_CHANNEL,
    }
