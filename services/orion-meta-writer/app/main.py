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
logger = logging.getLogger(settings.SERVICE_NAME)
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------
# App
# -----------------------
app = FastAPI(title=settings.SERVICE_NAME)

BAD_PAYLOAD_LOG = "/tmp/meta_writer_bad_messages.log"


def _persist_bad_payload(payload, reason: str) -> None:
    """Persist bad payloads to a local file for offline inspection."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "payload": payload,
        }
        with open(BAD_PAYLOAD_LOG, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        logger.exception("Failed to write bad payload to disk")


def listener_worker():
    """
    Creates its own thread-local bus connection, listens for tagged/triage
    messages, translates them, enriches, and republishes.
    """
    bus = OrionBus(url=settings.ORION_BUS_URL, enabled=settings.ORION_BUS_ENABLED)
    if not bus.enabled:
        logger.error("Bus is not initialized or is disabled. Listener cannot start.")
        return

    listen_channel = settings.CHANNEL_EVENTS_TAGGED
    publish_channel = settings.CHANNEL_EVENTS_ENRICHED

    logger.info(f"📡 Listening on {listen_channel}")
    logger.info(f"📨 Publishing enriched output to {publish_channel}")

    sub = bus.subscribe(listen_channel)

    for message in sub:
        try:
            data = message.get("data")
            if not data:
                continue

            # --- 1. Translate & Validate ---
            try:
                validated_input = translate_payload(data)
                logger.info("✅ Validated enrichment id=%s", getattr(validated_input, "id", "<no-id>"))
            except ValidationError as e:
                logger.error("Schema validation failed for message: %s", e)
                _persist_bad_payload(data, f"validation_error: {e}")
                continue

            # --- 2. Enrich Data ---
            try:
                enriched = EnrichmentOutput(
                    **validated_input.model_dump(),
                    processed_by=settings.SERVICE_NAME,
                    processed_version=settings.SERVICE_VERSION,
                    processed_at=datetime.now(timezone.utc),
                )
            except Exception as e:
                logger.exception("Failed to build EnrichmentOutput id=%s: %s", getattr(validated_input, "id", "<no-id>"), e)
                _persist_bad_payload(validated_input.model_dump(), f"output_build_error: {e}")
                continue

            # --- 3. Publish Downstream ---
            try:
                # CORRECTED: Use mode='json' to ensure all fields are JSON-serializable
                # before passing the dictionary to the bus for final encoding.
                bus.publish(publish_channel, enriched.model_dump(mode='json'))
                logger.info("📤 Published enriched id=%s → %s", getattr(validated_input, "id", "<no-id>"), publish_channel)
            except Exception as e:
                logger.exception("Failed to publish enriched id=%s: %s", getattr(validated_input, "id", "<no-id>"), e)
                _persist_bad_payload(enriched.model_dump(mode='json'), f"publish_error: {e}")
                continue

        except Exception:
            logger.exception("Unexpected listener error; continuing")
            continue


@app.on_event("startup")
def startup_event():
    """Initialize OrionBus and start listener thread."""
    if settings.ORION_BUS_ENABLED:
        logger.info("🚀 Starting listener thread...")
        threading.Thread(target=listener_worker, daemon=True, name="meta-writer-listener").start()
        logger.info("🧠 Meta-writer listener thread started")
    else:
        logger.warning("⚠️ OrionBus disabled — listener not started.")


@app.get("/health")
def health():
    """Liveness probe"""
    return {
        "ok": True,
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "bus_enabled": settings.ORION_BUS_ENABLED,
        "listen_channel": settings.CHANNEL_EVENTS_TAGGED,
        "publish_channel": settings.CHANNEL_EVENTS_ENRICHED,
    }

