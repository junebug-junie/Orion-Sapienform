# services/orion-meta-tags/app/main.py
import logging
import traceback
import uuid
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

import spacy
from fastapi import FastAPI

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter, Rabbit
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef

from .settings import settings
from .models import EventIn, Enrichment

# Setup Logging
logging.basicConfig(level=logging.getLevelName(settings.LOG_LEVEL))
logger = logging.getLogger(settings.SERVICE_NAME)

# Initialize NLP
try:
    nlp = spacy.load(settings.SPA_MODEL)
except OSError:
    from spacy.cli import download

    logger.info("Downloading spaCy model %s...", settings.SPA_MODEL)
    download(settings.SPA_MODEL)
    nlp = spacy.load(settings.SPA_MODEL)

# Global chassis references
meta_tagger: Hunter | None = None
meta_tagger_rpc: Rabbit | None = None


def _svc_ref() -> ServiceRef:
    return ServiceRef(
        name=settings.SERVICE_NAME,
        version=settings.SERVICE_VERSION,
        node=settings.NODE_NAME,
    )


def _normalize_observer(observer: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(observer or "")).encode("ascii", "ignore").decode("ascii")
    return normalized.strip().lower()


def _is_juniper(observer: Any) -> bool:
    return _normalize_observer(observer) == "juniper"


def _is_orion(observer: Any) -> bool:
    return _normalize_observer(observer) == "orion"


async def handle_meta_tags_rpc(env: BaseEnvelope) -> BaseEnvelope:
    """
    RPC handler for MetaTagsService (used as a Cortex-Exec plan step).

    IMPORTANT:
    - Must always return a stable non-null 'id' in payload so caller can join.
    - Do NOT assume upstream provides id.
    """
    raw = env.payload if isinstance(env.payload, dict) else {}

    # Determine a stable id:
    req_id = (
        raw.get("id")
        or raw.get("event_id")
        or (str(env.correlation_id) if env.correlation_id else None)
    )
    if not req_id:
        req_id = str(uuid.uuid4())

    # If EventIn requires id, ensure it exists.
    if "id" not in raw or not raw.get("id"):
        raw = dict(raw)
        raw["id"] = req_id

    # Normalize text through EventIn (your existing pattern)
    in_payload = EventIn(**raw)

    doc = nlp(in_payload.text or "")
    tags = [ent.text for ent in doc.ents]
    entities = [ent.label_ for ent in doc.ents]

    out = {
        "id": req_id,
        "tags": tags,
        "entities": entities,
    }

    return BaseEnvelope(
        kind="meta_tags.result.v1",
        source=_svc_ref(),
        correlation_id=env.correlation_id,
        payload=out,
    )


async def handle_triage_event(envelope: BaseEnvelope) -> None:
    """
    Handler for 'orion:collapse:triage' streaming path.
    Keeps your existing behavior: emits Enrichment -> tags.enriched.
    """
    global meta_tagger

    try:
        raw_payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        observer = _normalize_observer(raw_payload.get("observer"))
        logger.debug(
            "Meta-tags gate observer=%s action=%s",
            observer or "unknown",
            "skip" if _is_orion(observer) else "publish",
        )
        if _is_orion(observer):
            return

        # VALIDATION & TEXT EXTRACTION
        in_payload = EventIn(**raw_payload)
        logger.info("📨 Processing %s (Text len: %d)", in_payload.id, len(in_payload.text or ""))

        # NLP PROCESSING
        doc = nlp(in_payload.text or "")

        # Extract entities (as tags)
        tags = [ent.text for ent in doc.ents]

        # Basic sentiment heuristic
        sentiment_tag = "sentiment:neutral"
        positive_keywords = {"triumphant", "relief", "capable", "good", "success"}
        negative_keywords = {"anxious", "fear", "fail", "bad", "panic"}

        tokens = set((in_payload.text or "").lower().split())
        if tokens & positive_keywords:
            sentiment_tag = "sentiment:positive"
        elif tokens & negative_keywords:
            sentiment_tag = "sentiment:negative"

        tags.append(sentiment_tag)

        # Ensure collapse_id is never None (prevents downstream NOT NULL surprises)
        target_collapse_id = in_payload.collapse_id or in_payload.id
        if not target_collapse_id:
            logger.warning("⚠️  Missing collapse_id for event %s, sql-writer may fail.", in_payload.id)

        enrichment = Enrichment(
            id=in_payload.id,
            collapse_id=target_collapse_id,
            service_name=settings.SERVICE_NAME,
            service_version=settings.SERVICE_VERSION,
            enrichment_type="tagging",
            tags=tags,
            entities=[],
            salience=0.0,
            ts=datetime.now(timezone.utc).isoformat(),
            node=settings.NODE_NAME,
            correlation_id=str(envelope.correlation_id) if envelope.correlation_id else str(in_payload.id),
            source_message_id=str(envelope.id) if envelope.id else None,
        )

        # Wrap in standard envelope
        out_env = envelope.derive_child(
            kind="tags.enriched",
            source=_svc_ref(),
            payload=enrichment,
        )

        await meta_tagger.bus.publish(settings.CHANNEL_EVENTS_TAGGED, out_env)
        logger.info("✅ Published tags for %s -> %s", in_payload.id, settings.CHANNEL_EVENTS_TAGGED)

    except Exception as e:
        logger.error("❌ Error processing event: %s", e)
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global meta_tagger, meta_tagger_rpc

    # Configure Chassis
    config = ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        heartbeat_interval_sec=settings.HEARTBEAT_INTERVAL_SEC,
    )

    # Streaming triage consumer (keep)
    meta_tagger = Hunter(
        cfg=config,
        handler=handle_triage_event,
        patterns=[settings.CHANNEL_EVENTS_TRIAGE],
    )

    # RPC step-service (MetaTagsService)
    meta_tagger_rpc = Rabbit(
        config,
        request_channel="orion:exec:request:MetaTagsService",
        handler=handle_meta_tags_rpc,
    )

    await meta_tagger_rpc.start_background()
    await meta_tagger.start_background()
    logger.info("Meta-Tags starting... triage=%s rpc=%s", settings.CHANNEL_EVENTS_TRIAGE, "orion:exec:request:MetaTagsService")

    yield

    logger.info("Shutting down...")
    try:
        await meta_tagger.stop()
    finally:
        await meta_tagger_rpc.stop()


app = FastAPI(
    title="Orion Meta-Tags",
    version=settings.SERVICE_VERSION,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    status = "disconnected"
    if meta_tagger and meta_tagger.bus and meta_tagger.bus.redis:
        status = "connected"
    return {"status": "ok", "bus": status, "model": settings.SPA_MODEL}
