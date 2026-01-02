import logging
import os
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope
from orion.schemas.vector.schemas import VectorWriteRequest

from app.settings import settings

# Setup Logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(settings.SERVICE_NAME)

# --- Global Resources ---
chroma_client: Optional[chromadb.HttpClient] = None
embedding_model: Optional[SentenceTransformer] = None
hunter: Optional[Hunter] = None

# --- Configuration ---
def _cfg() -> ChassisConfig:
    return ChassisConfig(
        service_name=settings.SERVICE_NAME,
        service_version=settings.SERVICE_VERSION,
        node_name=settings.NODE_NAME,
        bus_url=settings.ORION_BUS_URL,
        bus_enabled=settings.ORION_BUS_ENABLED,
        health_channel=settings.HEALTH_CHANNEL,
        error_channel=settings.ERROR_CHANNEL,
    )

def _setup_resources():
    """Initialize ChromaDB connection and load ML models."""
    global chroma_client, embedding_model
    
    logger.info(f"🔌 Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}...")
    try:
        chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(allow_reset=True, anonymized_telemetry=False)
        )
        # Test connection
        chroma_client.heartbeat()
        logger.info("✅ ChromaDB Connected.")
    except Exception as e:
        logger.error(f"❌ Failed to connect to ChromaDB: {e}")
        # We don't raise here to allow the service to start, but writes will fail.

    logger.info(f"🧠 Loading embedding model: {settings.EMBEDDING_MODEL_NAME}...")
    try:
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("✅ Model loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        raise e

def normalize_to_request(env: BaseEnvelope) -> Optional[VectorWriteRequest]:
    """
    Adapts various incoming kinds to a unified VectorWriteRequest.
    """
    kind = env.kind
    payload = env.payload

    if not isinstance(payload, dict):
        return None

    # If it's already a direct vector write request
    if kind == "vector.write.request":
        try:
            return VectorWriteRequest.model_validate(payload)
        except Exception:
            return None

    # Normalization Map
    content = ""
    meta = {
        "source_node": env.source.node or "unknown",
        "kind": kind,
        "timestamp": env.created_at or "",
        "id": env.id
    }
    collection = "orion_general"

    if kind == "collapse.mirror":
        collection = "orion_collapse"
        content = f"{payload.get('summary', '')} {payload.get('mantra', '')} {payload.get('trigger', '')}"
        meta.update({
            "observer": payload.get("observer", "unknown"),
            "type": payload.get("type", "unknown")
        })
    elif kind in ("chat.message", "chat.history"):
        collection = "orion_chat"
        content = payload.get("content") or payload.get("message", "")
        meta.update({
            "role": payload.get("role", "unknown"),
            "session_id": payload.get("session_id", "")
        })
    elif kind == "rag.document":
        collection = "orion_knowledge"
        content = payload.get("text") or payload.get("content", "")
        meta.update({
            "filename": payload.get("filename", ""),
            "doc_id": payload.get("id", "")
        })
    else:
        # Fallback for generic text/event
        content = payload.get("text") or payload.get("summary") or ""

    if not content.strip():
        return None

    return VectorWriteRequest(
        id=env.id,
        kind=kind,
        content=content,
        metadata=meta,
        collection_name=collection
    )


# --- Bus Handler ---
async def handle_envelope(env: BaseEnvelope) -> None:
    """
    Receives messages from the bus, vectorizes the content, and upserts to ChromaDB.
    """
    if not chroma_client or not embedding_model:
        logger.warning("Skipping vector write: Resources not initialized.")
        return

    try:
        req = normalize_to_request(env)
        if not req:
            # logger.debug(f"Skipping {env.kind}: No valid content found.")
            return

        # 3. Generate Embedding (if not provided)
        vector_list = req.vector
        if not vector_list:
            # Run CPU-bound model in thread pool
            vector = await asyncio.to_thread(embedding_model.encode, req.content)
            vector_list = vector.tolist()

        # 4. Upsert to Chroma
        collection = chroma_client.get_or_create_collection(name=req.collection_name or "orion_general")
        
        await asyncio.to_thread(
            collection.upsert,
            ids=[req.id],
            embeddings=[vector_list],
            documents=[req.content],
            metadatas=[req.metadata]
        )
        
        logger.info(f"✨ Vectorized & stored {req.kind} -> {req.collection_name} (id={req.id})")

    except Exception as e:
        logger.exception(f"Error processing {env.kind}: {e}")


# --- Lifecycle & App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter
    _setup_resources()
    
    # Start Bus Listener
    config = _cfg()
    channels = settings.SUBSCRIBE_CHANNELS

    logger.info(f"🏹 Hunter starting. Subscribing to: {channels}")
    hunter = Hunter(config, patterns=channels, handler=handle_envelope)
    await hunter.start_background()
    
    yield
    
    logger.info("🛑 Stopping Hunter...")
    if hunter:
        await hunter.stop()

app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.SERVICE_VERSION,
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "chroma_connected": chroma_client is not None,
        "model_loaded": embedding_model is not None,
        "bus_connected": hunter.bus.is_connected if hunter and hunter.bus else False
    }
