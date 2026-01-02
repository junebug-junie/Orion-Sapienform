import logging
import os
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

from fastapi import FastAPI
import chromadb
from chromadb.config import Settings as ChromaSettings
import chromadb.utils.embedding_functions as ef
from sentence_transformers import SentenceTransformer

from orion.core.bus.bus_service_chassis import ChassisConfig, Hunter
from orion.core.bus.bus_schemas import BaseEnvelope
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

# --- Bus Handler ---
async def handle_envelope(env: BaseEnvelope) -> None:
    """
    Receives messages from the bus, vectorizes the content, and upserts to ChromaDB.
    """
    if not chroma_client or not embedding_model:
        logger.warning("Skipping vector write: Resources not initialized.")
        return

    kind = env.kind
    payload = env.payload
    
    if not isinstance(payload, dict):
        logger.warning(f"Skipping {kind}: Payload is not a dict.")
        return

    try:
        # 1. Determine Target Collection & Text Content
        collection_name = "orion_general" # Default
        text_content = ""
        metadata = {
            "source_node": env.source.node or "unknown",
            "kind": kind,
            "timestamp": env.created_at or "",
            "id": env.id
        }

        # --- Routing Logic ---
        if kind == "collapse.mirror":
            collection_name = "orion_collapse"
            # Combine fields for semantic search
            text_content = f"{payload.get('summary', '')} {payload.get('mantra', '')} {payload.get('trigger', '')}"
            metadata.update({
                "observer": payload.get("observer", "unknown"),
                "type": payload.get("type", "unknown")
            })

        elif kind == "chat.message" or kind == "chat.history":
            collection_name = "orion_chat"
            text_content = payload.get("content") or payload.get("message", "")
            metadata.update({
                "role": payload.get("role", "unknown"),
                "session_id": payload.get("session_id", "")
            })

        elif kind == "rag.document":
            collection_name = "orion_knowledge"
            text_content = payload.get("text") or payload.get("content", "")
            metadata.update({
                "filename": payload.get("filename", ""),
                "doc_id": payload.get("id", "")
            })

        # 2. Validation
        if not text_content or not text_content.strip():
            logger.debug(f"Skipping {kind}: No text content to embed.")
            return

        # 3. Generate Embedding
        # Run CPU-bound model in thread pool to avoid blocking asyncio loop
        vector = await asyncio.to_thread(embedding_model.encode, text_content)
        vector_list = vector.tolist()

        # 4. Upsert to Chroma
        collection = chroma_client.get_or_create_collection(name=collection_name)
        
        await asyncio.to_thread(
            collection.upsert,
            ids=[env.id],
            embeddings=[vector_list],
            documents=[text_content],
            metadatas=[metadata]
        )
        
        logger.info(f"✨ Vectorized & stored {kind} -> {collection_name} (id={env.id})")

    except Exception as e:
        logger.exception(f"Error processing {kind}: {e}")


# --- Lifecycle & App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global hunter
    _setup_resources()
    
    # Start Bus Listener
    config = _cfg()
    # Subscribe to configured channels (ensure settings returns a list)
    channels = settings.SUBSCRIBE_CHANNELS
    if isinstance(channels, str):
        channels = json.loads(channels) # Handle stringified JSON from .env if needed

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
