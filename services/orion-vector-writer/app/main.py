import logging
import threading
import json
import asyncio
import chromadb
import chromadb.utils.embedding_functions as ef
import time
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Dict, Any

from .settings import settings
from app.models import CollapseTriageEvent, ChatMessageEvent, RAGDocumentEvent
from orion.core.bus.consumer_worker import run_worker
from orion.core.bus.async_service import OrionBusAsync
from orion.core.bus.bus_schemas import BaseEnvelope

# --- Logging Setup ---
logging.basicConfig(level=settings.LOG_LEVEL.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(settings.SERVICE_NAME)

app = FastAPI(title=settings.SERVICE_NAME, version=settings.SERVICE_VERSION)

# A thread-safe queue for batching documents
_doc_queue: List[Dict[str, Any]] = []
_queue_lock = threading.Lock()

def _flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures all values in a metadata dictionary are simple types (str, int, float, bool)
    by JSON-encoding any complex types (lists, dicts).
    """
    flat_meta = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            flat_meta[key] = value
        elif value is not None:
            # Convert lists, dicts, etc., into a JSON string
            flat_meta[key] = json.dumps(value)
    return flat_meta

async def batch_upsert_worker_async(chroma_client, embedding_function, bus_url: str):
    """
    Periodically checks the queue and upserts a batch of documents to ChromaDB.
    """
    logger.info("⚙️ Batch upsert worker started. Batch size: %d", settings.BATCH_SIZE)

    # Use a separate bus for publishing confirmation
    bus = OrionBusAsync(bus_url, enabled=settings.ORION_BUS_ENABLED)
    await bus.connect()

    # Initialize collection
    collection = None
    for attempt in range(10):
        try:
            if getattr(settings, "VECTOR_DB_CREATE_IF_MISSING", True):
                collection = chroma_client.get_or_create_collection(
                    name=settings.VECTOR_DB_COLLECTION,
                    embedding_function=embedding_function,
                )
                logger.info(f"🧠 Connected to Chroma → collection '{settings.VECTOR_DB_COLLECTION}' (created if missing).")
            else:
                collection = chroma_client.get_collection(name=settings.VECTOR_DB_COLLECTION)
                logger.info(f"🧠 Connected to existing Chroma collection '{settings.VECTOR_DB_COLLECTION}'.")
            break
        except Exception as e:
            logger.warning(f"⏳ Waiting for ChromaDB... attempt {attempt+1}/10 ({e})")
            await asyncio.sleep(5)
    else:
        logger.critical("🚨 Failed to connect to ChromaDB after multiple attempts.")
        return

    # --- Upsert loop ---
    while True:
        await asyncio.sleep(2)
        with _queue_lock:
            if not _doc_queue:
                continue
            batch = _doc_queue[:settings.BATCH_SIZE]
            del _doc_queue[:settings.BATCH_SIZE]

        try:
            ids = [doc["id"] for doc in batch]
            texts = [doc["text"] for doc in batch]
            metadatas = [ _flatten_metadata(doc["metadata"]) for doc in batch ]

            # Chroma client is sync (usually), unless using the async client which is alpha
            # Since this runs in an asyncio loop, we should probably offload the sync call
            await asyncio.to_thread(collection.upsert, ids=ids, documents=texts, metadatas=metadatas)
            logger.info(f"✅ Upserted {len(batch)} documents into Chroma collection '{settings.VECTOR_DB_COLLECTION}'")

            # Publish confirmation
            for d in batch:
                await bus.publish(settings.PUBLISH_CHANNEL_VECTOR_CONFIRM, {"id": d["id"], "status": "stored"})

        except Exception as e:
            logger.error(f"❌ Failed to upsert batch to ChromaDB: {e}", exc_info=True)

# Map string keys from settings to actual classes
MODEL_MAP = {
    "CollapseTriageEvent": CollapseTriageEvent,
    "ChatMessageEvent": ChatMessageEvent,
    "RAGDocumentEvent": RAGDocumentEvent,
}

async def message_handler(envelope: BaseEnvelope):
    """
    Listens for messages, validates them, and adds them to a queue for batch processing.
    """

    # Route by kind
    route_key = settings.route_map.get(envelope.kind)

    if not route_key or route_key not in MODEL_MAP:
        logger.debug(f"Skipping unknown kind: {envelope.kind}")
        return

    model_class = MODEL_MAP[route_key]

    try:
        # Pydantic v2 validation
        if hasattr(model_class, "model_validate"):
            validated_event = model_class.model_validate(envelope.payload)
        else:
            validated_event = model_class.parse_obj(envelope.payload)

        doc_to_queue = validated_event.to_document()

        with _queue_lock:
            _doc_queue.append(doc_to_queue)

        logger.debug(f"📥 Queued document {doc_to_queue['id']} from kind {envelope.kind}")

    except Exception as e:
        logger.warning(f"Skipping invalid message kind {envelope.kind}: {e}")

def run_services_in_thread():
    """
    Starts the asyncio event loop for the consumers.
    """
    async def _runner():
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            embedding_function = ef.SentenceTransformerEmbeddingFunction(model_name=settings.EMBEDDING_MODEL)

            logger.info(f"Connecting to ChromaDB server at: {settings.VECTOR_DB_HOST}:{settings.VECTOR_DB_PORT}")
            chroma_client = chromadb.HttpClient(
                host=settings.VECTOR_DB_HOST,
                port=settings.VECTOR_DB_PORT
            )
            chroma_client.heartbeat() # check connection
        except Exception as e:
             logger.critical(f"Failed to init Chroma: {e}")
             return

        # Start the batch upsert worker
        asyncio.create_task(batch_upsert_worker_async(chroma_client, embedding_function, settings.ORION_BUS_URL))

        # Start the bus consumer
        await run_worker(
            service_name=settings.SERVICE_NAME,
            bus_url=settings.ORION_BUS_URL,
            channels=settings.VECTOR_WRITER_SUBSCRIBE_CHANNELS,
            handler=message_handler
        )

    asyncio.run(_runner())


@app.on_event("startup")
def startup_event():
    """
    Initializes all dependencies and starts the background worker threads.
    """
    logger.info(f"🚀 Starting {settings.SERVICE_NAME} v{settings.SERVICE_VERSION}")

    if settings.ORION_BUS_ENABLED:
        logger.info("Starting listener and batch worker threads...")
        threading.Thread(target=run_services_in_thread, daemon=True).start()
    else:
        logger.warning("Bus is disabled; vector writer will be idle.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "version": settings.SERVICE_VERSION,
        "queue_size": len(_doc_queue),
    }
