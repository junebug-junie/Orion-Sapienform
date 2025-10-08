from uuid import uuid4
from sqlalchemy.orm import Session

from app.settings import settings
from app.models import CollapseMirrorEntrySQL
from app.chroma_db import collection, embedder
from orion.core.bus.service import OrionBus
from orion.schemas.collapse_mirror import CollapseMirrorEntry

bus = OrionBus(url=settings.ORION_BUS_URL)

def log_and_persist(entry: CollapseMirrorEntry, db: Session):
    entry = entry.with_defaults()

    if not collection:
        raise RuntimeError("Chroma memory is not active in cloud mode.")

    entry_id = f"collapse_{uuid4().hex}"

    # Flatten metadata
    metadata = entry.dict()
    if isinstance(metadata["observer_state"], list):
        metadata["observer_state"] = ", ".join(metadata["observer_state"])
    metadata = {k: v for k, v in metadata.items() if v is not None}

    # Generate embedding
    embedding = embedder.encode(entry.summary)
    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    # Write to Chroma
    collection.add(
        documents=[entry.summary],
        metadatas=[metadata],
        embeddings=[embedding_list],
        ids=[entry_id],
    )

    # Write to SQL
    sql_entry = CollapseMirrorEntrySQL(id=entry_id, **metadata)
    db.add(sql_entry)
    db.commit()

    # Publish event downstream
    payload = {"id": entry_id, "service_name": settings.SERVICE_NAME, **metadata}
    bus.publish("collapse.events.raw", payload)  # let OrionBus handle JSON serialization
    print(f"📡 Published collapse {entry_id} to Redis bus")

    return {"id": entry_id, "status": "logged", "entry": metadata}
