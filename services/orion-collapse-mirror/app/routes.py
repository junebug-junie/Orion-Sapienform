from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from uuid import uuid4
import json

from app.schemas import CollapseMirrorEntry
from app.db import get_db
from app.models import CollapseMirrorEntrySQL
from app.chroma_db import collection, embedder
from app.bus import bus

router = APIRouter()


# 🪞 Collapse Mirror: Log new entry
@router.post("/log/collapse")
def log_collapse(entry: CollapseMirrorEntry, db: Session = Depends(get_db)):
    entry = entry.with_defaults()

    if not collection:
        raise RuntimeError("Chroma memory is not active in cloud mode.")

    entry_id = f"collapse_{uuid4().hex}"

    # Flatten metadata
    metadata = entry.dict()
    if isinstance(metadata["observer_state"], list):
        metadata["observer_state"] = ", ".join(metadata["observer_state"])
    metadata = {k: v for k, v in metadata.items() if v is not None}

    # ✅ Generate embedding
    embedding = embedder.encode(entry.summary)
    embedding_list = (
        embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    )

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

    # Publish to Redis (for enrichment + GDB client downstream)
    try:
        payload = {"id": entry_id, "service_name": "orion-collapse-mirror", **metadata}
        bus.publish("collapse.events.raw", json.dumps(payload))
        print(f"📡 Published collapse {entry_id} to Redis bus")
    except Exception as e:
        print(f"⚠️ Failed to publish collapse to Redis bus: {e}")

    return {"id": entry_id, "status": "logged", "entry": metadata}
