from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
from chromadb import PersistentClient
from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone
import os, json, redis

# 🧠 Local embeddings
from sentence_transformers import SentenceTransformer

router = APIRouter()

# 📦 SQLAlchemy setup
Base = declarative_base()
POSTGRES_URI = os.getenv("POSTGRES_URI", "sqlite:////mnt/storage/collapse-mirrors/collapse.db")
engine = create_engine(POSTGRES_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class CollapseMirrorEntrySQL(Base):
    __tablename__ = "collapse_mirror"
    id = Column(String, primary_key=True, index=True)
    observer = Column(String)
    trigger = Column(Text)
    observer_state = Column(String)
    field_resonance = Column(Text)
    type = Column(String)
    emergent_entity = Column(Text)
    summary = Column(Text)
    mantra = Column(Text)
    causal_echo = Column(Text, nullable=True)
    timestamp = Column(String, nullable=True)
    environment = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 🪞 Collapse Mirror entry schema
class CollapseMirrorEntry(BaseModel):
    observer: str
    trigger: str
    observer_state: List[str]
    field_resonance: str
    type: str
    emergent_entity: str
    summary: str
    mantra: str
    causal_echo: Optional[str] = None
    timestamp: Optional[str] = None
    environment: Optional[str] = None

    def with_defaults(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.environment:
            self.environment = os.getenv("CHRONICLE_ENVIRONMENT", "dev")
        return self

# 🧠 Chroma persistent memory
collection = None
embedder = None
if os.getenv("CHRONICLE_MODE") != "cloud":
    chroma_client = PersistentClient(path="/mnt/storage/collapse-mirrors/chroma")
    collection = chroma_client.get_or_create_collection("collapse_mirror")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        _ = embedder.encode("warmup")
        print("✅ Embedder warm-up success")
    except Exception as e:
        print("⚠️ Embedder warm-up failed:", e)

# 📡 Redis bus (initialized once)
REDIS_URL = os.getenv("REDIS_URL", "redis://orion-redis:6379/0")
bus = redis.from_url(REDIS_URL)

# 📥 Log a new Collapse Mirror
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
    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)

    # Write to Chroma
    collection.add(
        documents=[entry.summary],
        metadatas=[metadata],
        embeddings=[embedding_list],
        ids=[entry_id]
    )

    # Write to SQL
    sql_entry = CollapseMirrorEntrySQL(id=entry_id, **metadata)
    db.add(sql_entry)
    db.commit()

    # Publish event on Redis bus for GraphDB ingestion
    try:
        payload = {"id": entry_id, **metadata}
        bus.publish("collapse:new", json.dumps(payload))
        print(f"📡 Published collapse {entry_id} to Redis bus")
    except Exception as e:
        print(f"⚠️ Failed to publish collapse to Redis bus: {e}")

    return {"id": entry_id, "status": "logged", "entry": metadata}

# 🔍 Query collapse memory
@router.get("/log/query")
def query_memory(
    prompt: str = Query(...),
    observer: Optional[str] = None,
    type: Optional[str] = None,
    emergent_entity: Optional[str] = None,
    mantra: Optional[str] = None,
    n_results: int = 3
):
    filter_conditions = {}
    if observer:
        filter_conditions["observer"] = observer
    if type:
        filter_conditions["type"] = type
    if emergent_entity:
        filter_conditions["emergent_entity"] = emergent_entity
    if mantra:
        filter_conditions["mantra"] = mantra

    results = collection.query(
        query_texts=[prompt],
        n_results=n_results,
        where=filter_conditions or None
    )

    return {
        "query": prompt,
        "filter": filter_conditions or None,
        "results": [
            {"summary": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    }
