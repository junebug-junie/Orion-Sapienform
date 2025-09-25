from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
from uuid import uuid4
import json
import os
import redis
import requests
from chromadb import PersistentClient
from sqlalchemy import Column, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime, timezone

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

# 🧠 Connect to Chroma persistent memory
collection = None
if os.getenv("CHRONICLE_MODE") != "cloud":
    chroma_client = PersistentClient(path="/mnt/storage/collapse-mirrors/chroma")
    collection = chroma_client.get_or_create_collection("collapse_mirror")

# 📡 Redis bus (independent mesh backbone)
REDIS_URL = os.getenv("REDIS_URL", "redis://orion-redis:6379/0")
bus = redis.Redis.from_url(REDIS_URL)

# 📦 Initialize FastAPI router
router = APIRouter()

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
        """Ensure timestamp + environment are always set."""
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.environment:
            self.environment = os.getenv("CHRONICLE_ENVIRONMENT", "dev")
        return self

# 📥 Log a new Collapse Mirror
@router.post("/log/collapse")
def log_collapse(entry: CollapseMirrorEntry, db: Session = Depends(get_db)):
    entry = entry.with_defaults()
    if not collection:
        raise RuntimeError("Chroma memory is not active in cloud mode.")

    entry_id = f"collapse_{uuid4().hex}"

    # Flatten metadata for SQL/Chroma
    metadata = entry.dict()
    if isinstance(metadata["observer_state"], list):
        metadata["observer_state"] = ", ".join(metadata["observer_state"])
    metadata = {k: v for k, v in metadata.items() if v is not None}

    # Write to Chroma
    collection.add(
        documents=[entry.summary],
        metadatas=[metadata],
        ids=[entry_id]
    )

    # Write to SQL
    sql_entry = CollapseMirrorEntrySQL(id=entry_id, **metadata)
    db.add(sql_entry)
    db.commit()

    # 📡 Publish to Redis bus for downstream ingesters (e.g., orion-gdb-client)
    bus.publish("collapse:new", json.dumps({"id": entry_id, **metadata}))

    return {
        "id": entry_id,
        "status": "logged",
        "entry": metadata
    }

# 🔍 Query collapse memory by meaning + filters
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
            {
                "summary": doc,
                "metadata": meta
            }
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
    }

# 🧰 Local SDK-style function for CLI/test use
def query_collapse_from_sdk(prompt: str, base_url: str = "http://localhost:8090", **filters):
    params = {"prompt": prompt, **filters}
    response = requests.get(f"{base_url}/api/log/query", params=params)
    response.raise_for_status()
    return response.json()
