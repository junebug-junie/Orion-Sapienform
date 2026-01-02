from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

class VectorWriteRequest(BaseModel):
    """
    Standard request to write data to the vector database.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    kind: str # e.g., "collapse.mirror", "chat.message", "rag.document"
    content: str = Field(description="Text content to embed")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Optional vector override if pre-computed
    vector: Optional[List[float]] = None

    collection_name: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
