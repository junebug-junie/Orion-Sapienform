from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ConfigDict


class MemoryDocument(BaseModel):
    """
    Normalized chat or chunk document destined for vector storage.
    """
    model_config = ConfigDict(extra="ignore")

    doc_id: str
    kind: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection: Optional[str] = None
    chunk_id: Optional[str] = None
    finalized: bool = False
