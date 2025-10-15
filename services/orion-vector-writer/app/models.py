from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field
from typing import Dict, Any

from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator
from uuid import uuid4

class CollapseTriageEvent(BaseModel):
    id: str
    summary: str
    observer: Optional[str] = None
    trigger: Optional[str] = None
    observer_state: Optional[Union[List[str], str]] = None
    field_resonance: Optional[str] = None
    type: Optional[str] = None
    emergent_entity: Optional[str] = None
    mantra: Optional[str] = None
    causal_echo: Optional[str] = None
    timestamp: Optional[str] = None
    environment: Optional[str] = None

    class Config:
        extra = "allow"

    @field_validator("observer_state", mode="before")
    @classmethod
    def normalize_observer_state(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return [v]
        return v

    def to_document(self):
        # Flatten metadata for Chroma
        metadata = self.model_dump(exclude={"summary"})
        if self.observer_state:
            metadata["observer_state"] = ", ".join(self.observer_state)
        metadata["source_channel"] = "orion:collapse:triage"
        return {
            "id": self.id,
            "text": self.summary,
            "metadata": metadata,
        }

class ChatMessageEvent(BaseModel):
    """
    Defines the schema for chat messages that should be logged for long-term
    memory, coming from a channel like 'orion:chat:history:log'.
    """
    id: str
    user: str
    content: str
    timestamp: str

    def to_document(self) -> Dict[str, Any]:
        """Converts the chat message into a standard document format."""
        metadata = self.model_dump(exclude={'content'})
        metadata['source_channel'] = 'orion:chat:history:log' # Example channel
        return {
            "id": self.id,
            "text": f"User '{self.user}' said: {self.content}",
            "metadata": metadata
        }

class RAGDocumentEvent(BaseModel):
    """

    Defines the schema for explicitly adding a document to the RAG store,
    coming from a channel like 'orion:rag:document:add'.
    """
    id: str
    text: str
    metadata: Dict[str, Any]

    def to_document(self) -> Dict[str, Any]:
        """Converts the RAG document event into the standard format."""
        # The metadata is already well-structured for this event type.
        self.metadata['source_channel'] = 'orion:rag:document:add'
        return self.model_dump()

