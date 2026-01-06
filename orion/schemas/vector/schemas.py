from pydantic import AliasChoices, BaseModel, Field, ConfigDict
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


class VectorDocumentUpsertV1(BaseModel):
    """
    Canonical payload for vector upsert operations.
    Accepts pre-computed embeddings and optional latent references.
    """
    model_config = ConfigDict(extra="ignore")

    doc_id: str = Field(
        ...,
        validation_alias=AliasChoices("doc_id", "id"),
        description="Stable identifier for the document.",
    )
    kind: str = Field(..., description="Semantic type of the document (e.g., chat.message).")
    text: str = Field(
        ...,
        validation_alias=AliasChoices("text", "content"),
        description="Raw text content that was embedded.",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    collection: Optional[str] = Field(None, description="Target vector collection.")

    embedding: Optional[List[float]] = Field(default=None, description="Pre-computed embedding vector.")
    embedding_model: Optional[str] = Field(default=None, description="Identifier for the embedding model used.")
    embedding_dim: Optional[int] = Field(default=None, description="Dimensionality of the embedding vector.")

    latent_ref: Optional[str] = Field(default=None, description="Reference handle to latent artifact storage.")
    latent_summary: Optional[Dict[str, Any]] = Field(default=None, description="Lightweight stats/summary of latent.")


class EmbeddingGenerateV1(BaseModel):
    """
    Embedding generation request payload when talking to an embedding host.
    """
    model_config = ConfigDict(extra="ignore")

    doc_id: str
    text: str
    embedding_profile: str = "default"
    include_latent: bool = False


class EmbeddingResultV1(BaseModel):
    """
    Embedding generation response payload.
    """
    model_config = ConfigDict(extra="ignore")

    doc_id: str
    embedding: List[float]
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    latent_ref: Optional[str] = None
    latent_summary: Optional[Dict[str, Any]] = None
