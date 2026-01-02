from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

class RdfWriteRequest(BaseModel):
    """
    Direct request to write raw triples or triggers to the RDF writer.
    Used for bypassing Cortex.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    source: str = Field(description="Source service name")
    graph: Optional[str] = None
    triples: Optional[str] = Field(default=None, description="Raw Turtle/NTriples content if pre-generated")

    # If not providing raw triples, provide data to build them
    payload: Optional[Dict[str, Any]] = None
    kind: Optional[str] = None # e.g. "collapse.mirror", "chat.raw"

class RdfBuildRequest(BaseModel):
    """
    Contract for Cortex-Exec worker step 'rdf_build'.
    """
    model_config = ConfigDict(extra="ignore")

    # Standard worker request fields
    context: Dict[str, Any] # Contains messages, session_id, etc.
    args: Dict[str, Any] # Step args
    plan_id: Optional[str] = None
    step_id: Optional[str] = None

class RdfWriteResult(BaseModel):
    """
    Result returned to caller (RPC) or published to confirm stream.
    """
    model_config = ConfigDict(extra="ignore")

    ok: bool
    id: str
    triples_count: int = 0
    graph_name: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
