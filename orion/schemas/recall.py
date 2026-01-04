from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timezone

class RecallRequest(BaseModel):
    """
    Request to recall memory from various sources (Vector, SQL, RDF).
    """
    model_config = ConfigDict(extra="ignore")

    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Directives
    mode: Literal["hybrid", "vector", "rdf", "sql"] = "hybrid"
    limit: int = 5
    min_score: float = 0.0
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None

    context: Dict[str, Any] = Field(default_factory=dict)

class RecallResult(BaseModel):
    """
    Standardized memory fragment returned by Recall service.
    """
    model_config = ConfigDict(extra="ignore")

    id: str
    content: str
    source: str # "vector", "rdf", "sql"
    score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: Optional[str] = None
