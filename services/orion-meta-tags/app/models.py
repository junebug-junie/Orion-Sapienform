# services/orion-meta-writer/app/models.py
from pydantic import BaseModel, Field, root_validator
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

class EnrichmentInput(BaseModel):
    """
    Expected incoming message schema. This model is forgiving:
    - collapse_id is optional and derived from `id` when possible
    - tags/entities use default_factory to avoid shared mutable defaults
    - ts (ISO string) is accepted and parsed when provided
    """
    id: str
    collapse_id: Optional[str] = Field(None, description="Collapse id; may be derived from `id`")
    service_name: str
    service_version: str
    enrichment_type: str
    tags: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    salience: Optional[float] = None
    ts: Optional[datetime] = None

    @root_validator(pre=True)
    def derive_collapse_id_and_parse_ts(cls, values):
        # Derive collapse_id from id if it's missing and id looks like 'collapse_...'
        if not values.get("collapse_id"):
            id_val = values.get("id")
            if isinstance(id_val, str) and id_val.startswith("collapse_"):
                values["collapse_id"] = id_val

        # Parse ts if it's a string (ISO timestamp)
        ts_val = values.get("ts")
        if isinstance(ts_val, str):
            try:
                # Use fromisoformat or dateutil if necessary
                # This will accept RFC3339 / ISO formats like "2025-10-12T04:31:09.802298"
                values["ts"] = datetime.fromisoformat(ts_val)
            except Exception:
                # Last-resort: leave it as-is and let the app handle it downstream
                pass
        return values

class EnrichmentOutput(EnrichmentInput):
    """
    Outgoing schema: inherits input, adds processing metadata.
    processed_at defaults to current UTC time if caller doesn't set it.
    """
    processed_by: str
    processed_version: str
    processed_at: datetime = Field(default_factory=lambda: datetime.utcnow())
