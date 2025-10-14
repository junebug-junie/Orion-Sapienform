from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Any
from datetime import datetime

class EventIn(BaseModel):
    """
    Incoming event structure. This model now intelligently adapts to different
    possible text fields from upstream publishers.
    """
    id: str
    text: str  # This will be populated by the validator below.
    collapse_id: Optional[str] = None
    ts: Optional[datetime] = None
    # Keep the rest of the payload for potential future use
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def prepare_and_hydrate_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        This is the core of the adapter. It finds a suitable text field from
        the incoming message and assigns it to the `text` field.
        """
        # Prioritized list of fields to use for the main text content.
        text_source_fields = ["summary", "trigger", "text", "text_content"]

        found_text = ""
        for field in text_source_fields:
            if values.get(field):
                found_text = values[field]
                break # Stop at the first field we find

        if not found_text:
            # If no suitable field is found, the original validation error will occur
            # because 'text' is a required field in the model itself.
            pass

        values['text'] = found_text

        # Move all other fields into 'extra_data' for preservation.
        cls.model_fields.keys()
        known_fields = {'id', 'text', 'collapse_id', 'ts'}
        values['extra_data'] = {k: v for k, v in values.items() if k not in known_fields}

        # Normalize collapse_id
        if not values.get("collapse_id"):
            id_val = values.get("id")
            if isinstance(id_val, str) and id_val.startswith("collapse_"):
                values["collapse_id"] = id_val

        # Normalize timestamp
        ts_val = values.get("ts")
        if isinstance(ts_val, str):
            try:
                values["ts"] = datetime.fromisoformat(ts_val)
            except Exception:
                pass # Let Pydantic handle validation if it fails

        return values


class Enrichment(BaseModel):
    """
    Outgoing enrichment message published to downstream meta-writer.
    """
    id: str
    collapse_id: Optional[str] = None
    service_name: str
    service_version: str
    enrichment_type: str = "tagging"
    tags: List[str] = Field(default_factory=list)
    entities: List[Dict[str, str]] = Field(default_factory=list)
    salience: float = 0.0
    ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @model_validator(mode='before')
    @classmethod
    def ensure_ts_and_collapse(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures critical fields are present before final validation."""
        # Default collapse_id if missing
        if not values.get("collapse_id") and "id" in values:
            id_val = values["id"]
            if isinstance(id_val, str) and id_val.startswith("collapse_"):
                values["collapse_id"] = id_val
        # Ensure timestamp always present as ISO string
        if not values.get("ts"):
            values["ts"] = datetime.utcnow().isoformat()
        return values
