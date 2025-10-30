from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Any
from datetime import datetime

class EventIn(BaseModel):
    # ... (Model fields are unchanged) ...
    id: str
    text: str 
    collapse_id: Optional[str] = None
    ts: Optional[datetime] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def prepare_and_hydrate_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core adapter: Finds a suitable text field, prioritizing chat dialogues.
        """
        # Note: 'messages' often requires processing, so we check explicit text fields first.
        text_source_fields = ["summary", "trigger", "text", "text_content"]

        found_text = ""

        # --- 1. Check Chat Dialogue Structure ---
        if 'prompt' in values and 'response' in values:
            # Combine user prompt and assistant response into a single context block
            prompt = str(values.get('prompt', '')).strip()
            response = str(values.get('response', '')).strip()
            if prompt or response:
                # Use a clear dialogue separator for later processing/tagging
                found_text = f"User: {prompt}\nOrion: {response}"

        # --- 2. Fallback to Simple Text Fields (Only runs if no chat dialogue was found) ---
        if not found_text:
            for field in text_source_fields:
                if values.get(field):
                    found_text = values[field]
                    break

        # --- 3. Final Assignments and Normalization ---
        values['text'] = found_text

        # Move all other fields into 'extra_data' for preservation.
        known_fields = {'id', 'text', 'collapse_id', 'ts', 'prompt', 'response'}
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
