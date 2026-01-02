from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, model_validator

# Use shared schema for outbound enrichment
from orion.schemas.telemetry.meta_tags import MetaTagsPayload

class EventIn(BaseModel):
    """
    Ingress model for triage events. 
    automatically extracts the 'best' text representation from various payload shapes.
    """
    id: str
    text: str = ""
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
        text_source_fields = ["summary", "trigger", "text", "text_content", "content"]

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
                val = values.get(field)
                if val and isinstance(val, str):
                    found_text = val
                    break

        # --- 3. Final Assignments and Normalization ---
        values['text'] = found_text

        # Move all other fields into 'extra_data' for preservation.
        known_fields = {'id', 'text', 'collapse_id', 'ts', 'prompt', 'response', 'kind', 'source', 'correlation_id'}
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
                values["ts"] = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
            except Exception:
                pass # Let Pydantic handle validation if it fails or leave as None

        return values


# Re-export MetaTagsPayload as Enrichment for compatibility if needed,
# or simply use MetaTagsPayload directly in main.py
Enrichment = MetaTagsPayload
