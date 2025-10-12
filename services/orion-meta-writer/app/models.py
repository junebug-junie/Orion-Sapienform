from __future__ import annotations
from typing import Any, List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field, model_validator, ValidationError


class EnrichmentInput(BaseModel):
    """
    Incoming message schema (Pydantic v2-compatible).
    - collapse_id is optional and will be derived from `id` if missing and id starts with "collapse_".
    - tags/entities use default_factory to avoid shared mutable defaults.
    - ts supports ISO string and will be parsed to datetime if possible.
    """
    id: str
    collapse_id: Optional[str] = None
    service_name: str
    service_version: str
    enrichment_type: str
    tags: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    salience: Optional[float] = None
    ts: Optional[datetime] = None

    @model_validator(mode="before")
    def _prepare(cls, values: Any) -> Any:
        """
        Pre-validator: derive collapse_id from id when missing and try to parse ts string.
        This runs before field validation.
        """
        if not isinstance(values, dict):
            return values

        # derive collapse_id from id if missing and id looks like 'collapse_...'
        if not values.get("collapse_id"):
            idv = values.get("id")
            if isinstance(idv, str) and idv.startswith("collapse_"):
                values["collapse_id"] = idv

        # try parse ts if provided as string
        ts_val = values.get("ts")
        if isinstance(ts_val, str):
            try:
                values["ts"] = datetime.fromisoformat(ts_val)
            except Exception:
                # leave as-is; model validation will accept or reject later
                pass

        return values

    class Config:
        arbitrary_types_allowed = True


class EnrichmentOutput(EnrichmentInput):
    """
    Outgoing schema: inherits input, adds processing metadata.
    processed_at defaults to current UTC time.
    """
    processed_by: str
    processed_version: str
    processed_at: datetime = Field(default_factory=lambda: datetime.utcnow())


def translate_payload(payload: Any) -> EnrichmentInput:
    """
    Translate & validate a raw incoming payload into an EnrichmentInput instance.
    Raises pydantic.ValidationError on invalid payloads.

    Usage:
        validated = translate_payload(raw_payload)
    """
    # This uses pydantic v2 BaseModel validation entrypoint.
    return EnrichmentInput.model_validate(payload)
