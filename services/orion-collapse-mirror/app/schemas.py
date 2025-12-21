from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator


class CollapseMirrorEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    observer: str
    trigger: str
    observer_state: Union[str, List[str]]
    field_resonance: str
    type: str
    emergent_entity: str
    summary: str
    mantra: str
    causal_echo: Optional[str] = None
    timestamp: Optional[str] = None
    environment: Optional[str] = None

    @field_validator("observer_state", mode="before")
    @classmethod
    def _normalize_observer_state(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)
