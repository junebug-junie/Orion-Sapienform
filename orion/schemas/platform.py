from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class GenericPayloadV1(BaseModel):
    """Permissive payload schema for wildcard catalog entries."""

    model_config = ConfigDict(extra="allow")


class CoreEventV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    meta: Optional[Dict[str, Any]] = None



class SystemErrorV1(BaseModel):
    model_config = ConfigDict(extra="allow")

    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
