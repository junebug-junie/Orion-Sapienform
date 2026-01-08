from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VerbEffectV1(BaseModel):
    """Side-effect emitted by a verb execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: str = Field(..., description="Effect type identifier")
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerbRequestV1(BaseModel):
    """Serialized request for a verb execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    verb: str = Field(..., description="Registered verb trigger name")
    payload: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    caller: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class VerbResultV1(BaseModel):
    """Serialized result for a verb execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    verb: str
    ok: bool = True
    output: Optional[Dict[str, Any]] = None
    effects: List[VerbEffectV1] = Field(default_factory=list)
    error: Optional[str] = None
    request_id: Optional[str] = None
