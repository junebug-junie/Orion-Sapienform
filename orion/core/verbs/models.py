from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class VerbEffectV1(BaseModel):
    """Side-effect emitted by a verb execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    kind: str = Field(
        ...,
        description="Effect type identifier",
        validation_alias=AliasChoices("kind", "type"),
    )
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VerbRequestV1(BaseModel):
    """Serialized request for a verb execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trigger: str = Field(
        ...,
        description="Registered verb trigger name",
        validation_alias=AliasChoices("trigger", "verb"),
    )
    schema_id: Optional[str] = Field(
        None,
        description="Schema identifier for the payload model",
    )
    payload: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None
    caller: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def verb(self) -> str:
        return self.trigger


class VerbResultV1(BaseModel):
    """Serialized result for a verb execution."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    verb: str
    ok: bool = True
    output: Optional[Dict[str, Any]] = None
    effects: List[VerbEffectV1] = Field(default_factory=list)
    error: Optional[str] = None
    request_id: Optional[str] = None
