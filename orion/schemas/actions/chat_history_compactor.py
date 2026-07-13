from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatHistoryCompactorDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_summary: str = Field(..., min_length=1)
    journal_title: str = Field(default="")
    journal_body: str = Field(default="")
    turn_refs: List[str] = Field(default_factory=list)

    @field_validator("turn_refs")
    @classmethod
    def _normalize_turn_refs(cls, value: List[str]) -> List[str]:
        return [str(item).strip() for item in value if str(item).strip()]
