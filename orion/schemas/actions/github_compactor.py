from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GithubCompactorDigestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    card_summary: str = Field(..., min_length=1)
    journal_title: str = Field(..., min_length=1)
    journal_body: str = Field(..., min_length=1)
    pr_refs: List[str] = Field(default_factory=list)

    @field_validator("pr_refs")
    @classmethod
    def _normalize_pr_refs(cls, value: List[str]) -> List[str]:
        out = [str(item).strip() for item in value if str(item).strip()]
        return out
