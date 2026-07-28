from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class MetacogDraftWhatChangedV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: Optional[str] = None
    evidence: Optional[List[str]] = None


class MetacogDraftTextPatchV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mantra: Optional[str] = None
    summary: Optional[str] = None
    what_changed: Optional[MetacogDraftWhatChangedV1] = None
    tags_suggested: Optional[List[str]] = None
