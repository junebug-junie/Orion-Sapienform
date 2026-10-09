from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class MetacogDraftTextPatchV1(BaseModel):
    """The only fields the metacog draft LLM authors.

    what_changed is deliberately absent: it is computed from the trigger's own
    evidence (orion/metacog/evidence_map.py) and overwritten at publish. While
    it stayed here, a model that emitted it anyway in the wrong shape failed
    validation and took the whole draft -- a good summary and mantra -- down
    with it (live 2026-09-29: ~95% of baseline drafts, then dropped by the
    baseline firebreak). The sanitizer now strips it like any other unknown key.
    """

    model_config = ConfigDict(extra="forbid")

    mantra: Optional[str] = None
    summary: Optional[str] = None
    tags_suggested: Optional[List[str]] = None
