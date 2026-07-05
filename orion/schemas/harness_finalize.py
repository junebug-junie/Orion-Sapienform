from __future__ import annotations

from pydantic import BaseModel


class GrammarReceiptV1(BaseModel):
    step_index: int
    tool_name: str | None = None
    summary: str
    grammar_event_id: str | None = None
