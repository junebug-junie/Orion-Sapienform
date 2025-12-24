from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class VisionTask(BaseModel):
    """
    Bus-first task envelope (VisionHostService).

    Expected intake message shape (minimum viable):
    {
      "event": "vision_task",
      "service": "VisionHostService",
      "corr_id": "...",
      "reply_channel": "orion:vision:reply:<corr_id>",
      "task_type": "retina_fast" | "detect_open_vocab" | ...,
      "request": {...},
      "meta": {...}
    }
    """
    event: str = "vision_task"
    service: str = "VisionHostService"

    corr_id: str
    reply_channel: str

    task_type: str
    request: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)


class VisionResult(BaseModel):
    event: str = "vision_result"
    service: str = "VisionHostService"

    corr_id: str
    ok: bool = True

    task_type: str
    device: Optional[str] = None

    artifacts: Dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None

    meta: Dict[str, Any] = Field(default_factory=dict)
