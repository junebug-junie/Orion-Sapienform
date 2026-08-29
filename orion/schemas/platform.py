from __future__ import annotations

from typing import Any, Dict, Optional, Union

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
    # str is real: whisper-tts's tts_worker.py/stt_worker.py and
    # orion-substrate-runtime's finalize_appraisal_listener.py all publish
    # `"details": str(exc)` (a plain exception message), not a dict, on their
    # kind="system.error" replies. Confirmed live 2026-08-29. A strict
    # Dict[str, Any] here rejects every one of those real payloads.
    details: Union[str, Dict[str, Any]] = Field(default_factory=dict)
