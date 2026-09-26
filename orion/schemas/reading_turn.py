"""Admitted, read-only source turns. Queue persistence remains owned by Hub."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from orion.schemas.gpu_pool import GpuLeaseRefV1
from orion.schemas.reading import SourceFetchEvidenceV1

READING_WORKFLOW = "reading.turn"
READING_TURN_CHANNEL = "orion:reading:turn:request"
READING_TURN_REPLY_PREFIX = "orion:reading:turn:reply"
READING_TURN_REQUEST_KIND = "reading.turn.request.v1"
READING_TURN_RESULT_KIND = "reading.turn.result.v1"


class ReadingRunBriefV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed_id: str = Field(min_length=1)
    stage: Literal[1, 2]
    prompt: str = Field(min_length=1)
    session_id: str
    timeout_sec: float = Field(gt=0)
    fcc_model_label: str | None = None


class ReadingTurnRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reading.turn.request.v1"] = READING_TURN_REQUEST_KIND
    run_id: str
    correlation_id: str
    brief: ReadingRunBriefV1
    gpu_lease: GpuLeaseRefV1


class ReadingTurnResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["reading.turn.result.v1"] = READING_TURN_RESULT_KIND
    run_id: str
    correlation_id: str
    ok: bool
    text: str = ""
    error: str | None = None
    source_fetches: list[SourceFetchEvidenceV1] | None = None
