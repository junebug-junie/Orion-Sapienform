"""Admitted, read-only source turns. Queue persistence remains owned by Hub."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

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
    # Additive: what recall searches for during this reading turn --
    # "<source title> — <stage-1 claim>" (stage 1 has no claim yet: title only).
    # Stored in reading_durable_turn.request_json with the prompt, so it is as
    # durable as the prompt. ADDITIVE ON A `forbid` MODEL: deploy
    # orion-durable-runs before orion-hub.
    retrieval_query: str | None = Field(default=None, max_length=1000)

    @model_serializer(mode="wrap")
    def _omit_unset_retrieval_query(self, handler):
        # An older reader of this model forbids unknown keys, even null ones
        # (same rule as orion/schemas/reading.py's result selectors). Unset, the
        # key is absent from EVERY dump -- wire, stored request_json, checkpoint
        # -- so a new producer stays byte-compatible with an old consumer.
        data = handler(self)
        if isinstance(data, dict) and data.get("retrieval_query") is None:
            data.pop("retrieval_query", None)
        return data


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
