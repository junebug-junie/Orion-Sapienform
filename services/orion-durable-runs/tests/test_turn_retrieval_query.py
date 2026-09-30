"""The runner forwards the brief's retrieval_query onto the Hub turn request.

Recall retrieval design phase 3 (2026-09-29): a curiosity run's standing question is
carried on the checkpointed brief, so a Hub restart mid-run does not lose what recall
should search for. Omitted on the wire when unset (old Hub never sees the key).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from orion.schemas.durable_run import CuriosityRunBriefV1, CuriosityTurnRequestV1, CuriosityTurnResultV1  # noqa: E402

from app.graph import Deps, make_nodes  # noqa: E402

RUN_ID = "abc123def456"


def _deps(sent: list[CuriosityTurnRequestV1]) -> Deps:
    async def run_turn(req: CuriosityTurnRequestV1) -> CuriosityTurnResultV1:
        sent.append(req)
        return CuriosityTurnResultV1(run_id=req.run_id, correlation_id=req.correlation_id, text="found it")

    async def read_turn_result(run_id: str, **kwargs: Any) -> dict:
        return {}

    async def row(facts: dict) -> bool:
        return True

    async def journal(entry) -> str | None:
        return entry.entry_id

    return Deps(run_turn=run_turn, read_turn_result=read_turn_result, publish_attention_row=row, publish_journal=journal)


def _state(retrieval_query: str | None) -> dict[str, Any]:
    brief = CuriosityRunBriefV1(
        prompt="A long self-inquiry prompt.", session_id="orion_curiosity", timeout_sec=900.0,
        line="self_inquiry", retrieval_query=retrieval_query,
    )
    # The runner checkpoints the brief with model_dump(mode="json") (runner.py).
    return {"run_id": RUN_ID, "correlation_id": "trace-1", "attempt": 0, "brief": brief.model_dump(mode="json")}


def test_harness_turn_forwards_the_standing_question() -> None:
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](_state("What am I, when nobody asks?")))
    assert sent[0].retrieval_query == "What am I, when nobody asks?"
    assert sent[0].model_dump(mode="json", exclude_none=True)["retrieval_query"] == "What am I, when nobody asks?"


def test_unset_query_is_absent_on_the_wire() -> None:
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](_state(None)))
    assert sent[0].retrieval_query is None
    assert "retrieval_query" not in sent[0].model_dump(mode="json", exclude_none=True)


def test_a_checkpoint_from_before_the_field_still_resumes() -> None:
    state = _state(None)
    state["brief"].pop("retrieval_query", None)  # unset: already absent from the dump
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](state))
    assert sent[0].retrieval_query is None


# --- review finding (PR #2423 BLOCKER): new runner -> old Hub, reading turns ---
# durable-runs deploys before Hub. The runner's reading RPC dumped the whole request,
# so an unset brief.retrieval_query went out as `null`, the old Hub's forbid-model
# ReadingTurnRequestV1 rejected it, and every reading turn failed while holding a lease.

from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402
from typing import Literal  # noqa: E402

from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from app.runner import DurableRunner  # noqa: E402
from app.settings import Settings  # noqa: E402
from orion.core.bus.bus_schemas import BaseEnvelope, ServiceRef  # noqa: E402
from orion.core.bus.codec import OrionCodec  # noqa: E402
from orion.schemas.gpu_pool import GpuLeaseRefV1  # noqa: E402
from orion.schemas.reading_turn import (  # noqa: E402
    READING_TURN_RESULT_KIND,
    ReadingRunBriefV1,
    ReadingTurnRequestV1,
    ReadingTurnResultV1,
)


class _OldHubReadingBriefV1(BaseModel):
    """ReadingRunBriefV1 as the pre-#2423 Hub knows it: no retrieval_query."""

    model_config = ConfigDict(extra="forbid")
    seed_id: str = Field(min_length=1)
    stage: Literal[1, 2]
    prompt: str = Field(min_length=1)
    session_id: str
    timeout_sec: float = Field(gt=0)
    fcc_model_label: str | None = None


class _OldHubReadingTurnRequestV1(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["reading.turn.request.v1"] = "reading.turn.request.v1"
    run_id: str
    correlation_id: str
    brief: _OldHubReadingBriefV1
    gpu_lease: GpuLeaseRefV1


def _reading_payload_sent_by_runner(retrieval_query: str | None) -> dict:
    request = ReadingTurnRequestV1(
        run_id="reading-1", correlation_id="11111111-1111-4111-8111-111111111111",
        brief=ReadingRunBriefV1(seed_id="s", stage=1, prompt="read it", session_id="x",
                                timeout_sec=30.0, retrieval_query=retrieval_query),
        gpu_lease=GpuLeaseRefV1(lease_id="h", generation=1, role="agent", holder="durable-runs:reading-1"),
    )
    codec = OrionCodec()
    reply = BaseEnvelope(kind=READING_TURN_RESULT_KIND, source=ServiceRef(name="orion-hub"),
                         correlation_id=request.correlation_id,
                         payload=ReadingTurnResultV1(run_id=request.run_id, correlation_id=request.correlation_id,
                                                     ok=True, text="read").model_dump(mode="json"))
    bus = SimpleNamespace(codec=codec, rpc_request=AsyncMock(return_value={"data": codec.encode(reply)}))
    settings = Settings(_env_file=None, DURABLE_RUNS_GRAPH_HOST="", POSTGRES_URI="postgresql://unused",
                        ORION_BUS_ENABLED=False)
    asyncio.run(DurableRunner(settings, bus=bus, checkpointer=None)._run_reading_turn(request))
    return bus.rpc_request.await_args.args[1].payload


def test_new_runner_reading_turn_validates_on_an_old_hub() -> None:
    payload = _reading_payload_sent_by_runner(None)
    assert "retrieval_query" not in payload["brief"]
    _OldHubReadingTurnRequestV1.model_validate(payload)  # raised extra_forbidden before the fix


def test_reading_turn_still_carries_a_set_query() -> None:
    payload = _reading_payload_sent_by_runner("Chip packaging")
    assert payload["brief"]["retrieval_query"] == "Chip packaging"
    assert ReadingTurnRequestV1.model_validate(payload).brief.retrieval_query == "Chip packaging"


def test_unset_query_is_absent_from_every_dump_of_every_forbid_model() -> None:
    """The other new-producer -> old-consumer seams: Hub's stored reading request_json
    and its POST to the runner (model_dump_json / model_dump), and the runner's
    checkpointed curiosity brief."""
    import json

    reading = ReadingRunBriefV1(seed_id="s", stage=1, prompt="p", session_id="x", timeout_sec=1.0)
    assert "retrieval_query" not in reading.model_dump(mode="json")
    assert "retrieval_query" not in json.loads(reading.model_dump_json())
    brief = CuriosityRunBriefV1(prompt="p", session_id="x", timeout_sec=1.0)
    assert "retrieval_query" not in brief.model_dump(mode="json")
    turn = CuriosityTurnRequestV1(run_id="abcdef1", correlation_id="c", prompt="p", timeout_sec=1.0)
    assert "retrieval_query" not in turn.model_dump(mode="json")
    # Round trip still works when set.
    assert CuriosityRunBriefV1.model_validate(
        CuriosityRunBriefV1(prompt="p", session_id="x", timeout_sec=1.0, retrieval_query="q").model_dump(mode="json")
    ).retrieval_query == "q"
