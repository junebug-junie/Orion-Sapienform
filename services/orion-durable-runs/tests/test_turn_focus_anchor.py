"""The runner forwards the brief's curiosity focus onto the Hub turn request.

Hub hands it to the governor, which anchors it in the system prompt so CLI compaction cannot
lose it (live 2026-10-17: run 480ee593ead6 lost focus a21f96c9 that way). Absent from every dump
when unset, so an older Hub (forbid model) never sees the key.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Literal

import pytest

pytest.importorskip("langgraph")

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(SERVICE_ROOT)]

from pydantic import BaseModel, ConfigDict  # noqa: E402

from orion.schemas.curiosity_focus import CuriosityFocusAnchorV1  # noqa: E402
from orion.schemas.durable_run import (  # noqa: E402
    CuriosityRunBriefV1,
    CuriosityTurnRequestV1,
    CuriosityTurnResultV1,
    DurableRunRequestV1,
)

from app.graph import Deps, make_nodes  # noqa: E402

RUN_ID = "480ee593ead6"
FOCUS = CuriosityFocusAnchorV1(
    focus_id="a21f96c9-f1aa-5c19-9562-91ed82a64b80", run_id=RUN_ID,
    question="Do walkway regulars keep a weekly rhythm?", origin="self", graph="orion_worldview",
)


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


def _state(focus: CuriosityFocusAnchorV1 | None) -> dict[str, Any]:
    brief = CuriosityRunBriefV1(prompt="investigate", session_id="orion_curiosity", timeout_sec=900.0, focus=focus)
    return {"run_id": RUN_ID, "correlation_id": "trace-1", "attempt": 0, "brief": brief.model_dump(mode="json")}


def test_harness_turn_forwards_the_focus_from_the_checkpointed_brief() -> None:
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](_state(FOCUS)))
    assert sent[0].focus == FOCUS
    wire = sent[0].model_dump(mode="json", exclude_none=True)
    assert CuriosityTurnRequestV1.model_validate(wire).focus == FOCUS


def test_a_run_without_a_focus_sends_no_focus_key() -> None:
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](_state(None)))
    assert sent[0].focus is None
    assert "focus" not in sent[0].model_dump(mode="json")
    assert "focus" not in json.loads(sent[0].model_dump_json())


def test_unset_focus_is_absent_from_every_brief_dump() -> None:
    brief = CuriosityRunBriefV1(prompt="p", session_id="x", timeout_sec=1.0)
    assert "focus" not in brief.model_dump(mode="json")
    assert "focus" not in json.loads(brief.model_dump_json())
    request = DurableRunRequestV1(run_id=RUN_ID, workflow="curiosity.investigate", correlation_id="c", brief=brief)
    assert "focus" not in request.model_dump(mode="json")["brief"]


def test_a_set_focus_survives_the_durable_request_round_trip() -> None:
    brief = CuriosityRunBriefV1(prompt="p", session_id="x", timeout_sec=1.0, focus=FOCUS)
    request = DurableRunRequestV1(run_id=RUN_ID, workflow="curiosity.investigate", correlation_id="c", brief=brief)
    again = DurableRunRequestV1.model_validate(request.model_dump(mode="json", exclude_none=True))
    assert again.brief.focus == FOCUS


class _OldHubTurnRequestV1(BaseModel):
    """CuriosityTurnRequestV1 as a Hub from before this field knows it (forbid)."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["curiosity.turn.request.v1"] = "curiosity.turn.request.v1"
    run_id: str
    correlation_id: str
    prompt: str
    fcc_model_label: str | None = None
    timeout_sec: float
    source_tag: str = "curiosity_investigation"
    attempt: int = 1
    assigned_lane: str | None = None
    session_id: str | None = None
    retrieval_query: str | None = None


def test_new_runner_turn_without_focus_validates_on_an_old_hub() -> None:
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](_state(None)))
    _OldHubTurnRequestV1.model_validate(sent[0].model_dump(mode="json", exclude_none=True))
