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
    state["brief"].pop("retrieval_query")
    sent: list[CuriosityTurnRequestV1] = []
    asyncio.run(make_nodes(_deps(sent))["harness_turn"](state))
    assert sent[0].retrieval_query is None
