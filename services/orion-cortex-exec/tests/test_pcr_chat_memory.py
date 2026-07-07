from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.pcr_chat_memory import run_pcr_phase0_and_1
from app.settings import settings
from orion.core.bus.bus_schemas import ServiceRef
from orion.schemas.cortex.schemas import StepExecutionResult


@pytest.fixture(autouse=True)
def _enable_pcr(monkeypatch):
    monkeypatch.setattr(settings, "chat_pcr_enabled", True)
    monkeypatch.setattr(settings, "chat_pcr_skip_on_low_info", True)


def test_phase0_skips_recall_on_greeting(monkeypatch):
    recall_calls: list[dict] = []

    async def _fake_recall_step(*args, **kwargs):
        recall_calls.append(kwargs)
        return (
            StepExecutionResult(
                status="success",
                verb_name="chat_general",
                step_name="pcr_continuity_recall",
                order=-1,
                result={"RecallService": {"count": 1}},
                latency_ms=1,
                node="n",
                logs=[],
            ),
            {"count": 1},
            "continuity text",
        )

    monkeypatch.setattr("app.pcr_chat_memory.run_recall_step", _fake_recall_step)

    ctx = {
        "verb": "chat_general",
        "messages": [{"role": "user", "content": "hey Orion"}],
        "turn_change_appraisal": {"novelty_score": 0.1, "shift_kind": "NONE"},
    }

    pcr, recall_step, recall_debug = asyncio.run(
        run_pcr_phase0_and_1(
            object(),
            source=ServiceRef(name="x", version="0", node="n"),
            ctx=ctx,
            correlation_id="corr-greeting",
            recall_cfg={"enabled": True, "profile": "chat.general.v1"},
        )
    )

    assert len(recall_calls) == 0
    assert recall_step is None
    assert pcr.phase == "skip"
    assert pcr.retrieval_intent == "none"
    assert pcr.continuity_digest == ""
    assert pcr.belief_digest == ""
    assert "low_info_social" in pcr.skip_reasons
    assert ctx["continuity_digest"] == ""
    assert ctx["memory_digest"] == ""
    assert ctx["pcr_memory"] is pcr
    assert recall_debug.get("pcr_phase") == "skip"
